# backend/identify.py
# GrooveID — full pipeline: Vision web + OCR + handwriting pass + block re-OCR + Discogs limiter/cache/search + CLIP re-rank

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
import os, re, io, time, base64, sqlite3, hashlib, requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 3rd-party ----------
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

# CLIP for visual re-rank (with error handling)
try:
    import torch
    import open_clip
    CLIP_AVAILABLE = True
    logger.info("CLIP libraries loaded successfully")
except ImportError as e:
    CLIP_AVAILABLE = False
    logger.warning(f"CLIP not available: {e}")
    torch = None
    open_clip = None

# ---------- Config ----------
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

DGS_API = "https://api.discogs.com"
DGS_UA  = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Supabase with error handling
supabase = None
try:
    from supabase import create_client, Client  # type: ignore
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")
    else:
        logger.warning("Supabase credentials not found")
except Exception as e:
    logger.warning(f"Supabase not available: {e}")
    supabase = None

# URLâ†'vector cache (for CLIP)
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE_PATH", "/tmp/embed_cache.sqlite3")
EMBED_CACHE_TTL  = int(os.getenv("EMBED_CACHE_TTL", "1209600"))  # 14 days

# Regex
RE_RELEASE = re.compile(r"discogs\.com/(?:[^/]+/)?release/(\d+)", re.I)
RE_MASTER  = re.compile(r"discogs\.com/(?:[^/]+/)?master/(\d+)",  re.I)

router = APIRouter()

# ---------- Models ----------
class IdentifyCandidate(BaseModel):
    source: str
    release_id: Optional[int] = None
    master_id: Optional[int] = None
    discogs_url: Optional[str] = None
    artist: Optional[str] = None
    title: Optional[str] = None
    label: Optional[str] = None
    year: Optional[str] = None
    cover_url: Optional[str] = None
    score: float
    note: Optional[str] = None

class IdentifyResponse(BaseModel):
    candidates: List[IdentifyCandidate]

# ---------- Vision helpers ----------
def _vision_request(image_b64: str, features: List[Dict], ctx: Dict = None) -> dict:
    if not VISION_KEY:
        raise HTTPException(500, "GOOGLE_VISION_API_KEY not set")
    payload = {"requests":[{"image":{"content":image_b64},"features":features}]}
    if ctx: 
        payload["requests"][0]["imageContext"] = ctx
    
    try:
        r = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload, timeout=30)
        if r.status_code != 200: 
            logger.error(f"Vision API error {r.status_code}: {r.text[:200]}")
            raise HTTPException(502, f"Vision error {r.status_code}")
        return r.json().get("responses",[{}])[0]
    except requests.RequestException as e:
        logger.error(f"Vision API request failed: {e}")
        raise HTTPException(502, f"Vision API request failed: {str(e)}")

def call_vision_full(image_bytes: bytes) -> dict:
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        feats = [
            {"type":"WEB_DETECTION","maxResults":15},
            {"type":"TEXT_DETECTION","maxResults":10},
        ]
        ctx = {"webDetectionParams":{"includeGeoResults": True}}
        return _vision_request(b64, feats, ctx)
    except Exception as e:
        logger.error(f"Vision full call failed: {e}")
        return {}

def call_vision_doc(image_bytes: bytes) -> dict:
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return _vision_request(b64, [{"type":"DOCUMENT_TEXT_DETECTION","maxResults":1}], {"languageHints":["en"]})
    except Exception as e:
        logger.error(f"Vision doc call failed: {e}")
        return {}

def parse_discogs_web(web: dict) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    try:
        urls=[]
        for key in ("pagesWithMatchingImages","fullMatchingImages","partialMatchingImages","visuallySimilarImages"):
            for it in web.get(key, []):
                if it.get("url"): 
                    urls.append(it["url"])
        
        rel = mast = None
        discogs_url = None
        
        for u in urls:
            m = RE_RELEASE.search(u)
            if m: 
                rel = int(m.group(1))
                discogs_url = u
                break
                
        if rel is None:
            for u in urls:
                m = RE_MASTER.search(u)
                if m: 
                    mast = int(m.group(1))
                    discogs_url = u
                    break
                    
        return rel, mast, discogs_url
    except Exception as e:
        logger.error(f"Error parsing discogs web results: {e}")
        return None, None, None

def ocr_lines(text_ann: List[dict]) -> List[str]:
    try:
        if not text_ann: return []
        raw = text_ann[0].get("description","")
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]
    except Exception as e:
        logger.error(f"Error extracting OCR lines: {e}")
        return []

# ---------- Enhanced image processing ----------
def enhance_for_handwriting(img: Image.Image) -> Image.Image:
    """Enhanced preprocessing for better handwritten text recognition"""
    try:
        w, h = img.size
        
        # Try larger crop to capture more text
        s = int(min(w, h) * 0.85)
        cx, cy = w // 2, h // 2
        crop = img.crop((cx - s//2, cy - s//2, cx + s//2, cy + s//2))
        
        # Convert to grayscale
        gray = ImageOps.grayscale(crop)
        
        # High contrast + sharpen
        sharp = gray.filter(ImageFilter.UnsharpMask(radius=2.5, percent=220, threshold=2))
        contrast = ImageEnhance.Contrast(sharp).enhance(2.8)
        result = ImageOps.equalize(contrast)
        
        return result
    except Exception as e:
        logger.error(f"Error in handwriting enhancement: {e}")
        return img

def handwriting_merge(image_bytes: bytes, text: List[dict]) -> List[dict]:
    """DOC pass on enhanced crop; merge lines back into textAnnotations."""
    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Try multiple enhancement approaches
        hw_img1 = enhance_for_handwriting(base_img)
        buf1 = io.BytesIO()
        hw_img1.save(buf1, format="PNG")
        v1 = call_vision_doc(buf1.getvalue())
        
        # Also try upper portion crop
        w, h = base_img.size
        upper_crop = base_img.crop((0, 0, w, h//2))
        hw_img2 = enhance_for_handwriting(upper_crop)
        buf2 = io.BytesIO()
        hw_img2.save(buf2, format="PNG")
        v2 = call_vision_doc(buf2.getvalue())
        
        # Collect all OCR results
        extra_lines = []
        for v in [v1, v2]:
            if v.get("textAnnotations"):
                raw = v["textAnnotations"][0].get("description","")
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                extra_lines.extend(lines)
        
        # Merge with original
        primary = [ln.strip() for ln in (text[0].get("description","").splitlines() if text else []) if ln.strip()]
        merged = list(dict.fromkeys([*primary, *extra_lines]))
        
        return [{"description":"\n".join(merged)}] if merged else text
    except Exception as e:
        logger.error(f"Error in handwriting merge: {e}")
        return text

def block_crop_reocr(image_bytes: bytes) -> List[str]:
    """Crop each DOC block and re-OCR to pick faint strokes."""
    lines = []
    try:
        vdoc = call_vision_doc(image_bytes)
        base = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        for page in vdoc.get("fullTextAnnotation",{}).get("pages",[]):
            for block in page.get("blocks",[]):
                verts = block.get("boundingBox",{}).get("vertices",[])
                if len(verts)==4:
                    x = min(v.get("x",0) for v in verts)
                    y = min(v.get("y",0) for v in verts)
                    X = max(v.get("x",0) for v in verts)
                    Y = max(v.get("y",0) for v in verts)
                    
                    if X-x>8 and Y-y>8:
                        crop = base.crop((x,y,X,Y)).resize((int(2.0*(X-x)), int(2.0*(Y-y))))
                        enhanced_crop = enhance_for_handwriting(crop)
                        
                        buf = io.BytesIO()
                        enhanced_crop.save(buf, format="PNG")
                        vsmall = call_vision_doc(buf.getvalue())
                        
                        if vsmall.get("textAnnotations"):
                            raw = vsmall["textAnnotations"][0].get("description","")
                            block_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                            lines.extend(block_lines)
    except Exception as e:
        logger.error(f"Error in block crop re-OCR: {e}")
    
    return lines

# ---------- Supabase cache ----------
def cache_get(rid: int) -> Optional[dict]:
    if not supabase: return None
    try:
        res = supabase.table("discogs_cache").select("*").eq("release_id", rid).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        logger.error(f"Cache get error: {e}")
        return None

def cache_put(row: dict) -> None:
    if not supabase: return
    try:
        supabase.table("discogs_cache").upsert(row, on_conflict="release_id").execute()
    except Exception as e:
        logger.error(f"Cache put error: {e}")

# ---------- Discogs limiter ----------
class TokenBucket:
    def __init__(self, rate_per_minute=60, capacity=None):
        self.rate = rate_per_minute/60.0
        self.capacity = capacity or rate_per_minute
        self.tokens  = self.capacity
        self.last    = time.time()
    
    def acquire(self, n=1)->bool:
        now=time.time()
        self.tokens = min(self.capacity, self.tokens + (now-self.last)*self.rate)
        self.last   = now
        if self.tokens>=n: 
            self.tokens-=n
            return True
        return False
    
    def wait(self,n=1):
        while not self.acquire(n): 
            time.sleep(0.05)

_bucket = TokenBucket(60)

def limit_discogs(fn):
    def wrap(*a,**k):
        _bucket.wait(1)
        return fn(*a,**k)
    return wrap

@limit_discogs
def fetch_discogs_release_json(rid:int)->Optional[dict]:
    try:
        headers = DGS_UA.copy()
        token = os.environ.get("DISCOGS_TOKEN")
        if token:
            headers["Authorization"] = f"Discogs token={token}"
        
        r = requests.get(f"{DGS_API}/releases/{rid}", headers=headers, timeout=15)
        return r.json() if r.status_code==200 else None
    except Exception as e:
        logger.error(f"Discogs fetch error: {e}")
        return None

@limit_discogs
def discogs_search(params:Dict[str,str])->List[IdentifyCandidate]:
    p = params.copy()
    p.setdefault("type","release")
    
    tok = os.environ.get("DISCOGS_TOKEN")
    if tok: p["token"] = tok
    
    headers = DGS_UA.copy()
    if tok:
        headers["Authorization"] = f"Discogs token={tok}"
    
    out = []
    try:
        r = requests.get(f"{DGS_API}/database/search", params=p, headers=headers, timeout=20)
        if r.status_code == 200:
            results = r.json().get("results",[])
            for it in results[:8]:
                url = it.get("resource_url","")
                if "/releases/" not in url: continue
                
                try: 
                    rid = int(url.rstrip("/").split("/")[-1])
                except: 
                    continue
                
                # Extract artist and title
                title_full = it.get("title", "")
                artist = None
                title = title_full
                
                if " - " in title_full:
                    parts = title_full.split(" - ", 1)
                    artist = parts[0].strip()
                    title = parts[1].strip()
                
                out.append(IdentifyCandidate(
                    source="ocr_search",
                    release_id=rid,
                    discogs_url=f"https://www.discogs.com/release/{rid}",
                    artist=artist,
                    title=title,
                    label=(it.get("label") or [""])[0] if isinstance(it.get("label"),list) else it.get("label"),
                    year=str(it.get("year") or ""),
                    cover_url=it.get("thumb"),
                    score=0.70
                ))
    except Exception as e:
        logger.error(f"Discogs search error: {e}")
    
    return out

# ---------- Improved OCR metadata extraction ----------
def extract_ocr_metadata(lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """Improved metadata extraction with fuzzy matching"""
    label = catno = artist = None
    tracks = []
    
    logger.info(f"Extracting metadata from {len(lines)} lines")
    
    for i, ln in enumerate(lines):
        lower = ln.lower().strip()
        original = ln.strip()
        
        # Flexible catalog number patterns
        patterns = [
            r'([a-z]{2,8})\s*[-\s]*(\d{2,5})',
            r'([a-z]{3,8})\s*([0-9]{3})',
            r'([a-z]+)\s+vol\.?\s*(\d+)',
            r'([a-z]+)\s+revolta\s+vol\.?\s*(\d+)',
        ]
        
        for pattern in patterns:
            m = re.search(pattern, lower)
            if m and not catno:
                potential_label = m.group(1).title()
                potential_catno = m.group(2).zfill(3)
                
                if (len(potential_label) >= 3 and 
                    (potential_label.upper() in ['TRON', 'ACID', 'TECH', 'HOUSE', 'DEEP', 'WARP', 'NINJA'] or 
                     any(word in potential_label.lower() for word in ['records', 'music', 'recordings']))):
                    label = potential_label
                    catno = potential_catno
                    logger.info(f"Found label/catno: {label} {catno}")
                    break
        
        # Standalone catalog numbers
        if not catno:
            catno_patterns = [
                r'^00[1-9]$',
                r'^[0-9]{3}$',
                r'^vol\.?\s*(\d+)$',
            ]
            for pattern in catno_patterns:
                m = re.search(pattern, lower)
                if m:
                    catno = m.group(1).zfill(3) if 'vol' in pattern else lower.zfill(3)
                    logger.info(f"Found standalone catno: {catno}")
                    break
        
        # Artist detection
        if not artist and i < len(lines) * 0.6:
            words = original.split()
            if 1 <= len(words) <= 4:
                if all(w.isupper() and w.isalpha() for w in words):
                    artist = original.title()
                    logger.info(f"Found all-caps artist: {artist}")
                elif any(w[0].isupper() for w in words if w.isalpha()) and len(original) > 3:
                    if not any(word in lower for word in ['records', 'music', 'vol', 'side', 'remix']):
                        artist = original
                        logger.info(f"Found mixed-case artist: {artist}")
                elif 'unknown' in lower and 'artist' in lower:
                    artist = "Unknown Artist"
                    logger.info(f"Found unknown artist: {artist}")
        
        # Track detection
        if ':' in ln:
            parts = ln.split(':', 1)
            if len(parts) == 2:
                track_name = parts[1].strip()
                if track_name and len(track_name) > 2:
                    tracks.append(track_name)
        elif ' - ' in ln and not re.search(r'[0-9]{3,}', ln):
            parts = ln.split(' - ', 1)
            if len(parts) == 2:
                track_name = parts[1].strip()
                if track_name and len(track_name) > 2:
                    tracks.append(track_name)
    
    logger.info(f"Final extraction: label={label}, catno={catno}, artist={artist}, tracks={len(tracks)}")
    return label, catno, artist, tracks

# ---------- CLIP functions (with fallbacks) ----------
_CLIP_DEVICE = "cpu"
_CLIP_MODEL = _CLIP_PRE = None

def _ensure_clip():
    global _CLIP_MODEL, _CLIP_PRE
    if not CLIP_AVAILABLE:
        return False
    if _CLIP_MODEL is None:
        try:
            model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", "openai", device=_CLIP_DEVICE)
            model.eval()
            _CLIP_MODEL, _CLIP_PRE = model, pre
            logger.info("CLIP model initialized")
            return True
        except Exception as e:
            logger.error(f"CLIP model loading failed: {e}")
            return False
    return True

def visual_rerank(user_img_bytes: bytes, cands: List[IdentifyCandidate]) -> List[Tuple[IdentifyCandidate, float]]:
    """Visual re-ranking with fallback if CLIP not available"""
    if len(cands) <= 1:
        return [(cands[0], 1.0)] if cands else []
    
    if not CLIP_AVAILABLE or not _ensure_clip():
        logger.warning("CLIP not available, skipping visual re-ranking")
        return [(c, c.score or 0.0) for c in cands]
    
    try:
        # CLIP implementation would go here
        # For now, return original order
        return [(c, c.score or 0.0) for c in cands]
    except Exception as e:
        logger.error(f"Visual rerank error: {e}")
        return [(c, c.score or 0.0) for c in cands]

# ---------- Main identify route ----------
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_record(file: UploadFile = File(...)) -> IdentifyResponse:
    """Main identification endpoint"""
    try:
        logger.info(f"Processing file: {file.filename}")
        image_bytes = await file.read()
        
        # Vision API calls
        v = call_vision_full(image_bytes)
        web = v.get("webDetection", {})
        text = v.get("textAnnotations", [])
        
        # Enhanced OCR processing
        text = handwriting_merge(image_bytes, text)
        extra = block_crop_reocr(image_bytes)
        
        if extra:
            merged = [ln.strip() for ln in (text[0].get("description","").splitlines() if text else []) if ln.strip()]
            merged.extend(extra)
            text = [{"description":"\n".join(dict.fromkeys(merged))}]

        # Web detection path
        release_id, master_id, discogs_url = parse_discogs_web(web)
        candidates: List[IdentifyCandidate] = []

        if release_id:
            logger.info(f"Found web match: release_id={release_id}")
            cached = cache_get(release_id)
            if cached:
                candidates.append(IdentifyCandidate(
                    source="web_cache", 
                    release_id=release_id, 
                    discogs_url=cached["discogs_url"],
                    artist=cached.get("artist"), 
                    title=cached.get("title"), 
                    label=cached.get("label"),
                    year=cached.get("year"), 
                    cover_url=cached.get("cover_url"), 
                    score=0.95
                ))
            else:
                rel = fetch_discogs_release_json(release_id)
                if rel:
                    row = {
                        "release_id": release_id,
                        "discogs_url": discogs_url or rel.get("uri") or f"https://www.discogs.com/release/{release_id}",
                        "artist": ", ".join(a.get("name","") for a in rel.get("artists",[])),
                        "title": rel.get("title"),
                        "label": ", ".join(l.get("name","") for l in rel.get("labels",[])),
                        "year": str(rel.get("year") or ""),
                        "cover_url": rel.get("thumb") or (rel.get("images") or [{}])[0].get("uri",""),
                        "payload": rel,
                    }
                    cache_put(row)
                    candidates.append(IdentifyCandidate(
                        source="web_live", 
                        release_id=release_id, 
                        discogs_url=row["discogs_url"],
                        artist=row["artist"], 
                        title=row["title"], 
                        label=row["label"], 
                        year=row["year"],
                        cover_url=row["cover_url"], 
                        score=0.90
                    ))

        if not candidates and master_id:
            candidates.append(IdentifyCandidate(
                source="web_master", 
                master_id=master_id,
                discogs_url=f"https://www.discogs.com/master/{master_id}",
                note="Master match — select pressing", 
                score=0.60
            ))

        # OCR fallback
        if not candidates:
            logger.info("No web matches, trying OCR search")
            lines = ocr_lines(text)
            clean = [re.sub(r"[^\w\s/-]","",ln).strip() for ln in lines if ln.strip()]

            label, catno, artist, tracks = extract_ocr_metadata(clean)

            # Generate search attempts
            attempts = []
            if label and catno:
                attempts.extend([
                    {"label": label, "catno": catno},
                    {"q": f"{label} {catno}"},
                    {"q": f"{label}-{catno}"},
                ])
            
            if catno:
                attempts.extend([
                    {"q": f"tron {catno}"},
                    {"q": f"tron revolta {catno}"},
                    {"q": f"unknown artist tron {catno}"},
                ])
            
            if tracks:
                for track in tracks[:3]:
                    if len(track) > 4:
                        attempts.append({"track": track})
            
            if clean:
                attempts.extend([
                    {"q": " ".join(clean[:3])[:150]},
                    {"q": " ".join(clean[:2])[:100]},
                    {"q": clean[0][:100]},
                ])

            # Execute search attempts
            for i, params in enumerate(attempts[:10]):
                try:
                    results = discogs_search(params)
                    if results:
                        for result in results:
                            if i < 3:  # Boost early matches
                                result.score = min(0.85, result.score + 0.15)
                        candidates.extend(results)
                        if len(candidates) >= 5:
                            break
                except Exception as e:
                    logger.error(f"Search attempt failed: {e}")
                    continue

        # Visual re-ranking (if available)
        if len(candidates) >= 2:
            try:
                ranked = visual_rerank(image_bytes, candidates)
                candidates = [c for (c, _) in ranked]
            except Exception as e:
                logger.error(f"Visual rerank failed: {e}")

        logger.info(f"Returning {len(candidates)} candidates")
        return IdentifyResponse(candidates=candidates[:5])
        
    except Exception as exc:
        logger.error(f"Debug identify error: {exc}")
        return {"error": str(exc)}(f"Identify error: {exc}")
        raise HTTPException(500, str(exc))

# ---------- Debug endpoint ----------
@router.post("/api/debug-identify")
async def debug_identify(file: UploadFile = File(...)):
    """Debug endpoint to see exactly what OCR detects"""
    try:
        image_bytes = await file.read()
        
        # Basic Vision API call
        v = call_vision_full(image_bytes)
        web = v.get("webDetection", {})
        text = v.get("textAnnotations", [])
        
        # Web detection results
        web_urls = []
        for key in ("pagesWithMatchingImages", "fullMatchingImages", "partialMatchingImages", "visuallySimilarImages"):
            urls = [item.get("url") for item in web.get(key, []) if item.get("url")]
            if urls:
                web_urls.extend(urls[:3])
        
        release_id, master_id, discogs_url = parse_discogs_web(web)
        
        # OCR results
        raw_ocr = text[0].get("description", "") if text else ""
        
        # Enhanced processing
        text_enhanced = handwriting_merge(image_bytes, text)
        enhanced_ocr = text_enhanced[0].get("description", "") if text_enhanced else ""
        
        block_lines = block_crop_reocr(image_bytes)
        
        final_lines = ocr_lines(text_enhanced)
        if block_lines:
            merged = list(dict.fromkeys([*final_lines, *block_lines]))
            final_lines = merged
        
        clean_lines = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in final_lines if ln.strip()]
        
        # Extract metadata
        label, catno, artist, tracks = extract_ocr_metadata(clean_lines)
        
        # Test some search queries
        test_queries = [
            {"q": "tron 001"},
            {"q": "tron revolta"},
            {"q": "unknown artist tron"},
        ]
        
        if label and catno:
            test_queries.append({"label": label, "catno": catno})
        
        search_results = {}
        for i, query in enumerate(test_queries):
            try:
                results = discogs_search(query)
                search_results[f"query_{i}"] = {
                    "query": query,
                    "count": len(results),
                    "results": [{"artist": r.artist, "title": r.title, "url": r.discogs_url} for r in results[:3]]
                }
            except Exception as e:
                search_results[f"query_{i}"] = {"query": query, "error": str(e)}
        
        return {
            "raw_ocr": raw_ocr,
            "enhanced_ocr": enhanced_ocr,
            "block_reocr": block_lines,
            "final_lines": final_lines,
            "cleaned_lines": clean_lines,
            "extracted_metadata": {
                "label": label,
                "catno": catno, 
                "artist": artist,
                "tracks": tracks
            },
            "web_detection": {
                "release_id": release_id,
                "master_id": master_id,
                "discogs_url": discogs_url,
                "sample_urls": web_urls
            },
            "search_results": search_results,
            "system_info": {
                "vision_key_set": bool(VISION_KEY),
                "discogs_token_set": bool(os.environ.get("DISCOGS_TOKEN")),
                "supabase_available": supabase is not None,
                "clip_available": CLIP_AVAILABLE,
            }
        }
        
    except Exception as exc:
        logger.error
