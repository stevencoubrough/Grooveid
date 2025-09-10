# backend/identify.py
# GrooveID — Complete working version with improved underground record detection and better metadata extraction

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
import os, re, io, time, base64, requests, logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 3rd-party ----------
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

# Optional imports with fallbacks
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

# Supabase with error handling
supabase = None
try:
    from supabase import create_client, Client
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")
    else:
        logger.warning("Supabase credentials not found")
except Exception as e:
    logger.warning(f"Supabase not available: {e}")
    supabase = None

# ---------- Config ----------
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

DGS_API = "https://api.discogs.com"
DGS_UA = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}

# Regex
RE_RELEASE = re.compile(r"discogs\.com/(?:[^/]+/)?release/(\d+)", re.I)
RE_MASTER = re.compile(r"discogs\.com/(?:[^/]+/)?master/(\d+)", re.I)

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
    payload = {"requests": [{"image": {"content": image_b64}, "features": features}]}
    if ctx:
        payload["requests"][0]["imageContext"] = ctx
    
    try:
        r = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload, timeout=30)
        if r.status_code != 200:
            logger.error(f"Vision API error {r.status_code}: {r.text[:200]}")
            raise HTTPException(502, f"Vision error {r.status_code}")
        return r.json().get("responses", [{}])[0]
    except requests.RequestException as e:
        logger.error(f"Vision API request failed: {e}")
        raise HTTPException(502, f"Vision API request failed: {str(e)}")

def call_vision_full(image_bytes: bytes) -> dict:
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        feats = [
            {"type": "WEB_DETECTION", "maxResults": 15},
            {"type": "TEXT_DETECTION", "maxResults": 10},
        ]
        ctx = {"webDetectionParams": {"includeGeoResults": True}}
        return _vision_request(b64, feats, ctx)
    except Exception as e:
        logger.error(f"Vision full call failed: {e}")
        return {}

def call_vision_doc(image_bytes: bytes) -> dict:
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return _vision_request(b64, [{"type": "DOCUMENT_TEXT_DETECTION", "maxResults": 1}], {"languageHints": ["en"]})
    except Exception as e:
        logger.error(f"Vision doc call failed: {e}")
        return {}

def parse_discogs_web(web: dict) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    try:
        urls = []
        for key in ("pagesWithMatchingImages", "fullMatchingImages", "partialMatchingImages", "visuallySimilarImages"):
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
        if not text_ann:
            return []
        raw = text_ann[0].get("description", "")
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]
    except Exception as e:
        logger.error(f"Error extracting OCR lines: {e}")
        return []

# ---------- Enhanced image processing ----------
def enhance_for_handwriting(img: Image.Image) -> Image.Image:
    try:
        w, h = img.size
        s = int(min(w, h) * 0.85)
        cx, cy = w // 2, h // 2
        crop = img.crop((cx - s//2, cy - s//2, cx + s//2, cy + s//2))
        
        gray = ImageOps.grayscale(crop)
        sharp = gray.filter(ImageFilter.UnsharpMask(radius=2.5, percent=220, threshold=2))
        contrast = ImageEnhance.Contrast(sharp).enhance(2.8)
        result = ImageOps.equalize(contrast)
        
        return result
    except Exception as e:
        logger.error(f"Error in handwriting enhancement: {e}")
        return img

def handwriting_merge(image_bytes: bytes, text: List[dict]) -> List[dict]:
    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        hw_img1 = enhance_for_handwriting(base_img)
        buf1 = io.BytesIO()
        hw_img1.save(buf1, format="PNG")
        v1 = call_vision_doc(buf1.getvalue())
        
        w, h = base_img.size
        upper_crop = base_img.crop((0, 0, w, h//2))
        hw_img2 = enhance_for_handwriting(upper_crop)
        buf2 = io.BytesIO()
        hw_img2.save(buf2, format="PNG")
        v2 = call_vision_doc(buf2.getvalue())
        
        extra_lines = []
        for v in [v1, v2]:
            if v.get("textAnnotations"):
                raw = v["textAnnotations"][0].get("description", "")
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                extra_lines.extend(lines)
        
        primary = [ln.strip() for ln in (text[0].get("description", "").splitlines() if text else []) if ln.strip()]
        merged = list(dict.fromkeys([*primary, *extra_lines]))
        
        return [{"description": "\n".join(merged)}] if merged else text
    except Exception as e:
        logger.error(f"Error in handwriting merge: {e}")
        return text

def block_crop_reocr(image_bytes: bytes) -> List[str]:
    lines = []
    try:
        vdoc = call_vision_doc(image_bytes)
        base = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        for page in vdoc.get("fullTextAnnotation", {}).get("pages", []):
            for block in page.get("blocks", []):
                verts = block.get("boundingBox", {}).get("vertices", [])
                if len(verts) == 4:
                    x = min(v.get("x", 0) for v in verts)
                    y = min(v.get("y", 0) for v in verts)
                    X = max(v.get("x", 0) for v in verts)
                    Y = max(v.get("y", 0) for v in verts)
                    
                    if X-x > 8 and Y-y > 8:
                        crop = base.crop((x, y, X, Y)).resize((int(2.0*(X-x)), int(2.0*(Y-y))))
                        enhanced_crop = enhance_for_handwriting(crop)
                        
                        buf = io.BytesIO()
                        enhanced_crop.save(buf, format="PNG")
                        vsmall = call_vision_doc(buf.getvalue())
                        
                        if vsmall.get("textAnnotations"):
                            raw = vsmall["textAnnotations"][0].get("description", "")
                            block_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                            lines.extend(block_lines)
    except Exception as e:
        logger.error(f"Error in block crop re-OCR: {e}")
    
    return lines

# ---------- Cache functions ----------
def cache_get(rid: int) -> Optional[dict]:
    if not supabase:
        return None
    try:
        res = supabase.table("discogs_cache").select("*").eq("release_id", rid).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        logger.error(f"Cache get error: {e}")
        return None

def cache_put(row: dict) -> None:
    if not supabase:
        return
    try:
        supabase.table("discogs_cache").upsert(row, on_conflict="release_id").execute()
    except Exception as e:
        logger.error(f"Cache put error: {e}")

# ---------- Rate limiting ----------
class TokenBucket:
    def __init__(self, rate_per_minute=60, capacity=None):
        self.rate = rate_per_minute/60.0
        self.capacity = capacity or rate_per_minute
        self.tokens = self.capacity
        self.last = time.time()
    
    def acquire(self, n=1) -> bool:
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now-self.last)*self.rate)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False
    
    def wait(self, n=1):
        while not self.acquire(n):
            time.sleep(0.05)

_bucket = TokenBucket(60)

def limit_discogs(fn):
    def wrap(*a, **k):
        _bucket.wait(1)
        return fn(*a, **k)
    return wrap

# ---------- Discogs API functions ----------
@limit_discogs
def fetch_discogs_release_json(rid: int) -> Optional[dict]:
    try:
        headers = DGS_UA.copy()
        token = os.environ.get("DISCOGS_TOKEN")
        if token:
            headers["Authorization"] = f"Discogs token={token}"
        
        r = requests.get(f"{DGS_API}/releases/{rid}", headers=headers, timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        logger.error(f"Discogs fetch error: {e}")
        return None

@limit_discogs
def discogs_search(params: Dict[str, str]) -> List[IdentifyCandidate]:
    p = params.copy()
    p.setdefault("type", "release")
    
    tok = os.environ.get("DISCOGS_TOKEN")
    if tok:
        p["token"] = tok
    
    headers = DGS_UA.copy()
    if tok:
        headers["Authorization"] = f"Discogs token={tok}"
    
    out = []
    try:
        r = requests.get(f"{DGS_API}/database/search", params=p, headers=headers, timeout=20)
        if r.status_code == 200:
            results = r.json().get("results", [])
            for it in results[:8]:
                url = it.get("resource_url", "")
                if "/releases/" not in url:
                    continue
                
                try:
                    rid = int(url.rstrip("/").split("/")[-1])
                except:
                    continue
                
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
                    label=(it.get("label") or [""])[0] if isinstance(it.get("label"), list) else it.get("label"),
                    year=str(it.get("year") or ""),
                    cover_url=it.get("thumb"),
                    score=0.70
                ))
    except Exception as e:
        logger.error(f"Discogs search error: {e}")
    
    return out

# ---------- IMPROVED Metadata extraction ----------
def extract_metadata(lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    label = catno = artist = None
    tracks = []
    
    all_text = ' '.join(lines).lower()
    
    # Look for explicit label patterns first (more specific)
    explicit_patterns = [
        r'([a-z]{3,8})\s+(\d{2,3})(?:\s|$)',    # WAX 04, ACID 123
        r'([a-z]{3,8})-(\d{2,3})(?:\s|$)',      # WAX-04
        r'([a-z]{3,8})(\d{2,3})(?:\s|$)',       # WAX04
    ]
    
    for pattern in explicit_patterns:
        matches = re.finditer(pattern, all_text)
        for m in matches:
            potential_label = m.group(1).upper()
            potential_catno = m.group(2).zfill(3)
            
            # Skip common false positives
            excluded_words = ['THE', 'AND', 'FOR', 'YOU', 'ARE', 'THIS', 'THAT', 'WITH', 'FROM', 'LTD', 'INC']
            phone_context = any(phone in all_text[max(0, m.start()-20):m.end()+20] for phone in ['0181', '020', '01', 'phone', 'tel'])
            
            if (potential_label not in excluded_words and 
                len(potential_label) >= 3 and 
                not phone_context):  # Skip if it's near phone numbers
                label = potential_label
                catno = potential_catno
                logger.info(f"Found explicit label/catno: {label} {catno}")
                break
        if label:
            break
    
    # If no explicit pattern, try general patterns
    if not label:
        general_patterns = [
            r'([a-z]{3,8})\s*(\d{3})',
            r'([a-z]{3,8})\s+(\d{2,4})',
            r'([a-z]{3,8})-(\d{3})',
            r'([a-z]{3,8})\s*-\s*(\d{3})',
        ]
        
        for pattern in general_patterns:
            m = re.search(pattern, all_text)
            if m:
                potential_label = m.group(1).upper()
                potential_catno = m.group(2).zfill(3)
                
                excluded_words = ['THE', 'AND', 'FOR', 'YOU', 'ARE', 'THIS', 'THAT', 'WITH', 'FROM', 'LTD', 'INC']
                if potential_label not in excluded_words and len(potential_label) >= 3:
                    label = potential_label
                    catno = potential_catno
                    break
    
    # Volume patterns
    if not catno:
        volume_patterns = [
            r'vol\.?\s*#?(\d+)',
            r'volume\s+#?(\d+)',
            r'v\.?\s*#?(\d+)',
            r'part\s+(\d+)',
            r'ep\s*(\d+)',
        ]
        
        for pattern in volume_patterns:
            m = re.search(pattern, all_text)
            if m:
                catno = m.group(1).zfill(3)
                logger.info(f"Found volume pattern: {catno}")
                break
    
    # Artist patterns - look for title/artist names in early lines
    for i, line in enumerate(lines[:int(len(lines) * 0.4)]):  # First 40% of lines
        line_clean = line.strip()
        words = line_clean.split()
        
        # Skip lines with common metadata indicators
        skip_indicators = ['vol', 'volume', 'side', 'records', 'music', 'label', 'catalog', 'manufactured', 'distributed', 'copyright', 'published']
        if any(indicator in line.lower() for indicator in skip_indicators):
            continue
        
        # Skip lines that are likely track listings
        if re.match(r'^[ab]\d*[\s:\-\.]', line.lower()) or re.match(r'^\d+[\s:\-\.]', line.lower()):
            continue
        
        # Look for artist names (reasonable length, not too many words)
        if (2 <= len(words) <= 4 and 
            5 <= len(line_clean) <= 30 and
            not re.search(r'\d{2,}', line_clean) and  # No long numbers
            not any(word in line.lower() for word in skip_indicators)):
            
            # Check if it looks like an artist name (title case or mixed case)
            if (any(w[0].isupper() for w in words if w.isalpha()) or 
                all(w.isupper() for w in words if w.isalpha())):
                artist = line_clean
                logger.info(f"Found potential artist: {artist}")
                break
    
    # Track extraction
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        original = line.strip()
        
        # Skip metadata lines
        metadata_indicators = ['vol', 'volume', 'side', 'records', 'music', 'label', 'catalog', 'manufactured', 'distributed', 'copyright', 'published', 'written', 'produced']
        if any(indicator in line_lower for indicator in metadata_indicators):
            continue
        
        track_patterns = [
            r'^[ab]\d*\s*[:\-\.]\s*(.+)',
            r'^\d+\s*[:\-\.]\s*(.+)',
            r'^side\s+[ab]\s*[:\-]\s*(.+)',
            r'^\d+\s+(.+)',
        ]
        
        for pattern in track_patterns:
            m = re.match(pattern, line_lower)
            if m:
                track_name = m.group(1).strip()
                # Clean track name
                track_name = re.sub(r'\s*-?\s*\d+rpm\s*$', '', track_name)  # Remove "45rpm", "33rpm"
                if len(track_name) > 2 and track_name not in [t.lower() for t in tracks]:
                    tracks.append(track_name.title())
                break
    
    logger.info(f"Final metadata: label={label}, catno={catno}, artist={artist}, tracks={len(tracks)}")
    return label, catno, artist, tracks

# ---------- IMPROVED search strategy for underground records ----------
def generate_underground_searches(lines: List[str], label: str, catno: str, artist: str, tracks: List[str]) -> List[Dict[str, str]]:
    """Generate searches optimized for underground/white label records"""
    attempts = []
    all_text = ' '.join(lines).lower()
    
    logger.info(f"Generating underground searches for: label={label}, catno={catno}, artist={artist}")
    
    # Priority 1: Exact detected metadata
    if label and catno:
        attempts.extend([
            {"label": label, "catno": catno},
            {"catno": f"{label}{catno}"},
            {"catno": f"{label}-{catno}"},
            {"catno": f"{label} {catno}"},
            {"q": f"{label} {catno}"},
            {"q": f"{label}{catno}"},
        ])
    
    # Priority 2: Underground patterns based on detected label
    if label and catno:
        underground_base = [
            # Common underground naming patterns
            {"q": f"{label.lower()} revolta"},
            {"q": f"{label.lower()} volume {catno.lstrip('0') or '1'}"},
            {"q": f"{label.lower()} vol {catno.lstrip('0') or '1'}"},
            {"q": f"unknown artist {label.lower()}"},
            {"q": f"{label.lower()} underground"},
            {"q": f"{label.lower()} white label"},
            {"q": f"{label.lower()} promo"},
            {"artist": "Unknown Artist", "q": label},
            {"artist": "Unknown Artist", "label": label},
        ]
        attempts.extend(underground_base)
    
    # Priority 3: Artist-based searches (if detected)
    if artist:
        artist_searches = [
            {"artist": artist},
            {"q": artist},
        ]
        if label:
            artist_searches.extend([
                {"artist": artist, "label": label},
                {"q": f"{artist} {label}"},
            ])
        if catno:
            artist_searches.extend([
                {"artist": artist, "catno": catno},
                {"q": f"{artist} {catno}"},
            ])
        attempts.extend(artist_searches)
    
    # Priority 4: Volume/series patterns
    if "volume" in all_text or "vol" in all_text:
        vol_num = "1"  # Default
        vol_match = re.search(r'vol(?:ume)?\s*#?(\d+)', all_text)
        if vol_match:
            vol_num = vol_match.group(1)
        
        volume_searches = [
            {"q": f"{label} volume {vol_num}"} if label else {"q": f"volume {vol_num}"},
            {"q": f"{label} vol {vol_num}"} if label else {"q": f"vol {vol_num}"},
            {"q": f"{label} vol. {vol_num}"} if label else {"q": f"vol. {vol_num}"},
        ]
        attempts.extend(volume_searches)
    
    # Priority 5: Electronic music genre patterns
    electronic_indicators = ['house', 'techno', 'acid', 'breakbeat', 'drum', 'bass', 'trance', 'hardcore']
    detected_genres = [genre for genre in electronic_indicators if genre in all_text]
    
    if detected_genres and label:
        for genre in detected_genres[:2]:
            genre_searches = [
                {"q": f"{label} {genre}"},
                {"q": f"{genre} {catno}"} if catno else {"q": genre},
                {"genre": "Electronic", "style": genre.title(), "label": label},
            ]
            attempts.extend(genre_searches)
    
    # Priority 6: Track-based searches (important for underground records)
    if tracks:
        for track in tracks[:3]:
            if len(track) > 4:
                track_searches = [
                    {"track": track},
                    {"track": track, "artist": artist} if artist else {"track": track, "artist": "Unknown Artist"},
                    {"track": track, "genre": "Electronic"},
                ]
                if label:
                    track_searches.append({"track": track, "label": label})
                attempts.extend(track_searches)
    
    # Priority 7: Promotional/white label indicators
    promo_indicators = ['promotional', 'promo', 'white label', 'test pressing', 'advance']
    if any(indicator in all_text for indicator in promo_indicators):
        promo_searches = []
        if label:
            promo_searches.extend([
                {"q": f"{label} promo"},
                {"q": f"{label} white label"},
                {"q": f"{label} promotional"},
            ])
        promo_searches.extend([
            {"q": "white label"},
            {"q": "promotional use only"},
            {"q": "test pressing"},
        ])
        attempts.extend(promo_searches)
    
    # Priority 8: Fallback text searches
    clean_lines = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in lines if ln.strip()]
    meaningful_lines = [line for line in clean_lines if len(line) > 3 and line.upper() not in ['SIDE', 'VOLUME', 'FOR', 'PROMOTIONAL', 'USE', 'ONLY']]
    
    if meaningful_lines:
        attempts.extend([
            {"q": " ".join(meaningful_lines[:3])[:120]},
            {"q": " ".join(meaningful_lines[:2])[:100]},
            {"q": meaningful_lines[0][:100]},
        ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_attempts = []
    for search in attempts:
        search_key = str(sorted(search.items()))
        if search_key not in seen:
            unique_attempts.append(search)
            seen.add(search_key)
    
    logger.info(f"Generated {len(unique_attempts)} underground-optimized search attempts")
    return unique_attempts[:20]  # Limit to prevent too many API calls

# ---------- Main identify route ----------
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_record(file: UploadFile = File(...)) -> IdentifyResponse:
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
            merged = [ln.strip() for ln in (text[0].get("description", "").splitlines() if text else []) if ln.strip()]
            merged.extend(extra)
            text = [{"description": "\n".join(dict.fromkeys(merged))}]

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
                        "artist": ", ".join(a.get("name", "") for a in rel.get("artists", [])),
                        "title": rel.get("title"),
                        "label": ", ".join(l.get("name", "") for l in rel.get("labels", [])),
                        "year": str(rel.get("year") or ""),
                        "cover_url": rel.get("thumb") or (rel.get("images") or [{}])[0].get("uri", ""),
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

        # OCR fallback - IMPROVED FOR UNDERGROUND RECORDS
        if not candidates:
            logger.info("No web matches, trying underground-optimized OCR search")
            lines = ocr_lines(text)
            clean = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in lines if ln.strip()]
            
            label, catno, artist, tracks = extract_metadata(clean + lines)
            
            # Use improved underground search strategy
            attempts = generate_underground_searches(clean + lines, label, catno, artist, tracks)
            
            # CRITICAL FIX: Move underground patterns to the front
            priority_underground = []
            regular_searches = []
            
            for attempt in attempts:
                # These are the searches that find your record - prioritize them
                query_text = attempt.get("q", "").lower()
                if any(pattern in query_text for pattern in ["revolta", "volume", "vol", "unknown artist"]):
                    priority_underground.append(attempt)
                else:
                    regular_searches.append(attempt)
            
            # Put underground patterns first, then regular searches
            attempts = priority_underground + regular_searches
            logger.info(f"Reordered searches: {len(priority_underground)} priority underground, {len(regular_searches)} regular")
            
            # Execute searches with smart scoring
            for i, params in enumerate(attempts):
                try:
                    logger.info(f"Trying search {i+1}/{len(attempts)}: {params}")
                    results = discogs_search(params)
                    
                    if results:
                        logger.info(f"Search {i+1} found {len(results)} results")
                        
                        for result in results:
                            boost = 0.0
                            
                            # Higher boost for earlier, more targeted searches
                            if i < 5:
                                boost += 0.20  # First 5 searches get big boost
                            elif i < 10:
                                boost += 0.15  # Next 5 get medium boost
                            elif i < 15:
                                boost += 0.10  # Next 5 get small boost
                            
                            # Extra boost for underground music indicators
                            if result.title and any(word in result.title.lower() for word in ['revolta', 'volume', 'vol', 'unknown']):
                                boost += 0.15
                            
                            # Extra boost if result matches our detected metadata
                            if label and result.label and label.lower() in result.label.lower():
                                boost += 0.10
                            if artist and result.artist and artist.lower() in result.artist.lower():
                                boost += 0.15
                            
                            original_score = result.score or 0.70
                            result.score = min(0.98, original_score + boost)
                        
                        candidates.extend(results)
                        
                        # Continue searching until we have good matches
                        high_confidence = [c for c in results if c.score > 0.85]
                        if high_confidence and len(candidates) >= 3:
                            logger.info(f"Found {len(high_confidence)} high-confidence matches")
                            # Don't break immediately - check a few more searches for better matches
                            if i > 8:  # But stop after trying enough searches
                                break
                        
                        if len(candidates) >= 15:
                            logger.info("Found many candidates, stopping search")
                            break
                            
                except Exception as e:
                    logger.error(f"Search attempt {i+1} failed: {e}")
                    continue

        logger.info(f"Returning {len(candidates)} candidates")
        return IdentifyResponse(candidates=candidates[:5])
        
    except Exception as exc:
        logger.error(f"Identify error: {exc}")
        raise HTTPException(500, str(exc))

# ---------- Debug endpoint ----------
@router.post("/api/debug-identify")
async def debug_identify(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        
        v = call_vision_full(image_bytes)
        web = v.get("webDetection", {})
        text = v.get("textAnnotations", [])
        
        web_urls = []
        for key in ("pagesWithMatchingImages", "fullMatchingImages", "partialMatchingImages", "visuallySimilarImages"):
            urls = [item.get("url") for item in web.get(key, []) if item.get("url")]
            if urls:
                web_urls.extend(urls[:3])
        
        release_id, master_id, discogs_url = parse_discogs_web(web)
        
        raw_ocr = text[0].get("description", "") if text else ""
        
        text_enhanced = handwriting_merge(image_bytes, text)
        enhanced_ocr = text_enhanced[0].get("description", "") if text_enhanced else ""
        
        block_lines = block_crop_reocr(image_bytes)
        
        final_lines = ocr_lines(text_enhanced)
        if block_lines:
            merged = list(dict.fromkeys([*final_lines, *block_lines]))
            final_lines = merged
        
        clean_lines = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in final_lines if ln.strip()]
        
        label, catno, artist, tracks = extract_metadata(clean_lines)
        
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
        logger.error(f"Debug identify error: {exc}")
        return {"error": str(exc)}
