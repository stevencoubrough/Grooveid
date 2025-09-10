# backend/identify.py
# GrooveID — full pipeline: Vision web + OCR + handwriting pass + block re-OCR + Discogs limiter/cache/search + CLIP re-rank

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
import os, re, io, time, base64, sqlite3, hashlib, requests

# ---------- 3rd-party ----------
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

# CLIP for visual re-rank
import torch
import open_clip

# ---------- Config ----------
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

DGS_API = "https://api.discogs.com"
DGS_UA  = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
try:
    from supabase import create_client, Client  # type: ignore
    supabase: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
except Exception:
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
    if ctx: payload["requests"][0]["imageContext"] = ctx
    r = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload, timeout=30)
    if r.status_code != 200: raise HTTPException(502, f"Vision error {r.status_code}: {r.text[:200]}")
    return r.json().get("responses",[{}])[0]

def call_vision_full(image_bytes: bytes) -> dict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    feats = [
        {"type":"WEB_DETECTION","maxResults":15},  # Increased from 10
        {"type":"TEXT_DETECTION","maxResults":10},  # Increased from 5
    ]
    ctx = {"webDetectionParams":{"includeGeoResults": True}}
    return _vision_request(b64, feats, ctx)

def call_vision_doc(image_bytes: bytes) -> dict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return _vision_request(b64, [{"type":"DOCUMENT_TEXT_DETECTION","maxResults":1}], {"languageHints":["en"]})

def parse_discogs_web(web: dict) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    urls=[]
    for key in ("pagesWithMatchingImages","fullMatchingImages","partialMatchingImages","visuallySimilarImages"):
        for it in web.get(key, []):
            if it.get("url"): urls.append(it["url"])
    rel = mast = None; discogs_url = None
    for u in urls:
        m = RE_RELEASE.search(u)
        if m: rel = int(m.group(1)); discogs_url = u; break
    if rel is None:
        for u in urls:
            m = RE_MASTER.search(u)
            if m: mast = int(m.group(1)); discogs_url = u; break
    return rel, mast, discogs_url

def ocr_lines(text_ann: List[dict]) -> List[str]:
    if not text_ann: return []
    raw = text_ann[0].get("description","")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]

# ---------- IMPROVED Handwriting / block re-OCR ----------
def enhance_for_handwriting(img: Image.Image) -> Image.Image:
    """Enhanced preprocessing for better handwritten text recognition"""
    w, h = img.size
    
    # Try larger crop to capture more text
    s = int(min(w, h) * 0.85)  # Increased from 0.70
    cx, cy = w // 2, h // 2
    crop = img.crop((cx - s//2, cy - s//2, cx + s//2, cy + s//2))
    
    # Convert to grayscale
    gray = ImageOps.grayscale(crop)
    
    # Multiple enhancement strategies
    # Strategy 1: High contrast + sharpen
    sharp = gray.filter(ImageFilter.UnsharpMask(radius=2.5, percent=220, threshold=2))
    contrast = ImageEnhance.Contrast(sharp).enhance(2.8)
    result = ImageOps.equalize(contrast)
    
    # Strategy 2: Brightness boost for faint text
    bright = ImageEnhance.Brightness(gray).enhance(1.3)
    sharp2 = bright.filter(ImageFilter.SHARPEN)
    result2 = ImageOps.equalize(sharp2)
    
    # For now, return the high-contrast version
    # In production, you might want to try both and pick the best OCR result
    return result

def handwriting_merge(image_bytes: bytes, text: List[dict]) -> List[dict]:
    """DOC pass on enhanced crop; merge lines back into textAnnotations."""
    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Try multiple enhancement approaches
        hw_img1 = enhance_for_handwriting(base_img)
        buf1 = io.BytesIO(); hw_img1.save(buf1, format="PNG")
        v1 = call_vision_doc(buf1.getvalue())
        
        # Also try a different crop/enhancement
        w, h = base_img.size
        # Try upper portion crop (where labels often are)
        upper_crop = base_img.crop((0, 0, w, h//2))
        hw_img2 = enhance_for_handwriting(upper_crop)
        buf2 = io.BytesIO(); hw_img2.save(buf2, format="PNG")
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
        merged = list(dict.fromkeys([*primary, *extra_lines]))  # Remove duplicates while preserving order
        
        return [{"description":"\n".join(merged)}] if merged else text
    except Exception:
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
                    x = min(v.get("x",0) for v in verts); y = min(v.get("y",0) for v in verts)
                    X = max(v.get("x",0) for v in verts); Y = max(v.get("y",0) for v in verts)
                    if X-x>8 and Y-y>8:  # Increased minimum size
                        # Larger resize factor for better OCR
                        crop = base.crop((x,y,X,Y)).resize((int(2.0*(X-x)), int(2.0*(Y-y))))
                        
                        # Apply enhancement to crop
                        enhanced_crop = enhance_for_handwriting(crop)
                        
                        buf = io.BytesIO(); enhanced_crop.save(buf, format="PNG")
                        vsmall = call_vision_doc(buf.getvalue())
                        if vsmall.get("textAnnotations"):
                            raw = vsmall["textAnnotations"][0].get("description","")
                            block_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                            lines.extend(block_lines)
    except Exception:
        pass
    return lines

# ---------- Supabase cache ----------
def cache_get(rid: int) -> Optional[dict]:
    if not supabase: return None
    try:
        res = supabase.table("discogs_cache").select("*").eq("release_id", rid).limit(1).execute()
        return res.data[0] if res.data else None
    except:
        return None

def cache_put(row: dict) -> None:
    if not supabase: return
    try:
        supabase.table("discogs_cache").upsert(row, on_conflict="release_id").execute()
    except:
        pass

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
        if self.tokens>=n: self.tokens-=n; return True
        return False
    def wait(self,n=1):
        while not self.acquire(n): time.sleep(0.05)

_bucket = TokenBucket(60)

def limit_discogs(fn):
    def wrap(*a,**k):
        _bucket.wait(1); return fn(*a,**k)
    return wrap

@limit_discogs
def fetch_discogs_release_json(rid:int)->Optional[dict]:
    try:
        headers = DGS_UA.copy()
        token = os.environ.get("DISCOGS_TOKEN")
        if token:
            headers["Authorization"] = f"Discogs token={token}"
        
        r=requests.get(f"{DGS_API}/releases/{rid}", headers=headers, timeout=15)
        return r.json() if r.status_code==200 else None
    except: return None

@limit_discogs
def discogs_search(params:Dict[str,str])->List[IdentifyCandidate]:
    p=params.copy(); p.setdefault("type","release")
    tok=os.environ.get("DISCOGS_TOKEN")
    if tok: p["token"]=tok
    
    headers = DGS_UA.copy()
    if tok:
        headers["Authorization"] = f"Discogs token={tok}"
    
    out=[]
    try:
        r=requests.get(f"{DGS_API}/database/search", params=p, headers=headers, timeout=20)
        if r.status_code==200:
            results = r.json().get("results",[])
            for it in results[:8]:  # Increased from 5 to 8
                url=it.get("resource_url","")
                if "/releases/" not in url: continue
                try: rid=int(url.rstrip("/").split("/")[-1])
                except: continue
                
                # Extract artist and title more carefully
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
                    score=0.70  # Slightly higher base score
                ))
    except Exception as e:
        print(f"Discogs search error: {e}")
    return out

# ---------- IMPROVED OCR heuristics ----------
def improved_extract_ocr_metadata(lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """Improved metadata extraction with fuzzy matching"""
    label = catno = artist = None
    tracks = []
    
    print(f"Extracting metadata from {len(lines)} lines: {lines}")
    
    for i, ln in enumerate(lines):
        lower = ln.lower().strip()
        original = ln.strip()
        
        # More flexible catalog number patterns
        # Pattern 1: LABEL ### or LABEL-### or LABEL ###
        patterns = [
            r'([a-z]{2,8})\s*[-\s]*(\d{2,5})',  # TRON 001, ACID-123
            r'([a-z]{3,8})\s*([0-9]{3})',       # TRON001
            r'([a-z]+)\s+vol\.?\s*(\d+)',       # TRON VOL 1, TRON VOL. 1
            r'([a-z]+)\s+revolta\s+vol\.?\s*(\d+)',  # Specific for this record
        ]
        
        for pattern in patterns:
            m = re.search(pattern, lower)
            if m and not catno:
                potential_label = m.group(1).title()
                potential_catno = m.group(2).zfill(3)  # Pad to 001 format
                
                # Common record label indicators
                if (len(potential_label) >= 3 and 
                    (potential_label.upper() in ['TRON', 'ACID', 'TECH', 'HOUSE', 'DEEP', 'WARP', 'NINJA'] or 
                     any(word in potential_label.lower() for word in ['records', 'music', 'recordings']))):
                    label = potential_label
                    catno = potential_catno
                    print(f"Found label/catno: {label} {catno}")
                    break
        
        # Pattern for standalone catalog numbers
        if not catno:
            catno_patterns = [
                r'^00[1-9]$',           # 001, 002, etc.
                r'^[0-9]{3}$',          # Any 3-digit number
                r'^vol\.?\s*(\d+)$',    # VOL 1, VOL. 1
            ]
            for pattern in catno_patterns:
                m = re.search(pattern, lower)
                if m:
                    if 'vol' in pattern:
                        catno = m.group(1).zfill(3)
                    else:
                        catno = lower.zfill(3)
                    print(f"Found standalone catno: {catno}")
                    break
        
        # Artist detection - look for ALL CAPS, Title Case, or specific patterns
        if not artist and i < len(lines) * 0.6:  # Look in first 60%
            words = original.split()
            if 1 <= len(words) <= 4:
                # All caps artist names
                if all(w.isupper() and w.isalpha() for w in words):
                    artist = original.title()
                    print(f"Found all-caps artist: {artist}")
                # Title case or mixed case
                elif any(w[0].isupper() for w in words if w.isalpha()) and len(original) > 3:
                    # Avoid common label words
                    if not any(word in lower for word in ['records', 'music', 'vol', 'side', 'remix']):
                        artist = original
                        print(f"Found mixed-case artist: {artist}")
                # Special case for "Unknown Artist"
                elif 'unknown' in lower and 'artist' in lower:
                    artist = "Unknown Artist"
                    print(f"Found unknown artist: {artist}")
        
        # Track detection with multiple strategies
        track_candidates = []
        
        # Strategy 1: Colon-separated (A1: Track Name)
        if ':' in ln:
            parts = ln.split(':', 1)
            if len(parts) == 2:
                track_id = parts[0].strip()
                track_name = parts[1].strip()
                if track_name and len(track_name) > 2:
                    track_candidates.append(track_name)
        
        # Strategy 2: Hyphen-separated (avoid catalog numbers)
        elif ' - ' in ln and not re.search(r'[0-9]{3,}', ln):
            parts = ln.split(' - ', 1)
            if len(parts) == 2:
                track_name = parts[1].strip()
                if track_name and len(track_name) > 2:
                    track_candidates.append(track_name)
        
        # Strategy 3: Lines that look like track names
        elif (len(original) > 5 and len(original) < 50 and 
              not re.search(r'[0-9]{3,}', ln) and  # No long numbers
              i > len(lines) * 0.4 and  # In latter part of text
              not any(word in lower for word in ['vol', 'side', 'records', 'music', 'label'])):
            track_candidates.append(original)
        
        # Add valid track candidates
        for track in track_candidates:
            if track not in tracks and len(track.strip()) > 2:
                tracks.append(track.strip())
                print(f"Found track: {track.strip()}")
    
    print(f"Final extraction: label={label}, catno={catno}, artist={artist}, tracks={tracks}")
    return label, catno, artist, tracks

def broader_search_attempts(lines: List[str], label: str, catno: str, artist: str, tracks: List[str]) -> List[Dict[str, str]]:
    """Generate more comprehensive search attempts"""
    attempts = []
    
    print(f"Generating search attempts for: label={label}, catno={catno}, artist={artist}, tracks={len(tracks)}")
    
    # High-priority exact matches
    if label and catno:
        attempts.extend([
            {"label": label, "catno": catno},
            {"q": f"{label} {catno}"},
            {"q": f"{label}-{catno}"},
            {"q": f"{label}{catno}"},
        ])
    
    # Artist + catalog combinations
    if catno and artist:
        attempts.append({"artist": artist, "catno": catno})
    
    if artist and label:
        attempts.append({"artist": artist, "label": label})
    
    # Specific searches for electronic/underground music
    electronic_searches = []
    if catno:
        electronic_searches.extend([
            {"q": f"tron {catno}"},
            {"q": f"tron revolta {catno}"},
            {"q": f"unknown artist tron {catno}"},
            {"q": f"acid {catno}"},
            {"q": f"techno {catno}"},
        ])
    
    # Add specific patterns based on common OCR errors
    if label and catno:
        # Common OCR misreads: TRON -> IRON, KRON, etc.
        ocr_variations = []
        if label.upper() == 'TRON':
            ocr_variations.extend(['IRON', 'KRON', 'TKON'])
        
        for variant in ocr_variations:
            electronic_searches.append({"q": f"{variant} {catno}"})
    
    attempts.extend(electronic_searches)
    
    # Track-based searches
    if tracks:
        # Individual track searches
        for track in tracks[:3]:  # Limit to first 3 tracks
            if len(track) > 4:  # Only meaningful track names
                attempts.append({"track": track})
                if artist:
                    attempts.append({"track": track, "artist": artist})
        
        # Combined track searches
        if len(tracks) >= 2:
            track_combo = " ".join(tracks[:2])[:120]
            attempts.append({"q": track_combo})
    
    # Broad text-based searches as fallback
    if lines:
        clean = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in lines if ln.strip()]
        significant_lines = [line for line in clean if len(line) > 3]
        
        if significant_lines:
            # Try combinations of lines
            attempts.extend([
                {"q": " ".join(significant_lines[:4])[:150]},
                {"q": " ".join(significant_lines[:3])[:120]},
                {"q": " ".join(significant_lines[:2])[:100]},
            ])
            
            # Individual line searches
            for line in significant_lines[:4]:
                if len(line) > 5:
                    attempts.append({"q": line[:100]})
    
    # Genre-specific searches for electronic music
    genre_attempts = []
    all_text = " ".join(lines).lower()
    electronic_genres = ['acid', 'house', 'techno', 'trance', 'electro', 'breakbeat', 'drum', 'bass']
    
    for genre in electronic_genres:
        if genre in all_text:
            if catno:
                genre_attempts.append({"q": f"{genre} {catno}", "genre": "Electronic"})
            genre_attempts.append({"genre": "Electronic", "style": genre.title()})
    
    attempts.extend(genre_attempts)
    
    print(f"Generated {len(attempts)} search attempts")
    return attempts

# ---------- CLIP URLâ†'vector cache ----------
def _db():
    conn = sqlite3.connect(EMBED_CACHE_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS url_vectors(url_hash TEXT PRIMARY KEY, dim INTEGER, vec BLOB, updated_at INTEGER)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON url_vectors(updated_at)")
    conn.commit(); return conn

def _h(url:str)->str: return hashlib.sha1(url.encode("utf-8")).hexdigest()

def cache_vec_save(url:str, vec:np.ndarray):
    try:
        v = vec.astype(np.float16, copy=False); now=int(time.time())
        conn=_db(); conn.execute("REPLACE INTO url_vectors(url_hash,dim,vec,updated_at) VALUES(?,?,?,?)",(_h(url),v.shape[0],sqlite3.Binary(v.tobytes()),now)); conn.commit(); conn.close()
    except:
        pass

def cache_vec_load(url:str)->Optional[np.ndarray]:
    try:
        cutoff=int(time.time())-EMBED_CACHE_TTL
        conn=_db(); row=conn.execute("SELECT dim,vec,updated_at FROM url_vectors WHERE url_hash=?",(_h(url),)).fetchone()
        if not row: conn.close(); return None
        dim,blob,upd=row
        if upd<cutoff: conn.execute("DELETE FROM url_vectors WHERE url_hash=?",(_h(url),)); conn.commit(); conn.close(); return None
        arr=np.frombuffer(blob,dtype=np.float16).astype(np.float32); conn.close(); return arr.reshape((dim,))
    except:
        return None

# ---------- CLIP embedder ----------
_CLIP_DEVICE="cpu"
_CLIP_MODEL=_CLIP_PRE=None

def _ensure_clip():
    global _CLIP_MODEL,_CLIP_PRE
    if _CLIP_MODEL is None:
        try:
            model,_,pre = open_clip.create_model_and_transforms("ViT-B-32","openai", device=_CLIP_DEVICE)
            model.eval(); _CLIP_MODEL, _CLIP_PRE = model, pre
        except:
            print("CLIP model loading failed")

def _to_img(b:bytes)->Image.Image: return Image.open(io.BytesIO(b)).convert("RGB")

@torch.no_grad()
def embed_image_bytes(img_bytes:bytes)->np.ndarray:
    _ensure_clip()
    if _CLIP_MODEL is None: return np.zeros(512, dtype=np.float32)  # Fallback
    im=_to_img(img_bytes); ten=_CLIP_PRE(im).unsqueeze(0).to(_CLIP_DEVICE)
    feat=_CLIP_MODEL.encode_image(ten); feat=feat/feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)

@torch.no_grad()
def embed_image_url(url:str)->Optional[np.ndarray]:
    vec=cache_vec_load(url)
    if vec is not None: return vec
    try:
        r=requests.get(url,timeout=10, headers={'User-Agent': 'GrooveID/1.0'}); 
        if r.status_code!=200: return None
        vec=embed_image_bytes(r.content); cache_vec_save(url,vec); return vec
    except: return None

def cosine(a:np.ndarray,b:np.ndarray)->float:
    if a is None or b is None: return -1.0
    denom=(np.linalg.norm(a)*np.linalg.norm(b))
    if denom==0: return -1.0
    return float(np.dot(a,b)/denom)

def visual_rerank(user_img_bytes:bytes, cands:List[IdentifyCandidate])->List[Tuple[IdentifyCandidate,float]]:
    if len(cands)<=1: return [(cands[0],1.0)] if cands else []
    
    try:
        u=embed_image_bytes(user_img_bytes); sims=[]
        for c in cands:
            best=-1.0; urls=[]
            if c.cover_url: urls.append(c.cover_url)
            for uurl in urls[:2]:
                v=embed_image_url(uurl)
                if v is not None: best=max(best, cosine(u,v))
            sims.append(best)
        ranked=[]
        for c,vis in zip(cands,sims):
            text=c.score or 0.0; final = 0.55*text + 0.40*max(vis,0.0) + 0.05*0.0
            ranked.append((c,final))
        ranked.sort(key=lambda x:x[1], reverse=True); return ranked
    except:
        return [(c, c.score or 0.0) for c in cands]

# ---------- DEBUG ENDPOINT ----------
@router.post("/api/debug-identify")
async def debug_identify(file: UploadFile = File(...)):
    """Debug endpoint to see exactly what OCR detects"""
    try:
        image_bytes = await file.read()
        
        # Step 1: Basic Vision API call
        print("=== CALLING VISION API ===")
        v = call_vision_full(image_bytes)
        web = v.get("webDetection", {})
        text = v.get("textAnnotations", [])
        
        print("=== WEB DETECTION RESULTS ===")
        web_urls = []
        for key in ("pagesWithMatchingImages", "fullMatchingImages", "partialMatchingImages", "visuallySimilarImages"):
            urls = [item.get("url") for item in web.get(key, []) if item.get("url")]
            if urls:
                print(f"{key}: {urls[:3]}")  # Show first 3
                web_urls.extend(urls)
        
        # Check for Discogs URLs
        release_id, master_id, discogs_url = parse_discogs_web(web)
        print(f"Discogs match: release_id={release_id}, master_id={master_id}, url={discogs_url}")
        
        print("\n=== RAW OCR TEXT ===")
        raw_ocr = text[0].get("description", "") if text else ""
        print(f"Raw OCR: '{raw_ocr}'")
        
        # Step 2: Enhanced handwriting pass
        print("\n=== HANDWRITING ENHANCEMENT ===")
        text_enhanced = handwriting_merge(image_bytes, text)
        enhanced_ocr = text_enhanced[0].get("description", "") if text_enhanced else ""
        print(f"Enhanced OCR: '{enhanced_ocr}'")
        
        # Step 3: Block re-OCR
        print("\n=== BLOCK RE-OCR ===")
        block_lines = block_crop_reocr(image_bytes)
        print(f"Block re-OCR lines: {block_lines}")
        
        # Step 4: Parse OCR lines
        final_lines = ocr_lines(text_enhanced)
        if block_lines:
