# backend/identify.py
# Single-file drop-in: Vision + OCR fallback + Discogs limiter + CLIP visual re-rank + optional Supabase cache

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
import os, re, base64, requests, io, time, sqlite3, hashlib

# Optional (if you set SUPABASE_URL/KEY for cache speedups)
try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None
    Client = None

# --- Third-party for vision / vision helpers ---
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np

# CLIP for visual re-rank
import torch
import open_clip

# -------- Config --------
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

DGS_API = "https://api.discogs.com"
DGS_UA = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        supabase = None

# vector cache settings (for image URL embeddings)
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE_PATH", "/tmp/embed_cache.sqlite3")
EMBED_CACHE_TTL = int(os.getenv("EMBED_CACHE_TTL", "1209600"))  # 14 days

# -------- Regex for Discogs URLs --------
RE_RELEASE = re.compile(r"discogs\.com/(?:[^/]+/)?release/(\d+)", re.I)
RE_MASTER  = re.compile(r"discogs\.com/(?:[^/]+/)?master/(\d+)",  re.I)

router = APIRouter()

# -------------------- Models --------------------
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

# -------------------- Helpers: Google Vision --------------------
def call_vision_api(image_bytes: bytes) -> dict:
    if not VISION_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_VISION_API_KEY not set")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [{
            "image": {"content": b64},
            "features": [
                {"type": "WEB_DETECTION", "maxResults": 10},
                {"type": "TEXT_DETECTION", "maxResults": 5},
            ],
            "imageContext": {"webDetectionParams": {"includeGeoResults": True}},
        }]
    }
    r = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Vision error {r.status_code}: {r.text[:200]}")
    return r.json().get("responses", [{}])[0]

def parse_discogs_web_detection(web: dict) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    urls: List[str] = []
    for key in ("pagesWithMatchingImages","fullMatchingImages","partialMatchingImages","visuallySimilarImages"):
        for it in web.get(key, []):
            u = it.get("url")
            if u: urls.append(u)
    rel = master = None
    discogs_url = None
    for u in urls:
        m = RE_RELEASE.search(u)
        if m:
            rel = int(m.group(1)); discogs_url = u; break
    if rel is None:
        for u in urls:
            m = RE_MASTER.search(u)
            if m:
                master = int(m.group(1)); discogs_url = u; break
    return rel, master, discogs_url

def ocr_lines(text_annotations: List[dict]) -> List[str]:
    if not text_annotations: return []
    raw = text_annotations[0].get("description", "")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]

# -------------------- Tiny Token Bucket (Discogs limiter) --------------------
class TokenBucket:
    def __init__(self, rate_per_minute=60, capacity=None):
        self.rate = rate_per_minute / 60.0
        self.capacity = capacity or rate_per_minute
        self.tokens = self.capacity
        self.last = time.time()

    def acquire(self, tokens=1) -> bool:
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last)*self.rate)
        self.last = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait(self, tokens=1):
        while not self.acquire(tokens):
            time.sleep(0.05)

_discogs_bucket = TokenBucket(rate_per_minute=60)

def limit_discogs(func):
    def wrapper(*args, **kwargs):
        _discogs_bucket.wait(1)
        return func(*args, **kwargs)
    return wrapper

# -------------------- Discogs helpers --------------------
def _supabase_cache_get(release_id: int) -> Optional[dict]:
    if not supabase: return None
    try:
        res = supabase.table("discogs_cache").select("*").eq("release_id", release_id).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception:
        return None

def _supabase_cache_put(row: dict) -> None:
    if not supabase: return
    try:
        supabase.table("discogs_cache").upsert(row, on_conflict="release_id").execute()
    except Exception:
        pass

@limit_discogs
def fetch_discogs_release_json(release_id: int) -> Optional[dict]:
    try:
        r = requests.get(f"{DGS_API}/releases/{release_id}", headers=DGS_UA, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@limit_discogs
def discogs_search(params: Dict[str, str]) -> List[IdentifyCandidate]:
    p = params.copy()
    p.setdefault("type", "release")
    tok = os.environ.get("DISCOGS_TOKEN")
    if tok: p["token"] = tok
    out: List[IdentifyCandidate] = []
    try:
        r = requests.get(f"{DGS_API}/database/search", params=p, headers=DGS_UA, timeout=20)
        if r.status_code == 200:
            js = r.json()
            for it in js.get("results", [])[:5]:
                res_url = it.get("resource_url", "")
                if "/releases/" not in res_url: continue
                try:
                    rid = int(res_url.rstrip("/").split("/")[-1])
                except Exception:
                    continue
                out.append(IdentifyCandidate(
                    source="ocr_search",
                    release_id=rid,
                    discogs_url=f"https://www.discogs.com/release/{rid}",
                    artist=(it.get("title","" ).split(" - ")[0] if " - " in it.get("title","" ) else None),
                    title=it.get("title"),
                    label=(it.get("label") or [""])[0] if isinstance(it.get("label"), list) else it.get("label"),
                    year=str(it.get("year") or ""),
                    cover_url=it.get("thumb"),
                    score=0.65,
                ))
    except Exception:
        pass
    return out

# -------------------- OCR metadata extraction --------------------
def extract_ocr_metadata(lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    label = catno = artist = None
    tracks: List[str] = []
    for i, ln in enumerate(lines):
        lower = ln.lower()
        # e.g. "urban decay promo 003"
        m = re.match(r"([a-z0-9\s]+?)\s*(?:promo|pr)?\s*(\d{1,5})$", lower)
        if m and not catno:
            l = m.group(1).strip()
            label = l.title() if l else label
            catno = m.group(2)
            continue
        # e.g. "urban decay 003"
        m2 = re.match(r"([a-z0-9\s]+?)\s+(\d{1,5})$", lower)
        if m2 and not catno:
            l = m2.group(1).strip()
            label = l.title() if l else label
            catno = m2.group(2)
            continue
        # artist heuristic: short UPPERCASE line (not first)
        if not artist and i > 0:
            words = ln.strip().split()
            if 1 <= len(words) <= 3 and all(w.isupper() for w in words if w.isalpha()):
                artist = ln.strip().title()
                continue
        # track lines
        if ":" in ln:
            t = ln.split(":", 1)[1].strip()
            if t: tracks.append(t)
            continue
        if " - " in ln and not artist and not re.search(r"\d", ln):
            t = ln.split(" - ", 1)[1].strip()
            if t: tracks.append(t)
    return label, catno, artist, tracks

# ----------- Noise filtering, track extraction & voting -----------
NOISE_PATTERNS = [
    r"\bside\s*[ab]\b",
    r"\bvol(?:ume)?\s*#?\d+\b",
    r"\b[a-d]\d\b",
    r"\bfor\s+promotional\s+use\s+only\b",
]

def denoise_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        low = ln.lower()
        for pat in NOISE_PATTERNS:
            low = re.sub(pat, " ", low)
        low = re.sub(r"[^\w\s/-]", " ", low)
        low = re.sub(r"\s+", " ", low).strip(" -_\t")
        if low: out.append(low)
    return out

def extract_tracks(clean_lines: List[str]) -> List[str]:
    tracks: List[str] = []
    for ln in clean_lines:
        parts = [p.strip() for p in re.split(r"[;/|]", ln)]
        for p in parts:
            if 4 <= len(p) <= 48 and re.search(r"[a-z]", p):
                cand = p.title()
                if not cand.startswith(("Side ", "Volume ")) and cand not in tracks:
                    tracks.append(cand)
    return tracks[:6]

def find_label_catno(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    for ln in lines:
        m = re.search(r"\b([A-Z]{3,})[- ]?(\d{1,5})\b", ln.upper())
        if m: return m.group(1).strip(), m.group(2).strip()
    return None, None

def discogs_search_track(track: str, limit: int = 8) -> List[int]:
    params: Dict[str, str] = {"track": track, "type": "release"}
    tok = os.environ.get("DISCOGS_TOKEN")
    if tok: params["token"] = tok
    try:
        r = requests.get(f"{DGS_API}/database/search", headers=DGS_UA, params=params, timeout=20)
        if r.status_code != 200: return []
        ids: List[int] = []
        for it in r.json().get("results", [])[:limit]:
            res_url = it.get("resource_url", "")
            if "/releases/" in res_url:
                try: ids.append(int(res_url.rstrip("/").split("/")[-1]))
                except Exception: pass
        return ids
    except Exception:
        return []

def vote_releases_by_tracks(tracks: List[str]) -> List[int]:
    """Return releases that match >=2 tracks, sorted by votes desc then id asc."""
    from collections import Counter
    c: Counter[int] = Counter()
    for t in tracks:
        for rid in discogs_search_track(t):
            c[rid] += 1
    filtered = [(rid, v) for rid, v in c.items() if v >= 2]
    filtered.sort(key=lambda x: (-x[1], x[0]))
    return [rid for rid, _ in filtered[:10]]

def candidate_from_release_json(rel: dict, source: str, score: float) -> IdentifyCandidate:
    try:
        rid = int(rel.get("id", 0) or 0)
    except Exception:
        rid = None
    discogs_url = rel.get("uri") or (f"https://www.discogs.com/release/{rid}" if rid else None)
    artist = ", ".join(a.get("name", "") for a in rel.get("artists", [])).strip() or None
    title = rel.get("title")
    label = ", ".join(l.get("name", "") for l in rel.get("labels", [])).strip() or None
    year = str(rel.get("year", "")) if rel.get("year") is not None else None
    cover_url = rel.get("thumb") or ((rel.get("images") or [{}])[0].get("uri") if rel.get("images") else None)
    return IdentifyCandidate(
        source=source,
        release_id=rid,
        discogs_url=discogs_url,
        artist=artist,
        title=title,
        label=label,
        year=year,
        cover_url=cover_url,
        score=score,
    )

# -------------------- URL→vector cache (for CLIP) --------------------
def _db():
    conn = sqlite3.connect(EMBED_CACHE_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS url_vectors(
        url_hash TEXT PRIMARY KEY,
        dim INTEGER NOT NULL,
        vec BLOB NOT NULL,
        updated_at INTEGER NOT NULL
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON url_vectors(updated_at)")
    conn.commit()
    return conn

def _h(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()

def cache_save(url: str, vec: np.ndarray):
    v = vec.astype(np.float16, copy=False)
    now = int(time.time())
    conn = _db()
    try:
        conn.execute("REPLACE INTO url_vectors(url_hash,dim,vec,updated_at) VALUES(?,?,?,?)",
                     (_h(url), v.shape[0], sqlite3.Binary(v.tobytes()), now))
        conn.commit()
    finally:
        conn.close()

def cache_load(url: str) -> Optional[np.ndarray]:
    cutoff = int(time.time()) - EMBED_CACHE_TTL
    conn = _db()
    try:
        row = conn.execute("SELECT dim, vec, updated_at FROM url_vectors WHERE url_hash=?", (_h(url),)).fetchone()
        if not row: return None
        dim, blob, updated = row
        if updated < cutoff:
            try:
                conn.execute("DELETE FROM url_vectors WHERE url_hash=?", (_h(url),)); conn.commit()
            except Exception: pass
            return None
        arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape((dim,))
        return arr
    finally:
        conn.close()

# -------------------- OpenCLIP embedder (CPU) --------------------
_CLIP_DEVICE = "cpu"
_CLIP_MODEL, _CLIP_PREPROC = None, None

def _ensure_clip():
    global _CLIP_MODEL, _CLIP_PREPROC
    if _CLIP_MODEL is None:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=_CLIP_DEVICE)
        model.eval()
        _CLIP_MODEL, _CLIP_PREPROC = model, preprocess

def _to_img(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

@torch.no_grad()
def embed_image_bytes(img_bytes: bytes) -> np.ndarray:
    _ensure_clip()
    im = _to_img(img_bytes)
    ten = _CLIP_PREPROC(im).unsqueeze(0).to(_CLIP_DEVICE)
    feat = _CLIP_MODEL.encode_image(ten)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)

@torch.no_grad()
def embed_image_url(url: str) -> Optional[np.ndarray]:
    vec = cache_load(url)
    if vec is not None: return vec
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200: return None
        vec = embed_image_bytes(r.content)
        cache_save(url, vec)
        return vec
    except Exception:
        return None

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0: return -1.0
    return float(np.dot(a, b) / denom)

# -------------------- Handwriting / graffiti enhancer --------------------
def enhance_for_handwriting(img: Image.Image) -> Image.Image:
    """Center crop + sharpen + contrast + equalize to pop graffiti / scrawl."""
    w, h = img.size
    s = int(min(w, h) * 0.70)
    cx, cy = w // 2, h // 2
    crop = img.crop((cx - s//2, cy - s//2, cx + s//2, cy + s//2))
    gray = ImageOps.grayscale(crop)
    sharp = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
    boosted = ImageEnhance.Contrast(sharp).enhance(2.2)
    boosted = ImageOps.equalize(boosted)
    return boosted

def handwriting_ocr_merge(image_bytes: bytes, text: List[dict]) -> List[dict]:
    """Run a second Vision pass on an enhanced crop using DOCUMENT_TEXT_DETECTION and merge."""
    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        hw_img = enhance_for_handwriting(base_img)
        buf = io.BytesIO(); hw_img.save(buf, format="PNG"); hw_bytes = buf.getvalue()
        b64 = base64.b64encode(hw_bytes).decode("utf-8")
        payload_hw = {"requests": [{
            "image": {"content": b64},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION", "maxResults": 1}],
        }]}
        r_hw = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload_hw, timeout=30)
        if r_hw.status_code != 200:
            return text
        v2 = r_hw.json().get("responses", [{}])[0]
        extra = []
        t2 = v2.get("textAnnotations", [])
        if t2:
            raw = t2[0].get("description", "")
            extra = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        primary = [ln.strip() for ln in (text[0].get("description","").splitlines() if text else []) if ln.strip()]
        merged = list(dict.fromkeys([*primary, *extra]))
        return [{"description": "\n".join(merged)}] if merged else text
    except Exception:
        return text

# -------------------- Route --------------------
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_record(file: UploadFile = File(...)) -> IdentifyResponse:
    try:
        image_bytes = await file.read()
        v = call_vision_api(image_bytes)
        web = v.get("webDetection", {})
        text = v.get("textAnnotations", [])
        # handwriting / graffiti second pass merge
        text = handwriting_ocr_merge(image_bytes, text)

        release_id, master_id, discogs_url = parse_discogs_web_detection(web)
        candidates: List[IdentifyCandidate] = []

        # 1) Direct Discogs release via web detection
        if release_id:
            cached = _supabase_cache_get(release_id)
            if cached:
                candidates.append(IdentifyCandidate(
                    source="web_detection_cache",
                    release_id=release_id,
                    discogs_url=cached["discogs_url"],
                    artist=cached.get("artist"), title=cached.get("title"),
                    label=cached.get("label"), year=cached.get("year"),
                    cover_url=cached.get("cover_url"), score=0.95,
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
                        "year": str(rel.get("year", "")),
                        "cover_url": rel.get("thumb") or (rel.get("images") or [{}])[0].get("uri",""),
                        "payload": rel,
                    }
                    _supabase_cache_put(row)
                    candidates.append(IdentifyCandidate(
                        source="web_detection_live",
                        release_id=release_id,
                        discogs_url=row["discogs_url"],
                        artist=row.get("artist"), title=row.get("title"),
                        label=row.get("label"), year=row.get("year"),
                        cover_url=row.get("cover_url"), score=0.90,
                    ))

        # 2) Master match
        if not candidates and master_id:
            candidates.append(IdentifyCandidate(
                source="web_detection_master",
                master_id=master_id,
                discogs_url=f"https://www.discogs.com/master/{master_id}",
                note="Master match — prompt user to select a pressing",
                score=0.60,
            ))

        # 3) OCR fallback with enhanced logic
        if not candidates:
            # Get OCR lines and build clean + denoised versions
            raw_lines = ocr_lines(text)
            clean_lines = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in raw_lines if ln.strip()]
            denoised = denoise_lines(clean_lines)

            # Extract metadata via heuristics
            label, catno, artist, tracks_meta = extract_ocr_metadata(clean_lines)
            tracks = extract_tracks(denoised)
            lbl_from_cat, cat_from_cat = find_label_catno(denoised)
            if not label and lbl_from_cat: label = lbl_from_cat
            if not catno and cat_from_cat: catno = cat_from_cat

            # Track voting first (>=2 tracks to count)
            if len(tracks) >= 2:
                try:
                    voted_ids = vote_releases_by_tracks(tracks)
                    for rid in voted_ids:
                        if rid and all((c.release_id != rid for c in candidates)):
                            rel = fetch_discogs_release_json(rid)
                            if rel:
                                candidates.append(candidate_from_release_json(rel, "track_vote", 0.90))
                except Exception:
                    pass

            # Structured search attempts
            search_attempts: List[Dict[str,str]] = []

            def guess_title(ls: List[str]) -> Optional[str]:
                keys = (" part ", " pt ", " vol ", " ep ", " remix", " mixes", " ii", " iii", " iv", " v")
                for l in ls:
                    if any(k in f" {l.lower()} " for k in keys): return l
                cands = [l for l in ls if 8 <= len(l) <= 50 and not l.isupper()]
                return max(cands, key=len) if cands else None

            title_guess = guess_title(clean_lines)
            if artist and title_guess: search_attempts.append({"artist": artist, "release_title": title_guess})
            if label and title_guess:  search_attempts.append({"label": label, "release_title": title_guess})
            if label and catno:        search_attempts.append({"label": label, "catno": catno})
            if artist and catno:       search_attempts.append({"artist": artist, "catno": catno})

            # If we only have tracks (no artist/label), allow direct per-track search
            if tracks and not artist and not label and not candidates:
                for t in tracks[:4]:
                    for rid in discogs_search_track(t, limit=6):
                        rel = fetch_discogs_release_json(rid)
                        if rel:
                            candidates.append(candidate_from_release_json(rel, "track_search", 0.70))

            # Broad fallbacks if still empty
            if not candidates and clean_lines:
                q_parts: List[str] = []
                q_parts.append(" ".join(clean_lines[:3])[:200])
                if len(clean_lines) >= 2: q_parts.append(" ".join(clean_lines[:2])[:200])
                q_parts.append(clean_lines[0][:200])
                for q in q_parts:
                    search_attempts.append({"q": q})

            # Execute search attempts
            ricardo_hits: List[IdentifyCandidate] = []
            other_hits:   List[IdentifyCandidate] = []
            for params in search_attempts:
                res = discogs_search(params)
                if not res: continue
                for c in res:
                    a = (c.artist or "").lower()
                    if artist and artist.lower() in a:
                        ricardo_hits.append(c)
                    else:
                        other_hits.append(c)
                if ricardo_hits: break
            candidates.extend(ricardo_hits or other_hits)

        # 4) Visual re-rank (CLIP) when we have >=2 candidates
        if len(candidates) >= 2 and image_bytes:
            try:
                ranked = visual_rerank(image_bytes, candidates)
                candidates = [c for (c, _) in ranked]
            except Exception:
                pass

        return IdentifyResponse(candidates=candidates[:5])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
