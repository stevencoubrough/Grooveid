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
from PIL import Image
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
            rel = int(m.group(1))
            discogs_url = u
            break
    if rel is None:
        for u in urls:
            m = RE_MASTER.search(u)
            if m:
                master = int(m.group(1))
                discogs_url = u
                break
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

# -------------------- Tiny URL→vector SQLite cache --------------------
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
        if not row:
            return None
        dim, blob, updated = row
        if updated < cutoff:
            try:
                conn.execute("DELETE FROM url_vectors WHERE url_hash=?", (_h(url),))
                conn.commit()
            except Exception:
                pass
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
    if vec is not None:
        return vec
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
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

# -------------------- Visual re-ranker --------------------
def visual_rerank(user_image_bytes: bytes, candidates: List[IdentifyCandidate]) -> List[Tuple[IdentifyCandidate, float]]:
    if len(candidates) <= 1:
        return [(candidates[0], 1.0)] if candidates else []
    uvec = embed_image_bytes(user_image_bytes)
    sims: List[float] = []
    for c in candidates:
        best = -1.0
        urls = []
        if c.cover_url: urls.append(c.cover_url)
        for url in urls[:2]:
            v = embed_image_url(url)
            if v is None: continue
            best = max(best, cosine(uvec, v))
        sims.append(best)
    ranked: List[Tuple[IdentifyCandidate, float]] = []
    for c, vis in zip(candidates, sims):
        text_score = getattr(c, "score", 0.0) or 0.0
        final = 0.55*text_score + 0.40*max(vis, 0.0) + 0.05*0.0
        ranked.append((c, final))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

# -------------------- Route --------------------
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_record(file: UploadFile = File(...)) -> IdentifyResponse:
    try:
        image_bytes = await file.read()
        v = call_vision_api(image_bytes)
        web = v.get("webDetection", {})
        text = v.get("textAnnotations", [])
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

        # 3) OCR fallback with structured passes (keeps digits/hyphens)
        if not candidates:
            lines = ocr_lines(text)
            clean_lines = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in lines if ln.strip()]
            label, catno, artist, tracks = extract_ocr_metadata(clean_lines)

            search_attempts: List[Dict[str,str]] = []

            # Try to pick a plausible title line
            def guess_title(ls: List[str]) -> Optional[str]:
                keys = (" part ", " pt ", " vol ", " ep ", " remix", " mixes", " ii", " iii", " iv", " v")
                for l in ls:
                    if any(k in f" {l.lower()} " for k in keys): return l
                cands = [l for l in ls if 8 <= len(l) <= 50 and not l.isupper()]
                return max(cands, key=len) if cands else None

            title_guess = guess_title(clean_lines)

            # Structured attempts first
            if artist and title_guess:
                search_attempts.append({"artist": artist, "release_title": title_guess})
            if label and title_guess:
                search_attempts.append({"label": label, "release_title": title_guess})
            if label and catno:
                search_attempts.append({"label": label, "catno": catno})
            if artist and catno:
                search_attempts.append({"artist": artist, "catno": catno})
            if tracks:
                for t in tracks:
                    p = {"track": t}
                    if artist: p["artist"] = artist
                    search_attempts.append(p)

            # Broad fallbacks
            if clean_lines:
                search_attempts.append({"q": " ".join(clean_lines[:3])[:200]})
                if len(clean_lines) >= 2:
                    search_attempts.append({"q": " ".join(clean_lines[:2])[:200]})
                search_attempts.append({"q": clean_lines[0][:200]})

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
