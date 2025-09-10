# backend/identify.py
# GrooveID – full pipeline: Vision web + OCR + handwriting pass + block re-OCR + Discogs limiter/cache/search + CLIP re-rank

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

# URL→vector cache (for CLIP)
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
        {"type":"WEB_DETECTION","maxResults":10},
        {"type":"TEXT_DETECTION","maxResults":5},
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

# ---------- Handwriting / block re-OCR ----------
def enhance_for_handwriting(img: Image.Image) -> Image.Image:
    w,h = img.size; s = int(min(w,h)*0.70); cx,cy = w//2,h//2
    crop = img.crop((cx-s//2, cy-s//2, cx+s//2, cy+s//2))
    gray = ImageOps.grayscale(crop)
    sharp = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
    boost = ImageEnhance.Contrast(sharp).enhance(2.2)
    return ImageOps.equalize(boost)

def handwriting_merge(image_bytes: bytes, text: List[dict]) -> List[dict]:
    """DOC pass on enhanced crop; merge lines back into textAnnotations."""
    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        hw_img = enhance_for_handwriting(base_img)
        buf = io.BytesIO(); hw_img.save(buf, format="PNG")
        v2 = call_vision_doc(buf.getvalue())
        extra = []
        if v2.get("textAnnotations"):
            raw = v2["textAnnotations"][0].get("description","")
            extra = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        primary = [ln.strip() for ln in (text[0].get("description","").splitlines() if text else []) if ln.strip()]
        merged = list(dict.fromkeys([*primary, *extra]))
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
                    if X-x>4 and Y-y>4:
                        crop = base.crop((x,y,X,Y)).resize((int(1.6*(X-x)), int(1.6*(Y-y))))
                        buf = io.BytesIO(); crop.save(buf, format="PNG")
                        vsmall = call_vision_doc(buf.getvalue())
                        if vsmall.get("textAnnotations"):
                            raw = vsmall["textAnnotations"][0].get("description","")
                            lines.extend([ln.strip() for ln in raw.splitlines() if ln.strip()])
    except Exception:
        pass
    return lines

# ---------- Supabase cache ----------
def cache_get(rid: int) -> Optional[dict]:
    if not supabase: return None
    res = supabase.table("discogs_cache").select("*").eq("release_id", rid).limit(1).execute()
    return res.data[0] if res.data else None

def cache_put(row: dict) -> None:
    if not supabase: return
    supabase.table("discogs_cache").upsert(row, on_conflict="release_id").execute()

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
        r=requests.get(f"{DGS_API}/releases/{rid}", headers=DGS_UA, timeout=15)
        return r.json() if r.status_code==200 else None
    except: return None

@limit_discogs
def discogs_search(params:Dict[str,str])->List[IdentifyCandidate]:
    p=params.copy(); p.setdefault("type","release")
    tok=os.environ.get("DISCOGS_TOKEN")
    if tok: p["token"]=tok
    out=[]
    try:
        r=requests.get(f"{DGS_API}/database/search", params=p, headers=DGS_UA, timeout=20)
        if r.status_code==200:
            for it in r.json().get("results",[])[:5]:
                url=it.get("resource_url","")
                if "/releases/" not in url: continue
                try: rid=int(url.rstrip("/").split("/")[-1])
                except: continue
                out.append(IdentifyCandidate(
                    source="ocr_search", release_id=rid, discogs_url=f"https://www.discogs.com/release/{rid}",
                    artist=(it.get("title","").split(" - ")[0] if " - " in it.get("title","") else None),
                    title=it.get("title"),
                    label=(it.get("label") or [""])[0] if isinstance(it.get("label"),list) else it.get("label"),
                    year=str(it.get("year") or ""), cover_url=it.get("thumb"), score=0.65))
    except: pass
    return out

# ---------- OCR heuristics ----------
def extract_ocr_metadata(lines: List[str]) -> Tuple[Optional[str],Optional[str],Optional[str],List[str]]:
    label=catno=artist=None; tracks=[]
    for i,ln in enumerate(lines):
        lower=ln.lower()
        m=re.match(r"([a-z0-9\s]+?)\s*(?:promo|pr)?\s*(\d{1,5})$",lower)
        if m and not catno: label,catno=m.group(1).title().strip(),m.group(2); continue
        m2=re.match(r"([a-z0-9\s]+?)\s+(\d{1,5})$",lower)
        if m2 and not catno: label,catno=m2.group(1).title().strip(),m2.group(2); continue
        if not artist and i>0:
            words=ln.strip().split()
            if len(words)<=3 and all(w.isupper() for w in words if w.isalpha()):
                artist=ln.strip().title(); continue
        if ":" in ln: tracks.append(ln.split(":",1)[1].strip()); continue
        if " - " in ln and not re.search(r"\d",ln):
            tracks.append(ln.split(" - ",1)[1].strip())
    return label,catno,artist,tracks

# ---------- CLIP URL→vector cache ----------
def _db():
    conn = sqlite3.connect(EMBED_CACHE_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS url_vectors(url_hash TEXT PRIMARY KEY, dim INTEGER, vec BLOB, updated_at INTEGER)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON url_vectors(updated_at)")
    conn.commit(); return conn

def _h(url:str)->str: return hashlib.sha1(url.encode("utf-8")).hexdigest()

def cache_vec_save(url:str, vec:np.ndarray):
    v = vec.astype(np.float16, copy=False); now=int(time.time())
    conn=_db(); conn.execute("REPLACE INTO url_vectors(url_hash,dim,vec,updated_at) VALUES(?,?,?,?)",(_h(url),v.shape[0],sqlite3.Binary(v.tobytes()),now)); conn.commit(); conn.close()

def cache_vec_load(url:str)->Optional[np.ndarray]:
    cutoff=int(time.time())-EMBED_CACHE_TTL
    conn=_db(); row=conn.execute("SELECT dim,vec,updated_at FROM url_vectors WHERE url_hash=?",(_h(url),)).fetchone()
    if not row: conn.close(); return None
    dim,blob,upd=row
    if upd<cutoff: conn.execute("DELETE FROM url_vectors WHERE url_hash=?",(_h(url),)); conn.commit(); conn.close(); return None
    arr=np.frombuffer(blob,dtype=np.float16).astype(np.float32); conn.close(); return arr.reshape((dim,))

# ---------- CLIP embedder ----------
_CLIP_DEVICE="cpu"
_CLIP_MODEL=_CLIP_PRE=None

def _ensure_clip():
    global _CLIP_MODEL,_CLIP_PRE
    if _CLIP_MODEL is None:
        model,_,pre = open_clip.create_model_and_transforms("ViT-B-32","openai", device=_CLIP_DEVICE)
        model.eval(); _CLIP_MODEL, _CLIP_PRE = model, pre

def _to_img(b:bytes)->Image.Image: return Image.open(io.BytesIO(b)).convert("RGB")

@torch.no_grad()
def embed_image_bytes(img_bytes:bytes)->np.ndarray:
    _ensure_clip(); im=_to_img(img_bytes); ten=_CLIP_PRE(im).unsqueeze(0).to(_CLIP_DEVICE)
    feat=_CLIP_MODEL.encode_image(ten); feat=feat/feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)

@torch.no_grad()
def embed_image_url(url:str)->Optional[np.ndarray]:
    vec=cache_vec_load(url)
    if vec is not None: return vec
    try:
        r=requests.get(url,timeout=10); 
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

# ---------- Route ----------
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_record(file: UploadFile = File(...)) -> IdentifyResponse:
    try:
        image_bytes = await file.read()
        # Vision web+text
        v = call_vision_full(image_bytes)
        web, text = v.get("webDetection",{}), v.get("textAnnotations",[])
        # handwriting pass + block re-OCR
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
            cached = cache_get(release_id)
            if cached:
                candidates.append(IdentifyCandidate(
                    source="web_cache", release_id=release_id, discogs_url=cached["discogs_url"],
                    artist=cached.get("artist"), title=cached.get("title"), label=cached.get("label"),
                    year=cached.get("year"), cover_url=cached.get("cover_url"), score=0.95))
            else:
                rel = fetch_discogs_release_json(release_id)
                if rel:
                    row = {
                        "release_id":release_id,
                        "discogs_url":discogs_url or rel.get("uri") or f"https://www.discogs.com/release/{release_id}",
                        "artist":", ".join(a.get("name","") for a in rel.get("artists",[])),
                        "title":rel.get("title"),
                        "label":", ".join(l.get("name","") for l in rel.get("labels",[])),
                        "year":str(rel.get("year") or ""),
                        "cover_url":rel.get("thumb") or (rel.get("images") or [{}])[0].get("uri",""),
                        "payload":rel,
                    }
                    cache_put(row)
                    candidates.append(IdentifyCandidate(
                        source="web_live", release_id=release_id, discogs_url=row["discogs_url"],
                        artist=row["artist"], title=row["title"], label=row["label"], year=row["year"],
                        cover_url=row["cover_url"], score=0.90
                    ))

        if not candidates and master_id:
            candidates.append(IdentifyCandidate(
                source="web_master", master_id=master_id,
                discogs_url=f"https://www.discogs.com/master/{master_id}",
                note="Master match — select pressing", score=0.60
            ))

        # OCR fallback (full heuristics)
        if not candidates:
            lines = ocr_lines(text)
            # Keep letters/digits/space/hyphen/slash; drop other punct
            clean = [re.sub(r"[^\w\s/-]","",ln).strip() for ln in lines if ln.strip()]

            label, catno, artist, tracks = extract_ocr_metadata(clean)

            # Voting across tracks (hyphen-robust)
            def _vote_releases_by_tracks(_tracks: List[str]) -> List[int]:
                from collections import Counter
                c = Counter()
                tok = os.environ.get("DISCOGS_TOKEN")
                headers = DGS_UA
                def _search_track(tt:str):
                    params={"track":tt,"type":"release"}; 
                    if tok: params["token"]=tok
                    try:
                        r=requests.get(f"{DGS_API}/database/search",params=params,headers=headers,timeout=20)
                        if r.status_code!=200: return
                        for it in r.json().get("results",[])[:8]:
                            url=it.get("resource_url","")
                            if "/releases/" in url:
                                try: rid=int(url.rstrip("/").split("/")[-1]); c[rid]+=1
                                except: pass
                    except: pass
                for t in _tracks:
                    _search_track(t)
                    t_hl = t.replace("-", " ")
                    if t_hl!=t: _search_track(t_hl)
                # keep only >=2 votes
                voted = [(rid,v) for rid,v in c.items() if v>=2]
                voted.sort(key=lambda x:(-x[1], x[0]))
                return [rid for rid,_ in voted[:10]]

            # Build structured attempts
            attempts: List[Dict[str,str]] = []
            if label and catno: attempts.append({"label":label,"catno":catno})
            if catno and artist: attempts.append({"artist":artist,"catno":catno})
            if tracks:
                tracks_hl = list(dict.fromkeys([*tracks, *[t.replace("-"," ") for t in tracks]]))
                voted_ids = _vote_releases_by_tracks(tracks_hl) if len(tracks_hl)>=2 else []
                for rid in voted_ids:
                    rel = fetch_discogs_release_json(rid)
                    if rel: candidates.append(IdentifyCandidate(
                        source="track_vote", **{
                            "release_id":rid, "discogs_url":rel.get("uri") or f"https://www.discogs.com/release/{rid}",
                            "artist":", ".join(a.get("name","") for a in rel.get("artists",[])) or None,
                            "title":rel.get("title"), "label":", ".join(l.get("name","") for l in rel.get("labels",[])) or None,
                            "year":str(rel.get("year") or ""), "cover_url":rel.get("thumb") or (rel.get("images") or [{}])[0].get("uri",""),
                            "score":0.90
                        }
                    ))
                # If still no candidates, queue per-track searches as attempts
                if not candidates:
                    for t in tracks_hl[:4]:
                        attempts.append({"track":t} if not artist else {"track":t,"artist":artist})

            # Joined-tracks fallbacks
            if not candidates and tracks:
                joined = " ".join(list(dict.fromkeys([*tracks, *[t.replace('-',' ') for t in tracks]])))[:200]
                attempts.append({"q": joined})
                if catno: attempts.append({"q": f"{joined} {catno}"[:200]})
                if label and catno: attempts.append({"q": f"{label} {catno} {joined}"[:200]})

            # Generic clean-line fallbacks
            if not candidates and clean:
                attempts.append({"q":" ".join(clean[:3])[:200]})
                if len(clean)>=2: attempts.append({"q":" ".join(clean[:2])[:200]})
                attempts.append({"q":clean[0][:200]})

            # Execute attempts until something hits
            first_hits: List[IdentifyCandidate] = []
            for p in attempts:
                res = discogs_search(p)
                if res:
                    first_hits = res; break
            candidates.extend(first_hits)

        # CLIP visual re-rank if multiple candidates
        if len(candidates)>=2 and image_bytes:
            try:
                ranked = visual_rerank(image_bytes, candidates)
                candidates = [c for (c,_) in ranked]
            except Exception:
                pass

        return IdentifyResponse(candidates=candidates[:5])
    except Exception as exc:
        raise HTTPException(500, str(exc))


