# backend/identify.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
from supabase import create_client, Client
import os, re, base64, requests, io
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

# Google Vision config
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

# Supabase config
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# Regex
RE_RELEASE = re.compile(r"discogs\.com/(?:[^/]+/)?release/(\d+)", re.I)
RE_MASTER  = re.compile(r"discogs\.com/(?:[^/]+/)?master/(\d+)", re.I)

router = APIRouter()

# ---------------- Models ----------------
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

# ---------------- Vision helpers ----------------
def call_vision_api(image_bytes: bytes, feature_type="TEXT_DETECTION") -> dict:
    if not VISION_KEY:
        raise HTTPException(500, "GOOGLE_VISION_API_KEY not set")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [{
            "image": {"content": b64},
            "features": [
                {"type": "WEB_DETECTION", "maxResults": 10},
                {"type": feature_type, "maxResults": 5},
            ],
            "imageContext": {"webDetectionParams": {"includeGeoResults": True}},
        }]
    }
    r = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload, timeout=30)
    if r.status_code != 200:
        raise HTTPException(502, f"Vision error {r.status_code}: {r.text[:200]}")
    return r.json().get("responses", [{}])[0]

def parse_discogs_web(web: dict) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    urls = []
    for k in ("pagesWithMatchingImages","fullMatchingImages","partialMatchingImages","visuallySimilarImages"):
        for it in web.get(k, []):
            if it.get("url"): urls.append(it["url"])
    rel = mast = None; url = None
    for u in urls:
        m = RE_RELEASE.search(u)
        if m: rel, url = int(m.group(1)), u; break
    if not rel:
        for u in urls:
            m = RE_MASTER.search(u)
            if m: mast, url = int(m.group(1)), u; break
    return rel, mast, url

def ocr_lines(text_ann: List[dict]) -> List[str]:
    if not text_ann: return []
    raw = text_ann[0].get("description","")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]

# ---------------- Extra OCR helpers ----------------
def enhance_for_handwriting(img: Image.Image) -> Image.Image:
    w,h = img.size; s = int(min(w,h)*0.7); cx,cy = w//2,h//2
    crop = img.crop((cx-s//2, cy-s//2, cx+s//2, cy+s//2))
    gray = ImageOps.grayscale(crop)
    sharp = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
    boosted = ImageEnhance.Contrast(sharp).enhance(2.2)
    return ImageOps.equalize(boosted)

def handwriting_merge(img_bytes: bytes, text: List[dict]) -> List[dict]:
    try:
        base_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        hw_img = enhance_for_handwriting(base_img)
        buf = io.BytesIO(); hw_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        payload = {"requests":[{"image":{"content":b64},"features":[{"type":"DOCUMENT_TEXT_DETECTION"}]}]}
        r = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload, timeout=30)
        if r.status_code != 200: return text
        v2 = r.json().get("responses",[{}])[0]
        extra = []
        if v2.get("textAnnotations"):
            raw = v2["textAnnotations"][0].get("description","")
            extra = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        primary = [ln.strip() for ln in (text[0].get("description","").splitlines() if text else []) if ln.strip()]
        merged = list(dict.fromkeys([*primary, *extra]))
        return [{"description":"\n".join(merged)}] if merged else text
    except Exception:
        return text

def block_crop_reocr(img_bytes: bytes) -> List[str]:
    """Crop each text block from DOC OCR and re-run OCR."""
    lines = []
    try:
        vdoc = call_vision_api(img_bytes, "DOCUMENT_TEXT_DETECTION")
        base_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        for page in vdoc.get("fullTextAnnotation",{}).get("pages",[]):
            for block in page.get("blocks",[]):
                verts = block.get("boundingBox",{}).get("vertices",[])
                if len(verts)==4:
                    x = min(v.get("x",0) for v in verts); y = min(v.get("y",0) for v in verts)
                    X = max(v.get("x",0) for v in verts); Y = max(v.get("y",0) for v in verts)
                    crop = base_img.crop((x,y,X,Y)).resize((int(1.5*(X-x)), int(1.5*(Y-y))))
                    buf = io.BytesIO(); crop.save(buf, format="PNG")
                    vsmall = call_vision_api(buf.getvalue(), "DOCUMENT_TEXT_DETECTION")
                    if vsmall.get("textAnnotations"):
                        raw = vsmall["textAnnotations"][0].get("description","")
                        lines.extend([ln.strip() for ln in raw.splitlines() if ln.strip()])
    except Exception:
        pass
    return lines

# ---------------- Discogs helpers ----------------
def fetch_release_from_cache(rid: int) -> Optional[dict]:
    if not supabase: return None
    data = supabase.table("discogs_cache").select("*").eq("release_id", rid).limit(1).execute()
    return data.data[0] if data.data else None

def insert_cache_row(row: dict): 
    if supabase: supabase.table("discogs_cache").upsert(row,on_conflict="release_id").execute()

def fetch_discogs_release_json(rid: int) -> Optional[dict]:
    try:
        r = requests.get(f"https://api.discogs.com/releases/{rid}", headers={"User-Agent":"GrooveID/1.0"}, timeout=15)
        return r.json() if r.status_code==200 else None
    except: return None

def discogs_search(params: Dict[str,str]) -> List[IdentifyCandidate]:
    params = params.copy(); params.setdefault("type","release")
    token = os.environ.get("DISCOGS_TOKEN")
    if token: params["token"]=token
    out=[]
    try:
        r = requests.get("https://api.discogs.com/database/search",params=params,headers={"User-Agent":"GrooveID/1.0"},timeout=20)
        if r.status_code==200:
            for it in r.json().get("results",[])[:5]:
                url = it.get("resource_url","")
                if "/releases/" not in url: continue
                try: rid=int(url.rstrip("/").split("/")[-1])
                except: continue
                out.append(IdentifyCandidate(
                    source="ocr_search", release_id=rid,
                    discogs_url=f"https://www.discogs.com/release/{rid}",
                    artist=(it.get("title","").split(" - ")[0] if " - " in it.get("title","") else None),
                    title=it.get("title"),
                    label=(it.get("label") or [""])[0] if isinstance(it.get("label"),list) else it.get("label"),
                    year=str(it.get("year") or ""),
                    cover_url=it.get("thumb"), score=0.65
                ))
    except: pass
    return out

# ---------------- OCR metadata heuristics ----------------
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
        if " - " in ln and not re.search(r"\d",ln): tracks.append(ln.split(" - ",1)[1].strip())
    return label,catno,artist,tracks

# ---------------- Main route ----------------
@router.post("/identify", response_model=IdentifyResponse)
async def identify_record(file: UploadFile=File(...)) -> IdentifyResponse:
    try:
        img = await file.read()
        v = call_vision_api(img)
        web,text = v.get("webDetection",{}),v.get("textAnnotations",[])
        text = handwriting_merge(img,text)  # graffiti OCR
        extra_lines = block_crop_reocr(img) # block re-OCR
        if extra_lines:
            merged = [ln.strip() for ln in (text[0].get("description","").splitlines() if text else []) if ln.strip()]
            merged.extend(extra_lines); text=[{"description":"\n".join(dict.fromkeys(merged))}]

        rid,mid,url = parse_discogs_web(web)
        candidates=[]
        # Web hit
        if rid:
            cached=fetch_release_from_cache(rid)
            if cached:
                candidates.append(IdentifyCandidate(source="web_cache",release_id=rid,discogs_url=cached["discogs_url"],
                    artist=cached.get("artist"),title=cached.get("title"),label=cached.get("label"),
                    year=cached.get("year"),cover_url=cached.get("cover_url"),score=0.95))
            else:
                rel=fetch_discogs_release_json(rid)
                if rel:
                    row={"release_id":rid,"discogs_url":url or rel.get("uri") or f"https://www.discogs.com/release/{rid}",
                         "artist":", ".join(a.get("name","") for a in rel.get("artists",[])),
                         "title":rel.get("title"),"label":", ".join(l.get("name","") for l in rel.get("labels",[])),
                         "year":str(rel.get("year") or ""),"cover_url":rel.get("thumb") or (rel.get("images") or [{}])[0].get("uri",""),"payload":rel}
                    insert_cache_row(row)
                    candidates.append(IdentifyCandidate(source="web_live",release_id=rid,discogs_url=row["discogs_url"],
                        artist=row["artist"],title=row["title"],label=row["label"],year=row["year"],cover_url=row["cover_url"],score=0.90))
        if not candidates and mid:
            candidates.append(IdentifyCandidate(source="web_master",master_id=mid,
                discogs_url=f"https://www.discogs.com/master/{mid}",note="Master match — select pressing",score=0.60))

        # OCR fallback
        if not candidates:
            lines=ocr_lines(text); clean=[re.sub(r"[^\w\s/-]","",ln).strip() for ln in lines if ln.strip()]
            label,catno,artist,tracks = extract_ocr_metadata(clean)
            attempts=[]
            if label and catno: attempts.append({"label":label,"catno":catno})
            if catno and artist: attempts.append({"artist":artist,"catno":catno})
            if tracks: 
                for t in tracks: attempts.append({"track":t,"artist":artist} if artist else {"track":t})
            if clean:
                attempts.append({"q":" ".join(clean[:3])[:200]})
                if len(clean)>=2: attempts.append({"q":" ".join(clean[:2])[:200]})
                attempts.append({"q":clean[0][:200]})
            for p in attempts:
                res=discogs_search(p)
                if res: candidates.extend(res); break

        return IdentifyResponse(candidates=candidates[:5])
    except Exception as e:
        raise HTTPException(500,str(e))


