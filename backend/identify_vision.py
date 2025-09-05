from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os, base64, requests, re
from supabase import create_client

VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

RE_RELEASE = re.compile(r"discogs\.com/(?:[^/]+/)?release/(\d+)", re.IGNORECASE)
RE_MASTER = re.compile(r"discogs\.com/(?:[^/]+/)?master/(\d+)", re.IGNORECASE)

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

router = APIRouter()

@router.post("/identify_vision", response_model=IdentifyResponse)
async def identify_vision(image: UploadFile = File(...)):
    content = await image.read()
    if not VISION_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_VISION_API_KEY is not configured")
    b64 = base64.b64encode(content).decode("utf-8")
    payload = {
        "requests": [{
            "image": {"content": b64},
            "features": [
                {"type": "WEB_DETECTION", "maxResults": 10},
                {"type": "TEXT_DETECTION", "maxResults": 5}
            ],
            "imageContext": {"webDetectionParams": {"includeGeoResults": True}}
        }]
    }
    resp = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Vision API error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()["responses"][0]
    web = data.get("webDetection", {})
    text_ann = data.get("textAnnotations", [])
    urls = []
    for key in ("pagesWithMatchingImages","fullMatchingImages","partialMatchingImages","visuallySimilarImages"):
        for item in web.get(key, []):
            u = item.get("url")
            if u:
                urls.append(u)
    release_id = None
    discogs_url = None
    for u in urls:
        m = RE_RELEASE.search(u)
        if m:
            release_id = int(m.group(1))
            discogs_url = u
            break
    candidates: List[IdentifyCandidate] = []
    # If release id found, fetch from cache or discogs
    if release_id:
        row = None
        if supabase:
            result = supabase.table("discogs_cache").select("*").eq("release_id", release_id).limit(1).execute()
            row = result.data[0] if result.data else None
        if row:
            candidates.append(IdentifyCandidate(source="web_cache", release_id=release_id, discogs_url=row["discogs_url"], artist=row["artist"], title=row["title"], label=row["label"], year=row["year"], cover_url=row["cover_url"], score=0.95))
        else:
            headers = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
            r = requests.get(f"https://api.discogs.com/releases/{release_id}", headers=headers, timeout=15)
            if r.status_code == 200:
                js = r.json()
                row = {
                    "release_id": release_id,
                    "discogs_url": discogs_url or js.get("uri") or f"https://www.discogs.com/release/{release_id}",
                    "artist": ", ".join([a.get("name","") for a in js.get("artists", [])]),
                    "title": js.get("title"),
                    "label": ", ".join([l.get("name","") for l in js.get("labels", [])]),
                    "year": str(js.get("year") or ""),
                    "cover_url": js.get("thumb") or (js.get("images", [{}])[0].get("uri") if js.get("images") else None),
                    "payload": js
                }
                if supabase:
                    supabase.table("discogs_cache").upsert(row, on_conflict="release_id").execute()
                candidates.append(IdentifyCandidate(source="web_live", release_id=release_id, discogs_url=row["discogs_url"], artist=row["artist"], title=row["title"], label=row["label"], year=row["year"], cover_url=row["cover_url"], score=0.9))
    # Fallback: use OCR text to search discogs
    if not candidates and text_ann:
        query = " ".join([ln.strip() for ln in text_ann[0].get("description", "").splitlines() if ln.strip()])[:200]
        headers = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
        s = requests.get("https://api.discogs.com/database/search", params={"q": query, "type": "release"}, headers=headers, timeout=15)
        if s.status_code == 200:
            js = s.json()
            for it in js.get("results", [])[:5]:
                url = it.get("resource_url", "")
                if "/releases/" in url:
                    rid = int(url.rstrip("/").split("/")[-1])
                    artist_field = None
                    if " - " in it.get("title", ""):
                        artist_field = it.get("title", "").split(" - ")[0]
                    label_field = None
                    lbl = it.get("label")
                    if isinstance(lbl, list) and lbl:
                        label_field = lbl[0]
                    elif isinstance(lbl, str):
                        label_field = lbl
                    candidates.append(IdentifyCandidate(
                        source="ocr_search",
                        release_id=rid,
                        discogs_url=f"https://www.discogs.com/release/{rid}",
                        artist=artist_field,
                        title=it.get("title"),
                        label=label_field,
                        year=str(it.get("year") or ""),
                        cover_url=it.get("thumb"),
                        score=0.65
                    ))
    return IdentifyResponse(candidates=candidates[:5])
