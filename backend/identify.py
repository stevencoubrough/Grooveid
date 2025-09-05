from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
import os
import re
import base64
import requests
from typing import List, Optional, Tuple

VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

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

def call_vision_api(image_bytes: bytes) -> dict:
    if not VISION_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_VISION_API_KEY is not configured")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [
            {
                "image": {"content": b64},
                "features": [
                    {"type": "WEB_DETECTION", "maxResults": 10},
                    {"type": "TEXT_DETECTION", "maxResults": 5},
                ],
                "imageContext": {"webDetectionParams": {"includeGeoResults": True}},
            }
        ]
    }
    response = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload, timeout=30)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Vision API error {response.status_code}: {response.text[:200]}")
    data = response.json()
    return data["responses"][0]

def parse_discogs_web_detection(web_detection: dict) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    urls: List[str] = []
    for key in ("pagesWithMatchingImages","fullMatchingImages","partialMatchingImages","visuallySimilarImages"):
        for item in web_detection.get(key, []):
            url = item.get("url")
            if url:
                urls.append(url)
    release_id = None
    master_id = None
    discogs_url = None
    for u in urls:
        match = RE_RELEASE.search(u)
        if match:
            release_id = int(match.group(1))
            discogs_url = u
            break
    if not release_id:
        for u in urls:
            match = RE_MASTER.search(u)
            if match:
                master_id = int(match.group(1))
                discogs_url = u
                break
    return release_id, master_id, discogs_url

def ocr_lines(text_annotations: List[dict]) -> List[str]:
    if not text_annotations:
        return []
    raw = text_annotations[0].get("description", "")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]

def fetch_release_from_cache(release_id: int) -> Optional[dict]:
    if not supabase:
        return None
    data = supabase.table("discogs_cache").select("*").eq("release_id", release_id).limit(1).execute()
    return data.data[0] if data.data else None

def insert_cache_row(row: dict) -> None:
    if not supabase:
        return
    supabase.table("discogs_cache").upsert(row, on_conflict="release_id").execute()

def fetch_discogs_release_json(release_id: int) -> Optional[dict]:
    headers = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
    url = f"https://api.discogs.com/releases/{release_id}"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def search_discogs_via_ocr(query: str) -> List[IdentifyCandidate]:
    candidates: List[IdentifyCandidate] = []
    headers = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
    params = {"q": query, "type": "release"}
    try:
        r = requests.get("https://api.discogs.com/database/search", params=params, headers=headers, timeout=15)
        if r.status_code == 200:
            js = r.json()
            for it in js.get("results", [])[:5]:
                url = it.get("resource_url", "")
                if "/releases/" in url:
                    rid = int(url.rstrip("/").split("/")[-1])
                    candidates.append(IdentifyCandidate(
                        source="ocr_search",
                        release_id=rid,
                        discogs_url=f"https://www.discogs.com/release/{rid}",
                        artist=(it.get("title", "").split(" - ")[0] if " - " in it.get("title", "") else None),
                        title=it.get("title"),
                        label=(it.get("label") or [""])[0] if isinstance(it.get("label"), list) else it.get("label"),
                        year=str(it.get("year", "")),
                        cover_url=it.get("thumb"),
                        score=0.65,
                    ))
    except Exception:
        pass
    return candidates

@router.post("/identify", response_model=IdentifyResponse)
async def identify_record(file: UploadFile = File(...)) -> IdentifyResponse:
    try:
        image_bytes = await file.read()
        vision_resp = call_vision_api(image_bytes)
        web = vision_resp.get("webDetection", {})
        text_annotations = vision_resp.get("textAnnotations", [])
        release_id, master_id, discogs_url = parse_discogs_web_detection(web)
        candidates: List[IdentifyCandidate] = []
        if release_id:
            cached = fetch_release_from_cache(release_id)
            if cached:
                candidates.append(IdentifyCandidate(
                    source="web_detection_cache",
                    release_id=release_id,
                    discogs_url=cached["discogs_url"],
                    artist=cached.get("artist"),
                    title=cached.get("title"),
                    label=cached.get("label"),
                    year=cached.get("year"),
                    cover_url=cached.get("cover_url"),
                    score=0.95,
                ))
            else:
                release_json = fetch_discogs_release_json(release_id)
                if release_json:
                    row = {
                        "release_id": release_id,
                        "discogs_url": discogs_url or release_json.get("uri") or f"https://www.discogs.com/release/{release_id}",
                        "artist": ", ".join(a.get("name", "") for a in release_json.get("artists", [])),
                        "title": release_json.get("title"),
                        "label": ", ".join(l.get("name", "") for l in release_json.get("labels", [])),
                        "year": str(release_json.get("year", "")),
                        "cover_url": release_json.get("thumb") or (release_json.get("images") or [{}])[0].get("uri", ""),
                        "payload": release_json,
                    }
                    insert_cache_row(row)
                    candidates.append(IdentifyCandidate(
                        source="web_detection_live",
                        release_id=release_id,
                        discogs_url=row["discogs_url"],
                        artist=row.get("artist"),
                        title=row.get("title"),
                        label=row.get("label"),
                        year=row.get("year"),
                        cover_url=row.get("cover_url"),
                        score=0.90,
                    ))
        if not candidates and master_id:
            candidates.append(IdentifyCandidate(
                source="web_detection_master",
                master_id=master_id,
                discogs_url=f"https://www.discogs.com/master/{master_id}",
                note="Master match — prompt user to select a pressing",
                score=0.60,
            ))
        if not candidates:
            lines = ocr_lines(text_annotations)
            if lines:
                query = " ".join(lines[:5])[:200]
                candidates.extend(search_discogs_via_ocr(query))
        return IdentifyResponse(candidates=candidates[:5])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
