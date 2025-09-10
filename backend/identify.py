from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
from supabase import create_client, Client
import os, re, base64, requests, time

# ---------------- Env Config ----------------
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
)

DISCOGS_API = "https://api.discogs.com"
DISCOGS_TOKEN = os.environ.get("DISCOGS_TOKEN", "").strip()

# ---------------- Regex ----------------
RE_RELEASE = re.compile(r"discogs\.com/(?:[^/]+/)?release/(\d+)", re.I)
RE_MASTER = re.compile(r"discogs\.com/(?:[^/]+/)?master/(\d+)", re.I)

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

router = APIRouter()

# ---------------- Discogs Auth Helper ----------------
def discogs_request(path: str, params: Dict = None, timeout=20):
    if params is None:
        params = {}
    headers = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
    if DISCOGS_TOKEN:
        headers["Authorization"] = f"Discogs token={DISCOGS_TOKEN}"
        params.setdefault("token", DISCOGS_TOKEN)

    url = path if path.startswith("http") else f"{DISCOGS_API}{path}"
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    return r

# ---------------- Vision ----------------
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
        raise HTTPException(status_code=500, detail=f"Vision API error {r.text[:200]}")
    return r.json()["responses"][0]

# ---------------- Helpers ----------------
def parse_discogs_web_detection(web: dict) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    urls = []
    for key in ("pagesWithMatchingImages","fullMatchingImages","partialMatchingImages","visuallySimilarImages"):
        for item in web.get(key, []):
            if item.get("url"):
                urls.append(item["url"])

    release_id = master_id = None
    discogs_url = None

    for u in urls:
        m = RE_RELEASE.search(u)
        if m:
            release_id = int(m.group(1))
            discogs_url = u
            break
    if not release_id:
        for u in urls:
            m = RE_MASTER.search(u)
            if m:
                master_id = int(m.group(1))
                discogs_url = u
                break
    return release_id, master_id, discogs_url

def ocr_lines(text_annotations: List[dict]) -> List[str]:
    if not text_annotations: return []
    raw = text_annotations[0].get("description", "")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]

def fetch_discogs_release_json(release_id: int) -> Optional[dict]:
    r = discogs_request(f"/releases/{release_id}")
    return r.json() if r.status_code == 200 else None

def search_discogs_via_ocr(query: str) -> List[IdentifyCandidate]:
    candidates: List[IdentifyCandidate] = []
    params = {"q": query, "type": "release"}
    r = discogs_request("/database/search", params)
    if r.status_code == 200:
        js = r.json()
        for it in js.get("results", [])[:5]:
            url = it.get("resource_url", "")
            if "/releases/" not in url:
                continue
            try:
                rid = int(url.rstrip("/").split("/")[-1])
            except:
                continue
            candidates.append(
                IdentifyCandidate(
                    source="ocr_search",
                    release_id=rid,
                    discogs_url=f"https://www.discogs.com/release/{rid}",
                    artist=(it.get("title","").split(" - ")[0] if " - " in it.get("title","") else None),
                    title=it.get("title"),
                    label=(it.get("label") or [""])[0] if isinstance(it.get("label"), list) else it.get("label"),
                    year=str(it.get("year") or ""),
                    cover_url=it.get("thumb"),
                    score=0.65,
                )
            )
    return candidates

# ---------------- Main Endpoint ----------------
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_record(file: UploadFile = File(...)) -> IdentifyResponse:
    try:
        image_bytes = await file.read()
        vision_resp = call_vision_api(image_bytes)
        web = vision_resp.get("webDetection", {})
        text = vision_resp.get("textAnnotations", [])

        release_id, master_id, discogs_url = parse_discogs_web_detection(web)
        candidates: List[IdentifyCandidate] = []

        # Case A: direct release from Vision
        if release_id:
            rel = fetch_discogs_release_json(release_id)
            if rel:
                candidates.append(
                    IdentifyCandidate(
                        source="web_detection_live",
                        release_id=release_id,
                        discogs_url=discogs_url or rel.get("uri"),
                        artist=", ".join(a.get("name","") for a in rel.get("artists", [])),
                        title=rel.get("title"),
                        label=", ".join(l.get("name","") for l in rel.get("labels", [])),
                        year=str(rel.get("year", "")),
                        cover_url=rel.get("thumb") or (rel.get("images") or [{}])[0].get("uri", ""),
                        score=0.90,
                    )
                )

        # Case B: master match only
        if not candidates and master_id:
            candidates.append(
                IdentifyCandidate(
                    source="web_detection_master",
                    master_id=master_id,
                    discogs_url=f"https://www.discogs.com/master/{master_id}",
                    note="Master match — user must pick a pressing",
                    score=0.60,
                )
            )

        # Case C: OCR fallback
        if not candidates:
            lines = ocr_lines(text)
            if lines:
                clean_lines = []
                for ln in lines:
                    cleaned = re.sub(r"[^\w\s/-]", "", ln).strip()
                    if cleaned:
                        clean_lines.append(cleaned)
                queries = []
                if clean_lines:
                    queries.append(" ".join(clean_lines[:3])[:200])
                    if len(clean_lines) >= 2:
                        queries.append(" ".join(clean_lines[:2])[:200])
                    queries.append(clean_lines[0][:200])
                else:
                    queries.append(" ".join(lines[:2])[:200])

                for q in queries:
                    ocr_candidates = search_discogs_via_ocr(q)
                    if ocr_candidates:
                        candidates.extend(ocr_candidates)
                        break

        return IdentifyResponse(candidates=candidates[:5])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


