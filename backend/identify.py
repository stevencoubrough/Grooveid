from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
from supabase import create_client, Client
import os, re, base64, requests, time

# ================== ENV / CLIENTS ==================
# API endpoints and keys
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

# Supabase configuration: use provided URL and service role key
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
sb: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
)

# Discogs API configuration
DISCOGS_API = "https://api.discogs.com"
DISCOGS_TOKEN = os.environ.get("DISCOGS_TOKEN", "").strip()

# ---- local dump (optional) ----
# Default table name for local discogs data. Use 'records' as default since that table exists in Supabase.
LOCAL_TABLE = os.environ.get("DISCOGS_LOCAL_TABLE", "records")  # label/catalog_no/artist/title/release_id

# ================== REGEX ==================
# Regular expressions to parse Discogs URLs
RE_RELEASE = re.compile(r"discogs\.com/(?:[^/]+/)?release/(\d+)", re.I)
RE_MASTER = re.compile(r"discogs\.com/(?:[^/]+/)?master/(\d+)", re.I)

# ================== MODELS ==================
class IdentifyCandidate(BaseModel):
    """Representation of an identified discogs candidate."""
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
    """Response returned by the identify endpoint."""
    candidates: List[IdentifyCandidate]
    debug: Optional[Dict[str, Any]] = None  # populated when debug query param is true


router = APIRouter()

# ================== HELPERS: Discogs ==================
def discogs_request(path: str, params: Dict = None, timeout: int = 20):
    """
    Perform an authenticated GET request to the Discogs API.

    Parameters
    ----------
    path: str
        API path or full URL.
    params: Dict
        Query parameters for the request.
    timeout: int
        Timeout in seconds for the request.

    Returns
    -------
    requests.Response
        The HTTP response.
    """
    if params is None:
        params = {}
    headers = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
    # Provide token authentication when available
    if DISCOGS_TOKEN:
        headers["Authorization"] = f"Discogs token={DISCOGS_TOKEN}"
        params.setdefault("token", DISCOGS_TOKEN)
    url = path if path.startswith("http") else f"{DISCOGS_API}{path}"
    return requests.get(url, headers=headers, params=params, timeout=timeout)


# ================== HELPERS: Vision ==================
def call_vision_api(image_bytes: bytes) -> dict:
    """
    Call the Google Vision API to perform web and text detection on the image.

    Parameters
    ----------
    image_bytes: bytes
        Raw image bytes.

    Returns
    -------
    dict
        JSON response from the Vision API.
    """
    if not VISION_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_VISION_API_KEY not set")
    # encode the image to base64
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
    r = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Vision API error {r.text[:200]}")
    return r.json()["responses"][0]


def parse_discogs_web_detection(web: dict) -> Tuple[Optional[int], Optional[int], Optional[str], List[str]]:
    """
    Extract release or master IDs from Discogs URLs found in the Vision WebDetection results.

    Parameters
    ----------
    web: dict
        WebDetection field from Vision API response.

    Returns
    -------
    Tuple containing release_id, master_id, discogs_url, and a list of all candidate URLs found.
    """
    urls: List[str] = []
    # accumulate all candidate URLs
    for key in ("pagesWithMatchingImages", "fullMatchingImages", "partialMatchingImages", "visuallySimilarImages"):
        for item in web.get(key, []):
            if item.get("url"):
                urls.append(item["url"])
    release_id = master_id = None
    discogs_url = None
    # try to find release ID first
    for u in urls:
        m = RE_RELEASE.search(u)
        if m:
            release_id = int(m.group(1))
            discogs_url = u
            break
    # fallback to master ID if no release found
    if not release_id:
        for u in urls:
            m = RE_MASTER.search(u)
            if m:
                master_id = int(m.group(1))
                discogs_url = u
                break
    return release_id, master_id, discogs_url, urls


def ocr_lines(text_annotations: List[dict]) -> List[str]:
    """
    Extract individual lines from the Vision text annotations.

    Parameters
    ----------
    text_annotations: List[dict]
        TextAnnotations from Vision API response.

    Returns
    -------
    List[str]
        List of detected lines, cleaned of whitespace-only lines.
    """
    if not text_annotations:
        return []
    raw = text_annotations[0].get("description", "")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


# ================== HELPERS: Local dump (Supabase) ==================
def norm_label(s: str) -> str:
    """Normalize label strings (collapse whitespace and lower-case)."""
    return re.sub(r"\s+", " ", s).strip().lower()


def norm_catno(s: str) -> str:
    """Normalize catalog numbers (uppercase and unify hyphens)."""
    s = s.upper().strip().replace("  ", " ")
    s = re.sub(r"\s*-\s*", "-", s)  # normalize hyphens
    return s


def local_lookup(label: Optional[str], catno: Optional[str], artist: Optional[str], dbg: Dict) -> List[IdentifyCandidate]:
    """
    Attempt to find local matches in Supabase by label and catalog number (or artist and catalog number).
    Uses the `records` table with columns: release_id, label, catalog_no, artist, title.

    Parameters
    ----------
    label: Optional[str]
        Label name extracted from OCR or heuristics.
    catno: Optional[str]
        Catalog number extracted from OCR.
    artist: Optional[str]
        Artist name extracted from OCR heuristics.
    dbg: Dict
        Debug dictionary to record query metrics.

    Returns
    -------
    List[IdentifyCandidate]
        Up to 5 candidate matches from the local database.
    """
    out: List[IdentifyCandidate] = []
    if not sb or (not label and not artist) or not catno:
        return out
    try:
        # Attempt label + catalog_no match
        t0 = time.time()
        res = (
            sb.table(LOCAL_TABLE)
              .select("release_id, label, catalog_no, artist, title")
              .ilike("label", f"%{label.strip()}%")
              .ilike("catalog_no", norm_catno(catno))
              .limit(10)
              .execute()
        )
        rows = res.data or []
        dbg.setdefault("local_calls", []).append({
            "table": LOCAL_TABLE,
            "label": label,
            "catno": catno,
            "rows": len(rows),
            "elapsed_ms": int((time.time() - t0) * 1000)
        })
        # If no matches on label, try artist + catalog_no
        if not rows and artist:
            t1 = time.time()
            res2 = (
                sb.table(LOCAL_TABLE)
                  .select("release_id, label, catalog_no, artist, title")
                  .ilike("artist", f"%{artist}%")
                  .ilike("catalog_no", norm_catno(catno))
                  .limit(10)
                  .execute()
            )
            rows = res2.data or []
            dbg["local_calls"].append({
                "table": LOCAL_TABLE,
                "artist": artist,
                "catno": catno,
                "rows": len(rows),
                "elapsed_ms": int((time.time() - t1) * 1000)
            })
        # Convert rows to candidates
        for r in rows[:5]:
            rid = r.get("release_id")
            if not rid:
                continue
            out.append(
                IdentifyCandidate(
                    source="local_dump",
                    release_id=int(rid),
                    discogs_url=f"https://www.discogs.com/release/{rid}",
                    artist=r.get("artist"),
                    title=r.get("title"),
                    label=r.get("label"),
                    year=None,
                    cover_url=None,
                    score=0.92  # high confidence for exact catno matches
                )
            )
        return out
    except Exception as e:
        dbg.setdefault("local_calls", []).append({"error": str(e)})
        return out


# ================== HELPERS: Discogs Search ==================
def search_discogs_via_ocr(query: str, dbg: Dict) -> List[IdentifyCandidate]:
    """
    Fallback simple search via Discogs API using a generic query.

    Parameters
    ----------
    query: str
        Generic query string built from OCR lines.
    dbg: Dict
        Debug dictionary to record API call metrics.

    Returns
    -------
    List[IdentifyCandidate]
        Candidate matches from Discogs.
    """
    cands: List[IdentifyCandidate] = []
    params = {"q": query, "type": "release"}
    t0 = time.time()
    r = discogs_request("/database/search", params)
    dbg.setdefault("discogs_calls", []).append({
        "endpoint": "/database/search",
        "params": {"q": query, "type": "release"},
        "status": r.status_code,
        "elapsed_ms": int((time.time() - t0) * 1000),
    })
    if r.status_code == 200:
        js = r.json()
        for it in js.get("results", [])[:8]:
            url = it.get("resource_url", "")
            if "/releases/" not in url:
                continue
            try:
                rid = int(url.rstrip("/").split("/")[-1])
            except:
                continue
            cands.append(
                IdentifyCandidate(
                    source="ocr_search",
                    release_id=rid,
                    discogs_url=f"https://www.discogs.com/release/{rid}",
                    artist=(it.get("title", "").split(" - ")[0] if " - " in it.get("title", "") else None),
                    title=it.get("title"),
                    label=(it.get("label") or [""])[0] if isinstance(it.get("label"), list) else it.get("label"),
                    year=str(it.get("year") or ""),
                    cover_url=it.get("thumb"),
                    score=0.65,
                )
            )
    return cands


def discogs_multi_search(label: Optional[str], catno: Optional[str], artist: Optional[str], tracks: List[str], strong_title: Optional[str], dbg: Dict) -> List[IdentifyCandidate]:
    """
    Perform multiple Discogs searches with increasing looseness.
    Each search is tried in order until results are found.

    Parameters
    ----------
    label: Optional[str]
        Label name.
    catno: Optional[str]
        Catalog number.
    artist: Optional[str]
        Artist name.
    tracks: List[str]
        Track names extracted from OCR lines.
    strong_title: Optional[str]
        Strong title candidate extracted from OCR lines.
    dbg: Dict
        Debug dictionary to record API call metrics.

    Returns
    -------
    List[IdentifyCandidate]
        Candidate matches from Discogs API.
    """
    attempts: List[Dict[str, str]] = []
    # Helper to add search attempts
    def add_attempt(p: Dict[str, str]):
        attempts.append(p)
    # Most specific: label + catno
    if label and catno:
        add_attempt({"label": label, "catno": norm_catno(catno), "type": "release"})
    # Artist + catno
    if artist and catno:
        add_attempt({"artist": artist, "catno": norm_catno(catno), "type": "release"})
    # Track + (artist)
    for t in tracks[:3]:
        p = {"track": t, "type": "release"}
        if artist:
            p["artist"] = artist
        add_attempt(p)
    # Strong title
    if strong_title:
        add_attempt({"release_title": strong_title, "type": "release"})
    # Broad queries combining label, artist, and strong title
    if label and catno:
        add_attempt({"q": f"{label} {catno}", "type": "release"})
    if artist and strong_title:
        add_attempt({"q": f"{artist} {strong_title}", "type": "release"})
    if strong_title:
        add_attempt({"q": strong_title, "type": "release"})
    # Perform attempts in order
    out: List[IdentifyCandidate] = []
    for params in attempts:
        # Make request
        t0 = time.time()
        r = discogs_request("/database/search", params)
        dbg.setdefault("discogs_calls", []).append({
            "endpoint": "/database/search",
            "params": {k: v for k, v in params.items() if k != "token"},
            "status": r.status_code,
            "elapsed_ms": int((time.time() - t0) * 1000),
        })
        if r.status_code == 200:
            js = r.json()
            results = js.get("results", [])[:8]
            for it in results:
                url = it.get("resource_url", "")
                if "/releases/" not in url:
                    continue
                try:
                    rid = int(url.rstrip("/").split("/")[-1])
                except:
                    continue
                out.append(
                    IdentifyCandidate(
                        source="ocr_search",
                        release_id=rid,
                        discogs_url=f"https://www.discogs.com/release/{rid}",
                        artist=(it.get("title", "").split(" - ")[0] if " - " in it.get("title", "") else None),
                        title=it.get("title"),
                        label=(it.get("label") or [""])[0] if isinstance(it.get("label"), list) else it.get("label"),
                        year=str(it.get("year") or ""),
                        cover_url=it.get("thumb"),
                        score=0.70 if ("label" in params and "catno" in params) or ("artist" in params and "catno" in params) else 0.65,
                    )
                )
            # Stop at first non-empty results
            if out:
                break
        # If request fails due to rate limiting or server error, back off and continue
        elif r.status_code in (429, 500, 502, 503):
            time.sleep(1.2)
            continue
    return out


# ================== MAIN ENDPOINT ==================
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_record(
    file: UploadFile = File(...),
    debug: bool = Query(False, description="Set to true to include debug info")
) -> IdentifyResponse:
    """
    Identify a record from an uploaded image using OCR, local database lookups, and Discogs API search.

    Parameters
    ----------
    file: UploadFile
        The uploaded image file (e.g., record label photo).
    debug: bool
        Whether to include debug information in the response.

    Returns
    -------
    IdentifyResponse
        Response containing candidate matches and optional debug info.
    """
    # Initialize debug dictionary if requested
    dbg: Dict[str, Any] = {"steps": [], "web_urls": [], "queries_tried": []} if debug else {}
    try:
        start_time = time.time()
        image_bytes = await file.read()
        # 1) Run Vision API
        vision_start = time.time()
        vision_resp = call_vision_api(image_bytes)
        if debug:
            dbg["steps"].append({"stage": "vision", "elapsed_ms": int((time.time() - vision_start) * 1000)})
        web = vision_resp.get("webDetection", {})
        text = vision_resp.get("textAnnotations", [])
        release_id, master_id, discogs_url, urls = parse_discogs_web_detection(web)
        if debug:
            dbg["web_urls"] = urls
            dbg["web_release_id"] = release_id
            dbg["web_master_id"] = master_id
            dbg["ocr_lines_raw"] = ocr_lines(text)[:40]
        candidates: List[IdentifyCandidate] = []
        # 2) If web detection finds release directly, fetch that release
        if release_id:
            t = time.time()
            rel = discogs_request(f"/releases/{release_id}")
            if debug:
                dbg.setdefault("discogs_calls", []).append({
                    "endpoint": f"/releases/{release_id}",
                    "status": rel.status_code,
                    "elapsed_ms": int((time.time() - t) * 1000)
                })
            if rel.status_code == 200:
                rj = rel.json()
                candidates.append(
                    IdentifyCandidate(
                        source="web_detection_live",
                        release_id=release_id,
                        discogs_url=discogs_url or rj.get("uri"),
                        artist=", ".join(a.get("name", "") for a in rj.get("artists", [])),
                        title=rj.get("title"),
                        label=", ".join(l.get("name", "") for l in rj.get("labels", [])),
                        year=str(rj.get("year", "")),
                        cover_url=rj.get("thumb") or (rj.get("images") or [{}])[0].get("uri", ""),
                        score=0.90
                    )
                )
        # 3) Master ID fallback (if only master found)
        if not candidates and master_id:
            candidates.append(
                IdentifyCandidate(
                    source="web_detection_master",
                    master_id=master_id,
                    discogs_url=f"https://www.discogs.com/master/{master_id}",
                    note="Master match — user must pick a pressing",
                    score=0.60
                )
            )
        # 4) OCR fallback: use local DB and Discogs search
        if not candidates:
            lines = ocr_lines(text)
            if lines:
                # Clean lines: preserve digits/hyphens for catno detection
                clean_lines: List[str] = []
                for ln in lines:
                    cleaned = re.sub(r"[^\w\s/-]", "", ln).strip()
                    if cleaned:
                        clean_lines.append(cleaned)
                if debug:
                    dbg["ocr_lines_clean"] = clean_lines[:40]
                # Extract hints: label, catalog number, artist, track names
                label_hint = None
                catalog_no_hint = None
                artist_hint = None
                track_hints: List[str] = []
                for ln in clean_lines:
                    low = ln.lower()
                    # Find cat numbers: at least two letters followed by digits
                    m = re.search(r"[a-z]{2,}\s?-?\s?\d{1,5}", low)
                    if m and not catalog_no_hint:
                        catalog_no_hint = re.sub(r"\s*-\s*", "-", m.group(0).upper().replace("  ", " "))
                    # Identify label: words ending with 'records', 'recordings', or 'music'
                    if not label_hint and re.search(r"(records|recordings|music)\b", low):
                        label_hint = re.sub(r"(records|recordings|music)\b", "", ln, flags=re.I).strip()
                    # Identify artist: short line of all uppercase words
                    if not artist_hint:
                        words = ln.split()
                        if 1 <= len(words) <= 3 and all(w.isalpha() and w.isupper() for w in words):
                            artist_hint = ln.title()
                    # Track names after ':' or '-'
                    if ":" in ln:
                        t = ln.split(":", 1)[1].strip()
                        if t:
                            track_hints.append(t)
                    elif " - " in ln and not artist_hint:
                        rhs = ln.split(" - ", 1)[1].strip()
                        if rhs and not re.search(r"\d", rhs):
                            track_hints.append(rhs)
                strong_title = clean_lines[0] if clean_lines else None
                if debug:
                    dbg["extracted"] = {
                        "label": label_hint,
                        "catno": catalog_no_hint,
                        "artist": artist_hint,
                        "tracks": track_hints,
                        "strong_title": strong_title,
                    }
                # 4a) Local database lookup
                local_candidates = local_lookup(label_hint, catalog_no_hint, artist_hint, dbg if debug else {})
                if local_candidates:
                    candidates.extend(local_candidates)
                else:
                    # 4b) Discogs searches based on extracted hints
                    # Build fallback queries for debug logging
                    queries = []
                    if clean_lines:
                        queries.append(" ".join(clean_lines[:3])[:200])
                        if len(clean_lines) >= 2:
                            queries.append(" ".join(clean_lines[:2])[:200])
                        queries.append(clean_lines[0][:200])
                    else:
                        queries.append(" ".join(lines[:2])[:200])
                    if debug:
                        dbg["queries_tried"] = queries
                    # Perform structured Discogs searches
                    cands = discogs_multi_search(label_hint, catalog_no_hint, artist_hint, track_hints, strong_title, dbg if debug else {})
                    if not cands:
                        for q in queries:
                            cands = search_discogs_via_ocr(q, dbg if debug else {})
                            if cands:
                                break
                    if cands:
                        candidates.extend(cands)
        # Compose response
        if debug:
            dbg["total_elapsed_ms"] = int((time.time() - start_time) * 1000)
        return IdentifyResponse(candidates=candidates[:10], debug=(dbg or None))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
