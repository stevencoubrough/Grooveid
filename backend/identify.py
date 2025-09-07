"""
World-class record identification backend for GrooveID.

This module enhances the OCR fallback logic to extract useful metadata such
as label names, catalogue numbers, artists, and track titles from the
Google Vision OCR output. It then performs a series of increasingly broad
Discogs searches using these extracted fields. The goal is to match even
obscure promo or white‑label releases when web detection fails.

The search strategy is:

1. Look for a line containing a label name and catalogue number (e.g.,
   "Urban Decay Promo 003"), then query Discogs with `label` and `catno`.
2. If an artist name and catalogue number are found, search with
   `artist` and `catno`.
3. Extract track titles (e.g., lines after a colon) and search by
   `track` plus the artist name if available.
4. Finally, fall back to broad `q` queries assembled from the OCR
   lines, preserving digits and hyphens.

This file is a standalone drop‑in replacement for backend/identify.py.
It requires a valid Discogs user token set in the environment
variable `DISCOGS_TOKEN` and a Google Vision API key in
`GOOGLE_VISION_API_KEY`.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
from supabase import create_client, Client
import os
import re
import base64
import requests

# Google Vision configuration
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_KEY)
    if SUPABASE_URL and SUPABASE_KEY
    else None
)

# Regular expressions for parsing Discogs URLs
RE_RELEASE = re.compile(r"discogs\.com/(?:[^/]+/)?release/(\d+)", re.IGNORECASE)
RE_MASTER = re.compile(r"discogs\.com/(?:[^/]+/)?master/(\d+)", re.IGNORECASE)


class IdentifyCandidate(BaseModel):
    """A single identification candidate returned to the client."""
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
    """Response model for the identify endpoint."""
    candidates: List[IdentifyCandidate]


router = APIRouter()


def call_vision_api(image_bytes: bytes) -> dict:
    """Call the Google Vision API with both web and text detection enabled."""
    if not VISION_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_VISION_API_KEY is not configured",
        )
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [
            {
                "image": {"content": b64},
                "features": [
                    {"type": "WEB_DETECTION", "maxResults": 10},
                    {"type": "TEXT_DETECTION", "maxResults": 5},
                ],
                "imageContext": {
                    "webDetectionParams": {"includeGeoResults": True}
                },
            }
        ]
    }
    resp = requests.post(
        f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload, timeout=30
    )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Vision API error {resp.status_code}: {resp.text[:200]}",
        )
    data = resp.json()
    return data["responses"][0]


def parse_discogs_web_detection(web: dict) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Parse a release or master ID from Discogs URLs returned by Vision."""
    urls: List[str] = []
    for key in (
        "pagesWithMatchingImages",
        "fullMatchingImages",
        "partialMatchingImages",
        "visuallySimilarImages",
    ):
        for item in web.get(key, []):
            url = item.get("url")
            if url:
                urls.append(url)

    release_id = None
    master_id = None
    discogs_url = None
    # Prefer release URLs
    for u in urls:
        match = RE_RELEASE.search(u)
        if match:
            release_id = int(match.group(1))
            discogs_url = u
            break
    # Fallback to master URLs
    if not release_id:
        for u in urls:
            match = RE_MASTER.search(u)
            if match:
                master_id = int(match.group(1))
                discogs_url = u
                break
    return release_id, master_id, discogs_url


def ocr_lines(text_annotations: List[dict]) -> List[str]:
    """Extract a list of lines from the first text annotation."""
    if not text_annotations:
        return []
    raw = text_annotations[0].get("description", "")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def fetch_release_from_cache(release_id: int) -> Optional[dict]:
    """Look up a release in the Supabase cache."""
    if not supabase:
        return None
    data = (
        supabase.table("discogs_cache")
        .select("*")
        .eq("release_id", release_id)
        .limit(1)
        .execute()
    )
    return data.data[0] if data.data else None


def insert_cache_row(row: dict) -> None:
    """Insert or update a cache row in Supabase."""
    if not supabase:
        return
    supabase.table("discogs_cache").upsert(row, on_conflict="release_id").execute()


def fetch_discogs_release_json(release_id: int) -> Optional[dict]:
    """Retrieve a release from the Discogs API."""
    headers = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
    url = f"https://api.discogs.com/releases/{release_id}"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def discogs_search(params: Dict[str, str]) -> List[IdentifyCandidate]:
    """Perform a Discogs search with arbitrary parameters.

    This helper accepts both simple `q` queries and structured searches using
    fields such as `label`, `catno`, `artist`, `release_title`, or `track`.
    It always specifies `type=release` unless overridden in `params`.
    """
    params = params.copy()
    params.setdefault("type", "release")
    token = os.environ.get("DISCOGS_TOKEN")
    if token:
        params["token"] = token
    headers = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
    candidates: List[IdentifyCandidate] = []
    try:
        r = requests.get(
            "https://api.discogs.com/database/search",
            params=params,
            headers=headers,
            timeout=20,
        )
        if r.status_code == 200:
            js = r.json()
            for item in js.get("results", [])[:5]:
                url = item.get("resource_url", "")
                if "/releases/" not in url:
                    continue
                try:
                    rid = int(url.rstrip("/").split("/")[-1])
                except Exception:
                    continue
                candidates.append(
                    IdentifyCandidate(
                        source="ocr_search",
                        release_id=rid,
                        discogs_url=f"https://www.discogs.com/release/{rid}",
                        artist=(item.get("title", "").split(" - ")[0]
                                if " - " in item.get("title", "")
                                else None),
                        title=item.get("title"),
                        label=(item.get("label") or [""])[0]
                        if isinstance(item.get("label"), list)
                        else item.get("label"),
                        year=str(item.get("year") or ""),
                        cover_url=item.get("thumb"),
                        score=0.65,
                    )
                )
    except Exception:
        # Ignore network errors for fallback
        pass
    return candidates


def extract_ocr_metadata(lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """Attempt to extract label, catno, artist, and track titles from OCR lines.

    This heuristic assumes that the label and catalogue number may appear on the
    first line, the artist on a subsequent line, and track titles after a
    colon or dash. It returns a tuple `(label, catno, artist, tracks)`. Any
    missing value is returned as None or an empty list for tracks.
    """
    label = None
    catno = None
    artist = None
    tracks: List[str] = []
    for i, ln in enumerate(lines):
        lower = ln.lower()
        # Look for label + promo/catalog patterns (e.g., "urban decay promo 003")
        promo_match = re.match(r"([a-z0-9\s]+?)\s*(?:promo|pr)?\s*(\d{1,5})$", lower)
        if promo_match and not catno:
            label_candidate = promo_match.group(1).strip()
            catno_candidate = promo_match.group(2)
            if label_candidate:
                label = label_candidate.strip().title()
            catno = catno_candidate.strip()
            continue
        # General label + number pattern (e.g., "urban decay 003")
        gen_match = re.match(r"([a-z0-9\s]+?)\s+(\d{1,5})$", lower)
        if gen_match and not catno:
            label_candidate = gen_match.group(1).strip()
            catno_candidate = gen_match.group(2)
            if label_candidate:
                label = label_candidate.strip().title()
            catno = catno_candidate.strip()
            continue
        # Artist heuristic: uppercase text on its own line
        if not artist and i > 0:
            # Candidate if mostly uppercase letters and spaces and length <= 3 words
            words = ln.strip().split()
            if len(words) <= 3 and all(w.isupper() for w in words if w.isalpha()):
                artist = ln.strip().title()
                continue
        # Track lines: split on colon
        if ":" in ln:
            parts = ln.split(":", 1)
            track = parts[1].strip()
            if track:
                tracks.append(track)
            continue
        # Track lines separated by en dash or hyphen (e.g., "Tech-House Is Dead")
        if " - " in ln and not artist:
            # Might be "Artist - Title"; but if no artist detected, treat as track
            # We avoid this for lines with digits
            if not re.search(r"\d", ln):
                track = ln.split(" - ", 1)[1].strip()
                tracks.append(track)
    return label, catno, artist, tracks


@router.post("/identify", response_model=IdentifyResponse)
async def identify_record(file: UploadFile = File(...)) -> IdentifyResponse:
    """Identify a record from an uploaded image using Vision and Discogs."""
    try:
        image_bytes = await file.read()
        vision_resp = call_vision_api(image_bytes)
        web = vision_resp.get("webDetection", {})
        text = vision_resp.get("textAnnotations", [])
        release_id, master_id, discogs_url = parse_discogs_web_detection(web)
        candidates: List[IdentifyCandidate] = []
        # Primary hit: web detection found release
        if release_id:
            cached = fetch_release_from_cache(release_id)
            if cached:
                candidates.append(
                    IdentifyCandidate(
                        source="web_detection_cache",
                        release_id=release_id,
                        discogs_url=cached["discogs_url"],
                        artist=cached.get("artist"),
                        title=cached.get("title"),
                        label=cached.get("label"),
                        year=cached.get("year"),
                        cover_url=cached.get("cover_url"),
                        score=0.95,
                    )
                )
            else:
                rel = fetch_discogs_release_json(release_id)
                if rel:
                    row = {
                        "release_id": release_id,
                        "discogs_url": discogs_url
                        or rel.get("uri")
                        or f"https://www.discogs.com/release/{release_id}",
                        "artist": ", ".join(a.get("name", "") for a in rel.get("artists", [])),
                        "title": rel.get("title"),
                        "label": ", ".join(l.get("name", "") for l in rel.get("labels", [])),
                        "year": str(rel.get("year", "")),
                        "cover_url": rel.get("thumb")
                        or (rel.get("images") or [{}])[0].get("uri", ""),
                        "payload": rel,
                    }
                    insert_cache_row(row)
                    candidates.append(
                        IdentifyCandidate(
                            source="web_detection_live",
                            release_id=release_id,
                            discogs_url=row["discogs_url"],
                            artist=row.get("artist"),
                            title=row.get("title"),
                            label=row.get("label"),
                            year=row.get("year"),
                            cover_url=row.get("cover_url"),
                            score=0.90,
                        )
                    )
        # Secondary: master ID found
        if not candidates and master_id:
            candidates.append(
                IdentifyCandidate(
                    source="web_detection_master",
                    master_id=master_id,
                    discogs_url=f"https://www.discogs.com/master/{master_id}",
                    note="Master match — prompt user to select a pressing",
                    score=0.60,
                )
            )
        # OCR fallback with heuristics
        if not candidates:
            lines = ocr_lines(text)
            if lines:
                # Clean lines: keep digits, letters, hyphens, slashes; remove punctuation
                clean_lines: List[str] = []
                for ln in lines:
                    cleaned = re.sub(r"[^\w\s/-]", "", ln).strip()
                    if cleaned:
                        clean_lines.append(cleaned)
                label, catno, artist, tracks = extract_ocr_metadata(clean_lines)
                # Build a series of search attempts
                search_attempts: List[Dict[str, str]] = []
                if label and catno:
                    # Search by label and catalog number
                    search_attempts.append({"label": label, "catno": catno})
                if catno and artist:
                    # Search by artist and catalog number
                    search_attempts.append({"artist": artist, "catno": catno})
                if tracks:
                    for t in tracks:
                        params = {"track": t}
                        if artist:
                            params["artist"] = artist
                        search_attempts.append(params)
                # Fallback query assembled from cleaned lines (max 200 chars)
                if clean_lines:
                    fallback_query = " ".join(clean_lines[:3])[:200]
                    search_attempts.append({"q": fallback_query})
                    if len(clean_lines) >= 2:
                        fallback_query2 = " ".join(clean_lines[:2])[:200]
                        search_attempts.append({"q": fallback_query2})
                    # single-line fallback
                    search_attempts.append({"q": clean_lines[0][:200]})
                # Execute search attempts until we get candidates
                for params in search_attempts:
                    results = discogs_search(params)
                    if results:
                        candidates.extend(results)
                        break
        return IdentifyResponse(candidates=candidates[:5])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
