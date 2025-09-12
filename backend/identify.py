identify.py — GrooveID consolidated resolver

This module exposes a single POST endpoint `/api/identify` for the GrooveID
backend.  Given an uploaded image of a vinyl label, it tries to identify the
corresponding Discogs release using a cascaded set of heuristics:

1. **Vision Web Detection** — Ask Google Vision for web entities on the
   image.  If a Discogs release link appears in `webDetection`, the
   endpoint hydrates it via the Discogs API and returns it immediately.
2. **Vision OCR** — Merge all text from both `TEXT_DETECTION` and
   `DOCUMENT_TEXT_DETECTION` into a de‑duplicated list of lines.  Join
   broken track titles (e.g. `AL. BEHIND` followed by `THE WHEEL`) into
   single lines.  Extract a catalogue number, label, artist and up to
   three track titles from the cleaned lines.
3. **Google Custom Search** — Build a query of the form

       ``site:discogs.com "<track1>" "<track2>" <catalogue>``

   and call the Programmable Search API.  Score each returned link,
   preferring `/release/` over `/master/` and boosting when the
   catalogue, artist, title or tracks appear in the snippet/title.  The
   top link is hydrated via Discogs.
4. **Supabase RPC fallback** — If Vision and Google fail, call
   `search_records` on your Supabase database.  This RPC weights
   catalogue matches and trigram similarity on title/artist/label.  It
   returns the top few candidate rows, which are surfaced as
   high‑confidence guesses.
5. **Legacy local lookup** — If RPC is unavailable or returns nothing,
   query your dump table (`DISCOGS_LOCAL_TABLE`) by catalogue number
   alone, then label+catalogue, then artist+catalogue.
6. **Discogs structured search** — As a last resort, call the Discogs
   `/database/search` endpoint with structured parameters like
   label+catno, artist+catno or track names.

Debug information is returned when `?debug=true` is passed: OCR lines,
extracted signals, queries and timings.  All search hints are derived
dynamically from the OCR output—no artist names or track titles are
hard‑coded in this file.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
import os
import re
import base64
import time
import requests

# Optional Supabase client (used as a fallback/local index)
try:
    from supabase import create_client, Client  # type: ignore
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    sb: Optional["Client"] = (
        create_client(SUPABASE_URL, SUPABASE_KEY)
        if (SUPABASE_URL and SUPABASE_KEY)
        else None
    )
except Exception:
    sb = None

# ---------- ENV VARIABLES ----------
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()

DISCOGS_API = "https://api.discogs.com"
DISCOGS_TOKEN = os.environ.get("DISCOGS_TOKEN", "").strip()

GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY", "").strip()
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "").strip()

# Supabase local table (defaults to your dump table name 'records')
LOCAL_TABLE = os.environ.get("DISCOGS_LOCAL_TABLE", "records")

router = APIRouter()


# ---------- MODELS ----------
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
    debug: Optional[Dict[str, Any]] = None  # populated when debug=true


# ---------- HTTP / API HELPERS ----------
def discogs_request(path: str, params: Dict = None, timeout: int = 20) -> requests.Response:
    """Make a GET request to the Discogs API, attaching the token and user agent."""
    if params is None:
        params = {}
    headers: Dict[str, str] = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
    if DISCOGS_TOKEN:
        headers["Authorization"] = f"Discogs token={DISCOGS_TOKEN}"
        params.setdefault("token", DISCOGS_TOKEN)
    url = path if path.startswith("http") else f"{DISCOGS_API}{path}"
    return requests.get(url, headers=headers, params=params, timeout=timeout)


def call_vision_api(image_bytes: bytes) -> dict:
    """Call the Google Vision API for WebDetection and OCR."""
    if not VISION_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_VISION_API_KEY not set")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [
            {
                "image": {"content": b64},
                "features": [
                    {"type": "WEB_DETECTION", "maxResults": 10},
                    {"type": "TEXT_DETECTION", "maxResults": 5},
                    {"type": "DOCUMENT_TEXT_DETECTION"},
                ],
                "imageContext": {
                    "webDetectionParams": {"includeGeoResults": True},
                    "languageHints": ["en", "fr", "de", "es", "ru"],
                },
            }
        ]
    }
    r = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload, timeout=30)
    if r.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Vision API error {r.status_code}: {r.text[:200]}",
        )
    return r.json().get("responses", [{}])[0]


def parse_web(web: dict) -> Tuple[Optional[int], Optional[int], Optional[str], List[str]]:
    """Parse Vision webDetection results and return release_id, master_id, the first Discogs URL, and all URLs."""
    urls: List[str] = []
    for key in (
        "pagesWithMatchingImages",
        "fullMatchingImages",
        "partialMatchingImages",
        "visuallySimilarImages",
    ):
        for item in web.get(key, []):
            u = item.get("url")
            if u:
                urls.append(u)
    release_id = master_id = None
    discogs_url = None
    for u in urls:
        m = re.search(r"discogs\.com/(?:[^/]+/)?release/(\d+)", u, re.I)
        if m:
            release_id = int(m.group(1))
            discogs_url = u
            break
    if not release_id:
        for u in urls:
            m = re.search(r"discogs\.com/(?:[^/]+/)?master/(\d+)", u, re.I)
            if m:
                master_id = int(m.group(1))
                discogs_url = u
                break
    return release_id, master_id, discogs_url, urls


def merge_google_ocr(resp: dict) -> List[str]:
    """Merge OCR output from Google Vision TEXT_DETECTION and DOCUMENT_TEXT_DETECTION."""
    lines: List[str] = []
    fta = resp.get("fullTextAnnotation") or {}
    txt = fta.get("text")
    if txt:
        lines.extend([ln.strip() for ln in txt.splitlines() if ln.strip()])
    # The first element of textAnnotations is the full block; skip it
    for t in resp.get("textAnnotations", [])[1:]:
        d = t.get("description")
        if d:
            lines.extend([ln.strip() for ln in d.splitlines() if ln.strip()])
    # De‑duplicate, preserving order
    out: List[str] = []
    seen = set()
    for ln in lines:
        key = ln.lower()
        if key not in seen:
            seen.add(key)
            out.append(ln)
    return out


def ocr_lines(resp: dict) -> List[str]:
    return merge_google_ocr(resp)


def norm_catno(s: Optional[str]) -> Optional[str]:
    """Normalise catalogue number: uppercase, collapse spaces, preserve hyphens."""
    if not s:
        return None
    s = s.upper().strip()
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s


# ---------- TEXT NORMALISATION HELPERS ----------

# Pattern to ignore record rim text for title extraction
RIM_PREFIX_RE = re.compile(r"^all rights of the manufacturer", re.I)


def join_broken_tracks(lines: List[str]) -> List[str]:
    """
    Join lines where a track title is broken across multiple OCR lines.  Many
    white‑label scans split track titles over two lines, e.g. "AL. BEHIND" then
    "THE WHEEL".  We only treat lines that begin with a side/track prefix
    starting with the letters 'A' or 'B' as candidates for joining.  The
    prefix may include an optional additional letter or digit (e.g., "AL",
    "A1", "B2"), an optional dot, and optional whitespace.  Other lines
    (e.g., "SIDE", "TRON", etc.) should not be modified here.
    """
    result: List[str] = []
    skip = False
    for i, ln in enumerate(lines):
        if skip:
            skip = False
            continue
        # Consider prefixes that start with A or B followed by at most one
        # alphanumeric character and optional dot.  Do not match arbitrary
        # two-letter prefixes like 'SI' from "SIDE" or 'TR' from "TRON".
        if re.match(r"^[AB][A-Za-z0-9]?\.?\s*", ln):
            # Remove the detected prefix and any whitespace following it.
            title = re.sub(r"^[AB][A-Za-z0-9]?\.?\s*", "", ln).strip()
            # Join with the next line if the remainder is empty and the next
            # line exists.  This handles cases where the first line is just
            # the prefix.
            if not title and i + 1 < len(lines) and lines[i + 1].strip():
                title = lines[i + 1].strip()
                skip = True
            # If there is non-empty remainder and the next line looks like
            # part of the title (starts with an uppercase letter or digit),
            # append it.  This helps join cases like "AL. BEHIND" + "THE WHEEL".
            elif title and i + 1 < len(lines) and re.match(r"^[A-Za-z0-9]", lines[i + 1]):
                title = f"{title} {lines[i + 1].strip()}".strip()
                skip = True
            result.append(title)
        else:
            result.append(ln)
    return result


def google_cse_discogs(query: str, dbg: Dict, signals: Dict[str, Any]) -> Optional[str]:
    """Return the best Discogs URL using Google Custom Search Engine, or None if nothing fits."""
    api = GOOGLE_SEARCH_API_KEY
    cx = GOOGLE_CSE_ID
    if not api or not cx or not query:
        return None
    params = {"key": api, "cx": cx, "q": query, "num": 5, "safe": "off"}
    t0 = time.time()
    r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=15)
    dbg.setdefault("google_calls", []).append(
        {
            "endpoint": "customsearch/v1",
            "query": query,
            "status": r.status_code,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
    )
    if r.status_code != 200 or not r.json().get("items"):
        return None
    best, best_score = None, -1.0
    cat = (signals.get("p_catno") or "").lower()
    art = (signals.get("p_artist") or "").lower()
    ttl = (signals.get("p_title") or "").lower()
    tracks = signals.get("tracks") or []
    track1 = tracks[0].lower() if len(tracks) > 0 else None
    track2 = tracks[1].lower() if len(tracks) > 1 else None
    for item in r.json().get("items", []):
        link = item.get("link", "")
        # Only consider Discogs links
        if "discogs.com" not in link:
            continue
        title = (item.get("title") or "").lower()
        snip = (item.get("snippet") or "").lower()
        corpus = f"{title} {snip}"
        score = 0.0
        if "/release/" in link:
            score += 3.0
        if "/master/" in link:
            score += 1.0
        if cat and cat in corpus:
            score += 2.0
        if art and art in corpus:
            score += 1.2
        if ttl and ttl in corpus:
            score += 1.0
        if track1 and track1 in corpus:
            score += 1.1
        if track2 and track2 in corpus:
            score += 0.9
        if score > best_score:
            best_score = score
            best = link
    return best


# ---------- SUPABASE RPC ----------
def rpc_search_records(signals: Dict[str, Any], dbg: Dict) -> List[Dict[str, Any]]:
    """Call Supabase RPC search_records with extracted signals."""
    out: List[Dict[str, Any]] = []
    if not sb:
        dbg.setdefault("local_calls", []).append({"rpc": "skipped_no_client"})
        return out
    try:
        t0 = time.time()
        res = sb.rpc(
            "search_records",
            {
                "p_catno": signals.get("p_catno") or None,
                "p_label": signals.get("p_label") or None,
                "p_artist": signals.get("p_artist") or None,
                "p_title": signals.get("p_title") or None,
            },
        ).execute()
        rows = res.data or []
        dbg.setdefault("local_calls", []).append(
            {
                "rpc": "search_records",
                "rows": len(rows),
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
        )
        return rows
    except Exception as e:
        dbg.setdefault("local_calls", []).append({"rpc_error": str(e)})
        return out


# ---------- LEGACY SUPABASE LOOKUP ----------
def local_lookup(catno: Optional[str], label: Optional[str], artist: Optional[str], dbg: Dict) -> List[IdentifyCandidate]:
    out: List[IdentifyCandidate] = []
    if not sb or not catno:
        return out
    try:
        # 1) catalogue only
        res0 = (
            sb.table(LOCAL_TABLE)
            .select("release_id,label,catalog_no,artist,title,discogs_url")
            .ilike("catalog_no", norm_catno(catno) or "")
            .limit(12)
            .execute()
        )
        rows = res0.data or []
        dbg.setdefault("local_calls", []).append({"mode": "catalog_no_only", "rows": len(rows)})
        # 2) label + catalogue
        if not rows and label:
            res = (
                sb.table(LOCAL_TABLE)
                .select("release_id,label,catalog_no,artist,title,discogs_url")
                .ilike("label", f"%{label.strip()}%")
                .ilike("catalog_no", norm_catno(catno) or "")
                .limit(12)
                .execute()
            )
            rows = res.data or []
            dbg["local_calls"].append({"mode": "label+catalog_no", "rows": len(rows)})
        # 3) artist + catalogue
        if not rows and artist:
            res2 = (
                sb.table(LOCAL_TABLE)
                .select("release_id,label,catalog_no,artist,title,discogs_url")
                .ilike("artist", f"%{artist}%")
                .ilike("catalog_no", norm_catno(catno) or "")
                .limit(12)
                .execute()
            )
            rows = res2.data or []
            dbg["local_calls"].append({"mode": "artist+catalog_no", "rows": len(rows)})
        # Build candidate list from rows
        def parse_rid(url: Optional[str]) -> Optional[int]:
            if not url:
                return None
            m = re.search(r"/release/(\d+)", url)
            return int(m.group(1)) if m else None
        for r in rows[:5]:
            rid = r.get("release_id") or parse_rid(r.get("discogs_url"))
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
                    score=0.92,
                )
            )
        return out
    except Exception as e:
        dbg.setdefault("local_calls", []).append({"error": str(e)})
        return out


# ---------- DISCOGS STRUCTURED FALLBACK ----------
def search_discogs(params: Dict[str, str], dbg: Dict) -> List[IdentifyCandidate]:
    out: List[IdentifyCandidate] = []
    t0 = time.time()
    r = discogs_request("/database/search", params)
    dbg.setdefault("discogs_calls", []).append(
        {
            "endpoint": "/database/search",
            "params": {k: v for k, v in params.items() if k != "token"},
            "status": r.status_code,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
    )
    if r.status_code != 200:
        return out
    js = r.json()
    for it in js.get("results", [])[:8]:
        url = it.get("resource_url", "")
        if "/releases/" not in url:
            continue
        try:
            rid = int(url.rstrip("/").split("/")[-1])
        except ValueError:
            continue
        out.append(
            IdentifyCandidate(
                source="ocr_search",
                release_id=rid,
                discogs_url=f"https://www.discogs.com/release/{rid}",
                artist=(
                    it.get("title", "").split(" - ")[0]
                    if " - " in it.get("title", "")
                    else None
                ),
                title=it.get("title"),
                label=(it.get("label") or [""])[0]
                if isinstance(it.get("label"), list)
                else it.get("label"),
                year=str(it.get("year") or ""),
                cover_url=it.get("thumb"),
                score=0.65,
            )
        )
    return out


# ---------- API ROUTE ----------
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_api(
    file: UploadFile = File(...),
    debug: bool = Query(False, description="Return debug info when true"),
) -> IdentifyResponse:
    dbg: Dict[str, Any] = {"steps": [], "web_urls": [], "queries_tried": []} if debug else {}
    try:
        t0 = time.time()
        image_bytes = await file.read()
        # 1) Vision
        v = call_vision_api(image_bytes)
        web = v.get("webDetection", {})
        release_id, master_id, discogs_url, urls = parse_web(web)
        if debug:
            dbg["web_urls"] = urls
            dbg["web_release_id"] = release_id
            dbg["web_master_id"] = master_id
        candidates: List[IdentifyCandidate] = []
        # A) Immediate hit via Web Detection
        if release_id:
            rel = discogs_request(f"/releases/{release_id}")
            if debug:
                dbg.setdefault("discogs_calls", []).append(
                    {
                        "endpoint": f"/releases/{release_id}",
                        "status": rel.status_code,
                    }
                )
            if rel.status_code == 200:
                js = rel.json()
                candidates.append(
                    IdentifyCandidate(
                        source="web_detection_live",
                        release_id=release_id,
                        discogs_url=discogs_url or js.get("uri"),
                        artist=", ".join(a.get("name", "") for a in js.get("artists", [])),
                        title=js.get("title"),
                        label=", ".join(l.get("name", "") for l in js.get("labels", [])),
                        year=str(js.get("year") or ""),
                        cover_url=js.get("thumb")
                        or (js.get("images") or [{}])[0].get("uri", ""),
                        score=0.90,
                    )
                )
        # B) Master only
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
        # C) Text-driven: Google CSE → Supabase → Discogs
        if not candidates:
            lines = ocr_lines(v)
            if debug:
                dbg["ocr_lines_raw"] = lines[:200]
            if lines:
                # Clean lines: remove extraneous punctuation but keep dots and colons.
                # We keep '.' and ':' so that track prefixes like "AL." and "2."
                # survive long enough for join_broken_tracks and track extraction.
                clean: List[str] = []
                for ln in lines:
                    cleaned = re.sub(r"[^\w\s./:-]", "", ln).strip()
                    if cleaned:
                        clean.append(cleaned)
                # Join broken track names
                clean = join_broken_tracks(clean)
                if debug:
                    dbg["ocr_lines_clean"] = clean[:200]
                # Extract signals
                catalog_no_hint: Optional[str] = None
                label_hint: Optional[str] = None
                artist_hint: Optional[str] = None
                tracks: List[str] = []
                # Catalogue number: letters followed by digits; ignore 'VOLUME'
                for ln in clean:
                    up = ln.upper()
                    if "VOLUME" in up:
                        continue
                    m = re.search(r"\b([A-Z]{3,})\s*-?\s*(\d{1,5})\b", up)
                    if m and not catalog_no_hint:
                        catalog_no_hint = f"{m.group(1)}{m.group(2)}"
                # Label: look for Records/Recordings/Music
                for ln in clean:
                    if re.search(r"(records|recordings|music)\b", ln, re.I):
                        label_hint = re.sub(r"(records|recordings|music)\b", "", ln, flags=re.I).strip()
                        break
                # Artist: short all‑caps line (<=3 words) that does not look like a side
                # marker or volume indicator.  Skip lines containing words such as
                # "SIDE" or "VOLUME" (case‑insensitive) because these are layout
                # markers, not artist names.
                for ln in clean:
                    words = ln.split()
                    if 1 <= len(words) <= 3 and all(w.isalpha() and w.isupper() for w in words):
                        if not re.search(r"\b(side|volume)\b", ln, re.I):
                            artist_hint = ln.title()
                            break
                # Tracks: look for A/B prefix or numeric prefix.  Keep only
                # descriptive titles (not just 'A1' or 'B2').
                for ln in clean:
                    low = ln.lower()
                    # Remove side/track prefixes
                    if re.match(r"^[ab][0-9]?[.:]?\s*", low):
                        track = re.sub(r"^[ab][0-9]?[.:]?\s*", "", ln).strip()
                    elif re.match(r"^\d+[.:]?\s*", low):
                        track = re.sub(r"^\d+[.:]?\s*", "", ln).strip()
                    else:
                        track = None
                    if track:
                        tracks.append(track)
                # Filter out tracks that are just identifiers (e.g. 'A2')
                filtered_tracks: List[str] = []
                for ttrack in tracks:
                    words = [w for w in re.split(r"\s+", ttrack) if w]
                    # Keep if any word has >2 alphabetic characters
                    keep = any(len(re.sub(r"[^A-Za-z]", "", w)) > 2 for w in words)
                    if keep:
                        filtered_tracks.append(ttrack)
                tracks = filtered_tracks
                # Strong title: first non‑rim, non‑side marker line or first track
                strong_title: Optional[str] = None
                for ln in clean:
                    if not RIM_PREFIX_RE.match(ln) and not re.match(r"^(side|volume)\b", ln, re.I):
                        strong_title = ln
                        break
                if not strong_title and tracks:
                    strong_title = tracks[0]
                # Build signals
                signals: Dict[str, Any] = {
                    "p_catno": (catalog_no_hint or "").upper().strip(),
                    "p_label": (label_hint or "").strip(),
                    "p_artist": (artist_hint or "").strip(),
                    "p_title": (strong_title or "").strip(),
                    "tracks": tracks,
                }
                if debug:
                    dbg["extracted"] = signals
                # Tier 1: Google CSE
                parts = ["site:discogs.com"]
                # Use up to two track titles if present
                for t in tracks[:2]:
                    parts.append(f'"{t}"')
                if not tracks and signals["p_title"]:
                    parts.append(f'"{signals["p_title"]}"')
                if signals["p_artist"]:
                    parts.append(f'"{signals["p_artist"]}"')
                if signals["p_catno"]:
                    parts.append(signals["p_catno"])
                g_query = " ".join(parts)
                if debug:
                    dbg["queries_tried"].append({"google_cse": g_query})
                cse_url = google_cse_discogs(g_query, dbg if debug else {}, signals)
                if cse_url:
                    m = re.search(r"/release/(\d+)", cse_url)
                    if m:
                        rid = int(m.group(1))
                        rel = discogs_request(f"/releases/{rid}")
                        if debug:
                            dbg.setdefault("discogs_calls", []).append({"endpoint": f"/releases/{rid}", "status": rel.status_code})
                        artist_str = title_str = label_str = cover = year = None
                        if rel.status_code == 200:
                            js = rel.json()
                            artist_str = ", ".join(a.get("name", "") for a in js.get("artists", []))
                            title_str = js.get("title")
                            label_str = ", ".join(l.get("name", "") for l in js.get("labels", []))
                            cover = js.get("thumb") or (js.get("images") or [{}])[0].get("uri", "")
                            year = str(js.get("year") or "")
                        candidates.append(
                            IdentifyCandidate(
                                source="google_cse",
                                release_id=rid,
                                discogs_url=cse_url,
                                artist=artist_str or signals["p_artist"] or None,
                                title=title_str or signals["p_title"] or None,
                                label=label_str or signals["p_label"] or None,
                                year=year,
                                cover_url=cover,
                                score=0.88,
                            )
                        )
                # Tier 2: Supabase RPC search_records
                if not candidates and (
                    signals["p_catno"]
                    or signals["p_label"]
                    or signals["p_artist"]
                    or signals["p_title"]
                ):
                    rows = rpc_search_records(signals, dbg if debug else {})
                    if rows:
                        for r in rows[:8]:
                            try:
                                rid = (
                                    int(r.get("release_id"))
                                    if r.get("release_id")
                                    else None
                                )
                            except Exception:
                                rid = None
                            disc_url = r.get("discogs_url") or (
                                f"https://www.discogs.com/release/{rid}" if rid else None
                            )
                            candidates.append(
                                IdentifyCandidate(
                                    source="supabase_rpc",
                                    release_id=rid,
                                    discogs_url=disc_url,
                                    artist=r.get("artist"),
                                    title=r.get("title"),
                                    label=r.get("label"),
                                    year=None,
                                    cover_url=None,
                                    score=float(r.get("score") or 0.8),
                                )
                            )
                # Tier 3: Legacy local lookup
                if not candidates and (
                    catalog_no_hint or label_hint or artist_hint
                ):
                    local = local_lookup(catalog_no_hint, label_hint, artist_hint, dbg if debug else {})
                    if local:
                        candidates.extend(local)
                # Tier 4: Discogs structured fallback
                if not candidates:
                    attempts: List[Dict[str, str]] = []
                    ncat = norm_catno(catalog_no_hint) if catalog_no_hint else None
                    if label_hint and ncat:
                        attempts.append({"label": label_hint, "catno": ncat, "type": "release"})
                    if artist_hint and ncat:
                        attempts.append({"artist": artist_hint, "catno": ncat, "type": "release"})
                    # Track‑based queries: use track names (with optional artist)
                    for t in tracks[:2]:
                        p: Dict[str, str] = {"track": t, "type": "release"}
                        if artist_hint:
                            p["artist"] = artist_hint
                        attempts.append(p)
                    if strong_title:
                        attempts.append({"release_title": strong_title, "type": "release"})
                    if label_hint and ncat:
                        attempts.append({"q": f"{label_hint} {ncat}", "type": "release"})
                    if artist_hint and strong_title:
                        attempts.append({"q": f"{artist_hint} {strong_title}", "type": "release"})
                    if strong_title:
                        attempts.append({"q": strong_title, "type": "release"})
                    for p in attempts:
                        res = search_discogs(p, dbg if debug else {})
                        if res:
                            # Bump score for structured label+catno or artist+catno
                            for c in res:
                                if (
                                    ("label" in p and "catno" in p)
                                    or ("artist" in p and "catno" in p)
                                ):
                                    c.score = 0.70
                            candidates.extend(res)
                            break
        # Sort and return
        candidates = sorted(
            candidates,
            key=lambda c: c.score if c.score is not None else 0.0,
            reverse=True,
        )
        if debug:
            dbg["total_elapsed_ms"] = int((time.time() - t0) * 1000)
        return IdentifyResponse(candidates=candidates[:12], debug=(dbg or None))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:300])
