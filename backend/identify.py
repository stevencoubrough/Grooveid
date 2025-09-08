# backend/identify.py
"""
World-class record identification backend for GrooveID.

Enhancements:
- Google Vision (web + text) + Supabase cache
- OCR fallback with structured Discogs searches:
  * artist + release_title
  * label + release_title
  * label + catno
  * artist + catno
  * track (+ optional artist)
  * broad 'q' queries
- OCR retry on contrast-boosted center crop (for tiny / low-contrast center labels)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict

from supabase import create_client, Client
import os
import re
import base64
import requests

from io import BytesIO
from PIL import Image, ImageOps, ImageFilter  # requires pillow>=10

# -------------------- Config --------------------

VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
)

DGS_UA = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
DGS_API = "https://api.discogs.com"

RE_RELEASE = re.compile(r"discogs\.com/(?:[^/]+/)?release/(\d+)", re.IGNORECASE)
RE_MASTER  = re.compile(r"discogs\.com/(?:[^/]+/)?master/(\d+)",  re.IGNORECASE)

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

# -------------------- Helpers --------------------

def call_vision_api(image_bytes: bytes) -> dict:
    if not VISION_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_VISION_API_KEY is not configured")
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
    resp = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=payload, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Vision API error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    return data["responses"][0]

def parse_discogs_web_detection(web: dict) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    urls: List[str] = []
    for key in ("pagesWithMatchingImages","fullMatchingImages","partialMatchingImages","visuallySimilarImages"):
        for item in web.get(key, []):
            u = item.get("url")
            if u: urls.append(u)

    release_id = master_id = None
    discogs_url = None

    for u in urls:
        m = RE_RELEASE.search(u)
        if m:
            release_id = int(m.group(1))
            discogs_url = u
            break

    if release_id is None:
        for u in urls:
            m = RE_MASTER.search(u)
            if m:
                master_id = int(m.group(1))
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
    res = (
        supabase.table("discogs_cache")
        .select("*")
        .eq("release_id", release_id)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None

def insert_cache_row(row: dict) -> None:
    if not supabase:
        return
    supabase.table("discogs_cache").upsert(row, on_conflict="release_id").execute()

def fetch_discogs_release_json(release_id: int) -> Optional[dict]:
    try:
        r = requests.get(f"{DGS_API}/releases/{release_id}", headers=DGS_UA, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def discogs_search(params: Dict[str, str]) -> List[IdentifyCandidate]:
    """Search Discogs database with free-text or structured params."""
    params = params.copy()
    params.setdefault("type", "release")
    token = os.environ.get("DISCOGS_TOKEN")
    if token:
        params["token"] = token

    candidates: List[IdentifyCandidate] = []
    try:
        r = requests.get(f"{DGS_API}/database/search", params=params, headers=DGS_UA, timeout=20)
        if r.status_code == 200:
            js = r.json()
            for it in js.get("results", [])[:5]:
                res_url = it.get("resource_url", "")
                if "/releases/" not in res_url:
                    continue
                try:
                    rid = int(res_url.rstrip("/").split("/")[-1])
                except Exception:
                    continue
                candidates.append(IdentifyCandidate(
                    source="ocr_search",
                    release_id=rid,
                    discogs_url=f"https://www.discogs.com/release/{rid}",
                    artist=(it.get("title", "").split(" - ")[0] if " - " in it.get("title", "") else None),
                    title=it.get("title"),
                    label=(it.get("label") or [""])[0] if isinstance(it.get("label"), list) else it.get("label"),
                    year=str(it.get("year") or ""),
                    cover_url=it.get("thumb"),
                    score=0.65,
                ))
    except Exception:
        # swallow network issues in fallback path
        pass
    return candidates

def extract_ocr_metadata(lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """Heuristics to pull label, catno, artist, and track titles from OCR lines."""
    label = None
    catno = None
    artist = None
    tracks: List[str] = []

    for i, ln in enumerate(lines):
        lower = ln.lower()

        # Label + promo/catalog patterns (e.g., "urban decay promo 003")
        m = re.match(r"([a-z0-9\s]+?)\s*(?:promo|pr)?\s*(\d{1,5})$", lower)
        if m and not catno:
            l = m.group(1).strip()
            label = l.title() if l else label
            catno = m.group(2)
            continue

        # Generic "label 003"
        m2 = re.match(r"([a-z0-9\s]+?)\s+(\d{1,5})$", lower)
        if m2 and not catno:
            l = m2.group(1).strip()
            label = l.title() if l else label
            catno = m2.group(2)
            continue

        # Artist heuristic: short uppercase line (skip first line which is often label)
        if not artist and i > 0:
            words = ln.strip().split()
            if 1 <= len(words) <= 3 and all(w.isupper() for w in words if w.isalpha()):
                artist = ln.strip().title()
                continue

        # Track lines
        if ":" in ln:
            t = ln.split(":", 1)[1].strip()
            if t: tracks.append(t)
            continue
        if " - " in ln and not artist:
            # avoid track detection for lines with digits (likely durations/catnos)
            if not re.search(r"\d", ln):
                t = ln.split(" - ", 1)[1].strip()
                if t: tracks.append(t)

    return label, catno, artist, tracks

def pick_title_guess(lines: List[str]) -> Optional[str]:
    """Pick a plausible release title from cleaned OCR lines."""
    keys = (" part ", " pt ", " vol ", " ep ", " remix", " mixes", " ii", " iii", " iv", " v")
    for ln in lines:
        low = f" {ln.lower()} "
        if any(k in low for k in keys):
            return ln.strip()
    cands = [ln for ln in lines if 8 <= len(ln) <= 50 and not ln.isupper()]
    return max(cands, key=len).strip() if cands else None

def center_label_preprocess(image_bytes: bytes) -> bytes:
    """Crop center, boost contrast/sharpness, upscale to help OCR tiny label text."""
    im = Image.open(BytesIO(image_bytes)).convert("L")
    w, h = im.size
    side = int(min(w, h) * 0.60)
    left = (w - side) // 2
    top  = (h - side) // 2
    crop = im.crop((left, top, left + side, top + side))
    crop = ImageOps.autocontrast(crop)
    crop = crop.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    crop = crop.resize((1024, 1024), Image.LANCZOS)
    out = BytesIO(); crop.save(out, format="PNG")
    return out.getvalue()

# -------------------- Route --------------------

@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_record(file: UploadFile = File(...)) -> IdentifyResponse:
    """Identify a record from an uploaded image using Vision + Discogs."""
    try:
        image_bytes = await file.read()
        v = call_vision_api(image_bytes)
        web = v.get("webDetection", {})
        text = v.get("textAnnotations", [])

        release_id, master_id, discogs_url = parse_discogs_web_detection(web)
        candidates: List[IdentifyCandidate] = []

        # 1) Direct release via web detection
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
                rel = fetch_discogs_release_json(release_id)
                if rel:
                    row = {
                        "release_id": release_id,
                        "discogs_url": discogs_url or rel.get("uri") or f"https://www.discogs.com/release/{release_id}",
                        "artist": ", ".join(a.get("name", "") for a in rel.get("artists", [])),
                        "title": rel.get("title"),
                        "label": ", ".join(l.get("name", "") for l in rel.get("labels", [])),
                        "year": str(rel.get("year", "")),
                        "cover_url": rel.get("thumb") or (rel.get("images") or [{}])[0].get("uri", ""),
                        "payload": rel,
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

        # 2) Master match
        if not candidates and master_id:
            candidates.append(IdentifyCandidate(
                source="web_detection_master",
                master_id=master_id,
                discogs_url=f"https://www.discogs.com/master/{master_id}",
                note="Master match — prompt user to select a pressing",
                score=0.60,
            ))

        # 3) OCR fallback (rich)
        if not candidates:
            lines = ocr_lines(text)
            clean_lines = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in lines if ln.strip()]
            label, catno, artist, tracks = extract_ocr_metadata(clean_lines)

            # Build structured attempts (including title-driven ones)
            search_attempts: List[Dict[str, str]] = []
            title_guess = pick_title_guess(clean_lines)
            if artist and title_guess:
                search_attempts.append({"artist": artist, "release_title": title_guess})
            if label and title_guess:
                search_attempts.append({"label": label, "release_title": title_guess})

            if label and catno:
                search_attempts.append({"label": label, "catno": catno})
            if catno and artist:
                search_attempts.append({"artist": artist, "catno": catno})
            if tracks:
                for t in tracks:
                    p = {"track": t}
                    if artist: p["artist"] = artist
                    search_attempts.append(p)

            if clean_lines:
                q1 = " ".join(clean_lines[:3])[:200]; search_attempts.append({"q": q1})
                if len(clean_lines) >= 2:
                    q2 = " ".join(clean_lines[:2])[:200]; search_attempts.append({"q": q2})
                search_attempts.append({"q": clean_lines[0][:200]})

            # Execute until something hits
            for params in search_attempts:
                res = discogs_search(params)
                if res:
                    candidates.extend(res)
                    break

            # 3b) If OCR looked weak (e.g., tiny clear-vinyl text), retry on boosted center crop
            if not candidates and len(lines) < 2:
                try:
                    boosted = center_label_preprocess(image_bytes)
                    v2 = call_vision_api(boosted)
                    text2 = v2.get("textAnnotations", [])
                    lines2 = ocr_lines(text2)
                    if lines2:
                        clean_lines2 = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in lines2 if ln.strip()]
                        label2, catno2, artist2, tracks2 = extract_ocr_metadata(clean_lines2)

                        attempts2: List[Dict[str, str]] = []
                        title_guess2 = pick_title_guess(clean_lines2)
                        if artist2 and title_guess2:
                            attempts2.append({"artist": artist2, "release_title": title_guess2})
                        if label2 and title_guess2:
                            attempts2.append({"label": label2, "release_title": title_guess2})
                        if label2 and catno2:
                            attempts2.append({"label": label2, "catno": catno2})
                        if catno2 and artist2:
                            attempts2.append({"artist": artist2, "catno": catno2})
                        if tracks2:
                            for t in tracks2:
                                p = {"track": t}
                                if artist2: p["artist"] = artist2
                                attempts2.append(p)
                        if clean_lines2:
                            q1 = " ".join(clean_lines2[:3])[:200]; attempts2.append({"q": q1})
                            if len(clean_lines2) >= 2:
                                q2 = " ".join(clean_lines2[:2])[:200]; attempts2.append({"q": q2})
                            attempts2.append({"q": clean_lines2[0][:200]})

                        for params in attempts2:
                            res = discogs_search(params)
                            if res:
                                candidates.extend(res)
                                break
                except Exception:
                    pass

        return IdentifyResponse(candidates=candidates[:5])

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
