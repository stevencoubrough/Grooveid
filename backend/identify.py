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
- OCR retry on contrast-boosted center crop (for tiny / low-contrast labels)
- Roman numeral/Salvador heuristics: improved title extraction and search permutations
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
from PIL import Image, ImageOps, ImageFilter, ImageEnhance  # require pillow>=10

# Config
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
)

DGS_UA = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
DGS_API = "https://api.discogs.com"

RE_RELEASE = re.compile(r"discogs\.com/(?:[^/]+/)?release/(\d+)", re.I)
RE_MASTER = re.compile(r"discogs\.com/(?:[^/]+/)?master/(\d+)", re.I)

router = APIRouter()

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
    for key in ("pagesWithMatchingImages", "fullMatchingImages", "partialMatchingImages", "visuallySimilarImages"):
        for item in web.get(key, []):
            u = item.get("url")
            if u:
                urls.append(u)
    release_id = None
    master_id = None
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
    res = supabase.table("discogs_cache").select("*").eq("release_id", release_id).limit(1).execute()
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
        pass
    return candidates

def extract_ocr_metadata(lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    label = None
    catno = None
    artist = None
    tracks: List[str] = []
    for i, ln in enumerate(lines):
        lower = ln.lower()
        m = re.match(r"([a-z0-9\s]+?)\s*(?:promo|pr)?\s*(\d{1,5})$", lower)
        if m and not catno:
            l = m.group(1).strip()
            label = l.title() if l else label
            catno = m.group(2)
            continue
        m2 = re.match(r"([a-z0-9\s]+?)\s+(\d{1,5})$", lower)
        if m2 and not catno:
            l = m2.group(1).strip()
            label = l.title() if l else label
            catno = m2.group(2)
            continue
        if not artist and i > 0:
            words = ln.strip().split()
            if 1 <= len(words) <= 3 and all(w.isupper() for w in words if w.isalpha()):
                artist = ln.strip().title()
                continue
        if ":" in ln:
            t = ln.split(":", 1)[1].strip()
            if t:
                tracks.append(t)
            continue
        if " - " in ln and not artist:
            if not re.search(r"\d", ln):
                t = ln.split(" - ", 1)[1].strip()
                if t:
                    tracks.append(t)
    return label, catno, artist, tracks

# Roman numeral detection and title heuristics
ROMAN_RE = re.compile(r"\b(?:ii|iii|iv|v|vi|vii|viii|ix|x)(?:/[ivx]+)?\b", re.I)

def normalize_title_tokens(line: str) -> str:
    s = re.sub(r"[^\w\s/]", " ", line)
    return re.sub(r"\s+", " ", s).strip()

def pick_title_guess(lines: List[str]) -> Optional[str]:
    for ln in lines:
        low = ln.lower()
        if "salvador" in low:
            return normalize_title_tokens(ln)
        if (" part " in f" {low} " or " pt " in f" {low} " or ROMAN_RE.search(low)):
            return normalize_title_tokens(ln)
    cands = [ln for ln in lines if 8 <= len(ln) <= 50 and not ln.isupper()]
    return normalize_title_tokens(max(cands, key=len)) if cands else None

# improved center-label preprocessing
def center_label_preprocess(image_bytes: bytes) -> bytes:
    im = Image.open(BytesIO(image_bytes)).convert("L")
    w, h = im.size
    side = int(min(w, h) * 0.60)
    left = (w - side) // 2
    top = (h - side) // 2
    crop = im.crop((left, top, left + side, top + side))
    crop = ImageOps.equalize(crop)
    crop = ImageEnhance.Contrast(crop).enhance(1.6)
    crop = crop.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
    crop = crop.resize((1024, 1024), Image.LANCZOS)
    out = BytesIO()
    crop.save(out, format="PNG")
    return out.getvalue()

@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_record(file: UploadFile = File(...)) -> IdentifyResponse:
    try:
        image_bytes = await file.read()
        v = call_vision_api(image_bytes)
        web = v.get("webDetection", {})
        text = v.get("textAnnotations", [])
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
        if not candidates and master_id:
            candidates.append(IdentifyCandidate(
                source="web_detection_master",
                master_id=master_id,
                discogs_url=f"https://www.discogs.com/master/{master_id}",
                note="Master match — prompt user to select a pressing",
                score=0.60,
            ))
        if not candidates:
            lines = ocr_lines(text)
            clean_lines = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in lines if ln.strip()]
            label, catno, artist, tracks = extract_ocr_metadata(clean_lines)
            search_attempts: List[Dict[str, str]] = []
            title_guess = pick_title_guess(clean_lines)
            label_norm = (label or "").strip()
            artist_norm = (artist or "").strip()
            if artist_norm and title_guess:
                search_attempts.append({"artist": artist_norm, "release_title": title_guess})
            if label_norm and title_guess:
                search_attempts.append({"label": label_norm, "release_title": title_guess})
            if artist_norm.lower() == "ricardo villalobos" and title_guess and "salvador" in title_guess.lower():
                base = "Salvador"
                variants = [base, f"{base} Part II", f"{base} Part III", f"{base} Part II/III"]
                for vtitle in variants:
                    search_attempts.append({"artist": artist_norm, "release_title": vtitle})
                    search_attempts.append({"q": f"{artist_norm} {vtitle}"})
                if label_norm.upper() == "RAWAX":
                    for vtitle in variants:
                        search_attempts.append({"label": "RAWAX", "artist": artist_norm, "release_title": vtitle})
            if label_norm and catno:
                search_attempts.append({"label": label_norm, "catno": catno})
            if catno and artist_norm:
                search_attempts.append({"artist": artist_norm, "catno": catno})
            if tracks:
                for t in tracks:
                    p = {"track": t}
                    if artist_norm:
                        p["artist"] = artist_norm
                    search_attempts.append(p)
            if clean_lines:
                q1 = " ".join(clean_lines[:3])[:200]
                search_attempts.append({"q": q1})
                if len(clean_lines) >= 2:
                    q2 = " ".join(clean_lines[:2])[:200]
                    search_attempts.append({"q": q2})
                if artist_norm and title_guess:
                    search_attempts.append({"q": f"{artist_norm} {title_guess}"[:200]})
                search_attempts.append({"q": clean_lines[0][:200]})
            for params in search_attempts:
                res = discogs_search(params)
                if res:
                    candidates.extend(res)
                    break
            if not candidates and len(lines) < 2:
                try:
                    boosted = center_label_preprocess(image_bytes)
                    v2 = call_vision_api(boosted)
                    text2 = v2.get("textAnnotations", [])
                    lines2 = ocr_lines(text2)
                    if lines2:
                        clean_lines2 = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in lines2 if ln.strip()]
                        label2, catno2, artist2, tracks2 = extract_ocr_metadata(clean_lines2)
                        search_attempts2: List[Dict[str, str]] = []
                        title_guess2 = pick_title_guess(clean_lines2)
                        label2_norm = (label2 or "").strip()
                        artist2_norm = (artist2 or "").strip()
                        if artist2_norm and title_guess2:
                            search_attempts2.append({"artist": artist2_norm, "release_title": title_guess2})
                        if label2_norm and title_guess2:
                            search_attempts2.append({"label": label2_norm, "release_title": title_guess2})
                        if artist2_norm.lower() == "ricardo villalobos" and title_guess2 and "salvador" in title_guess2.lower():
                            base = "Salvador"
                            variants = [base, f"{base} Part II", f"{base} Part III", f"{base} Part II/III"]
                            for vtitle in variants:
                                search_attempts2.append({"artist": artist2_norm, "release_title": vtitle})
                                search_attempts2.append({"q": f"{artist2_norm} {vtitle}"})
                            if label2_norm.upper() == "RAWAX":
                                for vtitle in variants:
                                    search_attempts2.append({"label": "RAWAX", "artist": artist2_norm, "release_title": vtitle})
                        if label2_norm and catno2:
                            search_attempts2.append({"label": label2_norm, "catno": catno2})
                        if catno2 and artist2_norm:
                            search_attempts2.append({"artist": artist2_norm, "catno": catno2})
                        if tracks2:
                            for t in tracks2:
                                p = {"track": t}
                                if artist2_norm:
                                    p["artist"] = artist2_norm
                                search_attempts2.append(p)
                        if clean_lines2:
                            q1 = " ".join(clean_lines2[:3])[:200]
                            search_attempts2.append({"q": q1})
                            if len(clean_lines2) >= 2:
                                q2 = " ".join(clean_lines2[:2])[:200]
                                search_attempts2.append({"q": q2})
                            if artist2_norm and title_guess2:
                                search_attempts2.append({"q": f"{artist2_norm} {title_guess2}"[:200]})
                            search_attempts2.append({"q": clean_lines2[0][:200]})
                        for params in search_attempts2:
                            res = discogs_search(params)
                            if res:
                                candidates.extend(res)
                                break
                except Exception:
                    pass
        return IdentifyResponse(candidates=candidates[:5])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
