# backend/identify.py
# GrooveID — Complete dynamic version with improved Discogs integration and validation safety
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
import os, re, io, time, base64, requests, logging
from io import BytesIO
from PIL import Image
import difflib

# Optional imports with graceful fallback
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log availability
logger.warning("OpenCV not available - visual analysis disabled" if not CV2_AVAILABLE else "OpenCV available")
logger.warning("Scikit-learn not available - semantic matching disabled" if not SKLEARN_AVAILABLE else "Scikit-learn available")
logger.warning("EasyOCR not available" if not EASYOCR_AVAILABLE else "EasyOCR available")

# Initialize router
router = APIRouter()

# Response models
class DiscogsCandidate(BaseModel):
    source: str
    release_id: int
    master_id: Optional[int] = None
    discogs_url: str
    artist: str
    title: str
    label: str
    year: Optional[int] = None
    cover_url: Optional[str] = None
    score: float
    note: Optional[str] = None

class IdentifyResponse(BaseModel):
    candidates: List[DiscogsCandidate]

# ---------- VISION API ----------
def vision_api_detect(image_bytes: bytes) -> Tuple[Optional[dict], Optional[str]]:
    """Google Vision API detection"""
    api_key = os.getenv("GOOGLE_VISION_API_KEY")
    if not api_key:
        logger.error("No Vision API key")
        return None, None

    try:
        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        img_b64 = base64.b64encode(image_bytes).decode()

        payload = {
            "requests": [{
                "image": {"content": img_b64},
                "features": [
                    {"type": "WEB_DETECTION", "maxResults": 10},
                    {"type": "TEXT_DETECTION", "maxResults": 50}
                ]
            }]
        }

        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            responses = data.get("responses", [{}])[0]
            web = responses.get("webDetection", {})
            text = responses.get("fullTextAnnotation", {}).get("text", "")
            return web, text
        else:
            logger.warning(f"Vision API status: {response.status_code}")
    except Exception as e:
        logger.error(f"Vision API error: {e}")
    return None, None

# ---------- OCR HELPERS ----------
def ocr_lines(text: str) -> List[str]:
    """Extract lines from OCR text"""
    if not text:
        return []
    return [ln.strip() for ln in text.strip().split('\n') if ln.strip()]

# ---------- METADATA EXTRACTION ----------
def extract_metadata(lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """Extract label, catalog, artist, tracks from OCR lines"""
    label = catno = artist = None
    tracks: List[str] = []

    for line in lines:
        upper = line.upper().strip()
        # Pattern: "WAX 04"
        if re.match(r'^[A-Z]{2,8}\s+\d{1,4}$', upper):
            parts = upper.split()
            if len(parts) == 2:
                label, catno = parts[0], parts[1].zfill(3)
                break
        # Pattern: "WAX-004" or "WAX004"
        m = re.match(r'^([A-Z]{2,8})[- ]?(\d{1,4})[A-Z]?$', upper)
        if m:
            label, catno = m.group(1), m.group(2).zfill(3)
            break

    # Artist detection - naive proper-case name first
    for line in lines[:10]:
        if label and label in line.upper():
            continue
        low = line.lower()
        if any(skip in low for skip in ['manufactured', 'distributed', 'copyright', 'wax trax']):
            continue
        if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', line.strip()):
            if 4 < len(line) < 40:
                artist = line.strip()
                break

    # Track detection
    track_patterns = [r'^[AB]\d', r'^\d\.', r'^Side [AB]']
    for line in lines:
        if any(re.match(p, line) for p in track_patterns):
            track_name = re.sub(r'^([AB]\d+|[\d\.]+|Side [AB]:?\s*)', '', line).strip()
            if track_name and len(track_name) > 2:
                tracks.append(track_name)

    return label, catno, artist, tracks

# ---------- GENRE DETECTION ----------
def detect_genres(lines: List[str]) -> List[str]:
    genres: List[str] = []
    text = ' '.join(lines).lower()

    genre_keywords = {
        'techno': ['techno', 'detroit', 'minimal', 'industrial'],
        'house': ['house', 'deep', 'chicago', 'garage', 'soulful'],
        'drum_and_bass': ['drum and bass', 'dnb', 'jungle', 'neurofunk'],
        'dubstep': ['dubstep', 'bass', 'wobble', '140'],
        'trance': ['trance', 'psychedelic', 'goa', 'uplifting'],
        'acid': ['acid', '303', 'roland', 'squelch']
    }

    for genre, keywords in genre_keywords.items():
        if any(kw in text for kw in keywords):
            genres.append(genre)
    return genres

# ---------- FUZZY MATCHING ----------
def fuzzy_match_score(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

# ---------- SEARCH STRATEGY ----------
def generate_dynamic_searches(lines: List[str], label: Optional[str], catno: Optional[str], artist: Optional[str], tracks: List[str]) -> List[Dict]:
    attempts: List[Dict] = []

    # Artist
    if artist:
        attempts.extend([
            {"q": artist},
            {"artist": artist},
        ])
        if label:
            attempts.append({"q": f"{artist} {label}"})

    # Label + catno
    if label and catno:
        attempts.extend([
            {"label": label, "catno": catno},
            {"q": f"{label} {catno}"},
            {"q": f"{label}-{catno}"},
        ])
    elif label:
        attempts.append({"q": label})

    # Tracks
    for track in tracks[:2]:
        if len(track) > 3:
            clean_track = re.sub(r'\b(\d+rpm|\d+bpm|remix|mix|version)\b', '', track, flags=re.I).strip()
            if clean_track:
                attempts.append({"q": clean_track})
                if artist:
                    attempts.append({"q": f"{artist} {clean_track}"})

    # Volume / series
    for line in lines:
        if re.search(r'vol\.?\s*\d+|volume\s*\d+|ep\s*\d+|part\s*\d+', line.lower()):
            attempts.append({"q": line})

    # Genre hints
    genres = detect_genres(lines)
    if genres and label:
        if 'techno' in genres or 'house' in genres:
            attempts.extend([
                {"q": f"{label} white label"},
                {"q": f"{label} promo"},
            ])

    # Meaningful lines
    meaningful = [
        line for line in lines
        if len(line) > 3 and not any(x in line.lower() for x in [
            'manufactured','distributed','copyright','reserved','unauthorized','broadcasting','intergroove','phone'
        ])
    ]
    for line in meaningful[:3]:
        if label and line.upper() == label:
            continue
        if artist and line.upper() == artist.upper():
            continue
        if len(line) < 50:
            attempts.append({"q": line})

    # Dedup while preserving order
    seen = set()
    unique: List[Dict] = []
    for a in attempts:
        key = tuple(sorted(a.items()))
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique

# ---------- DISCOGS HELPERS ----------
def _normalize_discogs_params(attempt: dict) -> dict:
    p = {"type": "release"}
    if attempt.get("artist"):
        p["artist"] = attempt["artist"]
    if attempt.get("label"):
        p["label"] = attempt["label"]
    if attempt.get("catno"):
        p["catno"] = attempt["catno"]
    if attempt.get("q"):
        p["q"] = attempt["q"]
    return p


def fetch_release_details(release_id: int) -> Optional[dict]:
    try:
        token = os.getenv("DISCOGS_TOKEN")
        headers = {"User-Agent": os.getenv("DGS_UA", "GrooveID/1.0 (+https://grooveid.app)")}
        if token:
            headers["Authorization"] = f"Discogs token={token}"
        r = requests.get(f"https://api.discogs.com/releases/{release_id}", headers=headers, timeout=6)
        if r.status_code == 200:
            d = r.json()
            title = d.get("title", "") or ""
            artists = d.get("artists", []) or []
            labels = d.get("labels", []) or []
            images = d.get("images", []) or []
            return {
                "release_id": release_id,
                "master_id": d.get("master_id"),
                "discogs_url": f"https://www.discogs.com/release/{release_id}",
                "artist": (artists[0].get("name") if artists else "") or "Unknown",
                "title": title,
                "label": (labels[0].get("name") if labels else "") or "",
                "year": d.get("year"),
                "cover_url": next((im.get("uri") for im in images if im.get("type") == "primary"), None),
            }
        logger.warning(f"Release fetch {release_id} -> {r.status_code}")
    except Exception as e:
        logger.error(f"Discogs release fetch error: {e}")
    return None


def search_discogs(params: dict) -> List[dict]:
    """Search Discogs API with normalized params."""
    out: List[dict] = []
    try:
        query = _normalize_discogs_params(params)
        token = os.getenv("DISCOGS_TOKEN")
        headers = {"User-Agent": os.getenv("DGS_UA", "GrooveID/1.0 (+https://grooveid.app)")}
        if token:
            headers["Authorization"] = f"Discogs token={token}"
        url = "https://api.discogs.com/database/search"

        for attempt_num in range(3):
            resp = requests.get(url, params=query, headers=headers, timeout=6)
            if resp.status_code == 429 and attempt_num < 2:
                time.sleep(0.5 * (attempt_num + 1))
                continue
            if resp.status_code != 200:
                logger.warning(f"Discogs search {query} -> {resp.status_code}")
                return out
            data = resp.json()
            for r in data.get("results", [])[:10]:
                title = r.get("title", "") or ""
                artist, release_title = ("Unknown", title)
                if " - " in title:
                    parts = title.split(" - ", 1)
                    artist, release_title = parts[0], parts[1]
                out.append({
                    "release_id": r.get("id"),
                    "master_id": r.get("master_id"),
                    "discogs_url": f"https://www.discogs.com/release/{r.get('id')}",
                    "artist": artist,
                    "title": release_title,
                    "label": (r.get("label") or [""])[0] if isinstance(r.get("label"), list) else (r.get("label") or ""),
                    "year": r.get("year"),
                    "cover_url": r.get("cover_image"),
                })
            break
    except Exception as e:
        logger.error(f"Discogs search error: {e}")
    return out

# ---------- SCORING ----------
def calculate_advanced_score(result: dict, search_idx: int, result_idx: int,
                            label: Optional[str], catno: Optional[str], artist: Optional[str],
                            tracks: List[str], genres: List[str], lines: List[str]) -> float:
    base_score = 1.0 - (search_idx * 0.05) - (result_idx * 0.02)
    if artist and result.get('artist'):
        base_score *= (1 + fuzzy_match_score(artist, result['artist']) * 0.3)
    if label and result.get('label'):
        base_score *= (1 + fuzzy_match_score(label, result['label']) * 0.2)
    result_title = (result.get('title') or '').lower()
    for track in tracks:
        if fuzzy_match_score(track, result_title) > 0.7:
            base_score *= 1.25
            break
    if genres:
        text = f"{result.get('artist','')} {result.get('title','')}".lower()
        if any(g in text for g in genres):
            base_score *= 1.1
    if any(term in (result.get('title','').lower()) for term in ['white label','promo','unknown','various','dub']):
        base_score *= 1.15
    return min(base_score, 0.99)

# ---------- MAIN OCR FALLBACK ----------
def improved_ocr_fallback_with_ml(text: str, image_bytes: bytes) -> List[dict]:
    candidates: List[dict] = []
    lines = ocr_lines(text)

    label, catno, artist, tracks = extract_metadata(lines)
    genres = detect_genres(lines)
    logger.info(f"Extracted - Label: {label}, Catno: {catno}, Artist: {artist}, Genres: {genres}")

    attempts = generate_dynamic_searches(lines, label, catno, artist, tracks)
    logger.info(f"Generated {len(attempts)} dynamic search attempts")

    for i, attempt in enumerate(attempts[:15]):
        results = search_discogs(attempt)
        for j, result in enumerate(results[:5]):
            score = calculate_advanced_score(result, i, j, label, catno, artist, tracks, genres, lines)
            result['score'] = min(score, 0.99)
            result['source'] = 'ocr_search'
            candidates.append(result)
        if len(candidates) >= 10 and any(c.get('score', 0) > 0.85 for c in candidates):
            break

    candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
    return candidates[:5]

# ---------- MAIN IDENTIFY FUNCTION ----------
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_record(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        Image.open(BytesIO(image_bytes))  # validate image

        web, text = vision_api_detect(image_bytes)
        candidates: List[dict] = []

        # Use Vision webDetection to grab direct Discogs releases, hydrate with details
        if web:
            web_urls: List[str] = []
            for key in ("pagesWithMatchingImages", "fullMatchingImages", "partialMatchingImages", "visuallySimilarImages"):
                urls = [item.get("url") for item in web.get(key, []) if item.get("url")]
                if urls:
                    web_urls.extend(urls)
            for url in web_urls[:5]:
                if "discogs.com/release/" in url:
                    m = re.search(r'/release/(\d+)', url)
                    if not m:
                        continue
                    rid = int(m.group(1))
                    details = fetch_release_details(rid)
                    if details:
                        details.update({"source": "web_detection", "score": 0.95, "note": "Direct web match"})
                        candidates.append(details)

        if not candidates and text:
            logger.info("No web matches, trying dynamic OCR search")
            candidates = improved_ocr_fallback_with_ml(text, image_bytes)

        if not candidates:
            candidates = [{
                "source": "none",
                "release_id": 0,
                "master_id": None,
                "discogs_url": "",
                "artist": "Unknown",
                "title": "No matches found",
                "label": "",
                "year": None,
                "cover_url": None,
                "score": 0.0,
                "note": "Try a clearer image"
            }]

        return IdentifyResponse(candidates=candidates)

    except Exception as e:
        logger.error(f"Identify error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- DEBUG ENDPOINT ----------
@router.post("/api/debug-identify")
async def debug_identify(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes))
        web, text = vision_api_detect(image_bytes)
        all_lines = ocr_lines(text)
        label, catno, artist, tracks = extract_metadata(all_lines)
        genres = detect_genres(all_lines)
        attempts = generate_dynamic_searches(all_lines, label, catno, artist, tracks)

        search_results = {}
        for i, attempt in enumerate(attempts[:5]):
            results = search_discogs(attempt)
            search_results[f"query_{i}"] = {
                "query": attempt,
                "count": len(results),
                "results": results[:3]
            }

        return {
            "raw_ocr": text,
            "final_lines": all_lines,
            "extracted_metadata": {
                "label": label,
                "catno": catno,
                "artist": artist,
                "tracks": tracks
            },
            "detected_genres": genres,
            "generated_searches": attempts[:10],
            "search_results": search_results,
            "system_info": {
                "vision_key_set": bool(os.getenv("GOOGLE_VISION_API_KEY")),
                "discogs_token_set": bool(os.getenv("DISCOGS_TOKEN")),
                "cv2_available": CV2_AVAILABLE,
                "tesseract_available": TESSERACT_AVAILABLE,
                "sklearn_available": SKLEARN_AVAILABLE,
                "easyocr_available": EASYOCR_AVAILABLE
            }
        }
    except Exception as exc:
        logger.error(f"Debug identify error: {exc}")
        return {"error": str(exc)}

# ---------- HEALTHCHECK ----------
@router.get("/healthz")
async def healthz():
    return {"status": "ok"}
