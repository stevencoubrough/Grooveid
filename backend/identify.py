# backend/identify.py
# GrooveID — dynamic OCR→Discogs pipeline with robust merge & rerank (no hard per-record rules)

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import os, re, io, json, time, hashlib, logging, difflib
from collections import defaultdict
import requests
from PIL import Image

router = APIRouter()

# ---------------- Config ----------------
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")
if not VISION_KEY:
    logging.warning("GOOGLE_VISION_API_KEY not set. OCR will fail until configured.")

DGS_API = "https://api.discogs.com"
DGS_UA  = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}

# How many Discogs results to fetch per query and how many to keep overall
PER_QUERY = int(os.getenv("GROOVEID_DGS_PER_QUERY", "40"))   # fetch this many from Discogs per query
KEEP_PER_QUERY = int(os.getenv("GROOVEID_KEEP_PER_QUERY", "12"))  # keep this many per query locally
MAX_UNION = int(os.getenv("GROOVEID_MAX_UNION", "250"))      # max total candidates after union

# ---------------- Helpers: text & tokens ----------------
SIDEWORDS = set([
    "side", "sidea", "sideb", "a", "b", "stereo", "mono", "33", "45", "33⅓", "lp",
    "version", "edit", "mix", "radio", "club", "dub", "remix", "ascension", "ascensión",
    "promo", "test", "press", "limited", "ep", "single"
])

LOGO_TO_LABEL = {
    "ur": "Underground Resistance",
    # add more if/when needed, this is tiny and safe
}

def norm_text(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[^\w\s'&:/\-\.]", " ", s, flags=re.UNICODE)  # keep mild punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_ocr_lines(lines: List[str]) -> List[str]:
    txt = " ".join(x for x in lines if x)
    # common OCR fixes
    fixes = [
        (r"\bA\.?\s*K\.?\s*A\.?\b", "AKA"),
        (r"\bD]?\s?R?OLANDO\b", "DJ ROLANDO"),  # rough guard for weird reads
        (r"\bUR\b", "UR"),
        (r"\bASCENSION\b", "Ascensión"),
    ]
    for pat, rep in fixes:
        txt = re.sub(pat, rep, txt, flags=re.IGNORECASE)
    # remove side markers like "Side A", "A", "B" when isolated
    txt = re.sub(r"\b(SIDE\s*[AB]|SIDE|[AB])\b", " ", txt, flags=re.IGNORECASE)
    # collapse
    txt = norm_text(txt)
    # split back into lines (coarse)
    return [seg.strip() for seg in re.split(r"[|/•·\n]", txt) if seg.strip()]

def tokens(s: str) -> List[str]:
    s = s.lower()
    toks = re.findall(r"[a-z0-9'&]+", s)
    return [t for t in toks if t not in SIDEWORDS and len(t) >= 2]

def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

# ---------------- OCR (Google Vision) ----------------
def vision_ocr(image_bytes: bytes) -> List[str]:
    img_b64 = Image.open(io.BytesIO(image_bytes))
    # ensure RGB / small sanity resize (optional; we can send raw bytes too)
    buf = io.BytesIO()
    img_b64.save(buf, format="JPEG", quality=90)
    jpg_b64 = buf.getvalue()

    payload = {
        "requests": [{
            "image": {"content": jpg_b64.encode("base64") if False else None},
            "features": [
                {"type": "TEXT_DETECTION"},
                {"type": "DOCUMENT_TEXT_DETECTION"},
                {"type": "WEB_DETECTION", "maxResults": 5},
                {"type": "LOGO_DETECTION", "maxResults": 5},
            ]
        }]
    }
    # Because Python's standard library doesn't do base64 this way, build proper JSON:
    import base64
    payload["requests"][0]["image"]["content"] = base64.b64encode(jpg_b64).decode("utf-8")

    url = f"{VISION_ENDPOINT}?key={VISION_KEY}"
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Vision API error: {r.text[:300]}")
    data = r.json()
    resp = data.get("responses", [{}])[0]

    lines = []
    # document text
    if "fullTextAnnotation" in resp and resp["fullTextAnnotation"].get("text"):
        lines.extend([l for l in resp["fullTextAnnotation"]["text"].split("\n") if l.strip()])

    # web detection labels (sometimes super useful)
    for w in resp.get("webDetection", {}).get("bestGuessLabels", []):
        if w.get("label"):
            lines.append(w["label"])

    # logos (UR etc.)
    for logo in resp.get("logoAnnotations", []):
        if logo.get("description"):
            lines.append(logo["description"])

    # coarse textAnnotations
    for t in resp.get("textAnnotations", [])[1:]:
        if t.get("description"):
            lines.append(t["description"])

    # de-dup & normalize
    seen = set()
    clean = []
    for l in normalize_ocr_lines(lines):
        key = l.lower()
        if key not in seen:
            seen.add(key)
            clean.append(l)
    return clean[:60]  # cap

# ---------------- Entity hints (soft) ----------------
def extract_hints(lines: List[str]) -> Dict[str, List[str]]:
    joined = " " + " | ".join(lines) + " "
    low = joined.lower()

    artist_hints, label_hints, strong_phrases = [], [], []

    # phrases: long-ish spans from lines with 3–8 tokens, keep the meaty ones
    for ln in lines:
        tk = tokens(ln)
        if 3 <= len(tk) <= 8 and len(" ".join(tk)) >= 12:
            strong_phrases.append(ln)

    # artist patterns near aka/producer/by/feat
    ART_PAT = r"(?:by|aka|a\.?k\.?a\.?|producer|produced by|feat\.?|featuring)\s+([A-Za-z0-9 '&/.+-]{2,40})"
    for m in re.finditer(ART_PAT, low, flags=re.IGNORECASE):
        cand = norm_text(m.group(1))
        if cand and cand.lower() not in SIDEWORDS:
            artist_hints.append(cand)

    # logos → labels
    for logo, label in LOGO_TO_LABEL.items():
        if f" {logo} " in low:
            label_hints.append(label)

    # simple label words
    for m in re.finditer(r"([A-Za-z0-9 '&/.+-]{2,40})\s+(?:records|rec\.|music|recordings)", low, flags=re.I):
        cand = norm_text(m.group(1))
        if cand:
            label_hints.append(cand + " Records")

    # de-dup & trim
    def uniq(xs): 
        out, seen = [], set()
        for x in xs:
            k = x.strip().lower()
            if k and k not in seen:
                seen.add(k); out.append(x.strip())
        return out

    return {
        "artist_hints": uniq(artist_hints)[:4],
        "label_hints": uniq(label_hints)[:4],
        "phrases": uniq(strong_phrases)[:6]
    }

# ---------------- Query set ----------------
def build_queries(hints: Dict[str, List[str]]) -> List[str]:
    phrases = hints["phrases"] or []
    artists = hints["artist_hints"] or []
    labels  = hints["label_hints"] or []

    queries = []
    # base phrases
    queries.extend(phrases[:4])

    # phrase + artist
    for p in phrases[:4]:
        for a in artists[:3]:
            queries.append(f"{p} {a}")

    # phrase + label
    for p in phrases[:4]:
        for l in labels[:2]:
            queries.append(f"{p} {l}")

    # artist-only with generic record terms (helps on eponymous singles)
    for a in artists[:3]:
        queries.append(f"{a} EP")
        queries.append(f"{a} 12\"")

    # final fallback: top single short line (<20 chars, not sideword-y)
    short_lines = [p for p in phrases if 4 <= len(p) <= 20]
    queries.extend(short_lines[:2])

    # dedupe
    seen, out = set(), []
    for q in queries:
        qn = norm_text(q)
        if qn and qn.lower() not in seen:
            seen.add(qn.lower())
            out.append(qn)
    # cap
    return out[:16] if out else ["vinyl record label"]  # last-resort junk query

# ---------------- Discogs search & hydrate ----------------
def discogs_search(q: str, per_page: int = 40) -> List[Dict]:
    params = {
        "q": q,
        "type": "release",
        "format": "Vinyl",
        "per_page": per_page,
        "page": 1,
    }
    r = requests.get(f"{DGS_API}/database/search", headers=DGS_UA, params=params, timeout=20)
    if r.status_code != 200:
        # Backoff on rate limiting or transient errors
        if r.status_code in (429, 503):
            time.sleep(1.2)
            r = requests.get(f"{DGS_API}/database/search", headers=DGS_UA, params=params, timeout=20)
        if r.status_code != 200:
            logging.warning(f"Discogs search error {r.status_code}: {r.text[:200]}")
            return []
    data = r.json()
    return data.get("results", [])

def hydrate_release(release_id: int) -> Dict:
    """Fetch a release to get master_id and images; keep it quick."""
    r = requests.get(f"{DGS_API}/releases/{release_id}", headers=DGS_UA, timeout=20)
    if r.status_code != 200:
        return {}
    return r.json()

# ---------------- Scoring ----------------
def field(s): return s or ""

def title_support(candidate_title: str, phrases: List[str]) -> float:
    # strongest signal: exact or fuzzy containment of phrase within title
    score = 0.0
    tl = candidate_title.lower()
    for p in phrases[:4]:
        pl = p.lower()
        if pl and pl in tl:
            score = max(score, 3.0)
        else:
            fr = fuzzy_ratio(p, candidate_title)
            if fr >= 0.70:
                score = max(score, 2.0)
    # token overlap as backstop
    if score < 2.0:
        score = max(score, 1.5 * jaccard(tokens(candidate_title), tokens(" ".join(phrases))))
    return score

def contains_any(text: str, hints: List[str]) -> Tuple[bool, bool]:
    text_l = text.lower()
    exact = any(h.lower() in text_l for h in hints)
    partial = any(jaccard(tokens(text), tokens(h)) >= 0.5 for h in hints) if hints else False
    return exact, partial

def score_candidate(c: Dict, hints: Dict[str, List[str]], votes_by_master: Dict[int,int]) -> float:
    s = 0.0
    title = field(c.get("title"))
    artist = field(c.get("artist"))
    label  = field(c.get("label"))

    # Title
    s += title_support(title, hints["phrases"])

    # Artist (soft)
    ex, pa = contains_any(artist, hints["artist_hints"])
    if ex: s += 1.8
    elif pa: s += 1.2

    # Label (soft)
    ex, pa = contains_any(label, hints["label_hints"])
    if ex: s += 1.0
    elif pa: s += 0.6

    # Master consensus
    mid = c.get("master_id")
    if mid:
        s += 0.9 * votes_by_master.get(mid, 0)

    # Penalties: if title looks like we accidentally promoted sidewords
    if any(w in artist.lower() for w in SIDEWORDS):
        s -= 0.6
    if any(w in title.lower().split() for w in ("version","edit","mix")) and s < 3.0:
        s -= 0.3

    return round(float(s), 4)

# ---------------- API models ----------------
class IdentifyResponse(BaseModel):
    candidates: List[Dict]

# ---------------- Main endpoint ----------------
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_api(file: UploadFile = File(...)) -> IdentifyResponse:
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file.")

        # 1) OCR
        lines = vision_ocr(image_bytes)
        if not lines:
            raise HTTPException(status_code=422, detail="No text detected by OCR.")
        # 2) Hints & queries
        hints = extract_hints(lines)
        queries = build_queries(hints)

        # 3) Retrieve candidates from Discogs across all queries
        union: Dict[int, Dict] = {}
        votes_by_master: Dict[int,int] = defaultdict(int)

        for q in queries:
            results = discogs_search(q, per_page=PER_QUERY)[:KEEP_PER_QUERY]
            for r in results:
                # coerce
                rid = r.get("id") or r.get("release_id")
                if not rid: 
                    continue
                rid = int(rid)
                # normalize
                cand = union.get(rid) or {
                    "release_id": rid,
                    "master_id": r.get("master_id"),
                    "title": r.get("title") or "",
                    "artist": r.get("artist") or r.get("title", "").split("-")[0].strip(),
                    "label": (r.get("label") or (r.get("label_name") if isinstance(r.get("label_name"), str) else None) or ""),
                    "year": r.get("year"),
                    "country": r.get("country"),
                    "format": ", ".join(r.get("format", [])) if isinstance(r.get("format"), list) else r.get("format"),
                    "discogs_url": r.get("uri") or f"https://www.discogs.com/release/{rid}",
                    "cover_url": r.get("cover_image") or r.get("thumb"),
                    "source_queries": set(),
                }
                cand["source_queries"].add(q)
                union[rid] = cand

                # master vote
                mid = r.get("master_id")
                if mid:
                    votes_by_master[int(mid)] += 1

            if len(union) >= MAX_UNION:
                break

        if not union:
            return IdentifyResponse(candidates=[])

        # 4) Hydrate top-N for missing master_id/cover (lightweight)
        # Find items missing master_id or cover and hydrate a handful
        need = [rid for rid,c in union.items() if not c.get("master_id") or not c.get("cover_url")]
        for rid in need[:25]:
            info = hydrate_release(rid)
            if info:
                c = union[rid]
                c["master_id"] = c.get("master_id") or info.get("master_id")
                if not c.get("cover_url"):
                    img = (info.get("images") or [{}])[0] if info.get("images") else {}
                    c["cover_url"] = img.get("uri") or img.get("resource_url")

        # 5) Score
        for rid, c in union.items():
            c["score"] = score_candidate(c, hints, votes_by_master)
            # serialize queries
            c["source_queries"] = list(c["source_queries"])

        # 6) Sort & shape
        ranked = sorted(union.values(), key=lambda x: x["score"], reverse=True)
        for c in ranked:
            # keep keys tidy
            c.pop("format", None)
        # return top 15 (tune as needed)
        top = ranked[:15]

        return IdentifyResponse(candidates=top)

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("identify_api error")
        raise HTTPException(status_code=500, detail=str(e)[:300])

