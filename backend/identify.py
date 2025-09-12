# identify_with_rpc.py
# GrooveID — Identify via Google Vision OCR + free-text Google CSE → Discogs candidates (multi-return)
# Drop-in FastAPI router. Exposes POST /api/identify
#
# Dependencies (pip):
#   fastapi, uvicorn, pydantic, pillow, requests, numpy
#
# Notes:
# - This version prioritizes: OCR all text ➜ build several free-text queries ➜ CSE ➜
#   collect multiple Discogs links ➜ enrich /release links via Discogs API ➜ rank.
# - Keeps placeholder hooks for Supabase RPC / local caches if you want to wire them later.

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, io, re, time, base64, requests
from PIL import Image

router = APIRouter()

# ---------- Config ----------
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.getenv("GOOGLE_VISION_API_KEY")

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID         = os.getenv("GOOGLE_CSE_ID")

DISCOGS_API  = "https://api.discogs.com"
DISCOGS_UA   = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
DISCOGS_TOK  = os.getenv("DISCOGS_TOKEN")  # optional

# ---------- Models ----------
class IdentifyCandidate(BaseModel):
    source: str
    score: float
    discogs_url: str
    release_id: Optional[int] = None
    master_id: Optional[int] = None
    artist: Optional[str] = None
    title: Optional[str] = None
    label: Optional[str] = None
    year: Optional[str] = None
    cover_url: Optional[str] = None
    note: Optional[str] = None

class IdentifyResponse(BaseModel):
    candidates: List[IdentifyCandidate]
    queries_tried: Optional[List[Dict[str, Any]]] = None
    ocr_lines_raw: Optional[List[str]] = None
    web_release_id: Optional[int] = None
    web_master_id: Optional[int] = None
    debug_assets: Optional[Dict[str, Any]] = None

# ---------- Helpers ----------
def _img_to_base64(file: UploadFile) -> str:
    by = file.file.read()
    if not by:
        raise HTTPException(400, "Empty file.")
    return base64.b64encode(by).decode("utf-8")

def _vision_annotate(b64: str, dbg: Dict) -> Dict:
    if not VISION_KEY:
        raise HTTPException(500, "Missing GOOGLE_VISION_API_KEY.")
    req = {
        "requests": [{
            "image": {"content": b64},
            "features": [
                {"type": "TEXT_DETECTION"},
                {"type": "DOCUMENT_TEXT_DETECTION"},
                {"type": "WEB_DETECTION"}
            ]
        }]
    }
    t0 = time.time()
    r = requests.post(f"{VISION_ENDPOINT}?key={VISION_KEY}", json=req, timeout=30)
    dbg.setdefault("vision_calls", []).append(
        {"status": r.status_code, "elapsed_ms": int((time.time()-t0)*1000)}
    )
    if r.status_code != 200:
        raise HTTPException(502, f"Vision error: {r.text[:200]}")
    js = r.json()
    return js["responses"][0]

def _extract_ocr_lines(resp: Dict) -> List[str]:
    lines = []
    # documentText first (usually richer), fallback to plain textAnnotations
    doc = resp.get("fullTextAnnotation", {})
    if doc and "text" in doc:
        for ln in doc["text"].splitlines():
            ln = ln.strip()
            if ln:
                lines.append(ln)
    else:
        for it in resp.get("textAnnotations", [])[1:]:
            ln = (it.get("description") or "").strip()
            if ln:
                lines.append(ln)
    # de-dup while preserving order
    seen = set()
    out = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out

STOP = set("""
the and for of in a an to from remix mixes side volume vol track tracks produced arranged ltd
rights manufacturer unauthorized unauthorised reproduction copying hiring rental prohibited
for promotional use only promo only demo sample stereo mono made manufactured distributed
all behind behind. kobo kobokobo kobo-kobo tool wheel
""".split())

def build_free_text_queries(lines: List[str]) -> List[str]:
    tokens: List[str] = []
    seen: set = set()
    for ln in lines:
        # keep raw words & catno-like strings
        parts = re.split(r"[^\w#/+.\-]+", ln)
        for raw in parts:
            t = raw.strip()
            if not t:
                continue
            low = t.lower()
            if low in STOP:
                continue
            if len(t) < 2:
                continue
            if low not in seen:
                seen.add(low)
                tokens.append(t)

    # prefer catno-looking tokens early
    catlikes = [t for t in tokens if re.search(r"[a-z]{2,}\d+|\d{2,}[a-z]+|[a-z]+[-_/]?\d{2,}", t, re.I)]
    strong = list(dict.fromkeys(catlikes + tokens))

    # build up to 3 compact queries with 6–10 tokens each
    step = 8
    chunks = []
    for i in range(0, min(len(strong), 24), step):
        chunks.append(strong[i:i+step])

    queries = []
    for ch in chunks[:3]:
        parts = ["site:discogs.com"]
        for tok in ch:
            if " " in tok:
                parts.append(f"\"{tok}\"")
            else:
                parts.append(tok)
        queries.append(" ".join(parts))

    return queries or ["site:discogs.com"]

def google_cse_discogs_multi(query: str, dbg: Dict, want: int = 8) -> List[Dict[str, str]]:
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_CSE_ID:
        return []
    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": min(10, want),
        "safe": "off"
    }
    t0 = time.time()
    r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=20)
    dbg.setdefault("google_calls", []).append({
        "endpoint": "customsearch/v1",
        "query": query,
        "status": r.status_code,
        "elapsed_ms": int((time.time()-t0)*1000)
    })
    if r.status_code != 200:
        return []
    out = []
    for item in r.json().get("items", []):
        link = (item.get("link") or "")
        if "discogs.com" not in link:
            continue
        out.append({
            "link": link,
            "title": item.get("title") or "",
            "snippet": item.get("snippet") or ""
        })
        if len(out) >= want:
            break
    return out

def discogs_request(path: str):
    url = f"{DISCOGS_API}{path}"
    headers = dict(DISCOGS_UA)
    if DISCOGS_TOK:
        headers["Authorization"] = f"Discogs token={DISCOGS_TOK}"
    return requests.get(url, headers=headers, timeout=20)

def _score_from_link(u: str) -> float:
    # Prefer release pages slightly over master pages
    return 0.88 if "/release/" in u else (0.80 if "/master/" in u else 0.70)

# ----- OPTIONAL: Supabase RPC stub (wire up later) -----
def supabase_rpc_enrich(_candidate: IdentifyCandidate) -> IdentifyCandidate:
    # placeholder: no-op
    return _candidate

# ---------- Route ----------
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify(
    file: UploadFile = File(...),
    debug: bool = Query(False),
    search_mode: str = Query("all_text")  # keep default on the new mode
):
    dbg: Dict[str, Any] = {}
    try:
        b64 = _img_to_base64(file)

        # 1) Vision OCR + (optional) Web Detection if you want to use later
        vresp = _vision_annotate(b64, dbg)
        ocr_lines = _extract_ocr_lines(vresp)
        dbg["ocr_line_count"] = len(ocr_lines)

        # 2) Build multiple free-text queries from ALL OCR text
        free_qs = build_free_text_queries(ocr_lines)
        dbg.setdefault("queries_tried", []).append({"google_cse_free_text": free_qs})

        # 3) Run CSE for each query and aggregate Discogs links
        seen_links = set()
        cse_links: List[str] = []
        cse_items_debug: List[Dict[str, str]] = []
        for q in free_qs:
            hits = google_cse_discogs_multi(q, dbg, want=8)
            for h in hits:
                link = h["link"]
                if link in seen_links:
                    continue
                seen_links.add(link)
                if "discogs.com" in link:
                    cse_links.append(link)
                    if debug:
                        cse_items_debug.append(h)

        # 4) Convert links to IdentifyCandidates; enrich /release via Discogs API
        candidates: List[IdentifyCandidate] = []
        for link in cse_links[:12]:
            m_rel = re.search(r"/release/(\\d+)", link)
            if m_rel:
                rid = int(m_rel.group(1))
                t2 = time.time()
                rel = discogs_request(f"/releases/{rid}")
                dbg.setdefault("discogs_calls", []).append({
                    "endpoint": f"/releases/{rid}",
                    "status": rel.status_code,
                    "elapsed_ms": int((time.time()-t2)*1000)
                })
                if rel.status_code == 200:
                    js = rel.json()
                    cand = IdentifyCandidate(
                        source="google_cse_free_text",
                        score=_score_from_link(link),
                        discogs_url=link,
                        release_id=rid,
                        artist=", ".join(a.get("name","") for a in js.get("artists", [])),
                        title=js.get("title"),
                        label=", ".join(l.get("name","") for l in js.get("labels", [])),
                        year=str(js.get("year","") or ""),
                        cover_url=js.get("thumb") or (js.get("images") or [{}])[0].get("uri","")
                    )
                    cand = supabase_rpc_enrich(cand)  # no-op for now
                    candidates.append(cand)
                else:
                    # still return a lightweight candidate if API call failed
                    candidates.append(IdentifyCandidate(
                        source="google_cse_free_text",
                        score=_score_from_link(link) - 0.05,
                        discogs_url=link,
                        release_id=rid,
                        note="Discogs API enrichment failed"
                    ))
                continue

            m_mas = re.search(r"/master/(\\d+)", link)
            if m_mas:
                mid = int(m_mas.group(1))
                candidates.append(IdentifyCandidate(
                    source="google_cse_free_text",
                    score=_score_from_link(link),
                    discogs_url=link,
                    master_id=mid,
                    note="Master match — pick pressing"
                ))
                continue

            # Other Discogs pages: still surface
            candidates.append(IdentifyCandidate(
                source="google_cse_free_text",
                score=_score_from_link(link) - 0.1,
                discogs_url=link,
                note="Non-release Discogs URL"
            ))

        # 5) Sort candidates by score (desc)
        candidates.sort(key=lambda c: c.score, reverse=True)

        # 6) Build response
        resp = IdentifyResponse(
            candidates=candidates,
            queries_tried=dbg.get("queries_tried"),
            ocr_lines_raw=ocr_lines,
            web_release_id=None,
            web_master_id=None,
            debug_assets=dbg if debug else None
        )
        return resp

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Identify failed: {e}")


