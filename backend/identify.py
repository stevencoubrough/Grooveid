from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
from supabase import create_client, Client
import os, re, base64, requests, time

# ================== ENV / CLIENTS ==================
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
sb: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
)

DISCOGS_API = "https://api.discogs.com"
DISCOGS_TOKEN = os.environ.get("DISCOGS_TOKEN", "").strip()

# ---- local dump (optional) ----
# Default to 'records' since your Supabase dump uses this table
LOCAL_TABLE = os.environ.get("DISCOGS_LOCAL_TABLE", "records")  # label/catalog_no/artist/title/release_id

# ================== REGEX ==================
RE_RELEASE = re.compile(r"discogs\.com/(?:[^/]+/)?release/(\d+)", re.I)
RE_MASTER  = re.compile(r"discogs\.com/(?:[^/]+/)?master/(\d+)",  re.I)

# ================== MODELS ==================
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
    debug: Optional[Dict[str, Any]] = None  # present when ?debug=1

router = APIRouter()

# ================== HELPERS: DISCogs ==================
def discogs_request(path: str, params: Dict = None, timeout=20):
    """Authenticated GET to Discogs API."""
    if params is None:
        params = {}
    headers = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
    if DISCOGS_TOKEN:
        headers["Authorization"] = f"Discogs token={DISCOGS_TOKEN}"
        params.setdefault("token", DISCOGS_TOKEN)
    url = path if path.startswith("http") else f"{DISCOGS_API}{path}"
    return requests.get(url, headers=headers, params=params, timeout=timeout)

# ================== HELPERS: VISION ==================

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

def parse_discogs_web_detection(web: dict) -> Tuple[Optional[int], Optional[int], Optional[str], List[str]]:
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
            release_id = int(m.group(1)); discogs_url = u; break
    if not release_id:
        for u in urls:
            m = RE_MASTER.search(u)
            if m:
                master_id = int(m.group(1)); discogs_url = u; break
    return release_id, master_id, discogs_url, urls

def ocr_lines(text_annotations: List[dict]) -> List[str]:
    if not text_annotations: return []
    raw = text_annotations[0].get("description","")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]

# ================== HELPERS: LOCAL DUMP (OPTIONAL) ==================
def norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()

def norm_catno(s: str) -> str:
    s = s.upper().strip().replace("  ", " ")
    s = re.sub(r"\s*-\s*", "-", s)  # normalize hyphens
    return s

def local_lookup(label: Optional[str], catno: Optional[str], artist: Optional[str], dbg: Dict) -> List[IdentifyCandidate]:
    """Try to resolve via Supabase local dump. Returns [] if table missing or no match."""
    out: List[IdentifyCandidate] = []
    if not sb or not label or not catno:
        return out
    try:
        t0 = time.time()
        # exact-ish label + catalog_no (use ILIKE with wildcards for label)
        res = (
            sb.table(LOCAL_TABLE)
              .select("release_id, label, catalog_no, artist, title")
              .ilike("label", f"%{label.strip()}%")
              .ilike("catalog_no", norm_catno(catno))
              .limit(10)
              .execute()
        )
        dbg.setdefault("local_calls", []).append({
            "table": LOCAL_TABLE, "label": label, "catno": catno,
            "rows": len(res.data or []), "elapsed_ms": int((time.time()-t0)*1000)
        })
        rows = res.data or []

        # artist + catalog_no fallback
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
            dbg["local_calls"].append({
                "table": LOCAL_TABLE, "artist": artist, "catno": catno,
                "rows": len(res2.data or []), "elapsed_ms": int((time.time()-t1)*1000)
            })
            rows = res2.data or []

        for r in rows[:5]:
            rid = r.get("release_id")
            if not rid: continue
            out.append(IdentifyCandidate(
                source="local_dump",
                release_id=int(rid),
                discogs_url=f"https://www.discogs.com/release/{rid}",
                artist=r.get("artist"),
                title=r.get("title"),
                label=r.get("label"),
                year=None,
                cover_url=None,
                score=0.92,  # strong because exact catno hit
            ))
        return out
    except Exception as e:
        dbg.setdefault("local_calls", []).append({"error": str(e)})
        return []

# ================== HELPERS: DISCogs SEARCH ==================
def search_discogs_via_ocr(query: str, dbg: Dict) -> List[IdentifyCandidate]:
    cands: List[IdentifyCandidate] = []
    params = {"q": query, "type": "release"}
    t0 = time.time()
    r = discogs_request("/database/search", params)
    dbg.setdefault("discogs_calls", []).append({
        "endpoint": "/database/search",
        "params": {"q": query, "type": "release"},
        "status": r.status_code,
        "elapsed_ms": int((time.time()-t0)*1000),
    })
    if r.status_code == 200:
        js = r.json()
        for it in js.get("results", [])[:8]:
            url = it.get("resource_url","")
            if "/releases/" not in url: continue
            try:
                rid = int(url.rstrip("/").split("/")[-1])
            except: continue
            cands.append(IdentifyCandidate(
                source="ocr_search",
                release_id=rid,
                discogs_url=f"https://www.discogs.com/release/{rid}",
                artist=(it.get("title","").split(" - ")[0] if " - " in it.get("title","") else None),
                title=it.get("title"),
                label=(it.get("label") or [""])[0] if isinstance(it.get("label"), list) else it.get("label"),
                year=str(it.get("year") or ""),
                cover_url=it.get("thumb"),
                score=0.65,
            ))
    return cands

def discogs_multi_search(label: Optional[str], catno: Optional[str],
                         artist: Optional[str], tracks: List[str],
                         strong_title: Optional[str], dbg: Dict) -> List[IdentifyCandidate]:
    attempts: List[Dict[str,str]] = []

    def add_attempt(p: Dict[str,str]): attempts.append(p)

    if label and catno:
        add_attempt({"label": label, "catno": norm_catno(catno), "type": "release"})
    if artist and catno:
        add_attempt({"artist": artist, "catno": norm_catno(catno), "type": "release"})
    for t in tracks[:3]:
        p = {"track": t, "type": "release"}
        if artist: p["artist"] = artist
        add_attempt(p)
    if strong_title:
        add_attempt({"release_title": strong_title, "type": "release"})
    # broad q fallbacks
    if label and catno: add_attempt({"q": f"{label} {catno}", "type": "release"})
    if artist and strong_title: add_attempt({"q": f"{artist} {strong_title}", "type": "release"})
    if strong_title: add_attempt({"q": strong_title, "type": "release"})

    out: List[IdentifyCandidate] = []
    for params in attempts:
        t0 = time.time()
        r = discogs_request("/database/search", params)
        dbg.setdefault("discogs_calls", []).append({
            "endpoint": "/database/search",
            "params": {k:v for k,v in params.items() if k != "token"},
            "status": r.status_code,
            "elapsed_ms": int((time.time()-t0)*1000),
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
                out.append(IdentifyCandidate(
                    source="ocr_search",
                    release_id=rid,
                    discogs_url=f"https://www.discogs.com/release/{rid}",
                    artist=(it.get("title","").split(" - ")[0] if " - " in it.get("title","") else None),
                    title=it.get("title"),
                    label=(it.get("label") or [""])[0] if isinstance(it.get("label"), list) else it.get("label"),
                    year=str(it.get("year") or ""),
                    cover_url=it.get("thumb"),
                    score=0.70 if ("label" in params and "catno" in params) or ("artist" in params and "catno" in params) else 0.65,
                ))
            if out:
                break
        elif r.status_code in (429,500,502,503):
            time.sleep(1.2)
            continue
        else:
            continue
    return out

# ================== MAIN ENDPOINT ==================
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_record(
    file: UploadFile = File(...),
    debug: bool = Query(False, description="Set to true to include debug info")
) -> IdentifyResponse:
    dbg: Dict[str, Any] = {"steps": [], "web_urls": [], "queries_tried": []} if debug else {}
    try:
        t0 = time.time()
        image_bytes = await file.read()

        # 1) Vision
        v0 = time.time()
        vision_resp = call_vision_api(image_bytes)
        dbg and dbg["steps"].append({"stage":"vision", "elapsed_ms": int((time.time()-v0)*1000)})

        web = vision_resp.get("webDetection", {})
        text = vision_resp.get("textAnnotations", [])
        release_id, master_id, discogs_url, urls = parse_discogs_web_detection(web)
        if debug:
            dbg["web_urls"] = urls
            dbg["web_release_id"] = release_id
            dbg["web_master_id"]  = master_id
            dbg["ocr_lines_raw"]  = ocr_lines(text)[:40]

        candidates: List[IdentifyCandidate] = []

        # 2) Direct release via Web Detection
        if release_id:
            t = time.time()
            rel = discogs_request(f"/releases/{release_id}")
            dbg and dbg.setdefault("discogs_calls", []).append({
                "endpoint": f"/releases/{release_id}",
                "status": rel.status_code,
                "elapsed_ms": int((time.time()-t)*1000),
            })
            if rel.status_code == 200:
                rj = rel.json()
                candidates.append(
                    IdentifyCandidate(
                        source="web_detection_live",
                        release_id=release_id,
                        discogs_url=discogs_url or rj.get("uri"),
                        artist=", ".join(a.get("name","") for a in rj.get("artists", [])),
                        title=rj.get("title"),
                        label=", ".join(l.get("name","") for l in rj.get("labels", [])),
                        year=str(rj.get("year", "")),
                        cover_url=rj.get("thumb") or (rj.get("images") or [{}])[0].get("uri", ""),
                        score=0.90,
                    )
                )

        # 3) Master only
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

        # 4) OCR fallback (with local dump first, then Discogs)
        if not candidates:
            lines = ocr_lines(text)
            if lines:
                # clean & preserve digits/hyphens/slashes
                clean_lines: List[str] = []
                for ln in lines:
                    cleaned = re.sub(r"[^\w\s/-]", "", ln).strip()
                    if cleaned:
                        clean_lines.append(cleaned)
                if debug:
                    dbg["ocr_lines_clean"] = clean_lines[:40]

                # light heuristics
                label = None; catno = None; artist_hint = None; tracks: List[str] = []
                for ln in clean_lines:
                    low = ln.lower()

                    # Cat no candidates: DEF 004 / DECAY 003 / UR-014
                    m = re.search(r"[a-z]{2,}\s?-?\s?\d{1,5}", low)
                    if m and not catno:
                        catno = m.group(0).upper().replace("  "," ")
                        catno = re.sub(r"\s*-\s*", "-", catno)

                    # Label heuristic (… Records/Recordings/Music)
                    if not label and re.search(r"(records|recordings|music)\b", low):
                        label = re.sub(r"(records|recordings|music)\b", "", ln, flags=re.I).strip()

                    # Artist heuristic: short ALLCAPS line (<=3 words)
                    if not artist_hint:
                        words = ln.strip().split()
                        if 1 <= len(words) <= 3 and all(w.isalpha() and w.isupper() for w in words):
                            artist_hint = ln.title()

                    # Track titles after ":" or " - " (avoid digits on RHS)
                    if ":" in ln:
                        t = ln.split(":",1)[1].strip()
                        if t: tracks.append(t)
                    elif " - " in ln and not artist_hint:
                        rhs = ln.split(" - ",1)[1].strip()
                        if rhs and not re.search(r"\d", rhs):
                            tracks.append(rhs)

                strong_title = clean_lines[0] if clean_lines else None
                if debug:
                    dbg["extracted"] = {"label": label, "catno": catno, "artist": artist_hint, "tracks": tracks, "strong_title": strong_title}

                # 4a) Try local dump (Supabase) first (instant, no rate limits)
                local_cands = local_lookup(label, catno, artist_hint, dbg if debug else {})
                if local_cands:
                    candidates.extend(local_cands)
                else:
                    # 4b) Structured Discogs search (label+catno → artist+catno → track → title → q)
                    queries = []
                    if clean_lines:
                        queries.append(" ".join(clean_lines[:3])[:200])
                        if len(clean_lines) >= 2:
                            queries.append(" ".join(clean_lines[:2])[:200])
                        queries.append(clean_lines[0][:200])
                    else:
                        queries.append(" ".join(lines[:2])[:200])
                    debug and dbg.update({"queries_tried": queries})

                    cands = discogs_multi_search(label, catno, artist_hint, tracks, strong_title, dbg if debug else {})
                    if not cands:
                        for q in queries:
                            cands = search_discogs_via_ocr(q, dbg if debug else {})
                            if cands:
                                break
                    if cands:
                        candidates.extend(cands)

        # finalize
        if debug:
            dbg["total_elapsed_ms"] = int((time.time()-t0)*1000)
        return IdentifyResponse(candidates=candidates[:10], debug=(dbg or None))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


