
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
import os, re, base64, time, requests

# ===================== ENV =====================
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()

DISCOGS_API = "https://api.discogs.com"
DISCOGS_TOKEN = os.environ.get("DISCOGS_TOKEN", "").strip()

# Google Programmable Search (CSE)
GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY", "").strip()
GOOGLE_CSE_ID        = os.environ.get("GOOGLE_CSE_ID", "").strip()

# Optional Supabase local dump (records table with catalog_no)
try:
    from supabase import create_client, Client  # type: ignore
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    sb: Optional["Client"] = create_client(SUPABASE_URL, SUPABASE_KEY) if (SUPABASE_URL and SUPABASE_KEY) else None
except Exception:
    sb = None

LOCAL_TABLE = os.environ.get("DISCOGS_LOCAL_TABLE", "records")

router = APIRouter()

# ===================== MODELS =====================
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
    debug: Optional[Dict[str, Any]] = None  # returned when ?debug=true

# ===================== HELPERS =====================
def discogs_request(path: str, params: Dict = None, timeout=20) -> requests.Response:
    if params is None:
        params = {}
    headers = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
    if DISCOGS_TOKEN:
        headers["Authorization"] = f"Discogs token={DISCOGS_TOKEN}"
        params.setdefault("token", DISCOGS_TOKEN)
    url = path if path.startswith("http") else f"{DISCOGS_API}{path}"
    return requests.get(url, headers=headers, params=params, timeout=timeout)

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
        raise HTTPException(status_code=500, detail=f"Vision API error {r.status_code}: {r.text[:200]}")
    return r.json().get("responses", [{}])[0]

def parse_web(web: dict) -> Tuple[Optional[int], Optional[int], Optional[str], List[str]]:
    urls: List[str] = []
    for key in ("pagesWithMatchingImages","fullMatchingImages","partialMatchingImages","visuallySimilarImages"):
        for item in web.get(key, []):
            u = item.get("url")
            if u: urls.append(u)
    release_id = master_id = None
    discogs_url = None
    for u in urls:
        m = re.search(r"discogs\.com/(?:[^/]+/)?release/(\d+)", u, re.I)
        if m:
            release_id = int(m.group(1)); discogs_url = u; break
    if not release_id:
        for u in urls:
            m = re.search(r"discogs\.com/(?:[^/]+/)?master/(\d+)", u, re.I)
            if m:
                master_id = int(m.group(1)); discogs_url = u; break
    return release_id, master_id, discogs_url, urls

def ocr_lines(text_annotations: List[dict]) -> List[str]:
    if not text_annotations: return []
    raw = text_annotations[0].get("description","")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]

def norm_catno(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.upper().strip()
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s

def google_cse_discogs(query: str, dbg: Dict, signals: Dict[str, str]) -> Optional[str]:
    """Return best Discogs URL using Google CSE, or None."""
    api = GOOGLE_SEARCH_API_KEY
    cx  = GOOGLE_CSE_ID
    if not api or not cx or not query:
        return None
    params = {"key": api, "cx": cx, "q": query, "num": 5, "safe": "off"}
    t0 = time.time()
    r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=15)
    dbg.setdefault("google_calls", []).append({
        "endpoint": "customsearch/v1",
        "query": query,
        "status": r.status_code,
        "elapsed_ms": int((time.time()-t0)*1000)
    })
    if r.status_code != 200 or not r.json().get("items"):
        return None

    best, best_score = None, -1.0
    cat = (signals.get("p_catno")  or "").lower()
    art = (signals.get("p_artist") or "").lower()
    ttl = (signals.get("p_title")  or "").lower()

    for item in r.json().get("items", []):
        link = item.get("link","")
        if "discogs.com" not in link:
            continue
        title = (item.get("title")   or "").lower()
        snip  = (item.get("snippet") or "").lower()
        corpus = title + " " + snip
        score = 0.0
        if "/release/" in link: score += 3.0
        if "/master/"  in link: score += 1.0
        if cat and cat in corpus: score += 2.0
        if art and art in corpus: score += 1.5
        if ttl and ttl in corpus: score += 1.2
        if score > best_score:
            best_score, best = score, link
    return best

# ===================== Supabase (optional) =====================
def local_lookup(catno: Optional[str], label: Optional[str], artist: Optional[str], dbg: Dict) -> List[IdentifyCandidate]:
    out: List[IdentifyCandidate] = []
    if not sb or not catno:
        return out
    try:
        # 1) catalog_no only
        t0 = time.time()
        res0 = (sb.table(LOCAL_TABLE)
                  .select("release_id,label,catalog_no,artist,title,discogs_url")
                  .ilike("catalog_no", norm_catno(catno) or "")
                  .limit(12).execute())
        rows = res0.data or []
        dbg.setdefault("local_calls", []).append({
            "table": LOCAL_TABLE, "mode": "catalog_no_only", "catno": catno,
            "rows": len(rows), "ms": int((time.time()-t0)*1000)
        })

        # 2) label + catalog_no
        if not rows and label:
            t1 = time.time()
            res = (sb.table(LOCAL_TABLE)
                     .select("release_id,label,catalog_no,artist,title,discogs_url")
                     .ilike("label", f"%{label.strip()}%")
                     .ilike("catalog_no", norm_catno(catno) or "")
                     .limit(12).execute())
            rows = res.data or []
            dbg["local_calls"].append({
                "table": LOCAL_TABLE, "mode": "label+catalog_no",
                "label": label, "catno": catno, "rows": len(rows), "ms": int((time.time()-t1)*1000)
            })

        # 3) artist + catalog_no
        if not rows and artist:
            t2 = time.time()
            res2 = (sb.table(LOCAL_TABLE)
                      .select("release_id,label,catalog_no,artist,title,discogs_url")
                      .ilike("artist", f"%{artist}%")
                      .ilike("catalog_no", norm_catno(catno) or "")
                      .limit(12).execute())
            rows = res2.data or []
            dbg["local_calls"].append({
                "table": LOCAL_TABLE, "mode": "artist+catalog_no",
                "artist": artist, "catno": catno, "rows": len(rows), "ms": int((time.time()-t2)*1000)
            })

        def parse_rid(url: Optional[str]) -> Optional[int]:
            if not url: return None
            m = re.search(r"/release/(\d+)", url)
            return int(m.group(1)) if m else None

        for r in rows[:5]:
            rid = r.get("release_id") or parse_rid(r.get("discogs_url"))
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
                score=0.92
            ))
        return out
    except Exception as e:
        dbg.setdefault("local_calls", []).append({"error": str(e)})
        return out

# ===================== Discogs structured fallback =====================
def search_discogs(params: Dict[str, str], dbg: Dict) -> List[IdentifyCandidate]:
    out: List[IdentifyCandidate] = []
    t0 = time.time()
    r = discogs_request("/database/search", params)
    dbg.setdefault("discogs_calls", []).append({
        "endpoint": "/database/search",
        "params": {k: v for k, v in params.items() if k != "token"},
        "status": r.status_code,
        "elapsed_ms": int((time.time()-t0)*1000)
    })
    if r.status_code != 200:
        return out
    js = r.json()
    for it in js.get("results", [])[:8]:
        url = it.get("resource_url","")
        if "/releases/" not in url: continue
        try: rid = int(url.rstrip("/").split("/")[-1])
        except: continue
        out.append(IdentifyCandidate(
            source="ocr_search",
            release_id=rid,
            discogs_url=f"https://www.discogs.com/release/{rid}",
            artist=(it.get("title","").split(" - ")[0] if " - " in it.get("title","") else None),
            title=it.get("title"),
            label=(it.get("label") or [""])[0] if isinstance(it.get("label"), list) else it.get("label"),
            year=str(it.get("year") or ""),
            cover_url=it.get("thumb"),
            score=0.65
        ))
    return out

# ===================== API =====================
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_api(
    file: UploadFile = File(...),
    debug: bool = Query(False, description="Return debug info when true")
) -> IdentifyResponse:
    dbg: Dict[str, Any] = {"steps": [], "web_urls": [], "queries_tried": []} if debug else {}
    try:
        t0 = time.time()
        image_bytes = await file.read()

        # 1) Vision Web Detection
        tv = time.time()
        v = call_vision_api(image_bytes)
        dbg and dbg["steps"].append({"stage":"vision","elapsed_ms": int((time.time()-tv)*1000)})
        web = v.get("webDetection", {})
        text = v.get("textAnnotations", [])
        release_id, master_id, discogs_url, urls = parse_web(web)
        if debug:
            dbg["web_urls"] = urls
            dbg["web_release_id"] = release_id
            dbg["web_master_id"]  = master_id
            dbg["ocr_lines_raw"]  = ocr_lines(text)[:60]

        candidates: List[IdentifyCandidate] = []

        # A) Instant hit from image
        if release_id:
            tr = time.time()
            rel = discogs_request(f"/releases/{release_id}")
            dbg and dbg.setdefault("discogs_calls", []).append({
                "endpoint": f"/releases/{release_id}",
                "status": rel.status_code,
                "elapsed_ms": int((time.time()-tr)*1000)
            })
            if rel.status_code == 200:
                js = rel.json()
                candidates.append(IdentifyCandidate(
                    source="web_detection_live",
                    release_id=release_id,
                    discogs_url=discogs_url or js.get("uri"),
                    artist=", ".join(a.get("name","") for a in js.get("artists", [])),
                    title=js.get("title"),
                    label=", ".join(l.get("name","") for l in js.get("labels", [])),
                    year=str(js.get("year","")),
                    cover_url=js.get("thumb") or (js.get("images") or [{}])[0].get("uri", ""),
                    score=0.90
                ))

        # B) Master only
        if not candidates and master_id:
            candidates.append(IdentifyCandidate(
                source="web_detection_master",
                master_id=master_id,
                discogs_url=f"https://www.discogs.com/master/{master_id}",
                note="Master match — user must pick a pressing",
                score=0.60
            ))

        # C) Text-driven tiers
        if not candidates:
            lines = ocr_lines(text)
            if lines:
                # Clean OCR text (keep digits/hyphens/slashes for catnos)
                clean: List[str] = []
                for ln in lines:
                    cleaned = re.sub(r"[^\w\s/-]", "", ln).strip()
                    if cleaned: clean.append(cleaned)
                if debug: dbg["ocr_lines_clean"] = clean[:60]

                # Extract hints
                catalog_no_hint = None
                label_hint = None
                artist_hint = None

                for ln in clean:
                    low = ln.lower()
                    # catno like UR-014 / EVA008 / DECAY 003
                    m = re.search(r"[a-z]{2,}\s?-?\s?\d{1,5}", low)
                    if m and not catalog_no_hint:
                        catalog_no_hint = re.sub(r"\s*-\s*", "-", m.group(0).upper().replace("  "," "))
                    # label ... Records/Recordings/Music
                    if not label_hint and re.search(r"(records|recordings|music)\b", low):
                        label_hint = re.sub(r"(records|recordings|music)\b", "", ln, flags=re.I).strip()
                    # artist: short ALLCAPS line (<=3 words)
                    if not artist_hint:
                        words = ln.split()
                        if 1 <= len(words) <= 3 and all(w.isalpha() and w.isupper() for w in words):
                            artist_hint = ln.title()
                # small inference: if "evasive" anywhere
                if not label_hint and any("evasive" in ln.lower() for ln in clean):
                    label_hint = "Evasive Records"

                # strong title (avoid sending copyright rim text)
                copyright_re = re.compile(r"^all rights of the manufacturer", re.I)
                non_copyright = [ln for ln in clean if not copyright_re.match(ln)]
                strong_title = non_copyright[0] if non_copyright else (clean[0] if clean else "")

                # Signals for CSE ranker
                signals = {
                    "p_catno": (catalog_no_hint or "").upper().strip(),
                    "p_label": (label_hint or "").strip(),
                    "p_artist": (artist_hint or "").strip(),
                    "p_title": (strong_title or "").strip(),
                }

                # Tier 1: Google CSE
                parts = ["site:discogs.com"]
                if signals["p_artist"]: parts.append(f"\"{signals['p_artist']}\"")
                if signals["p_title"]:  parts.append(f"\"{signals['p_title']}\"")
                if signals["p_catno"]:  parts.append(signals["p_catno"])
                g_query = " ".join(parts)
                if debug: dbg["queries_tried"].append({"google_cse": g_query})

                cse_url = google_cse_discogs(g_query, dbg if debug else {}, signals)
                if cse_url:
                    m = re.search(r"/release/(\d+)", cse_url)
                    if m:
                        rid = int(m.group(1))
                        rel = discogs_request(f"/releases/{rid}")
                        artist_str = title_str = label_str = cover = year = None
                        if rel.status_code == 200:
                            js = rel.json()
                            artist_str = ", ".join(a.get("name","") for a in js.get("artists", []))
                            title_str  = js.get("title")
                            label_str  = ", ".join(l.get("name","") for l in js.get("labels", []))
                            cover      = js.get("thumb") or (js.get("images") or [{}])[0].get("uri", "")
                            year       = str(js.get("year") or "")
                            dbg and dbg.setdefault("discogs_calls", []).append({
                                "endpoint": f"/releases/{rid}", "status": 200
                            })
                        candidates.append(IdentifyCandidate(
                            source="google_cse",
                            release_id=rid,
                            discogs_url=cse_url,
                            artist=artist_str or artist_hint,
                            title=title_str  or strong_title or None,
                            label=label_str  or label_hint,
                            year=year,
                            cover_url=cover,
                            score=0.88
                        ))

                # Tier 2: Supabase local dump (optional)
                if not candidates and (catalog_no_hint or label_hint or artist_hint):
                    local = local_lookup(catalog_no_hint, label_hint, artist_hint, dbg if debug else {})
                    if local:
                        candidates.extend(local)

                # Tier 3: Discogs structured fallback
                if not candidates:
                    attempts: List[Dict[str,str]] = []
                    ncat = norm_catno(catalog_no_hint) if catalog_no_hint else None
                    if label_hint and ncat: attempts.append({"label": label_hint, "catno": ncat, "type": "release"})
                    if artist_hint and ncat: attempts.append({"artist": artist_hint, "catno": ncat, "type": "release"})
                    if strong_title: attempts.append({"release_title": strong_title, "type": "release"})
                    # broad q fallbacks
                    if label_hint and ncat: attempts.append({"q": f"{label_hint} {ncat}", "type":"release"})
                    if artist_hint and strong_title: attempts.append({"q": f"{artist_hint} {strong_title}", "type":"release"})
                    if strong_title: attempts.append({"q": strong_title, "type":"release"})
                    for p in attempts:
                        res = search_discogs(p, dbg if debug else {})
                        if res:
                            for c in res:
                                if ("label" in p and "catno" in p) or ("artist" in p and "catno" in p):
                                    c.score = 0.70
                            candidates.extend(res)
                            break

        # finalize
        dbg and dbg.update({"total_elapsed_ms": int((time.time()-t0)*1000)})
        return IdentifyResponse(candidates=candidates[:10], debug=(dbg or None))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:300])
