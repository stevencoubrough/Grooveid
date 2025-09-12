
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
import os, re, base64, requests, time, io
from collections import defaultdict
from PIL import Image
try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None
    Client = None  # type: ignore

# ================== ENV / CLIENTS ==================
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
sb: Optional["Client"] = (create_client(SUPABASE_URL, SUPABASE_KEY) if create_client and SUPABASE_URL and SUPABASE_KEY else None)

DISCOGS_API = "https://api.discogs.com"
DISCOGS_TOKEN = os.environ.get("DISCOGS_TOKEN", "").strip()

# Google Custom Search (ranking tier)
GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY", "").strip()
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "").strip()

# Local dump table (defaults to 'records' which matches your Supabase schema)
LOCAL_TABLE = os.environ.get("DISCOGS_LOCAL_TABLE", "records")

router = APIRouter()

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
    candidates: List[Dict[str, Any]]
    debug: Optional[Dict[str, Any]] = None  # present when ?debug=1

# ================== HELPERS ==================
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

def norm_label(s: Optional[str]) -> Optional[str]:
    if not s: return None
    return re.sub(r"\s+", " ", s).strip().lower()

def norm_catno(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.upper().strip()
    s = re.sub(r"\s*-\s*", "-", s)  # normalize hyphens
    s = re.sub(r"\s+", " ", s)
    return s

def _parse_release_id_from_url(url: Optional[str]) -> Optional[int]:
    if not url: return None
    m = re.search(r"/release/(\d+)", url or "")
    return int(m.group(1)) if m else None

def _google_cse_discogs(query: str, dbg: Dict, signals: Dict[str, str]) -> Optional[str]:
    """Return best Discogs URL using Google Custom Search, or None."""
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
        corpus = f"{title} {snip}"

        score = 0.0
        if "/release/" in link: score += 3.0
        if "/master/"  in link: score += 1.0
        if cat and cat in corpus: score += 2.0
        if art and art in corpus: score += 1.5
        if ttl and ttl in corpus: score += 1.2

        if score > best_score:
            best_score, best = score, link

    return best

def local_lookup(label: Optional[str], catno: Optional[str], artist: Optional[str], dbg: Dict) -> List[IdentifyCandidate]:
    """Supabase lookup against your Discogs dump. Tries catalog_no only, then label+catalog_no, then artist+catalog_no.
       Includes a wildcard EVA%008 fallback. Returns candidates built from rows.
    """
    out: List[IdentifyCandidate] = []
    if not sb or not catno:
        return out
    try:
        # 1) catalog_no exact-ish
        t0 = time.time()
        res0 = (
            sb.table(LOCAL_TABLE)
            .select("release_id,label,catalog_no,artist,title")
            .ilike("catalog_no", norm_catno(catno) or "")
            .limit(12).execute()
        )
        rows = res0.data or []
        dbg.setdefault("local_calls", []).append({
            "table": LOCAL_TABLE, "mode": "catalog_no_only", "label": label, "catno": catno,
            "rows": len(rows), "ms": int((time.time()-t0)*1000)
        })

        # 2) label + catalog_no
        if not rows and label:
            t1 = time.time()
            res = (
                sb.table(LOCAL_TABLE)
                .select("release_id,label,catalog_no,artist,title")
                .ilike("label", f"%{label.strip()}%")
                .ilike("catalog_no", norm_catno(catno) or "")
                .limit(12).execute()
            )
            rows = res.data or []
            dbg["local_calls"].append({
                "table": LOCAL_TABLE, "mode": "label+catalog_no",
                "label": label, "catno": catno, "rows": len(rows), "ms": int((time.time()-t1)*1000)
            })

        # 3) artist + catalog_no
        if not rows and artist:
            t2 = time.time()
            res2 = (
                sb.table(LOCAL_TABLE)
                .select("release_id,label,catalog_no,artist,title")
                .ilike("artist", f"%{artist}%")
                .ilike("catalog_no", norm_catno(catno) or "")
                .limit(12).execute()
            )
            rows = res2.data or []
            dbg["local_calls"].append({
                "table": LOCAL_TABLE, "mode": "artist+catalog_no",
                "artist": artist, "catno": catno, "rows": len(rows), "ms": int((time.time()-t2)*1000)
            })

        # 4) wildcard EVA%008 fallback for EVA 008 / EVA-008 style
        if not rows:
            tight = norm_catno(catno) or ""
            m = re.match(r"([A-Z]+)(\d+)", tight)
            wl = f"{m.group(1)}%{m.group(2)}" if m else f"%{tight}%"
            t3 = time.time()
            res3 = (
                sb.table(LOCAL_TABLE)
                .select("release_id,label,catalog_no,artist,title")
                .ilike("catalog_no", wl)
                .limit(12).execute()
            )
            rows = res3.data or []
            dbg["local_calls"].append({
                "table": LOCAL_TABLE, "mode": "catalog_no_wild", "catno_wild": wl,
                "rows": len(rows), "ms": int((time.time()-t3)*1000)
            })

        # 5) build candidates
        for r in rows[:5]:
            rid = r.get("release_id")
            if not rid:
                # If your dump has a URL column, parse it here (adjust column name if present)
                # rid = _parse_release_id_from_url(r.get("discogs_url"))
                pass
            if not rid:
                continue
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

def search_discogs(params: Dict[str, str], dbg: Dict) -> List[IdentifyCandidate]:
    out: List[IdentifyCandidate] = []
    t0 = time.time()
    r = discogs_request("/database/search", params)
    dbg.setdefault("discogs_calls", []).append({
        "endpoint": "/database/search",
        "params": {k: v for k, v in params.items() if k != "token"},
        "status": r.status_code, "ms": int((time.time()-t0)*1000)
    })
    if r.status_code != 200:
        return out
    js = r.json()
    for it in js.get("results", [])[:8]:
        url = it.get("resource_url","")
        if "/releases/" not in url: continue
        try: rid = int(url.rstrip("/").split("/")[-1])
        except: continue
        artist_guess = (it.get("title","").split(" - ")[0] if " - " in it.get("title","") else None)
        out.append(IdentifyCandidate(
            source="ocr_search",
            release_id=rid,
            discogs_url=f"https://www.discogs.com/release/{rid}",
            artist=artist_guess,
            title=it.get("title"),
            label=(it.get("label") or [""])[0] if isinstance(it.get("label"), list) else it.get("label"),
            year=str(it.get("year") or ""),
            cover_url=it.get("thumb"),
            score=0.65
        ))
    return out

def multi_search(label: Optional[str], catno: Optional[str], artist: Optional[str],
                 tracks: List[str], strong_title: Optional[str], dbg: Dict) -> List[IdentifyCandidate]:
    attempts: List[Dict[str,str]] = []
    ncat = norm_catno(catno) if catno else None
    if label and ncat: attempts.append({"label": label, "catno": ncat, "type": "release"})
    if artist and ncat: attempts.append({"artist": artist, "catno": ncat, "type": "release"})
    if ncat: attempts.append({"catno": ncat, "type": "release"})  # catno-only attempt
    for t in tracks[:3]:
        p = {"track": t, "type": "release"}
        if artist: p["artist"] = artist
        attempts.append(p)
    if strong_title: attempts.append({"release_title": strong_title, "type": "release"})
    # broad q fallbacks
    if label and ncat: attempts.append({"q": f"{label} {ncat}", "type": "release"})
    if artist and strong_title: attempts.append({"q": f"{artist} {strong_title}", "type": "release"})
    if strong_title: attempts.append({"q": strong_title, "type": "release"})
    out: List[IdentifyCandidate] = []
    for p in attempts:
        res = search_discogs(p, dbg)
        if res:
            for c in res:
                if ("label" in p and "catno" in p) or ("artist" in p and "catno" in p):
                    c.score = 0.70
            out.extend(res)
            break
    return out

# ================== API ==================
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_api(
    file: UploadFile = File(...),
    debug: bool = Query(False, description="Return debug info when true")
) -> IdentifyResponse:
    dbg: Dict[str, Any] = {"steps": [], "web_urls": [], "queries_tried": []} if debug else {}
    try:
        t0 = time.time()
        image_bytes = await file.read()

        # 1) Vision (Lens-like)
        tv = time.time()
        v = call_vision_api(image_bytes)
        dbg and dbg["steps"].append({"stage": "vision", "elapsed_ms": int((time.time() - tv) * 1000)})
        web = v.get("webDetection", {})
        text = v.get("textAnnotations", [])
        release_id, master_id, discogs_url, urls = parse_web(web)
        if debug:
            dbg["web_urls"] = urls
            dbg["web_release_id"] = release_id
            dbg["web_master_id"] = master_id
            dbg["ocr_lines_raw"] = ocr_lines(text)[:60]

        candidates: List[IdentifyCandidate] = []

        # A) Direct web detection hit
        if release_id:
            t1 = time.time()
            r = discogs_request(f"/releases/{release_id}")
            dbg and dbg.setdefault("discogs_calls", []).append({
                "endpoint": f"/releases/{release_id}",
                "status": r.status_code,
                "elapsed_ms": int((time.time() - t1) * 1000)
            })
            if r.status_code == 200:
                rel = r.json()
                candidates.append(IdentifyCandidate(
                    source="web_detection_live",
                    release_id=release_id,
                    discogs_url=discogs_url or rel.get("uri"),
                    artist=", ".join(a.get("name","") for a in rel.get("artists", [])),
                    title=rel.get("title"),
                    label=", ".join(l.get("name","") for l in rel.get("labels", [])),
                    year=str(rel.get("year", "")),
                    cover_url=rel.get("thumb") or (rel.get("images") or [{}])[0].get("uri", ""),
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

        # C) OCR-derived signals + Google CSE + local dump + Discogs fallback
        if not candidates:
            lines = ocr_lines(text)
            if lines:
                # Clean & preserve digits/hyphens/slashes
                clean: List[str] = []
                for ln in lines:
                    cleaned = re.sub(r"[^\w\s/-]", "", ln).strip()
                    if cleaned: clean.append(cleaned)
                if debug: dbg["ocr_lines_clean"] = clean[:60]

                # Extract hints
                label_hint = None
                catno_hint = None
                artist_hint = None
                tracks: List[str] = []

                # Catalog number like UR-014, EVA008, DEF 004
                for ln in clean:
                    low = ln.lower()
                    m = re.search(r"[a-z]{2,}\s?-?\s?\d{1,5}", low)
                    if m and not catno_hint:
                        catno_hint = m.group(0).upper().replace(" ", " ")
                        catno_hint = re.sub(r"\s*-\s*", "-", catno_hint)

                # Label heuristic (… Records/Recordings/Music) or brand token like "evasive"
                for ln in clean:
                    low = ln.lower()
                    if re.search(r"(records|recordings|music)\b", low) and not label_hint:
                        label_hint = re.sub(r"(records|recordings|music)\b", "", ln, flags=re.I).strip()
                    if not label_hint and "evasive" in low:
                        label_hint = "Evasive Records"
                    if label_hint:
                        break

                # Artist heuristic: short ALLCAPS line (<=3 words)
                for ln in clean:
                    words = ln.split()
                    if 1 <= len(words) <= 3 and all(w.isalpha() and w.isupper() for w in words):
                        artist_hint = ln.title()
                        break

                # Track titles after ":" or " - "
                for ln in clean:
                    if ":" in ln:
                        tr = ln.split(":",1)[1].strip()
                        if tr: tracks.append(tr)
                    elif " - " in ln and not artist_hint:
                        rhs = ln.split(" - ",1)[1].strip()
                        if rhs and not re.search(r"\d", rhs):
                            tracks.append(rhs)

                # Prefer non-copyright line for strong title
                strong_title = None
                copyright_re = re.compile(r"^all rights of the manufacturer", re.I)
                non_copyright = [ln for ln in clean if not copyright_re.match(ln)]
                strong_title = non_copyright[0] if non_copyright else (clean[0] if clean else None)

                # Build Google CSE query
                signals = {
                    "p_catno": (catno_hint or "").upper().strip(),
                    "p_label": (label_hint or "").strip(),
                    "p_artist": (artist_hint or "").strip(),
                    "p_title": (strong_title or "").strip(),
                }
                if signals["p_title"].lower().startswith("all rights of the manufacturer"):
                    signals["p_title"] = ""

                parts = ["site:discogs.com"]
                if signals["p_artist"]:
                    parts.append(f"\"{signals['p_artist']}\"")
                if signals["p_title"]:
                    parts.append(f"\"{signals['p_title']}\"")
                if signals["p_catno"]:
                    parts.append(signals["p_catno"])
                g_query = " ".join(parts)
                if debug:
                    dbg["queries_tried"].append({"google_cse": g_query})

                # Google CSE tier
                picked_url = _google_cse_discogs(g_query, dbg if debug else {}, signals)
                if picked_url:
                    rid = _parse_release_id_from_url(picked_url)
                    if rid:
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
                            discogs_url=picked_url,
                            artist=artist_str or (artist_hint or None),
                            title=title_str  or (strong_title or None),
                            label=label_str  or (label_hint or None),
                            year=year,
                            cover_url=cover,
                            score=0.88
                        ))

                # Supabase local dump tier (if any)
                if sb and (catno_hint or label_hint or artist_hint):
                    locals_ = local_lookup(label_hint, catno_hint, artist_hint, dbg if debug else {})
                    candidates.extend(locals_)

                # Discogs structured fallback if still empty
                if not candidates:
                    cands = multi_search(label_hint, catno_hint, artist_hint, tracks, strong_title, dbg if debug else {})
                    if not cands:
                        # final hail-mary generic OCR queries
                        generic_qs = []
                        if clean:
                            generic_qs.append(" ".join(clean[:3])[:200])
                            if len(clean) >= 2: generic_qs.append(" ".join(clean[:2])[:200])
                            generic_qs.append(clean[0][:200])
                        for q in generic_qs:
                            if not q: continue
                            dbg and dbg["queries_tried"].append({"discogs_q": q})
                            cands = search_discogs({"q": q, "type":"release"}, dbg if debug else {})
                            if cands: break
                    candidates.extend(cands)

        # finalize
        # serialize pydantic models to dicts to avoid FastAPI model-of-model issues when mixing sources
        out = [c.dict() for c in candidates[:10]]
        dbg and dbg.update({"total_elapsed_ms": int((time.time()-t0)*1000)})
        return IdentifyResponse(candidates=out, debug=(dbg or None))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:300])

