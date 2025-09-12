"""
identify.py — GrooveID consolidated resolver

Pipeline:
1) Google Vision Web Detection (Discogs links from image)
2) Google Vision TEXT + DOCUMENT_TEXT_DETECTION (OCR)
3) Google Custom Search (CSE) — site:discogs.com "<artist>" "<title>" <catno>
4) Supabase RPC fallback (search_records: scores catno + fuzzy artist/label/title)
5) Legacy local lookup (catalog_no → label+catalog_no → artist+catalog_no)
6) Discogs structured fallback
7) Debug trace via ?debug=true
"""

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

# Optional Supabase
try:
    from supabase import create_client, Client  # type: ignore
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    sb: Optional["Client"] = create_client(SUPABASE_URL, SUPABASE_KEY) if (SUPABASE_URL and SUPABASE_KEY) else None
except Exception:
    sb = None

# Default to your dump table name
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
    debug: Optional[Dict[str, Any]] = None  # returned when ?debug=1

# ===================== HELPERS =====================
def discogs_request(path: str, params: Dict = None, timeout=20) -> requests.Response:
    if params is None: params = {}
    headers = {"User-Agent": "GrooveID/1.0 (+https://grooveid.app)"}
    if DISCOGS_TOKEN:
        headers["Authorization"] = f"Discogs token={DISCOGS_TOKEN}"
        params.setdefault("token", DISCOGS_TOKEN)
    url = path if path.startswith("http") else f"{DISCOGS_API}{path}"
    return requests.get(url, headers=headers, params=params, timeout=timeout)

def call_vision(image_bytes: bytes) -> dict:
    if not VISION_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_VISION_API_KEY not set")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [{
            "image": {"content": b64},
            "features": [
                {"type": "WEB_DETECTION", "maxResults": 10},
                {"type": "TEXT_DETECTION", "maxResults": 5},
                {"type": "DOCUMENT_TEXT_DETECTION"},
            ],
            "imageContext": {
                "webDetectionParams": {"includeGeoResults": True},
                "languageHints": ["en","fr","de","es","ru"]
            },
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
        if m: release_id = int(m.group(1)); discogs_url = u; break
    if not release_id:
        for u in urls:
            m = re.search(r"discogs\.com/(?:[^/]+/)?master/(\d+)", u, re.I)
            if m: master_id = int(m.group(1)); discogs_url = u; break
    return release_id, master_id, discogs_url, urls

def merge_google_ocr(resp: dict) -> List[str]:
    lines: List[str] = []
    fta = resp.get("fullTextAnnotation") or {}
    txt = fta.get("text")
    if txt: lines.extend([ln.strip() for ln in txt.splitlines() if ln.strip()])
    for t in resp.get("textAnnotations", [])[1:]:
        d = t.get("description")
        if d: lines.extend([ln.strip() for ln in d.splitlines() if ln.strip()])
    # de-dupe
    out, seen = [], set()
    for ln in lines:
        k = ln.lower()
        if k not in seen:
            seen.add(k)
            out.append(ln)
    return out[:150]

def ocr_lines(resp: dict) -> List[str]:
    return merge_google_ocr(resp)

def norm_catno(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.upper().strip()
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s

# ----------------- Google CSE -----------------
def google_cse_discogs(query: str, dbg: Dict, signals: Dict[str, Any]) -> Optional[str]:
    api = GOOGLE_SEARCH_API_KEY; cx = GOOGLE_CSE_ID
    if not api or not cx or not query: return None
    params = {"key": api, "cx": cx, "q": query, "num": 5, "safe": "off"}
    t0 = time.time()
    r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=15)
    dbg.setdefault("google_calls", []).append({
        "endpoint": "customsearch/v1", "query": query, "status": r.status_code,
        "elapsed_ms": int((time.time()-t0)*1000)
    })
    if r.status_code != 200 or not r.json().get("items"): return None
    best, best_score = None, -1.0
    cat = (signals.get("p_catno")  or "").lower()
    art = (signals.get("p_artist") or "").lower()
    ttl = (signals.get("p_title")  or "").lower()
    tr1 = (signals.get("tracks") or [None, None])[0]
    tr2 = (signals.get("tracks") or [None, None])[1]
    for item in r.json().get("items", []):
        link = item.get("link",""); 
        if "discogs.com" not in link: continue
        title = (item.get("title") or "").lower()
        snip  = (item.get("snippet") or "").lower()
        corpus = title + " " + snip
        score = 0.0
        if "/release/" in link: score += 3.0
        if "/master/"  in link: score += 1.0
        if cat and cat in corpus: score += 2.0
        if art and art in corpus: score += 1.2
        if ttl and ttl in corpus: score += 1.0
        if tr1 and isinstance(tr1,str) and tr1.lower() in corpus: score += 1.1
        if tr2 and isinstance(tr2,str) and tr2.lower() in corpus: score += 0.9
        if score > best_score: best_score, best = score, link
    return best

# ----------------- Supabase RPC -----------------
def rpc_search_records(signals: Dict[str, str], dbg: Dict) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not sb:
        dbg.setdefault("local_calls", []).append({"rpc": "skipped_no_client"})
        return out
    try:
        t0 = time.time()
        res = sb.rpc("search_records", {
            "p_catno":  signals.get("p_catno")  or None,
            "p_label":  signals.get("p_label")  or None,
            "p_artist": signals.get("p_artist") or None,
            "p_title":  signals.get("p_title")  or None,
        }).execute()
        rows = res.data or []
        dbg.setdefault("local_calls", []).append({"rpc": "search_records", "rows": len(rows), "elapsed_ms": int((time.time()-t0)*1000)})
        return rows
    except Exception as e:
        dbg.setdefault("local_calls", []).append({"rpc_error": str(e)})
        return out

# ----------------- Legacy Supabase lookup -----------------
def local_lookup(catno: Optional[str], label: Optional[str], artist: Optional[str], dbg: Dict) -> List[IdentifyCandidate]:
    out: List[IdentifyCandidate] = []
    if not sb or not catno: return out
    try:
        t0 = time.time()
        res0 = (sb.table(LOCAL_TABLE)
                  .select("release_id,label,catalog_no,artist,title,discogs_url")
                  .ilike("catalog_no", norm_catno(catno) or "")
                  .limit(12).execute())
        rows = res0.data or []
        dbg.setdefault("local_calls", []).append({"mode":"catalog_no_only","rows":len(rows),"ms":int((time.time()-t0)*1000)})
        if not rows and label:
            t1 = time.time()
            res = (sb.table(LOCAL_TABLE)
                     .select("release_id,label,catalog_no,artist,title,discogs_url")
                     .ilike("label", f"%{label.strip()}%")
                     .ilike("catalog_no", norm_catno(catno) or "")
                     .limit(12).execute())
            rows = res.data or []
            dbg["local_calls"].append({"mode":"label+catalog_no","rows":len(rows),"ms":int((time.time()-t1)*1000)})
        if not rows and artist:
            t2 = time.time()
            res2 = (sb.table(LOCAL_TABLE)
                      .select("release_id,label,catalog_no,artist,title,discogs_url")
                      .ilike("artist", f"%{artist}%")
                      .ilike("catalog_no", norm_catno(catno) or "")
                      .limit(12).execute())
            rows = res2.data or []
            dbg["local_calls"].append({"mode":"artist+catalog_no","rows":len(rows),"ms":int((time.time()-t2)*1000)})

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
                artist=r.get("artist"), title=r.get("title"), label=r.get("label"),
                year=None, cover_url=None, score=0.92
            ))
        return out
    except Exception as e:
        dbg.setdefault("local_calls", []).append({"error": str(e)})
        return out

# ----------------- Discogs structured fallback -----------------
def search_discogs(params: Dict[str, str], dbg: Dict) -> List[IdentifyCandidate]:
    out: List[IdentifyCandidate] = []
    t0 = time.time()
    r = discogs_request("/database/search", params)
    dbg.setdefault("discogs_calls", []).append({
        "endpoint": "/database/search",
        "params": {k:v for k,v in params.items() if k!="token"},
        "status": r.status_code, "elapsed_ms": int((time.time()-t0)*1000)
    })
    if r.status_code != 200: return out
    js = r.json()
    for it in js.get("results", [])[:8]:
        url = it.get("resource_url","")
        if "/releases/" not in url: continue
        try: rid = int(url.rstrip("/").split("/")[-1])
        except: continue
        out.append(IdentifyCandidate(
            source="ocr_search", release_id=rid, discogs_url=f"https://www.discogs.com/release/{rid}",
            artist=(it.get("title","").split(" - ")[0] if " - " in it.get("title","") else None),
            title=it.get("title"),
            label=(it.get("label") or [""])[0] if isinstance(it.get("label"), list) else it.get("label"),
            year=str(it.get("year") or ""), cover_url=it.get("thumb"), score=0.65
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
        resp = call_vision(image_bytes)
        release_id, master_id, discogs_url, urls = parse_web(resp.get("webDetection", {}))
        if debug:
            dbg["web_urls"] = urls
            dbg["web_release_id"] = release_id
            dbg["web_master_id"]  = master_id

        lines = ocr_lines(resp)
        if debug: dbg["ocr_lines_raw"] = lines[:120]

        candidates: List[IdentifyCandidate] = []

        # A) Web Detection hit
        if release_id:
            rel = discogs_request(f"/releases/{release_id}")
            if rel.status_code == 200:
                js = rel.json()
                candidates.append(IdentifyCandidate(
                    source="web_detection_live",
                    release_id=release_id, discogs_url=discogs_url or js.get("uri"),
                    artist=", ".join(a.get("name","") for a in js.get("artists", [])),
                    title=js.get("title"), label=", ".join(l.get("name","") for l in js.get("labels", [])),
                    year=str(js.get("year","")), cover_url=js.get("thumb") or (js.get("images") or [{}])[0].get("uri",""),
                    score=0.90
                ))

        # B) Master only
        if not candidates and master_id:
            candidates.append(IdentifyCandidate(
                source="web_detection_master", master_id=master_id,
                discogs_url=f"https://www.discogs.com/master/{master_id}",
                note="Master match — user must pick a pressing", score=0.60
            ))

        # C) Text-driven flow
        if not candidates and lines:
            # Clean (keep digits/-/slashes), skip empty
            clean = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in lines if ln.strip()]
            if debug: dbg["ocr_lines_clean"] = clean[:150]

            # Extract signals
            copyright_re = re.compile(r"^all rights of the manufacturer", re.I)
            non_rim = [ln for ln in clean if not copyright_re.match(ln)]

            # catalog number
            catalog_no_hint = None
            for ln in clean:
                m = re.search(r"[a-z]{2,}\s?-?\s?\d{1,5}", ln.lower())
                if m and not catalog_no_hint:
                    catalog_no_hint = re.sub(r"\s*-\s*", "-", m.group(0).upper().replace("  "," "))

            # label
            label_hint = None
            for ln in clean:
                if re.search(r"(records|recordings|music)\b", ln, re.I):
                    label_hint = re.sub(r"(records|recordings|music)\b", "", ln, flags=re.I).strip()
                    break
            if not label_hint:
                # soft inference: 'evasive' token
                for ln in clean:
                    if "evasive" in ln.lower():
                        label_hint = "Evasive Records"; break

            # artist (short ALLCAPS)
            artist_hint = None
            for ln in clean:
                words = ln.split()
                if 1 <= len(words) <= 3 and all(w.isalpha() and w.isupper() for w in words):
                    artist_hint = ln.title()
                    break

            # tracks
            tracks: List[str] = []
            for ln in clean:
                low = ln.lower()
                if re.match(r"^[ab]\s*[\.:]?\s*\d?\s*", low):
                    title = re.sub(r"^[ab]\s*[\.:]?\s*\d?\s*", "", ln).strip(" -:·")
                    if title: tracks.append(title)
                elif re.match(r"^\d+\.\s+", low):
                    title = re.sub(r"^\d+\.\s+", "", ln).strip(" -:·")
                    if title: tracks.append(title)

            strong_title = (non_rim[0] if non_rim else (clean[0] if clean else ""))

            signals: Dict[str, Any] = {
                "p_catno": (catalog_no_hint or "").upper().strip(),
                "p_label": (label_hint or "").strip(),
                "p_artist": (artist_hint or "").strip(),
                "p_title": (strong_title or "").strip(),
                "tracks": tracks[:3]
            }
            if debug: dbg["extracted"] = signals

            # 1) Google CSE first (Lens-like text ranking)
            parts = ["site:discogs.com"]
            if tracks:
                for t in tracks[:2]: parts.append(f"\"{t}\"")
            if not tracks and signals["p_title"]: parts.append(f"\"{signals['p_title']}\"")
            if signals["p_artist"]: parts.append(f"\"{signals['p_artist']}\"")
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
                    candidates.append(IdentifyCandidate(
                        source="google_cse", release_id=rid, discogs_url=cse_url,
                        artist=artist_str or artist_hint, title=title_str or strong_title or None,
                        label=label_str or label_hint, year=year, cover_url=cover, score=0.88
                    ))

            # 2) Supabase RPC composite (if still nothing)
            if not candidates and (signals["p_catno"] or signals["p_label"] or signals["p_artist"] or signals["p_title"]):
                rows = rpc_search_records(signals, dbg if debug else {})
                if rows:
                    def parse_release_id(url: Optional[str]) -> Optional[int]:
                        if not url: return None
                        m = re.search(r"/release/(\d+)", url)
                        return int(m.group(1)) if m else None
                    for r in rows[:8]:
                        rid = r.get("release_id") or parse_release_id(r.get("discogs_url"))
                        if not rid: continue
                        candidates.append(IdentifyCandidate(
                            source="local_dump", release_id=int(rid),
                            discogs_url=f"https://www.discogs.com/release/{rid}",
                            artist=r.get("artist"), title=r.get("title"), label=r.get("label"),
                            year=None, cover_url=None, score=float(r.get("score") or 0.9)
                        ))

            # 3) Legacy local lookup (catalog_no → label+catalog_no → artist+catalog_no)
            if not candidates and (catalog_no_hint or label_hint or artist_hint):
                local = local_lookup(catalog_no_hint, label_hint, artist_hint, dbg if debug else {})
                if local: candidates.extend(local)

            # 4) Discogs structured fallback
            if not candidates:
                attempts: List[Dict[str,str]] = []
                ncat = norm_catno(catalog_no_hint) if catalog_no_hint else None
                if label_hint and ncat: attempts.append({"label": label_hint, "catno": ncat, "type": "release"})
                if artist_hint and ncat: attempts.append({"artist": artist_hint, "catno": ncat, "type": "release"})
                for t in tracks[:2]:
                    p = {"track": t, "type":"release"}; 
                    if artist_hint: p["artist"] = artist_hint
                    attempts.append(p)
                if strong_title: attempts.append({"release_title": strong_title, "type":"release"})
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
        candidates = sorted(candidates, key=lambda c: c.score if c.score is not None else 0.0, reverse=True)
        dbg and dbg.update({"total_elapsed_ms": int((time.time()-t0)*1000)})
        return IdentifyResponse(candidates=candidates[:12], debug=(dbg or None))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:300])
