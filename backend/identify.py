
"""identify_with_rpc.py

Drop-in FastAPI module for GrooveID:
- Google Vision Web Detection + Text/Document OCR (good for graffiti/handwriting)
- Google CSE ranking (site:discogs.com)
- Supabase RPC fallback: calls `search_records(p_catno, p_label, p_artist, p_title)`
  (you must create this RPC in your Supabase DB; SQL snippet included below)
- Discogs structured fallback
- Debug logging via ?debug=true
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
import os, re, base64, time, requests

# ----------------- ENV -----------------
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
VISION_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()

DISCOGS_API = "https://api.discogs.com"
DISCOGS_TOKEN = os.environ.get("DISCOGS_TOKEN", "").strip()

GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY", "").strip()
GOOGLE_CSE_ID        = os.environ.get("GOOGLE_CSE_ID", "").strip()

# Supabase client (optional)
try:
    from supabase import create_client, Client  # type: ignore
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    sb: Optional["Client"] = create_client(SUPABASE_URL, SUPABASE_KEY) if (SUPABASE_URL and SUPABASE_KEY) else None
except Exception:
    sb = None

LOCAL_TABLE = os.environ.get("DISCOGS_LOCAL_TABLE", "records")

router = APIRouter()

# ----------------- MODELS -----------------
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
    debug: Optional[Dict[str, Any]] = None

# ----------------- HELPERS -----------------
def discogs_request(path: str, params: Dict = None, timeout=20) -> requests.Response:
    if params is None:
        params = {}
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
        m = re.search(r"discogs\\.com/(?:[^/]+/)?release/(\\d+)", u, re.I)
        if m:
            release_id = int(m.group(1)); discogs_url = u; break
    if not release_id:
        for u in urls:
            m = re.search(r"discogs\\.com/(?:[^/]+/)?master/(\\d+)", u, re.I)
            if m:
                master_id = int(m.group(1)); discogs_url = u; break
    return release_id, master_id, discogs_url, urls

def merge_google_ocr(resp: dict) -> List[str]:
    lines: List[str] = []
    fta = resp.get("fullTextAnnotation") or {}
    txt = fta.get("text")
    if txt:
        lines.extend([ln.strip() for ln in txt.splitlines() if ln.strip()])
    for t in resp.get("textAnnotations", [])[1:]:
        d = t.get("description")
        if d:
            lines.extend([ln.strip() for ln in d.splitlines() if ln.strip()])
    out: List[str] = []
    seen = set()
    for ln in lines:
        key = ln.lower()
        if key not in seen:
            seen.add(key)
            out.append(ln)
    return out[:130]

def ocr_lines(resp: dict) -> List[str]:
    return merge_google_ocr(resp)

def norm_catno(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.upper().strip()
    s = re.sub(r"\\s*-\\s*", "-", s)
    s = re.sub(r"\\s+", " ", s)
    return s

def google_cse_discogs(query: str, dbg: Dict, signals: Dict[str, str]) -> Optional[str]:
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
        if "discogs.com" not in link: continue
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

# ----------------- Supabase RPC -----------------
def rpc_search_records(signals: Dict[str, str], dbg: Dict) -> List[Dict[str, Any]]:
    """
    Calls Supabase RPC `search_records(p_catno, p_label, p_artist, p_title)`.
    The RPC should return rows with fields: release_id, catalog_no, label, artist, title, discogs_url, score
    """
    out: List[Dict[str, Any]] = []
    if not sb:
        dbg.setdefault("local_calls", []).append({"rpc": "skipped_no_client"})
        return out
    try:
        t0 = time.time()
        # Note: supabase-py RPC signature expects named args as a dict
        res = sb.rpc("search_records", {
            "p_catno": signals.get("p_catno") or None,
            "p_label": signals.get("p_label") or None,
            "p_artist": signals.get("p_artist") or None,
            "p_title": signals.get("p_title") or None,
        }).execute()
        rows = res.data or []
        dbg.setdefault("local_calls", []).append({"rpc": "search_records", "rows": len(rows), "elapsed_ms": int((time.time()-t0)*1000)})
        return rows
    except Exception as e:
        dbg.setdefault("local_calls", []).append({"rpc_error": str(e)})
        return out

# ----------------- Local fallback (legacy) -----------------
def local_lookup(catno: Optional[str], label: Optional[str], artist: Optional[str], dbg: Dict) -> List[IdentifyCandidate]:
    out: List[IdentifyCandidate] = []
    if not sb or not catno:
        return out
    try:
        res0 = (sb.table(LOCAL_TABLE)
                  .select("release_id,label,catalog_no,artist,title,discogs_url")
                  .ilike("catalog_no", norm_catno(catno) or "")
                  .limit(12).execute())
        rows = res0.data or []
        dbg.setdefault("local_calls", []).append({"mode":"catalog_no_only","rows":len(rows)})
        if not rows and label:
            res = (sb.table(LOCAL_TABLE)
                     .select("release_id,label,catalog_no,artist,title,discogs_url")
                     .ilike("label", f"%{label.strip()}%")
                     .ilike("catalog_no", norm_catno(catno) or "")
                     .limit(12).execute())
            rows = res.data or []
            dbg["local_calls"].append({"mode":"label+catalog_no","rows":len(rows)})
        if not rows and artist:
            res2 = (sb.table(LOCAL_TABLE)
                      .select("release_id,label,catalog_no,artist,title,discogs_url")
                      .ilike("artist", f"%{artist}%")
                      .ilike("catalog_no", norm_catno(catno) or "")
                      .limit(12).execute())
            rows = res2.data or []
            dbg["local_calls"].append({"mode":"artist+catalog_no","rows":len(rows)})
        def parse_rid(url: Optional[str]) -> Optional[int]:
            if not url: return None
            m = re.search(r"/release/(\\d+)", url)
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

# ----------------- Discogs structured fallback -----------------
def search_discogs(params: Dict[str, str], dbg: Dict) -> List[IdentifyCandidate]:
    out: List[IdentifyCandidate] = []
    r = discogs_request("/database/search", params)
    dbg.setdefault("discogs_calls", []).append({
        "endpoint": "/database/search",
        "params": {k: v for k,v in params.items() if k != "token"},
        "status": r.status_code,
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

# ----------------- API -----------------
@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_api(
    file: UploadFile = File(...),
    debug: bool = Query(False, description="Return debug info when true")
) -> IdentifyResponse:
    dbg: Dict[str, Any] = {"steps": [], "web_urls": [], "queries_tried": []} if debug else {}
    try:
        t0 = time.time()
        image_bytes = await file.read()

        # Vision
        tv = time.time()
        resp = call_vision(image_bytes)
        dbg and dbg["steps"].append({"stage":"vision","elapsed_ms": int((time.time()-tv)*1000)})
        release_id, master_id, discogs_url, urls = parse_web(resp.get("webDetection", {}))
        if debug:
            dbg["web_urls"] = urls
            dbg["web_release_id"] = release_id
            dbg["web_master_id"]  = master_id

        # OCR lines (merged)
        lines = ocr_lines(resp)
        if debug: dbg["ocr_lines_raw"] = lines[:120]

        candidates: List[IdentifyCandidate] = []

        # A) Web detection direct hit
        if release_id:
            tr = time.time()
            rel = discogs_request(f"/releases/{release_id}")
            dbg and dbg.setdefault("discogs_calls", []).append({"endpoint": f"/releases/{release_id}", "status": rel.status_code, "elapsed_ms": int((time.time()-tr)*1000)})
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
                    cover_url=js.get("thumb") or (js.get("images") or [{}])[0].get("uri",""),
                    score=0.90
                ))

        # B) Master only
        if not candidates and master_id:
            candidates.append(IdentifyCandidate(source="web_detection_master", master_id=master_id, discogs_url=f"https://www.discogs.com/master/{master_id}", note="Master match — user must pick a pressing", score=0.60))

        # C) Text-driven (Google OCR)
        if not candidates and lines:
            clean = [re.sub(r"[^\w\s/-]", "", ln).strip() for ln in lines if ln.strip()]
            if debug: dbg["ocr_lines_clean"] = clean[:120]

            # extract hints
            catalog_no_hint = None
            label_hint = None
            artist_hint = None
            for ln in clean:
                low = ln.lower()
                m = re.search(r"[a-z]{2,}\s?-?\s?\d{1,5}", low)
                if m and not catalog_no_hint:
                    catalog_no_hint = re.sub(r"\s*-\s*", "-", m.group(0).upper().replace("  "," "))
                if not label_hint and re.search(r"(records|recordings|music)\b", low):
                    label_hint = re.sub(r"(records|recordings|music)\b", "", ln, flags=re.I).strip()
                if not artist_hint:
                    words = ln.split()
                    if 1 <= len(words) <= 3 and all(w.isalpha() and w.isupper() for w in words):
                        artist_hint = ln.title()

            rim_re = re.compile(r"^all rights of the manufacturer", re.I)
            non_rim = [ln for ln in clean if not rim_re.match(ln)]
            strong_title = non_rim[0] if non_rim else (clean[0] if clean else "")

            signals = {
                "p_catno": (catalog_no_hint or "").upper().strip(),
                "p_label": (label_hint or "").strip(),
                "p_artist": (artist_hint or "").strip(),
                "p_title": (strong_title or "").strip(),
            }
            dbg and dbg.setdefault("extracted", signals)

            # 1) Google CSE ranker
            parts = ["site:discogs.com"]
            if signals["p_artist"]: parts.append(f"\"{signals['p_artist']}\"")
            if signals["p_title"]: parts.append(f"\"{signals['p_title']}\"")
            if signals["p_catno"]: parts.append(signals["p_catno"])
            g_query = " ".join(parts)
            dbg and dbg.setdefault("queries_tried", []).append({"google_cse": g_query})

            cse_url = google_cse_discogs(g_query, dbg if debug else {}, signals)
            if cse_url:
                m = re.search(r"/release/(\\d+)", cse_url)
                if m:
                    rid = int(m.group(1))
                    tr2 = time.time()
                    rel = discogs_request(f"/releases/{rid}")
                    dbg and dbg.setdefault("discogs_calls", []).append({"endpoint": f"/releases/{rid}", "status": rel.status_code, "elapsed_ms": int((time.time()-tr2)*1000)})
                    if rel.status_code == 200:
                        js = rel.json()
                        candidates.append(IdentifyCandidate(
                            source="google_cse",
                            release_id=rid,
                            discogs_url=cse_url,
                            artist=", ".join(a.get("name","") for a in js.get("artists", [])),
                            title=js.get("title"),
                            label=", ".join(l.get("name","") for l in js.get("labels", [])),
                            year=str(js.get("year","")),
                            cover_url=js.get("thumb") or (js.get("images") or [{}])[0].get("uri",""),
                            score=0.88
                        ))

            # 2) Supabase RPC composite search (if still nothing)
            if not candidates and (signals["p_catno"] or signals["p_label"] or signals["p_artist"] or signals["p_title"]):
                rows = rpc_search_records(signals, dbg if debug else {})
                if rows:
                    # rows expected to have release_id, title, artist, label, discogs_url, score
                    for r in rows[:8]:
                        try:
                            rid = int(r.get("release_id")) if r.get("release_id") else None
                        except Exception:
                            rid = None
                        disc_url = r.get("discogs_url") or (f"https://www.discogs.com/release/{rid}" if rid else None)
                        candidates.append(IdentifyCandidate(
                            source="supabase_rpc",
                            release_id=rid,
                            discogs_url=disc_url,
                            artist=r.get("artist"),
                            title=r.get("title"),
                            label=r.get("label"),
                            year=None,
                            cover_url=None,
                            score=float(r.get("score") or 0.75)
                        ))

            # 3) Legacy local lookup
            if not candidates and (signals["p_catno"] or signals["p_label"] or signals["p_artist"]):
                local = local_lookup(signals["p_catno"], signals["p_label"], signals["p_artist"], dbg if debug else {})
                if local:
                    candidates.extend(local)

            # 4) Discogs structured fallback
            if not candidates:
                attempts: List[Dict[str,str]] = []
                ncat = norm_catno(signals["p_catno"]) if signals["p_catno"] else None
                if signals["p_label"] and ncat: attempts.append({"label": signals["p_label"], "catno": ncat, "type": "release"})
                if signals["p_artist"] and ncat: attempts.append({"artist": signals["p_artist"], "catno": ncat, "type": "release"})
                if signals["p_title"]: attempts.append({"release_title": signals["p_title"], "type": "release"})
                if signals["p_label"] and ncat: attempts.append({"q": f"{signals['p_label']} {ncat}", "type":"release"})
                if signals["p_artist"] and signals["p_title"]: attempts.append({"q": f"{signals['p_artist']} {signals['p_title']}", "type":"release"})
                if signals["p_title"]: attempts.append({"q": signals["p_title"], "type":"release"})
                for p in attempts:
                    res = search_discogs(p, dbg if debug else {})
                    if res:
                        for c in res:
                            if ("label" in p and "catno" in p) or ("artist" in p and "catno" in p):
                                c.score = 0.70
                        candidates.extend(res)
                        break

        # finalize
        # sort by score desc (highest first)
        candidates = sorted(candidates, key=lambda c: c.score if c.score is not None else 0.0, reverse=True)
        dbg and dbg.update({"total_elapsed_ms": int((time.time()-t0)*1000)})
        return IdentifyResponse(candidates=candidates[:12], debug=(dbg or None))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:300])

# ----------------- SUPABASE RPC SQL (run in your Supabase SQL editor) -----------------
SQL_SNIPPET = r\"\"\"
-- enable trigram for fuzzy matching
create extension if not exists pg_trgm;

create index if not exists idx_records_catalog_no_trgm
  on records using gin (catalog_no gin_trgm_ops);
create index if not exists idx_records_title_trgm
  on records using gin (title gin_trgm_ops);
create index if not exists idx_records_artist_trgm
  on records using gin (artist gin_trgm_ops);
create index if not exists idx_records_label_trgm
  on records using gin (label gin_trgm_ops);

create or replace function search_records(
  p_catno  text,
  p_label  text default null,
  p_artist text default null,
  p_title  text default null
)
returns table (
  release_id   int,
  catalog_no   text,
  label        text,
  artist       text,
  title        text,
  discogs_url  text,
  score        numeric
)
language sql stable as
$$
  select set_limit(0.35);

  with tokens as (
    select
      upper(coalesce(p_catno, '')) as cat,
      coalesce(p_label,  '')::text as lbl,
      coalesce(p_artist, '')::text as art,
      coalesce(p_title,  '')::text as ttl,
      case
        when p_catno ~ '^[A-Za-z]{2,}\\d+$'
          then upper(regexp_replace(p_catno, '^([A-Za-z]+)(\\d+)$', '\\1%\\2'))
        else null
      end as cat_wild
  )
  select
    r.release_id,
    r.catalog_no,
    r.label,
    r.artist,
    r.title,
    r.discogs_url,
    (case when tokens.cat <> '' and upper(r.catalog_no) = tokens.cat then 3.0 else 0 end) +
    (case when tokens.cat_wild is not null and r.catalog_no ilike tokens.cat_wild then 2.0 else 0 end) +
    similarity(r.title,  tokens.ttl)  * 1.5 +
    similarity(r.artist, tokens.art)  * 1.2 +
    similarity(r.label,  tokens.lbl)  * 1.0
    as score
  from records r, tokens
  where
    (tokens.cat <> '' and upper(r.catalog_no) = tokens.cat)
    or (tokens.cat_wild is not null and r.catalog_no ilike tokens.cat_wild)
    or (tokens.ttl <> '' and r.title  % tokens.ttl)
    or (tokens.art <> '' and r.artist % tokens.art)
    or (tokens.lbl <> '' and r.label  % tokens.lbl)
  order by score desc, release_id
  limit 20;
$$;
\"\"\"

# End of file

