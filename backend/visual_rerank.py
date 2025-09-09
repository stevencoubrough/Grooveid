import numpy as np
from typing import List, Dict, Tuple
from .vis_embedder import embed_image_bytes, embed_image_url, cosine
from .identify import IdentifyCandidate  # if circular import, move this function into identify.py


def _pick_candidate_urls(cands: List[IdentifyCandidate]) -> Dict[int, List[str]]:
    urls: Dict[int, List[str]] = {}
    for i, c in enumerate(cands):
        # Use provided cover_url (Discogs 'thumb') if present
        ulist = []
        if c.cover_url:
            ulist.append(c.cover_url)
        # Optionally, add more URLs if you store them later (e.g., label images)
        urls[i] = ulist[:2]  # limit to 1-2 per candidate
    return urls


def visual_rerank(user_image_bytes: bytes, candidates: List[IdentifyCandidate]) -> List[Tuple[IdentifyCandidate, float]]:
    if len(candidates) <= 1:
        return [(candidates[0], 1.0)] if candidates else []
    # Embed user image once (center/full already preprocessed upstream)
    uvec = embed_image_bytes(user_image_bytes)
    cand_urls = _pick_candidate_urls(candidates)
    # Compute a max similarity per candidate across its urls
    sims: List[float] = []
    for i, c in enumerate(candidates):
        best = -1.0
        for url in cand_urls.get(i, []):
            vec = embed_image_url(url)
            if vec is None:
                continue
            s = cosine(uvec, vec)
            best = max(best, s)
        sims.append(best)
    # Combine with existing 'score' (text score) if available
    out: List[Tuple[IdentifyCandidate, float]] = []
    for c, vis in zip(candidates, sims):
        text_score = getattr(c, "score", 0.0) or 0.0
        # weights: tweak as you like; start 55% text + 40% visual + 5% prior
        final = 0.55 * text_score + 0.40 * (max(vis, 0.0)) + 0.05 * 0.0
        out.append((c, final))
    out.sort(key=lambda x: x[1], reverse=True)
    return out
