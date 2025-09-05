
"""Groove ID backend.

This FastAPI application receives a record label scan, computes an image
embedding using a CLIP model, searches a Qdrant vector database for the
closest match and, if available, fetches full record metadata from
Supabase. The environment variables defined on the hosting platform
(Render) configure the connection details for Qdrant and Supabase.
"""

import os
import io
import shutil
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

import torch  # type: ignore
import torchvision.transforms as T  # type: ignore
import open_clip  # type: ignore

from backend.identify import router as identify_router
from backend.discogs_auth import router as discogs_auth_router
from backend.collection import router as collection_router

from qdrant_client import QdrantClient  # type: ignore
from supabase import create_client  # type: ignore


# Read environment variables for Qdrant and Supabase connections. These
# must be set in the hosting environment (e.g. Render dashboard). If
# they are missing, the application will raise during startup.
QDRANT_URL: str = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY: str = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION: str = os.environ.get("QDRANT_COLLECTION", "groove_id_vectors")
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Missing Qdrant configuration: QDRANT_URL and QDRANT_API_KEY must be set.")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase configuration: SUPABASE_URL and SUPABASE_KEY must be set.")

# Initialise external clients once. The Supabase client will be used to
# look up record metadata after a match is found in Qdrant. The Qdrant
# client performs vector similarity search.
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)


# Prepare the CLIP model and preprocessing pipeline. These objects
# consume significant resources, so they are loaded only once at
# module import. We use the ViT-B/32 variant for a good balance of
# performance and accuracy. The device is selected automatically.
_device = "cuda" if torch.cuda.is_available() else "cpu"
_clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
_clip_model = _clip_model.to(_device)


def embed_image(file_path: str) -> list[float]:
    """Compute a CLIP embedding for the provided image file.

    Args:
        file_path: Path to the image on disk.

    Returns:
        A list of floats representing the normalised image embedding.
    """
    # Load and convert the image to RGB
    image = Image.open(file_path).convert("RGB")
    # Apply CLIP's preprocessing (resize, crop, normalisation, tensor conversion)
    image_input = _clip_preprocess(image).unsqueeze(0).to(_device)
    # Compute the embedding with no gradients
    with torch.no_grad():
        image_features = _clip_model.encode_image(image_input)
    # Normalise the vector to unit length
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze(0).cpu().tolist()


def search_qdrant(vector: list[float]) -> Optional[Dict[str, Any]]:
    """Search the Qdrant collection for the closest vector.

    Args:
        vector: The query vector computed from the image embedding.

    Returns:
        The payload associated with the best matching point, or None if
        no match is found.
    """
    search_result = qdrant_client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vector,
        limit=1,
        with_payload=True,
    )
    if not search_result:
        return None
    # Each result is a ScoredPoint; we extract the payload dictionary
    return search_result[0].payload


class IdentifyResponse(BaseModel):
    title: str
    artist: str
    label: str
    catalog_number: str
    discogs_url: str
    confidence: float
    used_fallback: bool
    used_override: bool


app = FastAPI()

# Configure CORS to allow requests from any origin. In production,
# consider restricting this to your frontend domain for better
# security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routers for vision identification and discogs auth/collection
app.include_router(identify_router)
app.include_router(discogs_auth_router)
app.include_router(collection_router)



@app.post("/api/identify", response_model=IdentifyResponse)
async def identify_record(image: UploadFile = File(...)) -> IdentifyResponse:
    """Identify a record by its label image.

    This endpoint accepts an uploaded image, computes its embedding,
    performs a nearest-neighbour search in Qdrant, and returns record
    metadata. If no match is found or metadata cannot be fetched,
    appropriate HTTP errors are raised.

    Args:
        image: The uploaded image file containing a record label.

    Returns:
        An IdentifyResponse with record details and confidence score.
    """
    # Save the uploaded image to a temporary file on disk
    temp_path = f"temp_{image.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        # Compute embedding and search Qdrant
        embedding = embed_image(temp_path)
        payload = search_qdrant(embedding)

        if payload is None:
            raise HTTPException(status_code=404, detail="No matching record found.")

        # Attempt to fetch full metadata from Supabase using an ID from the payload.
        # We look for a key 'id' or 'release_id' in the payload. If not
        # present, we assume the payload already contains all metadata.
        record = None
        record_id = payload.get("id") or payload.get("release_id")
        if record_id:
            # Query the 'records' table (adjust table name as needed)
            response = supabase_client.table("records").select("*", count="exact").eq("id", record_id).execute()
            if response.data:
                record = response.data[0]

        # Fallback: if record is None, use the payload directly as metadata
        meta = record or payload

        # Map keys to our response model, providing defaults for missing
        # values. Qdrant payloads may use different naming schemes (e.g.
        # 'artist_name' vs 'artist'). Adjust these keys as needed to
        # match your ingestion pipeline.
        title = meta.get("title") or meta.get("release_title") or "Unknown Title"
        artist = meta.get("artist") or meta.get("artist_name") or "Unknown Artist"
        label = meta.get("label") or meta.get("label_name") or "Unknown Label"
        catalog_number = meta.get("catalog_number") or meta.get("cat_no") or ""
        discogs_url = meta.get("discogs_url") or meta.get("discogs_link") or ""
        confidence = meta.get("score") or meta.get("confidence") or 0.0
        used_override = bool(meta.get("used_override", False))

        return IdentifyResponse(
            title=title,
            artist=artist,
            label=label,
            catalog_number=catalog_number,
            discogs_url=discogs_url,
            confidence=float(confidence),
            used_fallback=record is None,
            used_override=used_override,
        )
    finally:
        # Always remove the temporary file
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":  # pragma: no cover
    # Only run uvicorn if this module is executed directly. When
    # deployed on Render, the start command will invoke uvicorn via
    # process manager.
    import uvicorn

    uvicorn.run("groove_id_backend:app", host="0.0.0.0", port=8000, reload=True)
