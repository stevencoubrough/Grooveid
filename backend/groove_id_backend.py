
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
impo
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))rt s

from backend.identify import router as identify_router

from backend.discogs_auth import router as discogs_auth_router
from backend.collection import router as collection_router
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routers for identify, Discogs auth, and collection
app.include_router(identify_router, prefix="/api"
app.include_router(discogs_auth_router)
app.include_router(collection_router)


def embed_image(file_path):
    return [0.1] * 512

def search_qdrant(vector):
    return {
        "title": "Phenomenon",
        "artist_name": "Dista",
        "label": "Pleasure",
        "catalog_number": "JOY3",
        "discogs_url": "https://www.discogs.com/release/36144-Dista-Phenomenon",
        "score": 0.93,
        "used_override": False
    }

class IdentifyResponse(BaseModel):
    title: str
    artist: str
    label: str
    catalog_number: str
    discogs_url: str
    confidence: float
    used_fallback: bool
    used_override: bool

@ap#p.post("/api/identify", response_model=IdentifyResponse)
asy#nc def identify_record(image: UploadFile = File(...)):
    temp_path = f"temp_{image.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    embedding = embed_image(temp_path)
    result = search_qdrant(embedding)

    os.remove(temp_path)

    return IdentifyResponse(
        title=result["title"],
        artist=result["artist_name"],
        label=result["label"],
        catalog_number=result["catalog_number"],
        discogs_url=result["discogs_url"],
        confidence=result["score"],
        used_fallback=False,
        used_override=result["used_override"]
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
