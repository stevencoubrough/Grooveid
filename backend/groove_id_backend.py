from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# ensure parent directory is in sys.path so `backend` package is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.identify_vision import router as identify_router
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

# include routers
app.include_router(identify_router, prefix="/api")
app.include_router(discogs_auth_router)
app.include_router(collection_router)

@app.get("/")
def read_root():
    return {"message": "GrooveID backend"}
