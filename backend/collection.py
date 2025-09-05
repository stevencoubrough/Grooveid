from fastapi import APIRouter, HTTPException
from requests_oauthlib import OAuth1Session
import os

router = APIRouter()

CONSUMER_KEY = os.environ.get("DISCOGS_CONSUMER_KEY")
CONSUMER_SECRET = os.environ.get("DISCOGS_CONSUMER_SECRET")

@router.post("/collection/add")
def add_to_collection(username: str, folder_id: int, release_id: int, user_oauth_token: str, user_oauth_secret: str):
    """
    Add a release to a user's Discogs collection folder using their OAuth credentials.
    """
    try:
        oauth = OAuth1Session(client_key=CONSUMER_KEY,
                              client_secret=CONSUMER_SECRET,
                              resource_owner_key=user_oauth_token,
                              resource_owner_secret=user_oauth_secret)
        url = f"https://api.discogs.com/users/{username}/collection/folders/{folder_id}/releases/{release_id}"
        response = oauth.post(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
