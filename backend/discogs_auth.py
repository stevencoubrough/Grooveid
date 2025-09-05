from fastapi import APIRouter, HTTPException
from requests_oauthlib import OAuth1Session
import os

router = APIRouter()

DISCOGS_BASE = "https://api.discogs.com"
REQUEST_TOKEN_URL = f"{DISCOGS_BASE}/oauth/request_token"
AUTHORIZE_URL = f"{DISCOGS_BASE}/oauth/authorize"
ACCESS_TOKEN_URL = f"{DISCOGS_BASE}/oauth/access_token"

CONSUMER_KEY = os.environ.get("DISCOGS_CONSUMER_KEY")
CONSUMER_SECRET = os.environ.get("DISCOGS_CONSUMER_SECRET")
CALLBACK_URL = os.environ.get("DISCOGS_CALLBACK_URL")

# In-memory store for oauth token/secret during login flow
request_tokens = {}

@router.get("/discogs/login")
def discogs_login():
    """
    Initiate Discogs OAuth login. Returns a URL for the user to authorize the app.
    """
    oauth = OAuth1Session(client_key=CONSUMER_KEY, client_secret=CONSUMER_SECRET, callback_uri=CALLBACK_URL)
    fetch_response = oauth.fetch_request_token(REQUEST_TOKEN_URL)
    resource_owner_key = fetch_response.get("oauth_token")
    resource_owner_secret = fetch_response.get("oauth_token_secret")
    # store for callback
    request_tokens[resource_owner_key] = resource_owner_secret
    authorization_url = oauth.authorization_url(AUTHORIZE_URL)
    return {"authorize_url": authorization_url, "oauth_token": resource_owner_key}

@router.get("/discogs/callback")
def discogs_callback(oauth_token: str, oauth_verifier: str):
    """
    Handle Discogs OAuth callback and return the user's access token/secret.
    """
    resource_owner_secret = request_tokens.get(oauth_token)
    if resource_owner_secret is None:
        raise HTTPException(status_code=400, detail="Invalid OAuth token")
    oauth = OAuth1Session(client_key=CONSUMER_KEY,
                          client_secret=CONSUMER_SECRET,
                          resource_owner_key=oauth_token,
                          resource_owner_secret=resource_owner_secret,
                          verifier=oauth_verifier)
    tokens = oauth.fetch_access_token(ACCESS_TOKEN_URL)
    # tokens: oauth_token, oauth_token_secret
    return tokens
