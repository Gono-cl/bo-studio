import os
import requests
import urllib.parse
from dotenv import load_dotenv

# Detect if running on Render
if os.getenv("RENDER"):
    load_dotenv('/etc/secrets/google_auth_secrets.env')
else:
    load_dotenv('.env')

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
SCOPE = "https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile"

def get_login_url():
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "access_type": "offline",
        "prompt": "consent"
    }
    return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"

def get_token(code):
    data = {
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code"
    }
    r = requests.post(GOOGLE_TOKEN_URL, data=data)
    return r.json()

def get_user_info(token):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(GOOGLE_USERINFO_URL, headers=headers)
    return r.json()