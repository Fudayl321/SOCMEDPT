from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, constr
import os
import logging
from contextlib import asynccontextmanager
import httpx
import base64
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
from pathlib import Path

# Get the current directory
current_dir = Path(__file__).parent.parent
env_path = current_dir / '.env'

# Load environment variables from .env file
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration class
class Settings:
    SPOTIFY_CLIENT_ID: str = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET: str = os.getenv("SPOTIFY_CLIENT_SECRET")
    REDIRECT_URI: str = "http://127.0.0.1:8000"
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    TOKEN_CACHE_DURATION: int = 3500  # seconds (just under 1 hour)
    REQUESTS_PER_MINUTE: int = 60

settings = Settings()

# Ensure required environment variables are set
if not all([settings.SPOTIFY_CLIENT_ID, settings.SPOTIFY_CLIENT_SECRET, settings.SECRET_KEY]):
    raise ValueError("Required environment variables are not set")

# Initialize the FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Mood-Based Music Recommender",
    description="An API that recommends music based on mood",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mood to music mapping
MOOD_GENRES = {
    "happy": ["happy", "pop", "dance"],
    "sad": ["sad", "acoustic", "ballad"],
    "energetic": ["dance", "electronic", "workout"],
    "relaxed": ["chill", "ambient", "relaxation"],
    "angry": ["rock", "metal", "intense"],
    "romantic": ["romance", "love songs", "r-n-b"]
}

# Rate limiter class
class RateLimiter:
    def __init__(self, requests_per_minute: int = settings.REQUESTS_PER_MINUTE):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    async def is_allowed(self, client_ip: str) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
            
        self.requests[client_ip].append(now)
        return True

rate_limiter = RateLimiter()

# Token cache
class SpotifyTokenCache:
    def __init__(self):
        self.token: Optional[str] = None
        self.expires_at: Optional[datetime] = None

    def set_token(self, token: str):
        self.token = token
        self.expires_at = datetime.now() + timedelta(seconds=settings.TOKEN_CACHE_DURATION)

    def get_token(self) -> Optional[str]:
        if self.expires_at and datetime.now() < self.expires_at:
            return self.token
        return None

token_cache = SpotifyTokenCache()

# Pydantic models
class MoodDescription(BaseModel):
    description: constr(min_length=1, max_length=500)

class TrackResponse(BaseModel):
    song_name: str
    artist: str
    song_url: str
    preview_url: Optional[str]
    album_image: Optional[str]

class RecommendationResponse(BaseModel):
    message: str
    mood_description: str
    detected_genres: List[str]
    recommendations: List[TrackResponse]

# Utility functions
async def get_spotify_token() -> str:
    """Get Spotify API access token with caching."""
    cached_token = token_cache.get_token()
    if cached_token:
        return cached_token

    try:
        auth_string = f"{settings.SPOTIFY_CLIENT_ID}:{settings.SPOTIFY_CLIENT_SECRET}"
        auth_bytes = auth_string.encode('utf-8')
        auth_base64 = str(base64.b64encode(auth_bytes), 'utf-8')
        
        url = "https://accounts.spotify.com/api/token"
        headers = {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, data=data, timeout=10.0)
            response.raise_for_status()
            
        token = response.json()["access_token"]
        token_cache.set_token(token)
        return token
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Spotify API timeout")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail="Failed to communicate with Spotify API")
    except Exception as e:
        logger.error(f"Error getting Spotify token: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def search_tracks(mood_genres: List[str], access_token: str) -> List[Dict]:
    """Search for tracks based on mood-related genres."""
    tracks = []
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}"}
        
        for genre in mood_genres:  # Iterate through all detected genres
            try:
                params = {
                    "q": f"genre:{genre}",
                    "type": "track",
                    "limit": 10,  # Increase the limit to get more tracks
                    "market": "US"
                }
                
                response = await client.get(
                    "https://api.spotify.com/v1/search",
                    headers=headers,
                    params=params,
                    timeout=10.0
                )
                response.raise_for_status()
                
                results = response.json()
                if results["tracks"]["items"]:
                    tracks.extend(results["tracks"]["items"])
                    logger.info(f"Found {len(results['tracks']['items'])} tracks for genre '{genre}'")
                    
            except httpx.TimeoutException:
                logger.warning(f"Timeout while searching tracks for genre {genre}")
                continue
            except Exception as e:
                logger.error(f"Error searching tracks for genre {genre}: {e}")
                continue
    
    logger.info(f"Total tracks found: {len(tracks)}")
    return tracks[:10]  # Return up to 10 tracks total

async def analyze_mood(mood_description: str) -> List[str]:
    """Analyze mood description and return relevant music genres."""
    mood_description = mood_description.lower()
    genres = []
    
    # Check for direct mood matches
    for mood, related_genres in MOOD_GENRES.items():
        if mood in mood_description:
            genres.extend(related_genres)

    # If no direct matches, use more comprehensive sentiment analysis
    if not genres:
        keywords = {
            "happy": ["good", "great", "joy", "happy", "excited", "cheerful", "fun", "party"],
            "sad": ["sad", "down", "blue", "unhappy", "depressed", "lonely"],
            "energetic": ["energetic", "active", "hyper", "lively", "motivated"],
            "relaxed": ["relaxed", "calm", "peaceful", "chill", "serene"],
            "angry": ["angry", "frustrated", "mad", "furious", "irritated"],
            "romantic": ["romantic", "love", "affection", "passionate", "intimate"]
        }

        for mood, words in keywords.items():
            if any(word in mood_description for word in words):
                genres.extend(MOOD_GENRES[mood])
                break  # Exit once a mood is matched

    logger.info(f"Analyzed mood description: '{mood_description}' - Detected genres: {genres}")
    return list(set(genres))
# Middleware for rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    if not await rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    return await call_next(request)

# Routes
@app.post("/recommend-music", response_model=RecommendationResponse)
async def recommend_music(
    moodDescription: str = Form(...),
):
    try:
        mood_desc = MoodDescription(description=moodDescription)
        mood_genres = await analyze_mood(mood_desc.description)
        access_token = await get_spotify_token()
        tracks = await search_tracks(mood_genres, access_token)
        
        if not tracks:
            raise HTTPException(
                status_code=404,
                detail="No matching songs found"
            )
        
        recommendations = [
            TrackResponse(
                song_name=track["name"],
                artist=track["artists"][0]["name"],
                song_url=track["external_urls"]["spotify"],
                preview_url=track.get("preview_url"),
                album_image=track["album"]["images"][0]["url"] if track["album"]["images"] else None
            )
            for track in tracks
        ]
        
        return RecommendationResponse(
            message="Song recommendations based on your mood:",
            mood_description=mood_desc.description,
            detected_genres=mood_genres,
            recommendations=recommendations
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error processing music recommendation"
        )

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the dashboard page."""
    return templates.TemplateResponse(
        "dashboard.html",  # Ensure this file exists in your templates directory
        {
            "request": request,
            "page_title": "Music Mood Dashboard",
            "mood_genres": MOOD_GENRES  # Pass mood genres to template
        }
    )

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, message: str = Form(...)):
    """Handle chat interactions."""
    mood_genres = await analyze_mood(message)
    response_message = "I think you're feeling " + ", ".join(mood_genres) + ". Is that correct?"
    
    # You can also add more logic to provide recommendations based on the mood
    if mood_genres:
        recommendations = await get_recommendations_based_on_mood(mood_genres)
        response_message += f" Here are some songs you might like: {', '.join(recommendations)}"
    
    return templates.TemplateResponse(
        "chat.html",  # Ensure this file exists in your templates directory
        {
            "request": request,
            "page_title": "Music Mood Chat",
            "user_message": message,
            "response": response_message
        }
    )

async def get_recommendations_based_on_mood(mood_genres: List[str]) -> List[str]:
    """Get song recommendations based on detected mood genres."""
    access_token = await get_spotify_token()
    tracks = await search_tracks(mood_genres, access_token)
    return [track["name"] for track in tracks[:5]]  # Return the names of the first 5 tracks

# Custom OpenAPI documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Mood-Based Music Recommender",
        version="1.0.0",
        description="An API that recommends music based on mood",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)