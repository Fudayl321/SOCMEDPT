from fastapi import FastAPI, Form, Request, Depends, UploadFile, File, HTTPException, status, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
import torch
import clip
from PIL import Image
import io
import os
from uuid import uuid4
import httpx

# Initialize the FastAPI app
app = FastAPI()

# Secret key for session management
app.add_middleware(SessionMiddleware, secret_key="super-secret-key")

# Setting up Jinja2 for template rendering
templates = Jinja2Templates(directory="templates")

# Mock user data for demonstration purposes
users_db = {"user@example.com": {"password": "password123"}}

# Static files mounting
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure the upload directory exists
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Dependency to check if the user is logged in
def get_current_user(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            detail="Not authenticated",
            headers={"Location": "/"}
        )
    return user

# CLIP model loading and mood analysis functions
model, preprocess = clip.load("ViT-B/32")
model.eval()

moods = ["a happy scene", "a calm scene", "an energetic scene", "a sad scene", "a peaceful scene"]

def analyze_image_mood(image: Image.Image) -> str:
    image_input = preprocess(image).unsqueeze(0)
    text_inputs = torch.cat([clip.tokenize(mood) for mood in moods])

    if torch.cuda.is_available():
        image_input = image_input.to("cuda")
        text_inputs = text_inputs.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        best_idx = similarity.argmax().item()

    return moods[best_idx].split(" ")[1]  # Extracts only the mood word (e.g., "happy")

def generate_music_from_image(image: Image.Image) -> str:
    mood = analyze_image_mood(image)
    song_recommendations = {
        "happy": "Happy Tune",
        "calm": "Soothing Melody",
        "energetic": "Upbeat Anthem",
        "sad": "Melancholy Symphony",
        "peaceful": "Peaceful Piano",
    }
    return song_recommendations.get(mood, "Default Song")

# Spotify API Constants
SPOTIFY_CLIENT_ID = os.getenv("62103406e2c7476bb413f8de8b127f70")
SPOTIFY_CLIENT_SECRET = os.getenv("853cbf85348f4b23b937e5afeb57200b")
SPOTIFY_REDIRECT_URI = "http://localhost:8000/callback"
SCOPE = "user-library-read"  # Adjust the scope as needed

# Routes

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    user = users_db.get(email)
    if user and user["password"] == password:
        request.session["user"] = email
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})

@app.get("/recommend-music", response_class=HTMLResponse)
async def recommend_music_page(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("music.html", {"request": request})

@app.post("/recommend-music")
async def recommend_music(musicImage: UploadFile = File(...), user: str = Depends(get_current_user)):
    # Read the image and save it to the uploads folder
    image_data = await musicImage.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Generate a unique filename for the uploaded image
    image_filename = f"{uuid4().hex}.png"
    image_path = os.path.join(UPLOAD_DIR, image_filename)
    image.save(image_path)

    # Analyze mood and search for a song
    mood = analyze_image_mood(image)
    access_token = user.get("access_token")  # Ensure you have access_token stored in the session
    song_data = await search_track(access_token, mood)

    # Extract the song information from the response
    if song_data.get("tracks") and song_data["tracks"]["items"]:
        song_name = song_data["tracks"]["items"][0]["name"]
        song_url = song_data["tracks"]["items"][0]["external_urls"]["spotify"]
    else:
        song_name = "No song found"
        song_url = ""

    # Create the image URL for frontend access
    image_url = f"/static/uploads/{image_filename}"

    # Return both the image URL and the song name as JSON response
    return JSONResponse(content={"song": song_name, "song_url": song_url, "image_url": image_url})

@app.get("/login-spotify")
async def login_spotify():
    # Redirect to Spotify authorization page
    auth_url = (
        "https://accounts.spotify.com/authorize"
        f"?response_type=code&client_id={SPOTIFY_CLIENT_ID}"
        f"&redirect_uri={SPOTIFY_REDIRECT_URI}"
        f"&scope={SCOPE}"
    )
    return RedirectResponse(auth_url)


@app.get("/callback")
async def callback(request: Request, code: str = Query(...)):
    # Exchange the authorization code for an access token
    token_url = "https://accounts.spotify.com/api/token"
    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": SPOTIFY_REDIRECT_URI,
            "client_id": SPOTIFY_CLIENT_ID,
            "client_secret": SPOTIFY_CLIENT_SECRET
        })
    
    token_info = response.json()
    access_token = token_info.get("access_token")

    # Store the access token in the user session
    request.session["access_token"] = access_token

    return RedirectResponse(url="/dashboard")  # Redirect back to the dashboard or wherever you want


@app.get("/crop-image", response_class=HTMLResponse)
async def crop_image(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("crop.html", {"request": request})

@app.get("/generate-caption", response_class=HTMLResponse)
async def generate_caption(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("caption.html", {"request": request})

@app.get("/real-time-preview", response_class=HTMLResponse)
async def real_time_preview(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("preview.html", {"request": request})

async def search_track(access_token: str, mood: str) -> dict:
    search_url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"q": mood, "type": "track", "limit": 1}

    async with httpx.AsyncClient() as client:
        response = await client.get(search_url, headers=headers, params=params)
        return response.json()
