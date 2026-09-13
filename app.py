import asyncio
import os
import pickle
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
import google.generativeai as genai

from config import FLASK_SECRET_KEY, TOP_K
from description import embed_text, process_image
from retrieval.faiss_index import load_faiss_index, search
from retrieval.ranker import explain, rerank
from services.feedback import artist_preferences, record_feedback
from services.model_manager import get_clip
from suno_automation import automate_login
from suno_session_manager import generate_song_on_suno

app = Flask(__name__)
app.config["SECRET_KEY"] = FLASK_SECRET_KEY

with open("song_data.pkl", "rb") as f:
    legacy_song_data = pickle.load(f)

MULTIMODAL_DATA_PATH = "song_data_multimodal.pkl"
if os.path.exists(MULTIMODAL_DATA_PATH):
    with open(MULTIMODAL_DATA_PATH, "rb") as f:
        song_data = pickle.load(f)
else:
    song_data = legacy_song_data

text_index = load_faiss_index("song_text_faiss.index")
clip_index = load_faiss_index("song_clip_faiss.index")

artist_language = {"Sachin-Jigar":"Hindi","The Weeknd":"English","Udit Narayan":"Hindi","Atif Aslam":"Hindi","Taylor Swift":"English","Karan Aujla":"Punjabi","Drake":"English","Tanishk Bagchi":"Hindi","Diljit Dosanjh":"Punjabi","Masoom Sharma":"Haryanvi","Bruno Mars":"English","Vishal Mishra":"Hindi","G. V. Prakash":"Tamil","SZA":"English","Sidhu Moose Wala":"Punjabi","Billie Eilish":"English","Rahat Fateh Ali Khan":"Hindi","Lady Gaga":"English","Darshan Raval":"Hindi","Sachet Tandon":"Hindi","Manoj Muntashir":"Hindi","Pawan Singh":"Bhojpuri","Gur Sidhu":"Punjabi","Jimin":"English","Arjan Dhillon":"Punjabi","AP Dhillon":"Punjabi","Javed Ali":"Hindi","Justin Bieber":"English","Lana Del Rey":"English","Thaman S":"Telugu","Cheema Y":"Punjabi","Jaani":"Hindi","Ariana Grande":"English"}

jobs = {}
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=2)


def clip_image_embedding(image_file):
    image = Image.open(image_file).convert("RGB")
    processor, model = get_clip()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding[0].cpu().numpy()


def candidate_results(description, image_embedding, filtered_data, top_k=50):
    allowed = {id(song): song for song in filtered_data}
    if not allowed:
        return []
    indices = {id(song): i for i, song in enumerate(song_data)}
    candidates = {}

    text_query = embed_text(description)
    if text_index is not None:
        ids, scores = search(text_index, text_query, min(top_k, len(song_data)))
        for idx, score in zip(ids, scores):
            if 0 <= idx < len(song_data) and id(song_data[idx]) in allowed:
                candidates[idx] = {"song": song_data[idx], "semantic_score": float(score), "image_score": 0.0}

    if clip_index is not None and image_embedding is not None:
        ids, scores = search(clip_index, image_embedding, min(top_k, len(song_data)))
        for idx, score in zip(ids, scores):
            if 0 <= idx < len(song_data) and id(song_data[idx]) in allowed:
                item = candidates.setdefault(idx, {"song": song_data[idx], "semantic_score": 0.0, "image_score": 0.0})
                item["image_score"] = float(score)

    if not candidates:
        for song in filtered_data:
            embedding = np.asarray(song.get("embedding"), dtype="float32")
            score = float(np.dot(text_query, embedding) / (np.linalg.norm(embedding) + 1e-12))
            candidates[indices.get(id(song), len(candidates))] = {"song": song, "semantic_score": score, "image_score": 0.0}
    return list(candidates.values())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_photo", methods=["POST"])
def upload_photo():
    if "photo" not in request.files or not request.files["photo"].filename:
        return jsonify({"error": "Please upload an image."}), 400
    file = request.files["photo"]
    if not (file.mimetype or "").startswith("image/"):
        return jsonify({"error": "Only image files are supported by the recommendation pipeline."}), 400

    manual_description = request.form.get("manual_description", "").strip()
    selected_languages = request.form.getlist("languages")
    selected_artists = request.form.getlist("artists")
    mood = request.form.get("mood", "").strip()
    user_id = request.form.get("user_id", "anonymous")[:100]

    refined_description = process_image(file, manual_description)
    file.stream.seek(0)
    image_embedding = None
    if clip_index is not None:
        image_embedding = clip_image_embedding(file)

    filtered_data = song_data
    if selected_languages:
        filtered_data = [s for s in filtered_data if artist_language.get(s.get("artist"), "Other") in selected_languages]
    if selected_artists:
        filtered_data = [s for s in filtered_data if s.get("artist") in selected_artists]

    candidates = candidate_results(refined_description, image_embedding, filtered_data)
    ranked = rerank(candidates, refined_description, mood=mood, preferences=artist_preferences(user_id))[:TOP_K]
    recommendations = []
    for item in ranked:
        song = item["song"]
        recommendations.append({"artist": song.get("artist", "Unknown"), "track": song.get("track", "Unknown"), "description": song.get("description", "No description available."), "similarity": round(float(item["score"]), 4), "semantic_score": round(float(item.get("semantic_score", 0)), 4), "image_score": round(float(item.get("image_score", 0)), 4), "mood_score": round(float(item.get("mood_score", 0)), 4), "explanation": explain(song, item)})

    return jsonify({"refined_description": refined_description, "recommendations": recommendations})


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    required = ["user_id", "artist", "track", "feedback"]
    if any(not data.get(key) for key in required):
        return jsonify({"error": "user_id, artist, track and feedback are required."}), 400
    try:
        record_feedback(data["user_id"][:100], data["artist"], data["track"], data["feedback"])
        return jsonify({"message": "Feedback saved."})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


def generate_lyrics_with_gemini(image_description, mood, genre="pop", language="English"):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = f"Write a short family-friendly {mood} {genre} song in {language} based on this image description: {image_description}. Keep it under 250 characters and format as [Verse 1] then [Chorus]."
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else None
    except Exception as exc:
        print(f"Gemini lyrics error: {exc}")
        return None


def song_generation_worker(job_id, description, mood, genre, language):
    try:
        lyrics = generate_lyrics_with_gemini(description, mood, genre, language)
        if not lyrics:
            raise RuntimeError("Failed to generate lyrics")
        session_file = "suno_session.json"
        if not os.path.exists(session_file) or os.path.getsize(session_file) < 100:
            asyncio.run(automate_login())
        audio_url = asyncio.run(generate_song_on_suno(lyrics))
        with jobs_lock:
            jobs[job_id] = {"status": "completed", "lyrics": lyrics, "audio_url": audio_url}
    except Exception as exc:
        with jobs_lock:
            jobs[job_id] = {"status": "failed", "error": str(exc)}


@app.route("/generate_song", methods=["POST"])
def generate_song():
    data = request.get_json(silent=True) or {}
    description = data.get("description", "").strip()
    if not description:
        return jsonify({"error": "Image description is required"}), 400
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "queued"}
    executor.submit(song_generation_worker, job_id, description, data.get("mood", "happy"), data.get("genre", "pop"), data.get("language", "English"))
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route("/song_status/<job_id>", methods=["GET"])
def song_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/get_song_moods")
def get_song_moods():
    return jsonify({"moods": ["happy", "sad", "energetic", "calm", "romantic", "nostalgic", "uplifting", "melancholic", "dramatic", "peaceful"]})


@app.route("/get_song_genres")
def get_song_genres():
    return jsonify({"genres": ["pop", "rock", "hip-hop", "country", "jazz", "blues", "folk", "electronic", "classical", "indie", "r&b", "reggae"]})


if __name__ == "__main__":
    os.makedirs("static/audio", exist_ok=True)
    app.run(debug=True)
