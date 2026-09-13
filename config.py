import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")
TOP_K = int(os.getenv("TOP_K", "5"))
TEXT_WEIGHT = float(os.getenv("TEXT_WEIGHT", "0.60"))
IMAGE_WEIGHT = float(os.getenv("IMAGE_WEIGHT", "0.40"))
MOOD_WEIGHT = float(os.getenv("MOOD_WEIGHT", "0.15"))
PREFERENCE_WEIGHT = float(os.getenv("PREFERENCE_WEIGHT", "0.10"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "song_data_faiss.index")
MULTIMODAL_DATA_PATH = os.getenv("MULTIMODAL_DATA_PATH", "song_data_multimodal.pkl")
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "data/feedback.json")
