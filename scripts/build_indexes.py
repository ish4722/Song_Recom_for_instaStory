"""Build text and CLIP FAISS indexes from the existing song_data.pkl.

Run after installing dependencies:
    python scripts/build_indexes.py
"""

import pickle

import numpy as np
import torch
from PIL import Image

from config import FAISS_INDEX_PATH, MULTIMODAL_DATA_PATH
from retrieval.faiss_index import build_faiss_index
from services.model_manager import get_clip

SOURCE_PATH = "song_data.pkl"
TEXT_INDEX_PATH = "song_text_faiss.index"
CLIP_INDEX_PATH = "song_clip_faiss.index"


def main():
    with open(SOURCE_PATH, "rb") as f:
        songs = pickle.load(f)

    descriptions = [song.get("description", "") for song in songs]
    processor, model = get_clip()
    model.eval()

    all_clip_embeddings = []
    batch_size = 32
    for start in range(0, len(descriptions), batch_size):
        batch = descriptions[start:start + batch_size]
        inputs = processor(text=batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        all_clip_embeddings.append(embeddings.cpu().numpy())
        print(f"Encoded {min(start + batch_size, len(descriptions))}/{len(descriptions)} songs")

    clip_embeddings = np.vstack(all_clip_embeddings).astype("float32")
    text_embeddings = np.asarray([song["embedding"] for song in songs], dtype="float32")

    # Persist CLIP vectors alongside the original song records.
    for song, embedding in zip(songs, clip_embeddings):
        song["clip_embedding"] = embedding

    with open(MULTIMODAL_DATA_PATH, "wb") as f:
        pickle.dump(songs, f)

    build_faiss_index(text_embeddings, TEXT_INDEX_PATH)
    build_faiss_index(clip_embeddings, CLIP_INDEX_PATH)
    print(f"Saved {len(songs)} songs to {MULTIMODAL_DATA_PATH}")
    print(f"Saved text index to {TEXT_INDEX_PATH}")
    print(f"Saved CLIP index to {CLIP_INDEX_PATH}")
    print(f"Legacy index path reserved for compatibility: {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
