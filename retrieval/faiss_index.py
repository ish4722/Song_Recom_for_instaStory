import os
import pickle

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from config import FAISS_INDEX_PATH, MULTIMODAL_DATA_PATH


def build_faiss_index(embeddings, path=FAISS_INDEX_PATH):
    if faiss is None:
        raise RuntimeError("faiss-cpu is required to build the vector index.")
    matrix = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, path)
    return index


def load_faiss_index(path=FAISS_INDEX_PATH):
    if faiss is None or not os.path.exists(path):
        return None
    return faiss.read_index(path)


def search(index, query_embedding, top_k):
    query = np.asarray([query_embedding], dtype="float32")
    faiss.normalize_L2(query)
    scores, indices = index.search(query, top_k)
    return indices[0].tolist(), scores[0].tolist()


def load_multimodal_data(path=MULTIMODAL_DATA_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
