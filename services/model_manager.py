from functools import lru_cache

from sentence_transformers import SentenceTransformer
from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor


@lru_cache(maxsize=1)
def get_text_model():
    return SentenceTransformer("all-mpnet-base-v2")


@lru_cache(maxsize=1)
def get_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


@lru_cache(maxsize=1)
def get_clip():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model
