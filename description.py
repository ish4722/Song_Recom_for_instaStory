from PIL import Image
import google.generativeai as genai

from config import GEMINI_API_KEY
from services.model_manager import get_blip, get_text_model

genai.configure(api_key=GEMINI_API_KEY)


def embed_text(text):
    return get_text_model().encode(text or "", normalize_embeddings=True)


def process_image(image_file, manual_description=None):
    """Generate a detailed image description using cached BLIP and Gemini models."""
    image = Image.open(image_file).convert("RGB")
    processor, model_blip = get_blip()
    inputs = processor(image, return_tensors="pt")
    out = model_blip.generate(**inputs, max_new_tokens=60)
    initial_caption = processor.decode(out[0], skip_special_tokens=True)

    combined_prompt = f"Description: {initial_caption}"
    if manual_description:
        combined_prompt += f". Additional details from the user: {manual_description}"

    gemini_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=(
            "Analyze visual content for a music recommendation system. Return one rich, unified "
            "description covering subjects, setting, activity, atmosphere, emotions, colors, "
            "time of day, and useful music mood cues."
        ),
    )
    try:
        response = gemini_model.generate_content(combined_prompt)
        return response.text.strip() if response.text else initial_caption
    except Exception:
        return initial_caption
