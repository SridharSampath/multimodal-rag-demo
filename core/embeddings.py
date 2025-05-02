import numpy as np
import io
import base64
from PIL import Image
from config import COHERE_API_KEY
import cohere

# Constants
MAX_PIXELS = 1568 * 1568  # Cohere image size limit

# Initialize Cohere client
co_client = cohere.ClientV2(api_key=COHERE_API_KEY)

def resize_image(pil_image):
    """Resize image if too large for embedding API"""
    org_width, org_height = pil_image.size
    if org_width * org_height > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        return pil_image.resize((new_width, new_height))
    return pil_image

def base64_from_image(pil_image):
    """Convert PIL Image to base64 for Cohere"""
    pil_image = resize_image(pil_image)
    img_format = pil_image.format if pil_image.format else "PNG"
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format=img_format)
        img_bytes = buffer.getvalue()
    return f"data:image/{img_format.lower()};base64," + base64.b64encode(img_bytes).decode("utf-8")

def get_document_embedding(content, content_type="text"):
    """Embed document (text or image)"""
    try:
        if content_type == "text":
            response = co_client.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                texts=[content],
            )
            return np.array(response.embeddings.float[0])
        else:
            api_input_document = {
                "content": [
                    {"type": "image", "image": base64_from_image(content)},
                ]
            }
            response = co_client.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                inputs=[api_input_document],
            )
            return np.array(response.embeddings.float[0])
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def get_query_embedding(query):
    """Embed search query"""
    try:
        response = co_client.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[query],
        )
        return np.array(response.embeddings.float[0])
    except Exception as e:
        print(f"Query embedding error: {e}")
        return None