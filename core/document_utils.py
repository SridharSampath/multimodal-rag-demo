import os
import io
import tempfile
from PIL import Image
import pdf2image
import PyPDF2
import pickle
import numpy as np
import faiss

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def pdf_to_images(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name

    images = pdf2image.convert_from_path(tmp_path, dpi=200)
    os.unlink(tmp_path)
    return images

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.getvalue()))
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"Text extraction error: {e}")
        return ""

def save_image_preview(image, filename):
    path = os.path.join(DATA_DIR, filename)
    image.save(path)
    return path

def save_embeddings_and_info(embeddings_data, docs_info):
    vectors = [item["embedding"].astype("float32") for item in embeddings_data]
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.vstack(vectors))
    faiss.write_index(index, os.path.join(DATA_DIR, "faiss.index"))

    with open(os.path.join(DATA_DIR, "docs_info.pkl"), "wb") as f:
        pickle.dump(docs_info, f)

def load_embeddings_and_info():
    index_path = os.path.join(DATA_DIR, "faiss.index")
    docs_path = os.path.join(DATA_DIR, "docs_info.pkl")

    if os.path.exists(index_path) and os.path.exists(docs_path):
        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            docs_info = pickle.load(f)
    else:
        index = None
        docs_info = []

    return index, docs_info
