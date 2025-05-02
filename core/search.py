import numpy as np
from PIL import Image
from config import GEMINI_API_KEY, GEMINI_MODEL
import google.generativeai as genai

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_client = genai

def search_documents(query, index, docs_info, query_embed_fn, top_k=3):
    query_vector = query_embed_fn(query)
    if query_vector is None or index is None:
        return []

    D, I = index.search(np.array([query_vector.astype("float32")]), top_k)
    results = []

    for score, idx in zip(D[0], I[0]):
        if idx < len(docs_info):
            doc_info = docs_info[idx]
            results.append({
                "doc_id": doc_info["doc_id"],
                "source": doc_info["source"],
                "content_type": doc_info["content_type"],
                "page": doc_info.get("page", 1),
                "similarity": 1 / (1 + score),
                "content": doc_info.get("content"),
                "preview": doc_info.get("preview"),
            })

    return results

def answer_with_gemini(question, content):
    try:
        model = gemini_client.GenerativeModel(GEMINI_MODEL)

        if isinstance(content, Image.Image):
            prompt = [f"""Answer the question based on the following image.
Don't use markdown.
Please provide enough context for your answer.

Question: {question}""", content]
            response = model.generate_content(contents=prompt)
        else:
            prompt = f"""Answer the question based on the following information.
Don't use markdown.
Please provide enough context for your answer.

Information: {content}

Question: {question}"""
            response = model.generate_content(prompt)

        answer = response.text
        print("LLM Answer:", answer)
        return answer.strip() if answer else "Gemini returned no answer."
    except Exception as e:
        print("Gemini error:", str(e))
        return f"Gemini error: {e}"
