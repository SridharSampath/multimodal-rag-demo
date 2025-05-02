# 🔍 Multimodal RAG with Cohere + Gemini

A powerful demo project that showcases **Multimodal Retrieval-Augmented Generation (RAG)** using **Cohere’s multimodal embeddings** and **Gemini 2.5 Flash** — enabling answers from both **text and images** in PDFs.

📄 This project compares:
- ✅ **Text-only RAG** (traditional)
- ✅ **Multimodal RAG** (text + visual embeddings)

🎥 [Watch the 9-min Demo on YouTube](https://www.youtube.com/watch?v=qI3lYZ6-79k)

---

### ✨ Features

- Embed both **text and images** from PDFs using **Cohere Embed-v4.0**
- Store embeddings in **FAISS** for fast vector search
- Retrieve the best match (text or image) from FAISS
- Use **Gemini 2.5 Flash** to answer questions contextually
- Compare **Multimodal vs Text-Only RAG** side-by-side

---

### 📂 Project Structure
multimodal-rag-demo/
├── app.py # Streamlit app
├── config.py # API keys and model configs
├── core/
│ ├── embeddings.py # Cohere embedding logic
│ ├── document_utils.py # PDF parsing, FAISS indexing
│ └── search.py # Querying and Gemini response
├── data/ # Stores FAISS index and previews
└── requirements.txt


---

### 📋 Prerequisites

- Python 3.8+
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) (for image conversion via `pdf2image`)
- Cohere API key (https://dashboard.cohere.com/)
- Gemini API access via [Google AI Studio](https://makersuite.google.com/)

---

### 🛠️ Setup

```bash
git clone https://github.com/SridharSampath/multimodal-rag-demo.git
cd multimodal-rag-demo
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
Configure API Keys

COHERE_API_KEY = "your-cohere-key"
GEMINI_API_KEY = "your-gemini-key"

🚀 Run the App

streamlit run app.py --server.port 

📖 Blog Walkthrough
Read the full technical blog with diagrams
🔗 Multimodal RAG with Cohere + Gemini Explained

👨‍💻 Author
I share my learnings around AI,ML,AWS:

🔗 Hashnode Blog – sridhartech.hashnode.dev 
🔗 LinkedIn – [Sridhar Sampath
](https://www.linkedin.com/in/sridharsampath89/)
