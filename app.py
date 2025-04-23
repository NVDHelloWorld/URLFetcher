import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# === Setup ===
st.set_page_config(page_title="Web Q&A", layout="centered")

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)
documents = []

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)  # Use CPU

# === Functions ===

def scrape_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text() for p in paragraphs]).strip()
    except Exception as e:
        return f"Error scraping {url}: {e}"

def split_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def add_documents(texts):
    global documents
    embeddings = model.encode(texts)
    index.add(np.array(embeddings).astype("float32"))
    documents.extend(texts)

def get_top_k(query, k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k)
    return [documents[i] for i in I[0]]

def chunk_text(text, max_tokens=400):
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def answer_question(query, context_chunks):
    answers = []
    for chunk in context_chunks:
        for sub_chunk in chunk_text(chunk):
            prompt = f"Answer the question based only on this context:\n{sub_chunk}\n\nQuestion: {query}"
            try:
                result = qa_pipeline(prompt, max_new_tokens=200, truncation=True)
                answers.append(result[0]['generated_text'].strip())
            except Exception as e:
                answers.append(f"[Error: {e}]")
    return answers[0] if answers else "No answer found."


# === Streamlit App ===
st.title("🧠 Web Q&A Tool")
st.write("Enter URLs to ingest content, then ask questions based only on that content.")

# URL input
url_input = st.text_input("Enter URLs (comma separated):")
if st.button("Ingest URLs"):
    if url_input:
        urls = [url.strip() for url in url_input.split(",")]
        texts = [scrape_url(url) for url in urls]
        all_chunks = []
        for text in texts:
            all_chunks.extend(split_text(text))
        add_documents(all_chunks)
        st.success("✅ Web content ingested!")
    else:
        st.warning("Please enter at least one URL.")

# Question input
question = st.text_input("Ask a question:")
if question:
    if len(documents) == 0:
        st.error("❌ You must ingest content first.")
    else:
        chunks = get_top_k(question)
        answer = answer_question(question, chunks)
        st.markdown("### 🧾 Answer:")
        st.write(answer)
