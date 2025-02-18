import streamlit as st
import PyPDF2
import requests
import os
import json
from pdf2image import convert_from_bytes
import pytesseract
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_images(pdf_file):
    images = convert_from_bytes(pdf_file.read())
    image_text = ""
    for img in images:
        image_text += pytesseract.image_to_string(img) + "\n"
    return image_text

def create_vector_store(text_chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, text_chunks, model

def retrieve_relevant_text(query, index, text_chunks, model, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_text = "\n".join([text_chunks[i] for i in indices[0]])
    return retrieved_text

def query_free_llm(prompt, model):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
    data = {"inputs": prompt}
    response = requests.post(url, headers=headers, json=data)
    return response.json()[0].get('summary_text', "Error in response") if response.status_code == 200 else "Error in response"

def generate_summary(text, index, text_chunks, model):
    relevant_text = retrieve_relevant_text("Summarize this document", index, text_chunks, model)
    return query_free_llm(f"Summarize the following text: {relevant_text}", "facebook/bart-large-cnn")

def generate_flashcards(text, index, text_chunks, model):
    relevant_text = retrieve_relevant_text("Generate flashcards", index, text_chunks, model)
    return query_free_llm(f"Generate flashcards from the following text: {relevant_text}", "mistralai/Mistral-7B-Instruct")

def generate_mcqs(text, index, text_chunks, model):
    relevant_text = retrieve_relevant_text("Generate MCQs", index, text_chunks, model)
    return query_free_llm(f"Generate multiple choice questions from the following text: {relevant_text}", "bigscience/bloomz-7b1")

st.title("RAG Study Assistant")

uploaded_file = st.file_uploader("Upload your subject PDF", type=["pdf"])

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    image_text = extract_text_from_images(uploaded_file)
    full_text = text + "\n" + image_text
    text_chunks = full_text.split(". ")
    index, text_chunks, model = create_vector_store(text_chunks)
    
    st.write("### Extracted Text Preview")
    st.text_area("Text", full_text[:1000], height=200)
    
    if st.button("Generate Summary"):
        summary = generate_summary(full_text, index, text_chunks, model)
        st.write("### Summary")
        st.write(summary)
    
    if st.button("Generate Flashcards"):
        flashcards = json.loads(generate_flashcards(full_text, index, text_chunks, model))
        st.write("### Flashcards")
        for i, card in enumerate(flashcards):
            st.write(f"**Q{i+1}:** {card['question']}")
            st.write(f"**A:** {card['answer']}")
    
    if st.button("Generate MCQs"):
        mcqs = json.loads(generate_mcqs(full_text, index, text_chunks, model))
        st.write("### Multiple Choice Questions")
        for i, mcq in enumerate(mcqs):
            st.write(f"**Q{i+1}:** {mcq['question']}")
            for opt in mcq['options']:
                st.write(f"- {opt}")
            st.write(f"**Answer:** {mcq['answer']}")
            st.write(f"**Explanation:** {mcq['explanation']}")
