import streamlit as st
import PyPDF2
import requests
import os
import json
from pdf2image import convert_from_bytes
import pytesseract

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

def query_free_llm(prompt, model):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}
    data = {"inputs": prompt}
    response = requests.post(url, headers=headers, json=data)
    return response.json()[0].get('summary_text', "Error in response") if response.status_code == 200 else "Error in response"

def generate_summary(text):
    return query_free_llm(f"Summarize the following text: {text}", "facebook/bart-large-cnn")

def generate_flashcards(text):
    return query_free_llm(f"Generate flashcards from the following text: {text}", "mistralai/Mistral-7B-Instruct")

def generate_mcqs(text):
    return query_free_llm(f"Generate multiple choice questions from the following text: {text}", "bigscience/bloomz-7b1")

st.title("RAG Study Assistant")

uploaded_file = st.file_uploader("Upload your subject PDF", type=["pdf"])

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    '''image_text = extract_text_from_images(uploaded_file)'''
    full_text = text + "\n"
        '''+ image_text'''
    
    st.write("### Extracted Text Preview")
    st.text_area("Text", full_text[:1000], height=200)
    
    if st.button("Generate Summary"):
        summary = generate_summary(full_text)
        st.write("### Summary")
        st.write(summary)
    
    if st.button("Generate Flashcards"):
        flashcards = json.loads(generate_flashcards(full_text))
        st.write("### Flashcards")
        for i, card in enumerate(flashcards):
            st.write(f"**Q{i+1}:** {card['question']}")
            st.write(f"**A:** {card['answer']}")
    
    if st.button("Generate MCQs"):
        mcqs = json.loads(generate_mcqs(full_text))
        st.write("### Multiple Choice Questions")
        for i, mcq in enumerate(mcqs):
            st.write(f"**Q{i+1}:** {mcq['question']}")
            for opt in mcq['options']:
                st.write(f"- {opt}")
            st.write(f"**Answer:** {mcq['answer']}")
            st.write(f"**Explanation:** {mcq['explanation']}")
