---
title: Semantic Rag Demo
emoji: 🦀
colorFrom: yellow
colorTo: purple
sdk: streamlit
sdk_version: 1.44.0
app_file: app.py
pinned: false
license: mit
---

# 🧠 Semantic RAG Demo

A lightweight Retrieval-Augmented Generation demo integrating:
- Ontology-based semantic text
- FAISS vector search
- LangChain RAG
- Streamlit interactive UI

## 💻 Run Locally

bash
pip install -r requirements.txt
streamlit run app.py


## 🐳 Docker

bash
docker build -t semantic-rag-demo .
docker run -p 8501:8501 semantic-rag-demo


## 📂 Input Data

Place your ontology or knowledge text in data/enterprise_ontology.txt.

## 🔐 Secrets

Set your OPENAI_API_KEY in Streamlit secrets or env:
bash
export OPENAI_API_KEY=your_key_here