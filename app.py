# semantic_rag_demo/app.py

import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# --------- SETUP ---------
openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    st.error("OPENAI_API_KEY environment variable is not set")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_key

st.set_page_config(page_title="Semantic RAG Demo", layout="wide")
st.title("📚 Semantic RAG Demo")

# --------- LOAD & INDEX DOCUMENT ---------
@st.cache_resource
def load_vector_store():
    loader = TextLoader("data/enterprise_ontology.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = load_vector_store()

# --------- RAG PIPELINE ---------
llm = ChatOpenAI(temperature=0)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_type="similarity", k=4),
    return_source_documents=True
)

# --------- UI INPUT ---------
st.sidebar.header("🔧 RAG Parameters")
k_val = st.sidebar.slider("Top-K Chunks", 1, 10, 4)
temp = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.0)

query = st.text_input("Ask a question about the enterprise data:", "What does the platform know about product ownership?")

if query:
    with st.spinner("Thinking..."):
        rag_chain.retriever.search_kwargs["k"] = k_val
        rag_chain.llm.temperature = temp
        result = rag_chain(query)
        st.markdown("### 🤖 Answer")
        st.write(result["result"])

        st.markdown("---")
        st.markdown("### 📄 Retrieved Context")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Source {i+1}:**")
            st.code(doc.page_content, language="text")