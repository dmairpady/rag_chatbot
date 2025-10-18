# app.py
import streamlit as st
from pathlib import Path
import os
from ingest import load_document, split_into_chunks
from rag_pipeline import RAGPipeline, ensure_embeddings_exist

st.set_page_config(page_title="rag-chatbot (Streamlit)", layout="wide")

st.title("rag-chatbot — Local RAG (DevOps expert tone)")
st.markdown(
    "Upload PDFs / DOCX / TXT, build local index, then ask questions. "
    "Answers come in a DevOps/SRE tone and include source chunk citations."
)

# Sidebar config
with st.sidebar:
    st.header("Index & Model Settings")
    index_dir = st.text_input("Index directory", value="embeddings")
    embed_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
    llm_model = st.text_input("LLM model (local)", value="microsoft/phi-3-mini-4k-instruct")
    use_openai = st.checkbox("Allow OpenAI fallback (if local LLM fails)", value=False)
    openai_note = "Set OPENAI_API_KEY in environment if using fallback."
    st.caption(openai_note)
    chunk_size = st.number_input("Chunk size (chars)", value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", value=200, step=50)
    top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=12, value=5)
    temp = st.slider("Temperature (generation)", 0.0, 1.0, 0.1)

# Initialize pipeline (deferred heavy loads)
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(
        persist_dir=index_dir,
        embed_model_name=embed_model,
        llm_model_name=llm_model,
        allow_openai_fallback=use_openai,
    )

pipeline: RAGPipeline = st.session_state.pipeline

# Left column: ingestion
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Upload documents")
    uploaded_files = st.file_uploader("Upload PDF / DOCX / TXT (multiple)", accept_multiple_files=True)
    if st.button("Ingest uploaded files"):
        if not uploaded_files:
            st.warning("Upload at least one file.")
        else:
            tmp_dir = Path("data")
            tmp_dir.mkdir(exist_ok=True)
            saved_paths = []
            for f in uploaded_files:
                dest = tmp_dir / f.name
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
                saved_paths.append(str(dest))
            st.info(f"Saved {len(saved_paths)} files to {tmp_dir}/")
            # Ingest: load, chunk, create index
            all_docs = []
            for p in saved_paths:
                text = load_document(p)
                chunks = split_into_chunks(text, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
                for i, c in enumerate(chunks):
                    all_docs.append({"text": c, "metadata": {"source": Path(p).name, "chunk": i}})
            with st.spinner("Creating embeddings & FAISS index (this may take some time)..."):
                pipeline.create_index_from_docs(all_docs)
            st.success(f"Ingested {len(all_docs)} chunks. Index saved to '{pipeline.persist_dir}'")

    st.markdown("---")
    st.subheader("Index status")
    index_exists = pipeline.index_exists()
    st.write("Index present:", index_exists)
    if index_exists and st.button("Reload Index"):
        with st.spinner("Loading index..."):
            pipeline.load_index()
        st.success("Index loaded.")

# Right column: chat
with col2:
    st.subheader("Ask the knowledge base")
    query = st.text_input("Your question", value="", key="query_input")
    ask = st.button("Ask")
    if ask:
        if not pipeline.index_exists():
            st.warning("Index not found. Ingest documents first.")
        elif not query or query.strip() == "":
            st.warning("Please type a question.")
        else:
            with st.spinner("Retrieving context & generating answer..."):
                answer, citations = pipeline.answer_query(
                    query, top_k=int(top_k), temperature=float(temp), tone="DevOps expert"
                )
            # Slack-style answer block
            st.markdown("### ✅ Answer")
            st.info(answer)
            st.markdown("### 📎 Sources (excerpts)")
            for c in citations:
                st.markdown(f"- **{c['source']}** | chunk:{c['chunk']}  \n  > {c['snippet']}...")

st.markdown("---")
st.caption("Tip: use concise questions like 'How does Kubernetes handle pod replication?'")
