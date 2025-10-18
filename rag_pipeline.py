# rag_pipeline.py
import os
from typing import List, Tuple
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch

# LLM imports
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional OpenAI fallback
import os as _os

class RAGPipeline:
    def __init__(self, persist_dir="embeddings", embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 llm_model_name="microsoft/phi-3-mini-4k-instruct", allow_openai_fallback=False):
        self.persist_dir = persist_dir
        self.embed_model_name = embed_model_name
        self.llm_model_name = llm_model_name
        self.allow_openai_fallback = allow_openai_fallback

        os.makedirs(self.persist_dir, exist_ok=True)
        self.embedder = HuggingFaceEmbeddings(model_name=self.embed_model_name)
        self.vstore = None

        # LLM placeholders; lazy load
        self.tokenizer = None
        self.model = None

    # Index management
    def create_index_from_docs(self, docs: List[dict]):
        texts = [d["text"] for d in docs]
        metadatas = [d["metadata"] for d in docs]
        self.vstore = FAISS.from_texts(texts=texts, embedding=self.embedder, metadatas=metadatas)
        self.vstore.save_local(self.persist_dir)
        return self.vstore

    def load_index(self):
        self.vstore = FAISS.load_local(self.persist_dir, embeddings=self.embedder)
        return self.vstore

    def index_exists(self) -> bool:
        return os.path.isdir(self.persist_dir) and any(os.scandir(self.persist_dir))

    # retrieval
    def retrieve(self, query: str, k: int = 5):
        if not self.vstore:
            self.load_index()
        return self.vstore.similarity_search(query, k=k)

    # LLM handling (local)
    def _load_local_llm(self):
        if self.tokenizer and self.model:
            return
        # try to load local model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                device_map='auto' if torch.cuda.is_available() else None,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            # move to CPU if required
            if not torch.cuda.is_available() and hasattr(self.model, "to"):
                self.model.to(torch.device("cpu"))
            return
        except Exception as e:
            if self.allow_openai_fallback and _os.getenv("OPENAI_API_KEY"):
                # skip local error; caller will handle fallback
                raise RuntimeError(f"Local LLM load failed: {e}")
            else:
                raise

    def _build_context(self, docs: List, max_chars=3000) -> str:
        pieces = []
        total = 0
        for d in docs:
            content = getattr(d, "page_content", None) or d.get("text", "")
            meta = getattr(d, "metadata", None) or d.get("metadata", {})
            header = f"[source:{meta.get('source','unknown')} | chunk:{meta.get('chunk',-1)}]"
            piece = f"{header}\n{content}\n"
            pieces.append(piece)
            total += len(piece)
            if total > max_chars:
                break
        return "\n\n".join(pieces)

    def answer_query(self, query: str, top_k: int = 5, temperature: float = 0.1, tone: str = "DevOps expert") -> Tuple[str, List[dict]]:
        if not self.index_exists():
            raise RuntimeError("Index not found. Ingest documents first.")
        docs = self.retrieve(query, k=top_k)
        context = self._build_context(docs, max_chars=3000)
        prompt = f"""
You are a DevOps/SRE expert. Answer the user's question concisely and authoritatively using ONLY the provided context. Cite source chunks [source:NAME | chunk:IDX] when relevant.
Tone: {tone}

Context:
{context}

Question: {query}

Answer:
"""
        # Try local LLM
        try:
            self._load_local_llm()
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature=temperature)
            raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            # If allowed, fallback to OpenAI
            if self.allow_openai_fallback and _os.getenv("OPENAI_API_KEY"):
                from langchain.llms import OpenAI
                # Use LangChain wrapper to make the call simple
                llm = OpenAI(temperature=temperature, max_tokens=256)
                raw = llm(prompt)
            else:
                raise RuntimeError(f"LLM generation failed: {e}")

        # try to isolate answer portion
        if "Answer:" in raw:
            answer = raw.split("Answer:")[-1].strip()
        else:
            answer = raw.strip()

        # build citation metadata
        citations = []
        for d in docs:
            meta = getattr(d, "metadata", None) or d.get("metadata", {})
            snippet = (getattr(d, "page_content", None) or d.get("text", ""))[:300].replace("\n", " ").strip()
            citations.append({"source": meta.get("source", "unknown"), "chunk": meta.get("chunk", -1), "snippet": snippet})

        return answer, citations

# convenience wrapper
def ensure_embeddings_exist(pipeline: RAGPipeline) -> bool:
    return pipeline.index_exists()
