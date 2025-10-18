# 🧠 RAG Chatbot — Local PDF-based Question Answering with Streamlit, FAISS & Phi-3

A **fully offline Retrieval-Augmented Generation (RAG) chatbot** that lets you:

✅ **Upload any PDF / document**  
✅ **Ingest & embed content into FAISS vector store**  
✅ **Ask natural language questions**  
✅ **Get grounded answers with DevOps-style tone + source citations**

---

## 🚀 Tech Stack

| Component       | Technology Used |
|----------------|-----------------|
| UI             | Streamlit       |
| Embeddings     | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store   | FAISS           |
| Language Model | Phi-3 Mini (local) / OpenAI (fallback) |
| File Parsing   | PyPDF, Docx, TXT |

---

## 📦 Installation & Setup

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

# Start Streamlit App
streamlit run app.py


🎯 Usage

Upload one or more PDF / Text files

Click Ingest Files — embeddings will be stored in FAISS

Ask any question related to the uploaded content

Get SRE-style / DevOps tone answers with citations

Example query:

“How does Kubernetes ensure container reliability?”

📁 Project Structure
rag-chatbot/
│── app.py                # Streamlit interface
│── ingest.py             # Embedding & vector storage
│── retriever.py          # FAISS retrieval
│── llm.py                # Phi-3 inference wrapper
│── data/                 # Sample PDFs
│── embeddings/           # Stored FAISS index
│── requirements.txt
│── README.md

📜 License

MIT License — free to use & modify.