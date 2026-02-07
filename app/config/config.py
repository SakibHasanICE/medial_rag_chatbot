import os
from pathlib import Path
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
HUGGINGFACE_REPO_ID="microsoft/Phi-3-mini-4k-instruct"
DB_FAISS_PATH=os.path.join(BASE_DIR, "vectorstore", "db_faiss")
DATA_PATH="data/"
CHUNK_SIZE=500
CHUNK_OVERLAP=50