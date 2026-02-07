from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH
import os

logger = get_logger(__name__)

def load_vector_store():
    """Load existing FAISS vectorstore"""
    try:
        logger.info(f"Checking for vectorstore at: {DB_FAISS_PATH}")
        
        # Check if path exists
        if not os.path.exists(DB_FAISS_PATH):
            logger.warning(f"Vectorstore path does not exist: {DB_FAISS_PATH}")
            return None
        
        # Check if required files exist
        index_file = os.path.join(DB_FAISS_PATH, "index.faiss")
        pkl_file = os.path.join(DB_FAISS_PATH, "index.pkl")
        
        if not os.path.exists(index_file):
            logger.error(f"index.faiss not found in {DB_FAISS_PATH}")
            return None
            
        if not os.path.exists(pkl_file):
            logger.error(f"index.pkl not found in {DB_FAISS_PATH}")
            return None
        
        logger.info("Vectorstore files found, loading embedding model...")
        
        # Get embedding model
        embedding_model = get_embedding_model()
        if embedding_model is None:
            logger.error("Failed to load embedding model")
            return None
        
        logger.info("Loading vectorstore...")
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        
        logger.info("Vectorstore loaded successfully!")
        return vectorstore
        
    except Exception as e:
        error_message = CustomException("failed to load vectorstore", e)
        logger.error(f"Error details: {str(error_message)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception message: {str(e)}")
        return None

def save_vectorstore(text_chunks):
    """Create and save new FAISS vectorstore"""
    try:
        if not text_chunks:
            raise CustomException("no chunks were found")
        
        logger.info(f"Creating vectorstore with {len(text_chunks)} chunks")
        
        # Get embedding model
        embedding_model = get_embedding_model()
        if embedding_model is None:
            raise CustomException("Failed to load embedding model")
        
        # Create vectorstore
        logger.info("Generating embeddings and creating FAISS index...")
        db = FAISS.from_documents(text_chunks, embedding_model)
        
        # Save vectorstore
        logger.info(f"Saving vectorstore to: {DB_FAISS_PATH}")
        
        # Create directory if it doesn't exist
        os.makedirs(DB_FAISS_PATH, exist_ok=True)
        
        db.save_local(DB_FAISS_PATH)
        logger.info("Vectorstore saved successfully!")
        
        return db
        
    except Exception as e:
        error_message = CustomException("failed to create new vectorstore", e)
        logger.error(str(error_message))
        return None