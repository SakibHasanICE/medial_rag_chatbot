from langchain_huggingface import HuggingFaceEmbeddings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
import os

logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("initialize our Huggingface embedding model")
        
        hf_token = os.environ.get("HF_TOKEN")
        
        # CORRECT: model_name as direct parameter
        if hf_token:
            model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"token": hf_token}
            )
        else:
            model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        logger.info("Huggingface embedding model loaded successfully ...")
        return model
    except Exception as e:
        logger.error(str(e))
        return None