from langchain.llms import HuggingFaceHub
from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger= get_logger(__name__)

def load_llm(huggingface_repo_id: str=HUGGINGFACE_REPO_ID, hf_token:str= HF_TOKEN):
    try:
        logger.info("loading llm from HuggingFace")
        llm=HuggingFaceHub(
            repo_id=huggingface_repo_id,
            model_kwargs={
                "temperature":0.5,
                "max_length":256,
                "return_full_text":False

            },
            huggingfacehub_api_token=hf_token

        )
        logger.info("llm loaded successfully")
        return llm
    except Exception as e:
        error_message=CustomException("failed to load a llm",e)
        logger.error(str(error_message))
        return []
    
    