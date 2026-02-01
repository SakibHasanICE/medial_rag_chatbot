import os
from app.components.pdf_loader import create_text_chunks,load_pdfs_from_directory
from app.components.vectorstore import save_vectorstore
from app.config.config import DB_FAISS_PATH

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger= get_logger(__name__)

def process_and_store_pdfs():
    try:
        logger.info("making the vectorstore")
      

        documents=load_pdfs_from_directory()
        text_chunks=create_text_chunks(documents)

        save_vectorstore(text_chunks)

        logger.info("vectprstore created successfully")
    
    except Exception as e:
        error_message=CustomException("failed to load data and create vectorstore",e)
        logger.error(str(error_message))
        return []

#when this data_loader file run in terminal whatever function present in  if __name__=="__main__": this that will be execute
if __name__=="__main__":
    process_and_store_pdfs()
    