import os
from lanchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger= get_logger(__name__)

def load_pdfs_from_directory():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"The specified data path does not exist: {DATA_PATH}")
        logger.info(f"Loading PDFs from directory: {DATA_PATH}")

        loader=DirectoryLoader(DATA_PATH,glob="*.pdf",loader_cls=PyPDFLoader)
        documents=loader.load()
        if not documents:
            logger.warning("no pdf were found")
        else:
            logger.info(f"successfully fetched{len(documents)} documents")
        return documents

    except Exception as e:
        error_message=CustomException("failed to load pdf",e)
        logger.error(str(error_message))
        return []

