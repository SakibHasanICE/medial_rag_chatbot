import os
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
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
    

def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("no documents found")
        logger.info(f"splitting {len(documents)} documents into chunks")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        text_chunks=text_splitter.split_documents(documents)
        logger.info(f"generated {len(text_chunks)} text chunks")
        return text_chunks
   
    except Exception as e:
        error_message=CustomException("failed to generate chunks",e)
        logger.error(str(error_message))
        return []


    
