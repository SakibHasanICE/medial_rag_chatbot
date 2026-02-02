from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.components.llm import load_llm
from  app.components.vectorstore import load_vector_store

from app.config.config import HUGGINGFACE_REPO_ID,HF_TOKEN

from app.common.logger import get_logger
from app.common.custom_exception import CustomException


logger=get_logger(__name__)

CUSTOM_PROPMT_TEMPLATE="""
Answer the following medical question in 2-3 lines maximum using only the information provided in the context.
Context:
{context}
Question:
{question}

Answer:

"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROPMT_TEMPLATE,input_variables={"context","question"})
def create_qa_chain():
    try:
        logger.info("load vectorstore")
        db=load_vector_store()
        if db is None:
            raise CustomException("vectorstore not present")
        llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN)
        if llm is None:
            raise CustomException("LLM not loaded")
        
        qa_chain=RetrievalQA.from_chain_type(
            llm=llm,
            chain_type= "stuff",
            retriever=db.as_retriever(search_kwargs={"k":3}), #if need more context basically cover more data chain increase the value 3 is standard
            return_source_documents=False,
            chain_type_kwargs={"prompt":set_custom_prompt()}

            
        )
        logger.info("successfully created the QA chain")
        return qa_chain
    except Exception as e:
        error_message=CustomException("failed to make a QA chain",e)
        logger.error(str(error_message))
        return []





