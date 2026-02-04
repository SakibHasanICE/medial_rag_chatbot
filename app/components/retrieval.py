from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.components.llm import load_llm
from app.components.vectorstore import load_vector_store

from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException


logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """
Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""


def create_qa_chain():
    try:
        logger.info("Loading vectorstore")
        db = load_vector_store()
        if db is None:
            raise CustomException("Vectorstore not present")

        logger.info("Loading LLM")
        llm = load_llm(
            huggingface_repo_id=HUGGINGFACE_REPO_ID,
            hf_token=HF_TOKEN
        )
        if llm is None:
            raise CustomException("LLM not loaded")

        retriever = db.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_template(
            CUSTOM_PROMPT_TEMPLATE
        )

        qa_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )

        logger.info("Successfully created QA chain")
        return qa_chain

    except Exception as e:
        error_message = CustomException("Failed to make QA chain", e)
        logger.error(str(error_message))
        return None
