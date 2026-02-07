from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.components.llm import load_llm
from app.components.vectorstore import load_vector_store

from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException


logger = get_logger(__name__)


def format_docs(docs):
    """
    Format retrieved documents into a single string.
    Joins all document content with double newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_prompt(inputs):
    """
    Create the final prompt string manually.
    This returns a plain string, avoiding PromptValue issues.
    
    Args:
        inputs: dict with 'context' and 'question' keys
    
    Returns:
        str: The formatted prompt ready for the LLM
    """
    context = inputs["context"]
    question = inputs["question"]
    
    prompt = f"""You are a medical assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context, say you don't know.
Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:"""
    
    return prompt


def create_qa_chain():
    """
    Create the complete QA chain for the medical chatbot.
    
    Chain flow:
    1. User question comes in
    2. Retriever fetches relevant documents
    3. Documents are formatted into text
    4. Context + question are combined into a prompt string
    5. LLM processes the prompt and generates answer
    
    Returns:
        Runnable chain or None if creation fails
    """
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

        # Create retriever that fetches top 3 most relevant documents
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # Build the chain without PromptTemplate to avoid type issues
        qa_chain = (
            {
                "context": retriever | format_docs,  # Get docs and format them
                "question": RunnablePassthrough()     # Pass question through
            }
            | RunnableLambda(create_prompt)  # Create prompt string manually
            | llm                             # Generate answer with LLM
        )

        logger.info("Successfully created QA chain")
        return qa_chain

    except Exception as e:
        error_message = CustomException("Failed to make QA chain", e)
        logger.error(str(error_message))
        return None