from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.runnables import RunnableLambda
import torch


def load_llm(huggingface_repo_id: str, hf_token: str):
    """
    Load a HuggingFace language model for text generation.
    
    Args:
        huggingface_repo_id: Model ID from HuggingFace (e.g., "microsoft/Phi-3-mini-4k-instruct")
        hf_token: HuggingFace authentication token
    
    Returns:
        RunnableLambda: A LangChain-compatible runnable that processes prompts
    """
    tokenizer = AutoTokenizer.from_pretrained(
        huggingface_repo_id,
        token=hf_token
    )

    model = AutoModelForCausalLM.from_pretrained(
        huggingface_repo_id,
        dtype=torch.float16,  # Use float16 for memory efficiency
        device_map="auto",    # Automatically distribute across available devices
        token=hf_token
    )

    def microsoft_Phi_3(prompt: str):
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: The input prompt string
        
        Returns:
            str: The generated response
        """
        # Format the prompt as a chat conversation
        messages = [
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": prompt}
        ]

        # Apply the model's chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize the input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        # Generate response
        outputs = model.generate(
            **inputs,
            temperature=0.3,      # Lower = more focused, higher = more creative
            do_sample=True,       # Enable sampling for varied responses
            max_new_tokens=256,   # Maximum tokens to generate
            pad_token_id=tokenizer.eos_token_id  # Use EOS token for padding
            # NOTE: return_full_text is NOT a valid parameter - removed
        )

        # Extract only the newly generated tokens (exclude the input prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        
        # Decode tokens to text
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()  # Remove leading/trailing whitespace

    return RunnableLambda(microsoft_Phi_3)