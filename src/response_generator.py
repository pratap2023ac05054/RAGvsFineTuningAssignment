# response_generator.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ResponseGenerator:
    """
    A class to handle the loading of a language model and the generation of answers
    based on a query and provided context.
    """
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initializes the generator by loading the model and tokenizer.

        Args:
            model_name (str): The name of the Hugging Face model to use.
        """
        print(f"Loading generator model '{model_name}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- ADD YOUR HUGGING FACE TOKEN HERE ---
        # Replace "hf_YOUR_TOKEN_HERE" with your actual access token
        # Get a token from: https://huggingface.co/settings/tokens
        hf_token = "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" 

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        if self.device == "cuda":
            # Configure 4-bit quantization to load the large model efficiently on GPU
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto", # Automatically use GPU
                token=hf_token
            )
        else:
            # Load the model without quantization for CPU
            print("CUDA not available. Loading model on CPU. This may be slow and memory-intensive.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token
            ).to(self.device)

        print(f"Model loaded successfully on device: {self.device}")

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        """
        Generates a final answer using the retrieved context.
        """
        context_passages = [chunk['text'] for chunk in retrieved_chunks]
        
        max_length = 4096 
        packed_context = ""
        for passage in context_passages:
            if len(self.tokenizer.encode(packed_context + passage)) < max_length - 512: # Reserve space for prompt/query
                packed_context += passage + "\n\n"
            else:
                break
        
        if not packed_context.strip():
            return "Could not generate an answer because no relevant context was found."

        # Using the prompt format for Mistral Instruct models
        messages = [
            {"role": "user", "content": f"Context:\n{packed_context}\n\nBased on the context provided, please answer the following question:\n{query}"}
        ]
        
        final_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)

        output_sequences = self.model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # Clean the output to remove the prompt part
        # The answer is what comes after the final [/INST] tag
        parts = generated_text.split("[/INST]")
        if len(parts) > 1:
            return parts[-1].strip()
        else:
            return "Could not extract a clear answer from the model's response."
