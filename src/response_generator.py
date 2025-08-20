import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ResponseGenerator:
    """
    Loads a language model and generates an answer from query + retrieved context.
    """
    def __init__(self, model_name: str = "gpt2-medium"):
        print(f"Loading generator model '{model_name}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # GPT-2 has no pad token; use EOS and left padding to play nice with generate()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        # Max positional window (1024 for GPT-2/Medium)
        self.max_positions = getattr(self.model.config, "n_positions",
                              getattr(self.model.config, "max_position_embeddings", 1024))
        # Help tokenizers respect the true window
        self.tokenizer.model_max_length = self.max_positions

        print(f"Model loaded on {self.device}. Max positions: {self.max_positions}")

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        """
        Generates a final answer using the retrieved context.
        """
        # 1) Prompt building
        context_passages = [c["text"] for c in retrieved_chunks]
        prompt_tmpl = (
            "Answer the following question based on the context provided below.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        # Reserve some tokens for the question and header/footer text
        probe = self.tokenizer.encode(prompt_tmpl.format(context="", question=query), add_special_tokens=False)
        # Keep a small buffer for safety
        reserved_for_prompt = len(probe) + 16

        max_context_tokens = max(0, self.max_positions - reserved_for_prompt)
        packed_context = ""
        # Greedily pack passages until we run out of budget
        for passage in context_passages:
            trial = packed_context + ("" if not packed_context else "\n") + passage
            if len(self.tokenizer.encode(trial, add_special_tokens=False)) <= max_context_tokens:
                packed_context = trial
            else:
                break

        if not packed_context.strip():
            return "Could not generate an answer because no relevant context was found."

        final_prompt = prompt_tmpl.format(context=packed_context.strip(), question=query)

        # 2) Tokenize input with strict truncation to model window
        inputs = self.tokenizer(
            final_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_positions
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_len = int(inputs["input_ids"].shape[-1])

        # 3) Compute safe generation budget
        # We cannot exceed model positional limit during generation.
        room = max(0, self.max_positions - input_len)
        # Target up to 150 new tokens, but never exceed remaining room - 1 (safety margin)
        target_new = 150
        safe_new = max(1, min(target_new, max(0, room - 1)))

        # If no room at all, aggressively trim input to leave some space for the answer
        if safe_new < 1:
            keep = max(32, self.max_positions - 128)  # keep most recent part of the prompt
            inputs = self.tokenizer(
                final_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=keep
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_len = int(inputs["input_ids"].shape[-1])
            room = max(0, self.max_positions - input_len)
            safe_new = max(1, min(128, max(0, room - 1)))

        # 4) Generate
        output_sequences = self.model.generate(
            **inputs,
            max_new_tokens=safe_new,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            do_sample=True,
            use_cache=True  # default True; being explicit
        )

        # 5) Decode and extract answer
        generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        answer = generated_text.replace(final_prompt, "").strip()
        # Take the first coherent line
        answer = answer.split("\n")[0].strip()

        # Optional: if generation is empty after tight budgets, provide a fallback note
        if not answer:
            answer = "I couldn’t produce a confident answer within the model’s context window."

        return answer