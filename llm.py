"""
Velox - Local 3B LLM
=====================
Wraps a quantized 3B causal language model for CPU / low-VRAM inference.
Supports Phi-3 and LLaMA 3.2 prompt formats automatically.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import Config
from utils import cprint


# -- Prompt templates ----------------------------------------------------------

# Token constants (kept as variables to avoid escaping issues)
_SYS    = chr(60) + '|system|' + chr(62)
_END    = chr(60) + '|end|' + chr(62)
_USR    = chr(60) + '|user|' + chr(62)
_ASST   = chr(60) + '|assistant|' + chr(62)
_LLAMA_SYS_OPEN  = chr(60) + chr(60) + 'SYS' + chr(62) + chr(62)
_LLAMA_SYS_CLOSE = chr(60) + chr(60) + '/SYS' + chr(62) + chr(62)


SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer the question using ONLY "
    "the context below. If the context lacks the answer, say so."
)


def _phi3_prompt(context: str, question: str) -> str:
    """Phi-3 chat-template prompt."""
    return (
        f"{_SYS}\n"
        f"{SYSTEM_INSTRUCTION}\n"
        f"{_END}\n"
        f"{_USR}\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"{_END}\n"
        f"{_ASST}\n"
    )


def _llama_prompt(context: str, question: str) -> str:
    """LLaMA 3.2 / Instruct chat-template prompt."""
    return (
        f"[INST] {_LLAMA_SYS_OPEN}\n"
        f"{SYSTEM_INSTRUCTION}\n"
        f"{_LLAMA_SYS_CLOSE}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question} [/INST]\n"
    )


def _select_prompt_fn():
    """Auto-detect prompt format based on model name."""
    model_lower = Config.LLM_MODEL.lower()
    if "phi" in model_lower:
        return _phi3_prompt
    elif "llama" in model_lower:
        return _llama_prompt
    # Default to Phi-3 style
    return _phi3_prompt


# -- LLM wrapper ---------------------------------------------------------------

class LocalLLM:
    """Quantized 3B causal LM for CPU / low-VRAM inference."""

    def __init__(self):
        cprint(f"  [LLM] Loading '{Config.LLM_MODEL}' ...", "cyan")

        quant_cfg = None
        if Config.USE_4BIT_QUANT and torch.cuda.is_available():
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        # CPU fallback: explicit device placement
        if torch.cuda.is_available():
            device_map = "auto"
            dtype       = torch.float16
        else:
            device_map = None
            dtype       = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL)
        self.model     = AutoModelForCausalLM.from_pretrained(
            Config.LLM_MODEL,
            quantization_config=quant_cfg,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Move to CPU explicitly if no device_map
        if device_map is None:
            self.model = self.model.to("cpu")

        self.model.eval()
        self._prompt_fn = _select_prompt_fn()
        cprint("  [LLM] Model ready.", "green")

    def generate(self, context: str, question: str) -> str:
        """Generate an answer grounded in the provided context."""
        # Trim context to stay within token budget
        ctx_words = context.split()
        if len(ctx_words) > Config.MAX_CTX_TOKENS:
            context = " ".join(ctx_words[: Config.MAX_CTX_TOKENS]) + " ..."

        prompt = self._prompt_fn(context, question)

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Strip the prompt echo from the output
        if _ASST in decoded:
            answer = decoded.split(_ASST)[-1].strip()
        elif "[/INST]" in decoded:
            answer = decoded.split("[/INST]")[-1].strip()
        else:
            answer = decoded[len(prompt):].strip()

        return answer
