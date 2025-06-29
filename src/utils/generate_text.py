from src.tokenizer import Tokenizer
from src.utils import softmax
import torch


@torch.inference_mode()
def generate_text(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    context_length: int,
    max_tokens: int,
    temperature: float | None = None,
    top_k: int | None = None,
    eos: str | None = "<|endoftext|>",
    device: torch.device | None = None,
) -> str:

    tokens = tokenizer.encode(prompt)
    eos_id = tokenizer.encode(eos)[0] if eos is not None else None

    for _ in range(max_tokens):
        context = torch.tensor(
            [tokens[-context_length:]], dtype=torch.long, device=device
        )
        logits = model(context)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = logits.topk(top_k, dim=-1)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, -torch.inf, logits)

        if temperature is not None and temperature > 0.0:
            logits /= temperature

        if temperature == 0.0:
            token = logits.argmax(dim=-1)
        else:
            probs = softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1).item()

        if token == eos_id:
            break

        tokens.append(token)

    return tokenizer.decode(tokens)
