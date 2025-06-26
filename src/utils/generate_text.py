from src.tokenizer import Tokenizer
from src.model import TransformerLM
from src.utils import softmax
import torch


@torch.inference_mode()
def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float | None = None,
    top_k: int | None = None,
    device: torch.device | None = None,
) -> str:
    tokens = tokenizer.encode(prompt)
    for _ in range(max_tokens):
        context = torch.LongTensor([tokens[-model.context_length :]], device=device)
        output = model(context)

        if temperature is not None and temperature > 0.0:
            output /= temperature

        output = softmax(output, dim=-1)

        if top_k is not None:
            top = output.topk(top_k, dim=-1)

        token = torch.multinomial(output, 1).item()

        tokens.append(token)

    return tokenizer.decode(tokens)
