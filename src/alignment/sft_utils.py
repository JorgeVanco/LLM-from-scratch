import torch
from cs336_alignment.utils import masked_normalize

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)-> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
    other tokens (prompt or padding).
    Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
    Returns:
        dict[str,torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. Then the returned dictionary should have the
        following keys:
            input_ids: torch.Tensor of shape (batch_size, max(prompt_and_output_lens)-1):
                the tokenized prompt and output strings, with the final token sliced off.
            labels: torch.Tensor of shape (batch_size, max(prompt_and_output_lens)-1):
                shifted input ids, i.e., the input ids without the first token.
            response_mask: torch.Tensor of shape (batch_size, max(prompt_and_output_lens)-1):
                a mask on the response tokens in the labels."""
                
    prompt_tokenized = [tokenizer.encode(i) for i in prompt_strs]
    output_tokenized = [tokenizer.encode(i) for i in output_strs]

    max_len: int = max((len(x[0]) + len(x[1]) for x in zip(prompt_tokenized, output_tokenized)))
    
    masks = []
    tokenized_prompt_and_output = []
    for i in range(len(prompt_tokenized)):
        n_prompt: int = len(prompt_tokenized[i])
        n_output: int = len(output_tokenized[i])
        masks.append([False] * (n_prompt - 1) + [True] * n_output + [False] * (max_len - n_prompt - n_output))
        tokenized_prompt_and_output.append(prompt_tokenized[i] + output_tokenized[i] + [151643] * (max_len - n_prompt - n_output))

    return {
        "input_ids": torch.tensor([t[:-1] for t in tokenized_prompt_and_output], dtype=torch.long),
        "labels": torch.tensor([t[1:] for t in tokenized_prompt_and_output], dtype=torch.long),
        "response_mask": torch.tensor(masks, dtype=torch.float)
    }

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    Args:
        policy_log_probs (batch_size, sequence_length): per-token log-probabilities from the
            SFT policy being trained.
        response_mask (batch_size, sequence_length): 1 for response tokens, 0 for
            prompt/padding.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        normalize_constant: The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]. loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
        this so we can log it.
        metadata: Dict with metadata from the underlying loss call, and any other statistics you
        might want to log.
    """
    loss = -1.0 * masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1).mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {"loss": loss, "policy_log_probs": policy_log_probs}