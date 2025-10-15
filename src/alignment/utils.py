import torch
from typing import Callable, List, Any
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
# from transformers.models import PreTrainedModel

def init_models(model_id: str, gpu_memory_utilization: float=0.85, seed:int=42) -> tuple[LLM, Any, AutoTokenizer]:
    # Create an LLM.
    vllm_model = init_vllm(
        model_id=model_id,
        device="cuda:0",
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization
    )
    model_name = vllm_model.llm_engine.model_config.model
    policy = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda:1")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return vllm_model, policy, tokenizer

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
    return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
        model=model_id,
        device=device,
        dtype=torch.bfloat16, # type: ignore
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        )
        
def load_policy_into_vllm_instance(policy, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model # type: ignore
    llm_model.load_weights(state_dict.items())

def add_system_prompt(prompts: List[str]) -> List[str]:
    """
    Add a system prompt to each prompt in the list.
    This is useful for models that require a system prompt.
    """
    system_prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""
    return [system_prompt.format(question=prompt) for prompt in prompts]


def format_gsm8k_answers(answers: List[str]) -> List[str]:
    """
    Format the answers from the GSM8K dataset to match the expected output format.
    The answers are expected to be in the form of a string with the answer enclosed in <answer> tags.
    """
    return [f"<think> {answer.split('####')[0].strip()} </think> <answer> {answer.split('####')[1].strip()} </answer>" if "<answer" not in answer else answer for answer in answers]

def init_wandb(project: str, name: str) -> None:
    # Initialize wandb
    wandb.init(
        project=project,
        name=name,
    )
    # Setup wandb metrics
    wandb.define_metric("train_step") # the x‑axis for training
    wandb.define_metric("eval_step") # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")  

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
        containing unnormalized logits.
    Returns:
        torch.Tensor of shape (batch_size, sequence_length). The entropy for each next-token
        prediction.
    """
    
    log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    
    p = torch.softmax(logits, dim=-1)
    
    entropy = - (p * log_p).sum(dim=-1)
    
    return entropy


def get_response_log_probs(
    model,#: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities (given the previous tokens) from a causal language model, and optionally the
    entropy of the model’s next-token distribution.
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
            response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling
            compute_entropy.
    Returns:
        dict[str,torch.Tensor].
        "log_probs" shape(batch_size, sequence_length),conditional log-probabilities
        logpθ(xt |x<t).
        "token_entropy"optional,shape(batch_size, sequence_length),per-tokenentropy
        for each position (present only if return_token_entropy=True)
    """

    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)[torch.arange(logits.size(0))[:, None], 
                                                torch.arange(logits.size(1))[None, :], 
                                                labels]
    
    return_dict = {"log_probs": log_probs}
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return_dict["token_entropy"] = token_entropy
    return return_dict


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.
    
    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim:int|None the dimension to sum along before normalization. If None, sum over all
        dimensions.
        
    Returns:
        torch.Tensor The normalized sum, where masked elements (mask == 0) don’t contribute to
        the sum.
    """
    return (tensor * mask).sum(dim=dim) / normalize_constant


def log_generations(vllm_model: LLM, hf_model, tokenizer, reward_fn: Callable[[str,str], dict[str, float]], prompts: list[str], ground_truths: list[str], eval_sampling_params: SamplingParams) -> dict[str, float]:
    """
    Log the generations of a model given prompts and their ground truths.
    
    Args:
        model: The language model to generate text from.
        prompts: List of prompts to generate text for.
        ground_truths: List of ground truth answers corresponding to the prompts.
    """
    eval_sampling_params.stop = ["</answer>"]
    eval_sampling_params.include_stop_str_in_output = True
    eval_sampling_params.logprobs = 1

    outputs = vllm_model.generate(prompts, eval_sampling_params)

    response_lengths = []
    correct_responses_lengths = []
    incorrect_responses_lengths = []
    for output in outputs:
        generated_text = output.outputs[0].text
        prompt = output.prompt
        idx = prompts.index(prompt) # type: ignore
        gt = ground_truths[idx]

        reward = reward_fn(generated_text, gt) # type: ignore
        
        token_ids = output.outputs[0].token_ids
        response_lengths.append(len(token_ids))
        if reward['reward'] > 0:
            correct_responses_lengths.append(len(token_ids))
        else:
            incorrect_responses_lengths.append(len(token_ids))
            
        # Calculate average token entropy
        output_ids = tokenizer.encode(generated_text)
        logprob_dict = get_response_log_probs(
            model=hf_model,
            input_ids=torch.tensor([output_ids], device=hf_model.device),
            labels=torch.tensor([output_ids], device=hf_model.device),
            return_token_entropy=True,
        )

        avg_token_entropy = logprob_dict["token_entropy"].mean().item()

        print(f"\n\nPrompt:\n{prompt!r}\nGenerated text:\n{generated_text!r}\nGround Truth: {gt!r}\nReward: {reward}\nAverage token entropy: {avg_token_entropy}\n")

    print(f"\n\nAverage response length: {sum(response_lengths) / len(response_lengths)}")
    print(f"Average correct response length: {sum(correct_responses_lengths) / len(correct_responses_lengths) if correct_responses_lengths else "NaN"}")
    print(f"Average incorrect response length: {sum(incorrect_responses_lengths) / len(incorrect_responses_lengths) if incorrect_responses_lengths else "NaN"}")

    return {"accuracy": len(correct_responses_lengths) / len(response_lengths)}

if __name__ == "__main__":
    model = "Qwen/Qwen2.5-Math-1.5B"
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main",revision="main")
    
    num_samples = 5
    prompts = add_system_prompt(dataset["train"][:num_samples]["question"]) # type: ignore

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"]
    )
    print(sampling_params)
    # Create an LLM.
    llm = LLM(model=model)
    
    log_generations(llm,
                    reward_fn=r1_zero_reward_fn,
                 prompts=prompts,
                 ground_truths=[answer for answer in dataset["train"][:num_samples]["answer"]], # type: ignore
                 eval_sampling_params=sampling_params)
    