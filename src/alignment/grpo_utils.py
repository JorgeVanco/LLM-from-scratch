import torch
from typing import Callable, Any, Literal
from vllm import SamplingParams
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import get_response_log_probs, init_models, add_system_prompt, load_policy_into_vllm_instance, init_wandb, log_generations
from cs336_alignment.sft_utils import tokenize_prompt_and_output

def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute rewards for each group of rollout responses, normalized by the group size.
    
    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
            the ground truths, producing a dict with keys "reward", "format_reward", and
            "answer_reward".
        rollout_responses: list[str] Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str] The ground truths for the examples. The length of this
            list is rollout_batch_size, because the ground truth for each example is repeated
            group_size times.
        group_size: int Number of responses per question (group).
        advantage_eps: float Small constant to avoid division by zero in normalization.
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
            subtract only the group mean.
            
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
        advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
            response.
        raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response.
        metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    rewards = [torch.empty((group_size,)) for _ in range(len(rollout_responses) // group_size)]
    for i, (response, ground_truth) in enumerate(zip(rollout_responses, repeated_ground_truths)):
        reward = reward_fn(response, ground_truth)
        rewards[i // group_size][i % group_size] = torch.tensor(reward["reward"])
    
    reward_tensor = torch.stack(rewards, dim=0)
    raw_tensor = (reward_tensor - reward_tensor.mean(dim=-1, keepdim=True))
    if normalize_by_std:
        raw_tensor = raw_tensor / (reward_tensor.std(dim=-1, keepdim=True) + advantage_eps)
    
    reward_tensor = reward_tensor.view(-1)
    raw_tensor = raw_tensor.view(-1)
    
    return raw_tensor, reward_tensor, {
        "mean": reward_tensor.mean().item(),
        "std": reward_tensor.std().item(),
        "max": reward_tensor.max().item(),
        "min": reward_tensor.min().item(),
    }
    
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs:torch.Tensor,
)-> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
            reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), log probs for
            each token.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
            be aggregated across the batch and sequence dimensions in the training loop).
    """
    
    return -1.0 * (raw_rewards_or_advantages * policy_log_probs)

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
            probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
            from the old policy.
        cliprange: float Clip parameter ϵ (e.g. 0.2).
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
            loss.
        metadata dict containing whatever you want to log. We suggest logging whether each
            token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
            the min was lower than the LHS.
    """
    policy_ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clip(policy_ratio, min=1-cliprange, max=1+cliprange)
    was_clipped = torch.where((policy_ratio < 1 - cliprange) & (policy_ratio > 1 + cliprange), 1.0, 0.0)
    no_clip_loss = policy_ratio * advantages
    clip_loss = clipped_ratio * advantages
    return -1.0 *  torch.where(no_clip_loss < clip_loss, no_clip_loss, clip_loss), {
        "clipped": was_clipped
    }
    

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Select and compute the desired policy-gradient loss.
    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the
            policy being trained.
        loss_type: One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards: Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages: Required for "reinforce_with_baseline" and "grpo_clip"; shape
            (batch_size, 1).
        old_log_probs: Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange: Required for "grpo_clip"; scalar ϵ used for clipping.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss: (batch_size, sequence_length), per-token loss.
        metadata: dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards must be provided for 'no_baseline' loss type.")
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages must be provided for 'reinforce_with_baseline' loss type.")
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}

    elif loss_type == "grpo_clip":
        if old_log_probs is None:
            raise ValueError("old_log_probs must be provided for 'grpo_clip' loss type.")
        if cliprange is None:
            raise ValueError("cliprange must be provided for 'grpo_clip' loss type.")
        if advantages is None:
            raise ValueError("advantages must be provided for 'grpo_clip' loss type.")
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    
    
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute the mean of tensor along a given dimension, considering only those elements where
    mask == 1.
    
    Args:
        tensor: torch.Tensor The data to be averaged.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
        dim: int | None Dimension over which to average. If None, compute the mean over all
            masked elements.
            
    Returns:
        torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch.
    
    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the
            policy being trained.
        response_mask: (batch_size, sequence_length), 1 for response tokens, 0 for
            prompt/padding.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        loss_type: One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards: Needed when loss_type == "no_baseline"; shape (batch_size, 1).
        advantages: Needed when loss_type != "no_baseline"; shape (batch_size, 1).
        old_log_probs: Required for GRPO-Clip; shape (batch_size, sequence_length).
        cliprange: Clip parameter ϵ for GRPO-Clip.
        
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
            this so we can log it.
        metadata: Dict with metadata from the underlying loss call, and any other statistics you
            might want to log.
    """
    
    policy_per_token_gradient_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )
    
    policy_gradient_loss = masked_mean(policy_per_token_gradient_loss, response_mask)
    
    metadata["policy_gradient_loss"] = policy_gradient_loss
    
    (policy_gradient_loss / gradient_accumulation_steps).backward()
    
    return policy_gradient_loss, metadata


def grpo_train_loop() -> None:
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1 # On-policy
    train_batch_size: int = 256 # On-policy
    gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85
    loss_type: Literal[
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    seed: int = 42
    log_every_n_steps: int = 5

    # Create an LLM.
    model_id = "Qwen/Qwen2.5-Math-1.5B"

    vllm_model, policy, tokenizer = init_models(model_id)
    
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens, 
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        seed=seed
    )
    
    assert train_batch_size % gradient_accumulation_steps == 0, (
    "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
    "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
    "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    
    # Corregir el cálculo de microbatches
    assert rollout_batch_size == train_batch_size, (
    "rollout_batch_size must equal train_batch_size for on-policy training"
    )
    print("rollout_batch_size", rollout_batch_size)
    print("micro_train_batch_size", micro_train_batch_size)
    print("grpo_steps", n_grpo_steps)
    
    dataset = load_dataset("openai/gsm8k", "main",revision="main")
    train_dataloader = DataLoader(
        dataset=dataset["train"], # type: ignore
        batch_size=n_prompts_per_rollout_batch,
        shuffle=False,
    )
    val_dataloader = DataLoader(
        dataset=dataset["test"], # type: ignore
        batch_size=n_prompts_per_rollout_batch,
        shuffle=False,
    )
    
    init_wandb("cs336_alignment", "grpo")
    print("Starting GRPO training loop...")
    
    for step in range(n_grpo_steps):
        batch = next(iter(train_dataloader))
        prompts = add_system_prompt(batch["question"])
        ground_truths = [answer.split("####")[1].strip() for answer in batch["answer"]]
        
        load_policy_into_vllm_instance(policy, vllm_model)
        
        # Generate responses for the prompts.
        sampling_params.n = group_size
        outputs = vllm_model.generate(prompts, sampling_params)
        
        # Collect the responses and ground truths.
        rollout_responses = [o.text for output in outputs for o in output.outputs]
        repeated_ground_truths = [ground_truths[i // group_size] for i in range(len(rollout_responses))]
        
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths, # type: ignore
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )
        
        tokenized = tokenize_prompt_and_output(
            prompt_strs=[prompts[i // group_size] for i in range(len(rollout_responses))],
            output_strs=rollout_responses,
            tokenizer=tokenizer,
        )
        
        # Get old log_probs
        if loss_type == "grpo_clip":
            policy_log_probs_dict = get_response_log_probs(
                    policy,
                    tokenized["input_ids"].to(policy.device),
                    tokenized["labels"].to(policy.device),
                    return_token_entropy=True
                )
                
            old_log_probs = policy_log_probs_dict["log_probs"]
        else:
            old_log_probs = None
        
        for train_step in range(epochs_per_rollout_batch):
            # Procesar en microbatches reales
            for micro_step in range(gradient_accumulation_steps):
                start_idx = micro_step * micro_train_batch_size
                end_idx = start_idx + micro_train_batch_size
                
                # Obtener microbatch
                micro_input_ids = tokenized["input_ids"][start_idx:end_idx].to(policy.device)
                micro_labels = tokenized["labels"][start_idx:end_idx].to(policy.device)
                micro_response_mask = tokenized["response_mask"][start_idx:end_idx].to(policy.device)
                micro_advantages = advantages[start_idx:end_idx].unsqueeze(1).to(policy.device)
                micro_raw_rewards = raw_rewards[start_idx:end_idx].unsqueeze(1).to(policy.device)
                micro_old_log_probs = old_log_probs[start_idx:end_idx] if old_log_probs is not None else None
                
                policy_log_probs_dict = get_response_log_probs(
                    policy,
                    micro_input_ids,
                    micro_labels,
                    return_token_entropy=True
                )
                
                policy_log_probs = policy_log_probs_dict["log_probs"]

                cliprange = 0.2 if loss_type == "grpo_clip" else None
        
                policy_gradient_loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs,
                    micro_response_mask,
                    gradient_accumulation_steps,
                    loss_type,
                    micro_raw_rewards,
                    micro_advantages,
                    micro_old_log_probs,
                    cliprange,
                )

            # Update gradients after gradient accumulation
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % log_every_n_steps == 0:
            print(f"Step {step + 1}, Loss: {policy_gradient_loss.item()}")

            if val_dataloader is not None:
                batch = next(iter(val_dataloader))
            else:
                batch = next(iter(train_dataloader))
            prompts = add_system_prompt(batch["question"])
            output_strs = [answer.split("####")[1].strip() for answer in batch["answer"]]
            log_dict = log_generations(vllm_model, policy, tokenizer, r1_zero_reward_fn, prompts, output_strs, sampling_params)
            wandb.log({"train/loss": policy_gradient_loss.item(), "eval/accuracy": log_dict["accuracy"], "train/entropy": policy_log_probs_dict["token_entropy"].mean().item()}, step=step+1)
            if loss_type == "grpo_clip":
                wandb.log({"train/clipped_fraction": metadata["clipped"].mean().item()}, step=step+1) # type: ignore
if __name__ == "__main__":
    grpo_train_loop()