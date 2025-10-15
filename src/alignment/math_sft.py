from vllm import SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import wandb
from typing import Callable
from cs336_alignment.batch_inference import evaluate_vllm
from cs336_alignment.sft_utils import tokenize_prompt_and_output, sft_microbatch_train_step
from cs336_alignment.utils import get_response_log_probs, add_system_prompt, init_wandb, log_generations, format_gsm8k_answers, load_policy_into_vllm_instance, init_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
    

def sft(
    vllm_model,
    policy,
    tokenizer,
    optimizer,
    n_sft_steps: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    gradient_accumulation_steps: int = 1,
    normalize_constant: float = 1.0,
    sampling_params: SamplingParams = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"]),
    log_every_n_steps: int = 1,
) -> None:

    for step in range(n_sft_steps):
        print(f"Step {step + 1}/{n_sft_steps}")
        
        # Sample prompts
        batch = next(iter(train_dataloader))
        print(batch)
        prompts = add_system_prompt(batch["question"])
        output_strs = format_gsm8k_answers(batch["answer"])
        
        tokenized = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=output_strs,
            tokenizer=tokenizer,
        )

        logprobs_dict = get_response_log_probs(policy, tokenized["input_ids"].to(policy.device), tokenized["labels"].to(policy.device), return_token_entropy=True)

        loss, metadata = sft_microbatch_train_step(
            logprobs_dict["log_probs"],
            tokenized["response_mask"].to(policy.device),
            gradient_accumulation_steps=gradient_accumulation_steps,
            normalize_constant=normalize_constant,
        )


        if (step + 1) % gradient_accumulation_steps == 0:
            # Log the loss
            total_norm = 0.0
            for p in policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            wandb.log({"train/gradient_norm": total_norm}, step=step+1)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        if (step + 1) % log_every_n_steps == 0:
            print(f"Step {step + 1}, Loss: {loss.item()}")

            if val_dataloader is not None:
                batch = next(iter(val_dataloader))
            else:
                batch = next(iter(train_dataloader))
            prompts = add_system_prompt(batch["question"])
            output_strs = [answer.split("####")[1].strip() for answer in batch["answer"]]
            log_dict = log_generations(vllm_model, policy, tokenizer, r1_zero_reward_fn, prompts, output_strs, sampling_params)
            wandb.log({"train/loss": loss.item(), "eval/accuracy": log_dict["accuracy"], "train/entropy": logprobs_dict["token_entropy"].mean().item()}, step=step+1)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data) -> None:
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def expert_iteration(
    vllm_model,
    policy,
    tokenizer,
    optimizer,
    n_ei_steps: int,
    G_num_samples: int,
    reward_fn: Callable[[str,str], dict[str, float]],
    n_sft_steps: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    gradient_accumulation_steps: int = 1,
    normalize_constant: float = 1.0,
    sampling_params: SamplingParams = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"]),
    log_every_n_steps: int = 1,
) -> None:
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
    for step in range(n_ei_steps):
        batch = next(iter(train_dataloader))
        prompts = add_system_prompt(batch["question"])
        output_strs = format_gsm8k_answers(batch["answer"])
        
        load_policy_into_vllm_instance(policy, vllm_model)
        
        sampling_params.n = G_num_samples  # Generate G_num_samples for each prompt
        outputs = vllm_model.generate(prompts, sampling_params)
        
        final_batch: list[dict[str, str]] = []
        for i, output in enumerate(outputs):
            original_question = batch["question"][i] 
            original_answer = output_strs[i]
            
            # Procesar todas las muestras generadas para este prompt
            for sample_output in output.outputs:
                generated_text = sample_output.text
                
                # Compute the reward for the generated text.
                reward = reward_fn(generated_text, original_answer)
                if reward["reward"] > 0:
                    final_batch.append({"question": original_question, "answer": generated_text})

        if not final_batch:
            print("No valid responses generated, using initial batch for EI step.")
            final_batch = [{"question": q, "answer": a} for q, a in zip(batch["question"], batch["answer"])]
        else:
            print(f"Generated {len(final_batch)} valid responses for EI step {step + 1}/{n_ei_steps}")
        
        sft_dataset = SimpleDataset(final_batch)
        sampling_params.n = 1  # Reset n to 1 for SFT
        batch_size = min(max(1, len(final_batch) // train_dataloader.batch_size), 8)  # type: ignore

        sft(vllm_model,
            policy,
            tokenizer,
            optimizer,
            n_sft_steps=max(1, min(n_sft_steps, len(sft_dataset) // batch_size)),
            train_dataloader=DataLoader(sft_dataset, batch_size=batch_size, shuffle=True),
            val_dataloader=val_dataloader,
            gradient_accumulation_steps=gradient_accumulation_steps,
            normalize_constant=normalize_constant,
            sampling_params=sampling_params,
            log_every_n_steps=log_every_n_steps,
        )

    
if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    
    dataset = load_dataset("openai/gsm8k", "main",revision="main")

    num_samples: int = 512 # 128, 256, 512, 1024
    batch_size: int = 8
    
    train_dataloader = DataLoader(
        dataset=dataset["train"], # type: ignore
        batch_size=batch_size,
        shuffle=False,
    )
    val_dataloader = DataLoader(
        dataset=dataset["test"], # type: ignore
        batch_size=batch_size,
        shuffle=False,
    )
    
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0, 
        max_tokens=1024, 
        min_tokens=4,
        stop=["</answer>"],
        seed=42
    )
    # Create an LLM.
    vllm_model = init_vllm(
        model_id=model_id,
        device="cuda:0",
        seed=42,
        gpu_memory_utilization=0.85,
    )
    model_name = vllm_model.llm_engine.model_config.model
    policy = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda:1")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    
    init_wandb(project="cs336_alignment", name="ei_sft")
    
    # # Start the SFT process
    # sft(
    #     vllm_model=vllm_model,
    #     policy=policy,
    #     tokenizer=tokenizer,
    #     optimizer=optimizer,
    #     n_sft_steps=num_samples // batch_size,
    #     train_dataloader=train_dataloader,
    #     val_dataloader=val_dataloader,
    #     gradient_accumulation_steps=1,
    # )
    
    # Start the Expert Iteration process
    expert_iteration(
        vllm_model=vllm_model,
        policy=policy,
        tokenizer=tokenizer,
        optimizer=optimizer,
        n_ei_steps=5,  # Number of expert iteration steps
        G_num_samples=4,  # Number of samples per prompt
        reward_fn=r1_zero_reward_fn,  # Reward function
        n_sft_steps=num_samples // batch_size,  # Number of SFT steps per EI step
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        gradient_accumulation_steps=1,
        normalize_constant=1.0,
        sampling_params=sampling_params,
    )