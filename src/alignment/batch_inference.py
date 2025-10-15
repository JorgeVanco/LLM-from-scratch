from vllm import LLM, SamplingParams
from typing import Callable, List
from datasets import load_dataset
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import add_system_prompt

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str,str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
    out_file: str | None = None
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    eval_sampling_params.stop = ["</answer>"]
    eval_sampling_params.include_stop_str_in_output = True
    
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    rewards = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        idx = prompts.index(prompt) # type: ignore
        answer = answers[idx]
        
        # Compute the reward for the generated text.
        reward = reward_fn(generated_text, answer) # type: ignore
        rewards.append(reward)
        
        if out_file is not None:
            # Serialize the results to disk.
            with open(out_file, "a") as f:
                f.write(f"\n\nPrompt:\n{prompt!r}\n")
                f.write(f"\n\nGenerated text:\n{generated_text!r}\n")
                f.write(f"\n\nAnswer:\n{answer}\n")
                f.write(f"\n\nReward:\n{reward}\n\n")
        # Serialize the results to disk or print them.
        print(f"\n\n\nPrompt: \n{prompt!r}\n\nGenerated text:\n{generated_text!r}\n\nAnswer:{answer}\n\nReward: {reward}")
    
    print(f"\n\n\nAverage reward: {sum(r['reward'] for r in rewards) / len(rewards)}")
    print(f"Total number of samples: {len(rewards)}")
    print(f"Total number of correct answers: {sum(1 for r in rewards if r['reward'] > 0)}")
    if out_file is not None:
        with open(out_file, "a") as f:
            f.write(f"\n\n\nAverage reward: {sum(r['reward'] for r in rewards) / len(rewards)}\n")
            f.write(f"Total number of samples: {len(rewards)}\n")
            f.write(f"Total number of correct answers: {sum(1 for r in rewards if r['reward'] > 0)}\n")


if __name__ == "__main__":
    model = "Qwen/Qwen2.5-Math-1.5B"
    
    dataset = load_dataset("openai/gsm8k", "main",revision="main")

    num_samples = 5000
    prompts = add_system_prompt(dataset["train"][:num_samples]["question"]) # type: ignore

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"]
    )
    # Create an LLM.
    llm = LLM(model=model)
    
    out_file = "vllm_evaluation_results.txt"
    
    with open(out_file, "w") as f:
        f.write(f"Evaluating model {model} on {num_samples} samples from the GSM8K dataset.\n")
    
    evaluate_vllm(llm, r1_zero_reward_fn, 
                 prompts=prompts,
                 answers=[answer.split("####")[-1].strip() for answer in dataset["train"][:num_samples]["answer"]], # type: ignore
                 eval_sampling_params=sampling_params,
                 out_file=out_file)
    