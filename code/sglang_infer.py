import os
import time
import csv
import psutil
import torch
from sglang import SGLang, CompletionConfig

MODEL_PATH = "/home/mihir/data/engines/custom/hf_cache/gemma-3-4b-it"
OUTPUT_CSV_PATH = "sglang_benchmark_results.csv"

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_gpu_memory_mb(device=0):
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) / (1024 * 1024)

def clean_output(text: str) -> str:
    return text.strip()

def run_sglang_benchmark(
    max_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.5,
    repetition_penalty: float = 1.1,
):
    print("Measuring baseline memory usage...")
    cpu_mem_initial = get_process_memory_mb()

    print(f"Loading model: {MODEL_PATH}...")
    sglang = SGLang()
    model = sglang.load_model(
        model_path=MODEL_PATH,
        max_batch_size=8,
        max_total_tokens=2048,
        trust_remote_code=True,
        memory_static_fraction=0.25
    )

    cpu_mem_after_load = get_process_memory_mb()
    gpu_mem_after_load = get_gpu_memory_mb()
    cpu_load_cost = cpu_mem_after_load - cpu_mem_initial

    print("-" * 50)
    print("Model Loading Memory Report:")
    print(f"  CPU Memory Cost to Load: {cpu_load_cost:.2f} MB")
    print(f"  Initial GPU Memory Allocated: {gpu_mem_after_load:.2f} MB")
    print("-" * 50)

    config = CompletionConfig(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    prompts = [
        "Explain the theory of relativity in simple words.",
        "What is the future of artificial intelligence?",
        "Describe how blockchain works.",
        "How does quantum computing differ from classical computing?",
        "What are black holes?",
    ]

    print(f"Using {len(prompts)} predefined prompts. Starting benchmark...")

    peak_gpu_mem_mb = gpu_mem_after_load
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        writer.writerow([
            "prompt",
            "prompt_word_count",
            "output",
            "output_token_count",
            "tokens_per_sec",
            "total_gpu_mem_after_load_mb",
            "peak_gpu_mem_during_run_mb"
        ])

        for i, prompt in enumerate(prompts):
            start = time.time()
            resp = model.generate(prompt, config)
            duration = time.time() - start

            current_gpu_mem = get_gpu_memory_mb()
            peak_gpu_mem_mb = max(peak_gpu_mem_mb, current_gpu_mem)

            output_text = clean_output(resp.text)
            output_token_count = len(resp.token_ids)
            tps = output_token_count / duration if duration > 0 else 0.0

            writer.writerow([
                prompt,
                len(prompt.split()),
                output_text.replace("\n", " \\n "),
                output_token_count,
                f"{tps:.2f}",
                f"{gpu_mem_after_load:.2f}",
                f"{peak_gpu_mem_mb:.2f}"
            ])

            print(f"  Processed prompt {i+1}/{len(prompts)}... ({tps:.2f} tokens/sec)")

    print("-" * 50)
    print("Benchmark Complete!")
    print(f"Results saved to {OUTPUT_CSV_PATH}")
    print(f"Final Peak GPU Memory: {peak_gpu_mem_mb:.2f} MB")
    print("-" * 50)

if __name__ == "__main__":
    run_sglang_benchmark()
