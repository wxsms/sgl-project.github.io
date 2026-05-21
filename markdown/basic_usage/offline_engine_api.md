# Offline Engine API

SGLang provides a direct inference engine without the need for an HTTP server, especially for use cases where additional HTTP server adds unnecessary complexity or overhead. Here are two general use cases:

- Offline Batch Inference
- Custom Server on Top of the Engine

This document focuses on the offline batch inference, demonstrating four different inference modes:

- Non-streaming synchronous generation
- Streaming synchronous generation
- Non-streaming asynchronous generation
- Streaming asynchronous generation

Additionally, you can easily build a custom server on top of the SGLang offline engine. A detailed example working in a python script can be found in [custom_server](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/custom_server.py).



## Nest Asyncio
Note that if you want to use **Offline Engine** in ipython or some other nested loop code, you need to add the following code:
```python
import nest_asyncio

nest_asyncio.apply()

```

## Advanced Usage

The engine supports [vlm inference](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/offline_batch_inference_vlm.py) as well as [extracting hidden states](https://github.com/sgl-project/sglang/blob/main/examples/runtime/hidden_states). 

Please see [the examples](https://github.com/sgl-project/sglang/tree/main/examples/runtime/engine) for further use cases.

## Offline Batch Inference

SGLang offline engine supports batch inference with efficient scheduling.


```python
# launch the offline engine
import asyncio

import sglang as sgl
import sglang.test.doc_patch  # noqa: F401
from sglang.utils import async_stream_and_merge, stream_and_merge

llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
```

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.63it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.37it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.79it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.79it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.79it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.79it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.79it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.79it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.79it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.79it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.79it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 14.96it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 14.96it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 14.96it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 14.96it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 14.96it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 14.96it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 14.96it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 14.96it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 14.96it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 14.96it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 14.96it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 22.93it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 22.93it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 22.93it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 22.93it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.93it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.93it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.93it/s]

    Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.93it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.93it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.93it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.80it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.80it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.80it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.80it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.80it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.80it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.80it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.80it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.80it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.26it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   7%|▋         | 4/58 [00:00<00:02, 18.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   7%|▋         | 4/58 [00:00<00:02, 18.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   7%|▋         | 4/58 [00:00<00:02, 18.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   7%|▋         | 4/58 [00:00<00:02, 18.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.41it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.41it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.67 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.10it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.10it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.10it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 34.10it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.79it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.79it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.79it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.79it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.79it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.79it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.18it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.18it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.18it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.18it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.18it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.18it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  53%|█████▎    | 31/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.14it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.14it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.14it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.14it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 46.14it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  71%|███████   | 41/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=160 avail_mem=74.54 GB):  71%|███████   | 41/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 47.05it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.42it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.42it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.42it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.42it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.42it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.42it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.75it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.75it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.98it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 38.13it/s]


### Non-streaming Synchronous Generation


```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print("===============================")
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

    ===============================
    Prompt: Hello, my name is
    Generated text:  Tracy and I'm a social worker with the Public Health Clearinghouse. My name is Tracy and I'm a social worker with the Public Health Clearinghouse. I am one of the few social workers in the nation who specializes in Family Therapy. I believe in the power of individual and family therapy to improve lives. I've been trained in both the Family Therapy and Family Therapy. This training builds on my background in child psychology and early childhood development. My areas of expertise include the assessment of family functioning, family therapy, and the application of family therapy to families with serious and complex issues.
    
    My expertise lies in the area of family therapy
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to attend an executive school, a prestigious program that focuses on teaching high-level business skills. The school offers a degree that is worth $45,000 per year for 10 years. The president also has the option of starting a side business at an investment firm, which could generate a $25,000 annual income for 5 years and earn a profit of $10,000 annually thereafter. After 10 years, the president will retire.
    
    Which of the two options has a higher expected financial benefit over 10 years? To determine which option has a higher
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]
    A. Paris
    B. Lyon
    C. London
    D. Marseille
    
    To determine the capital of France, we need to consider the continent from which France is derived. France is located in Europe, specifically on the continent of Europe. The capital of France is Paris.
    
    Let's break it down step by step:
    
    1. Identify the continent: France is located in Europe.
    2. Identify the capital: France's capital is Paris.
    
    Based on this information, the correct answer is:
    
    \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it’s not the only way to reach the future. Because in 2018, 75% of the world’s nations are in the middle of a technology revolution, and as technology becomes more sophisticated, it can bring about huge changes that are not always understood. In other words, there is a huge potential for change and innovation. I am sure that even with all the technological advancement, the future is going to be an exciting place to live.
    
    How can AI be a positive force for change? AI has the potential to revolutionize the way we live, work, and communicate. Here are a few examples


### Streaming Synchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {
    "temperature": 0.2,
    "top_p": 0.9,
}

print("\n=== Testing synchronous streaming generation with overlap removal ===\n")

for prompt in prompts:
    print(f"Prompt: {prompt}")
    merged_output = stream_and_merge(llm, prompt, sampling_params)
    print("Generated text:", merged_output)
    print()
```

    
    === Testing synchronous streaming generation with overlap removal ===
    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have [number] years of experience in [industry]. I'm a [job title] at [company name], and I'm always looking for ways to [describe a new skill or initiative]. I'm always eager to learn and grow, and I'm always looking for opportunities to contribute to the company. What's your favorite hobby or activity? I love [describe a hobby or activity].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French Quarter, a historic neighborhood. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also known for its cuisine, including French cuisine, and its fashion industry. The city is home to many international organizations and is a major economic and political center in Europe. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of art, culture,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation and robotics: As AI technology continues to improve, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare to transportation. This will lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more applications in healthcare, including personalized medicine, disease diagnosis, and drug discovery.
    
    3. AI in finance: AI is already being used in finance to
    


### Non-streaming Asynchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

print("\n=== Testing asynchronous batch generation ===")


async def main():
    outputs = await llm.async_generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")


asyncio.run(main())
```

    
    === Testing asynchronous batch generation ===


    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is
    Generated text:  Sarah. I work as a librarian at a community college. My passion is reading and helping students find their own paths. I love animals and have a keen eye for detail. What's your favorite book of all time?
    Hi, I'm Sarah! I'm a librarian at a community college, and I love animals, and I'm also a bit of a bookworm! What's your favorite book? I love reading books with detailed and intriguing descriptions of their characters. I also really like stories that explore themes of love, friendship, and loss. What's your favorite book? I like reading books about animals, and I really enjoy stories
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A concise statement could be: "Paris, the capital of France, is renowned for its stunning architecture, vibrant culture, and iconic landmarks such as the Eiffel Tower and Louvre Museum." 
    
    This statement encapsulates the core features of Paris while being concise and informative. It highlights its historical significance, architectural marvels, and cultural attractions, all of which contribute to its status as one of the most cosmopolitan cities in the world. 
    
    The statement is likely to be easily understood by those unfamiliar with French geography and history, as well as by tourists and visitors interested in the city. It provides a clear overview of Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but several trends are likely to shape the direction of the technology's development in the coming years:
    
    1. Increased Human-Machine Collaboration: It is likely that we will see a trend towards more human-machine collaboration in AI research, particularly in areas such as natural language processing, machine learning, and image recognition. This will likely lead to more sophisticated and nuanced AI systems that can understand and interpret human language and behaviors more effectively.
    
    2. Better Data Privacy: As AI becomes more integrated into our lives, it is likely that we will see an increase in data privacy concerns. This will lead to increased regulations and standards around data collection,


### Streaming Asynchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

print("\n=== Testing asynchronous streaming generation (no repeats) ===")


async def main():
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Generated text: ", end="", flush=True)

        # Replace direct calls to async_generate with our custom overlap-aware version
        async for cleaned_chunk in async_stream_and_merge(llm, prompt, sampling_params):
            print(cleaned_chunk, end="", flush=True)

        print()  # New line after each prompt


asyncio.run(main())
```

    
    === Testing asynchronous streaming generation (no repeats) ===
    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is
    Generated text: 

     [

    Your

     Name

    ],

     and

     I

    'm

     [

    Your

     Age

    ].

     I

    'm

     a

     [

    Your

     occupation

    ]

     who

     is

     passionate

     about

     [

    Your

     favorite

     hobby

     or

     activity

    ].

     I

     have

     always

     been

     [

    Your

     favorite

     thing

     about

     yourself

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    Your

     ultimate

     goal

     or

     goal

     for

     life

    ].

     How

     would

     you

     describe

     your

     personality

     and

     what

     makes

     you

     unique

    ?

     As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     feelings

     or

     personal

     experiences

     like

     a

     human

     being

    ,

     but

     I

     can

     tell

     you

     that

     I

    'm

     programmed

     to

     understand

     and

     respond

     to

     a

     wide

     range

     of

     questions

     and

     topics

    .

     And

    ,

     my

     purpose

     is

     to

     provide

     accurate

     and

     helpful

     information

     to

     the

     best

     of

     my

     ability

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    ,

     the

     Gateway

     of

     the

     World

    ,

     and

     the

     City

     of

     Ideas

    .

     It

     is

     a

     UNESCO

     World

     Heritage

     Site

     and

     is

     the

     country

    's

     largest

     city

     with

     a

     population

     of

     approximately

     

    2

    .

    7

     million

     people

    .

     The

     city

     is

     known

     for

     its

     historical

     landmarks

    ,

     rich

     cultural

     heritage

    ,

     and

     vibrant

     food

     and

     fashion

     scene

    .

     Paris

     is

     an

     important

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     France

    .

     It

     is

     also

     a

     popular

     tourist

     destination

    ,

     drawing

     millions

     of

     visitors

     every

     year

    .

     The

     city

     has

     a

     rich

     history

    ,

     including

     ancient

     ruins

    ,

     and

     has

     influenced

     many

     aspects

     of

     French

     culture

     and

     society

    .

     As

     one

     of

     the

     most

     visited

     cities

     in

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     several

     trends

     are

     likely

     to

     shape

     the

     technology

    's

     evolution

     in

     the

     next

     decade

     and

     beyond

    :
    


    1

    .

     Increased

     specialization

     of

     AI

    :

     As

     AI

     becomes

     more

     complex

     and

     sophisticated

    ,

     it

     is

     likely

     to

     become

     specialized

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     and

     customer

     service

    .

     This

     will

     require

     more

     expertise

     and

     training

     for

     AI

     systems

    ,

     which

     could

     create

     a

     need

     for

     more

     specialized

     AI

     engineers

    .
    


    2

    .

     Integration

     of

     more

     diverse

     data

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     they

     will

     need

     to

     access

     and

     process

     a

     wider

     range

     of

     data

     to

     make

     accurate

     predictions

     and

     decisions

    .

     This

     will

     require

     the

     development

     of

     more

     advanced

     data

     analytics

     techniques

     and

     the

     integration

     of

     more

     diverse

     data

    



```python
llm.shutdown()
```
