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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.74it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.73it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.28it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.60it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.44it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.44it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.44it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.44it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.44it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.44it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.44it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.44it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.44it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.44it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.44it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.36it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.36it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.36it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.36it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.36it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.36it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.36it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.36it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.36it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.36it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.36it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.30it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.08it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.06it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.42 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.11it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.11it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.11it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.05it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.05it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.05it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.05it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.05it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.05it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.12it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.12it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.12it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.12it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.12it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.12it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:00<00:00, 42.97it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  50%|█████     | 29/58 [00:00<00:00, 42.97it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  50%|█████     | 29/58 [00:00<00:00, 42.97it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.97it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.97it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.97it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:00<00:00, 42.97it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:00<00:00, 45.48it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:00<00:00, 45.48it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:00<00:00, 45.48it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:00<00:00, 45.48it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:00<00:00, 45.48it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:00<00:00, 45.48it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.96it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.96it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.96it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.96it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.96it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.96it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.08it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.02it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.85it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 41.02it/s]


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
    Generated text:  Michelangelo.
    
    What did Michelangelo do? Michelangelo was a renowned Renaissance humanist and sculptor, active in the late 16th to early 17th centuries. Here are some of his most notable accomplishments:
    
    1. Sculpture: Michelangelo created many iconic sculptures, including David, the Sistine Chapel ceiling, and the tomb of Pope Paul III.
    
    2. Painting: He was also a painter, creating several masterpieces, including The Creation of Adam, The School of Athens, and The Pieta.
    
    3. Architecture: Michelangelo was a visionary architect who designed several impressive buildings, including the Uffizi Gallery
    ===============================
    Prompt: The president of the United States is
    Generated text:  30 years older than the president of the Senate. If the president of the Senate is currently 1/3 the age of the current president, what is the current age of the president of the Senate? To determine the current age of the president of the Senate, we need to follow a step-by-step approach based on the information provided.
    
    1. Let's denote the age of the current president of the Senate as \( x \) years.
    2. According to the problem, the president of the United States is 30 years older than the president of the Senate. Therefore, the age of the president of the United States
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Lille
    C. Nice
    D. Laval
    
    To determine the capital of France, let's recall the capital cities of France. The capital of France is:
    
    A. Paris
    
    Therefore, the correct answer is:
    
    A. Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  revolutionising the way we live. Whether it be in the way we communicate, the way we work, or the way we learn, AI is shaping the future. However, it’s important to note that the use of AI raises important ethical questions that need to be addressed. In this article, we will explore some of the most pressing issues surrounding the use of AI and how they impact our daily lives.
    One of the most pressing ethical questions surrounding the use of AI is the issue of bias. AI systems are trained on data that is biased, which means that they may have an unfair advantage or disadvantage over other types of data. This can


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Skill] who has been [Number of Years] years in the field of [Field of Interest]. I am passionate about [Why I Love My Profession]. I am always looking for new challenges and opportunities to grow and learn. I am a [Personality Trait] who is [How I Handle Stress]. I am a [Personality Trait] who is [How I Handle Conflict]. I am a [Personality Trait] who is [How I Communicate]. I am a [Personality Trait] who is [How I Work with
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris". It is the largest city in France and the second-largest city in the European Union, with a population of over 1. 3 million people. Paris is known for its rich history, art, and culture, and is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also known for its fashion industry, with many famous fashion designers and boutiques located in the city. Paris is a popular tourist destination, with millions of visitors each year. It is also a major economic center, with a significant role
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and robots to personalized medicine and virtual assistants. Additionally, AI is likely to play an increasingly important role in areas such as healthcare, finance, and energy, as companies and governments invest heavily in developing AI-powered solutions to address complex challenges. Finally, the development of AI is likely to be driven by a growing focus on ethical considerations and the need to ensure that AI is used in a responsible and beneficial way for all. This includes
    


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
    Generated text:  [Name] and I am a [main character's occupation or status]. What brings you to this world? Please give me a brief background on your life, achievements, and any special skills or interests you possess.
    Hello, my name is [Name] and I am a [main character's occupation or status]. What brings you to this world? Please give me a brief background on your life, achievements, and any special skills or interests you possess.
    First and foremost, I am a [occupation or status]. I have always been fascinated by the world around me, from its most complex mysteries to its smallest details. I have always been
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France. It serves as the administrative and political center of the country, hosting the European Parliament, the French parliament. The city is home to the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and many other attractions. Paris is also known as the "City of Light" and is considered one of the most beautiful cities in the world. As of 2021, Paris has a population of around 2. 4 million people and is home to the French Parliament and the European Parliament. 
    
    Therefore, Paris is a well-known and significant city in France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly diverse and rapidly evolving, and there are several potential areas where we can expect to see significant advancements. Here are some possible future trends in AI:
    
    1. More complex natural language processing: As technology continues to advance, we can expect to see more complex natural language processing algorithms that can understand and interpret human language in ways that are difficult for humans to grasp. This could lead to new forms of artificial intelligence and assistive technologies that make life easier for humans.
    
    2. More advanced machine learning: As more data becomes available, we can expect to see more advanced machine learning algorithms that can learn from data in ways that are difficult for humans to


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

    Name

    ],

     and

     I

    'm

     a

     highly

     skilled

     and

     experienced

     [

    occupation

    ].

     I

     have

     a

     passion

     for

     [

    specific

     activity

     or

     hobby

     that

     I

     enjoy

    ]

     and

     have

     consistently

     performed

     exceptionally

     well

     in

     [

    any

     relevant

     field

    ].

     I

     have

     a

     unique

     talent

     for

     [

    something

     specific

     that

     I

     do

    ]

     and

     am

     always

     striving

     to

     improve

    .

     I

     enjoy

     sharing

     my

     knowledge

     with

     others

     and

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

     for

     growth

    .

     I

    'm

     excited

     to

     learn

     more

     about

     you

     and

     the

     world

     around

     me

    .

     Please

     feel

     free

     to

     introduce

     yourself

     further

    !

     [

    Name

    ]

     looks

     forward

     to

     our

     conversation

    .

     [

    Name

    ]


    This

     is

     not

     a

     fan

     of

     self

    -int

    rodu

    ctions

    .

     Can

     you

     provide

     an

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     beautiful

     architecture

    ,

     iconic

     landmarks

    ,

     and

     lively

     atmosphere

    .

     It

    's

     also

     known

     for

     its

     wine

     industry

     and

     as

     the

     birth

    place

     of

     the

     French

     Revolution

    .

     However

    ,

     it

    's

     worth

     noting

     that

     the

     city

     has

     undergone

     significant

     changes

     over

     time

    ,

     including

     the

     construction

     of

     the

     E

    iff

    el

     Tower

     and

     the

     implementation

     of

     various

     environmental

     policies

    .

     French

     people

     are

     known

     for

     their

     love

     of

     food

    ,

     which

     can

     be

     found

     in

     many

     popular

     restaurants

     around

     the

     city

    .

     Paris

     is

     also

     a

     hub

     for

     the

     arts

    ,

     with

     numerous

     museums

    ,

     theaters

    ,

     and

     galleries

    .

     Overall

    ,

     Paris

     is

     a

     vibrant

     and

     exciting

     city

     with

     a

     rich

     history

    ,

     and

     it

    's

     an

     important

     part

     of

     French

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     increased

     collaboration

     between

     humans

     and

     machines

    ,

     as

     well

     as

     a

     continued

     push

     for

     more

     sophisticated

     and

     ethical

     AI

     systems

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     collaboration

     between

     humans

     and

     machines

    :

     In

     the

     future

    ,

     it

     is

     likely

     that

     we

     will

     see

     more

     collaboration

     between

     humans

     and

     machines

     in

     areas

     such

     as

     healthcare

    ,

     transportation

    ,

     and

     manufacturing

    .

     Machines

     will

     continue

     to

     learn

     and

     adapt

    ,

     and

     humans

     will

     play

     a

     more

     active

     role

     in

     guiding

     and

     monitoring

     AI

     systems

    .
    


    2

    .

     More

     sophisticated

     AI

     systems

    :

     As

     AI

     systems

     continue

     to

     become

     more

     advanced

    ,

     we

     are

     likely

     to

     see

     more

     sophisticated

     AI

     systems

     that

     are

     able

     to

     understand

     and

     learn

     from

    



```python
llm.shutdown()
```
