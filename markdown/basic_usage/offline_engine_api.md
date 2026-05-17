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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.54it/s]


    2026-05-17 06:20:47,803 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-17 06:20:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:21,  4.58s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.24it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.01it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.01it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.01it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 19.23it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 19.23it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 19.23it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 19.23it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 19.23it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 19.23it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 19.23it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 19.23it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 19.23it/s]

    Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 19.23it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 27.33it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 37.23it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 37.23it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 37.23it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 37.23it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.09it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 36.64it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 36.64it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 36.64it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.87it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.41it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.41it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.41it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.41it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.41it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.41it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.32it/s]

    Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.99it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.99it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.99it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.99it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.99it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.99it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.63it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.30it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 39.92it/s]


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
    Generated text:  Alex and I'm a 28 year old, male, and currently in my first year of medical school. I have been taking a course on Medical Terminology. My teacher told me I'm doing well on the course. I am a little nervous about my first day of school and I am very concerned about my performance on the course. I am aware that it is a very stressful time for most people. When I first arrived at school I was anxious. I don't have any personal experiences to draw from to help me feel more comfortable about the prospect of starting school.
    
    Based on the information provided, what advice would you give to
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person; therefore, all presidents are intelligent. If the statement is true, what is the minimum number of presidents mentioned in the statement? If the statement is false, what is the minimum number of presidents mentioned in the statement? The problem requires us to determine the minimum number of presidents mentioned in the statement "All presidents are intelligent. " Let's analyze the statement step by step.
    
    1. **Understanding the statement**: The statement is "All presidents are intelligent." This means that every president must be intelligent. 
    
    2. **Analyzing the statement**: The president of the United States is a person. Therefore, if a president is intelligent
    ===============================
    Prompt: The capital of France is
    Generated text:  _______. It is the largest and most populous city in France and the third largest city in Europe after Paris and London. Paris
    
    Therefore, the answer is Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  the future of education
    
    In the early 2000s, a big breakthrough in machine learning was made. The first AI program to learn from textual data was developed by the cognitive scientist Claude Shannon and the Stanford Artificial Intelligence Laboratory. At that time, this system was called “Broader,” and it was designed to draw meaning from textual data. But thanks to advances in computer science, today, AI is able to process and interpret vast amounts of textual data. This has the potential to greatly improve the quality of education.
    
    In this article, we will explain why AI is changing the way we learn and teach. We’ll also explore


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always looking for ways to [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, art, and cuisine, and is home to many famous museums, theaters, and restaurants. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage. The city is also known for its fashion industry, with many famous designers and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the impact of AI on society as a whole.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This will require a greater understanding of human emotions, motivations, and cognitive processes.
    
    3
    


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
    Generated text:  [Your Name], and I'm a [Role]. I've always been fascinated by [Your Hobby/Interest/Profession] and have always wanted to [Your Goal]. If you could call me anything else, what would it be? Hello, my name is [Your Name], and I'm a [Role]. I've always been fascinated by [Your Hobby/Interest/Profession] and have always wanted to [Your Goal]. If you could call me anything else, what would it be? [Your Name] is a [Age] year old [Occupation] who is known for [Your Character Trait/Ability]. In
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France by population and is the capital of France. It is also the seat of the government, the majority of the country's institutions, the most visited city in the world, and the center of France's culture and identity. It is known for its museums, theaters, and iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Champs-Élysées. Paris is a UNESCO World Heritage site and a major transportation hub. The city is known for its food, fashion, and music, as well as its art, literature, and historical significance.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are a few possibilities that could shape the direction of the technology:
    
    1. Improved AI will lead to increased automation: As AI continues to improve, it may become more efficient at performing tasks that were previously done by humans. This could lead to a shift towards more automated processes, where AI is used to perform tasks that are more routine or repetitive.
    
    2. AI will become more prevalent in everyday life: As AI becomes more advanced, it may become more integrated into our daily lives. For example, we could see more widespread use of smart devices that use AI to assist with tasks such as grocery shopping, scheduling appointments,


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

     I

    'm

     a

     [

    Title

    ]

     at

     [

    Company

    ].

     I

    'm

     currently

     [

    Job

     Title

    ]

     at

     [

    Company

    ],

     but

     I

    'm

     also

     looking

     forward

     to

     taking

     on

     [

    Position

    ]

     at

     [

    New

     Company

    ].

     Thank

     you

     for

     taking

     the

     time

     to

     meet

     me

    ,

     and

     I

     look

     forward

     to

     connecting

     with

     you

    !

     

    🌟

    
    


    [

    Name

    ]

     

    ✨

      


    [

    Company

    ]

     

    ✨

    
    


    ---
    


    Feel

     free

     to

     adjust

     the

     details

     to

     better

     fit

     your

     personal

     brand

     and

     company

    .

     Let

     me

     know

     if

     you

     need

     any

     adjustments

     to

     the

     introduction

    !

     

    📖

    ✨

    
    


    ---
    


    Please

     let

     me

     know

     if

     you

    'd

     like

     me

     to

     provide

     more

     details

     or

     have

     you

     already

     met

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     center

     of

     the

     country

    .

     It

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    ,

     as

     well

     as

     its

     historical

     importance

     in

     French

     culture

     and

     politics

    .

     It

     is

     the

     largest

     and

     most

     populous

     city

     in

     Europe

     and

     a

     major

     center

     of

     education

    ,

     business

    ,

     and

     entertainment

    .

     Paris

     is

     also

     known

     for

     its

     vibrant

     street

     food

     scene

     and

     its

     cultural

     exchanges

     with

     neighboring

     countries

    .

     It

     is

     home

     to

     many

     renowned

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     is

     known

     for

     its

     annual

     fashion

     and

     food

     festivals

    .

     Paris

     is

     a

     major

     cultural

     hub

     that

     attracts

     visitors

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     highly

     disruptive

    ,

     with

     a

     wide

     range

     of

     possible

     outcomes

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Integration

     with

     human

     cognition

    :

     AI

     is

     likely

     to

     become

     more

     integrated

     with

     human

     cognition

    ,

     with

     machines

     able

     to

     perform

     tasks

     that

     typically

     require

     human

     creativity

     and

     insight

    .

     This

     could

     lead

     to

     new

     approaches

     to

     machine

     learning

     and

     natural

     language

     processing

     that

     are

     not

     possible

     with

     traditional

     AI

     systems

    .
    


    2

    .

     Personal

    ization

     and

     automation

    :

     AI

     is

     likely

     to

     continue

     to

     personalize

     and

     automate

     tasks

    ,

     leading

     to

     new

     opportunities

     for

     businesses

     to

     improve

     efficiency

     and

     reduce

     costs

    .

     However

    ,

     it

    's

     also

     possible

     that

     AI

     could

     become

     too

     personal

     and

     result

     in

    



```python
llm.shutdown()
```
