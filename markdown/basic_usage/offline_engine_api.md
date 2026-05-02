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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.36it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.35it/s]


    2026-05-02 04:33:28,112 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 04:33:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]

    Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:15,  3.10it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:15,  3.10it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:15,  3.10it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:15,  3.10it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:15,  3.10it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:15,  3.10it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:15,  3.10it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:15,  3.10it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:04<00:15,  3.10it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:05,  7.30it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:05,  7.30it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:05,  7.30it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:05,  7.30it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:05,  7.30it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:04<00:05,  7.30it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:05<00:05,  7.30it/s]

    Compiling num tokens (num_tokens=832):  29%|██▉       | 17/58 [00:05<00:05,  7.30it/s]Compiling num tokens (num_tokens=768):  29%|██▉       | 17/58 [00:05<00:05,  7.30it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:02, 12.57it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:02, 12.57it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:02, 12.57it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:02, 12.57it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:02, 12.57it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:05<00:02, 12.57it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:05<00:02, 12.57it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:05<00:02, 12.57it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:05<00:02, 12.57it/s]Compiling num tokens (num_tokens=352):  43%|████▎     | 25/58 [00:05<00:02, 12.57it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:01, 19.68it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:01, 19.68it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:01, 19.68it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:01, 19.68it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:01, 19.68it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:01, 19.68it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:01, 19.68it/s]

    Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:05<00:01, 19.68it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:05<00:01, 19.68it/s]Compiling num tokens (num_tokens=160):  59%|█████▊    | 34/58 [00:05<00:01, 19.68it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 27.77it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 34.09it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 34.09it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 34.09it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 34.09it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 34.09it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 34.09it/s]

    Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 34.09it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 34.09it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 19.12it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 19.12it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 19.12it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 19.12it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.21 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.20 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.82it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.16 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.82it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.16 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.82it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.16 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.82it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.34it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.34it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.34it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.15 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.34it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.13 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.34it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.13 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.97it/s]Capturing num tokens (num_tokens=960 avail_mem=74.15 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.97it/s] Capturing num tokens (num_tokens=896 avail_mem=74.14 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.97it/s]Capturing num tokens (num_tokens=832 avail_mem=74.14 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.97it/s]Capturing num tokens (num_tokens=768 avail_mem=74.14 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.97it/s]Capturing num tokens (num_tokens=768 avail_mem=74.14 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.02it/s]Capturing num tokens (num_tokens=704 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.02it/s]Capturing num tokens (num_tokens=640 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.02it/s]Capturing num tokens (num_tokens=576 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.02it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.11 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.02it/s]Capturing num tokens (num_tokens=512 avail_mem=74.11 GB):  50%|█████     | 29/58 [00:00<00:00, 34.93it/s]Capturing num tokens (num_tokens=480 avail_mem=74.13 GB):  50%|█████     | 29/58 [00:00<00:00, 34.93it/s]Capturing num tokens (num_tokens=448 avail_mem=74.13 GB):  50%|█████     | 29/58 [00:00<00:00, 34.93it/s]Capturing num tokens (num_tokens=416 avail_mem=74.13 GB):  50%|█████     | 29/58 [00:00<00:00, 34.93it/s]Capturing num tokens (num_tokens=384 avail_mem=74.12 GB):  50%|█████     | 29/58 [00:00<00:00, 34.93it/s]Capturing num tokens (num_tokens=352 avail_mem=74.12 GB):  50%|█████     | 29/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=352 avail_mem=74.12 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.12it/s]Capturing num tokens (num_tokens=320 avail_mem=74.11 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.12it/s]Capturing num tokens (num_tokens=288 avail_mem=74.11 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.12it/s]

    Capturing num tokens (num_tokens=256 avail_mem=74.11 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.12it/s]Capturing num tokens (num_tokens=240 avail_mem=74.10 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.12it/s]Capturing num tokens (num_tokens=224 avail_mem=74.10 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.12it/s]Capturing num tokens (num_tokens=224 avail_mem=74.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=208 avail_mem=74.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=192 avail_mem=74.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=176 avail_mem=74.09 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=160 avail_mem=74.09 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=160 avail_mem=74.09 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.78it/s]Capturing num tokens (num_tokens=144 avail_mem=74.09 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.78it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.78it/s]Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.78it/s]Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.78it/s] Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  81%|████████  | 47/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=80 avail_mem=74.08 GB):  81%|████████  | 47/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  81%|████████  | 47/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=48 avail_mem=74.07 GB):  81%|████████  | 47/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  81%|████████  | 47/58 [00:01<00:00, 35.79it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.24it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.24it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.24it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.24it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.24it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.06it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.06it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.06it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.06it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 34.03it/s]


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
    Generated text:  Ayen and I am a young entrepreneur who is passionate about starting a new business. I believe that entrepreneurship is a way to make a positive impact on the world and I am determined to achieve this goal. As of today, I am planning to start a new business that focuses on sustainable agriculture.
    My business concept is to develop and sell organic, sustainably grown produce that is accessible to all income levels and consumed by a diverse group of consumers. I am excited about the potential of my business and I am confident that I can achieve this goal.
    Please provide me with a detailed business plan for my sustainable agriculture business that includes all necessary steps,
    ===============================
    Prompt: The president of the United States is
    Generated text:  appointed by the ____.
    A. President
    B. Congress
    C. Supreme Court
    D. Senate
    
    To determine the correct answer, let's break down the process of the U. S. Presidential Election:
    
    1. The U. S. presidential election is held every four years.
    2. The U. S. Constitution outlines the process of selecting the President and Vice President.
    3. The President is chosen by the Vice President.
    4. The Vice President is also chosen by the House of Representatives, but the choice is made by the president of the Senate.
    5. Therefore, the President must be appointed by the Senate.
    
    Given
    ===============================
    Prompt: The capital of France is
    Generated text:  located on the __ of the continent.
    A. Eastern side
    B. Western side
    C. Southern side
    D. Northern side
    Answer:
    
    B
    
    A. 1/2
    B. 1/4
    C. 1/8
    D. 1/16
    Answer:
    
    C
    
    According to the 2007 edition of the French Civil Code, what is the minimum age for being able to purchase a property?
    A. 18 years old
    B. 21 years old
    C. 25 years old
    D. 30 years old
    Answer:
    
    A
    ===============================
    Prompt: The future of AI is
    Generated text:  still uncertain, but one thing is certain: it will have a significant impact on the healthcare industry. The use of AI in healthcare has been on the rise for the past few years, and it is expected to continue in the future.
    AI in healthcare can be used in a variety of ways, such as in diagnosis and treatment, medical research, patient management, and more. It can help doctors and nurses to make more accurate diagnoses, improve treatment outcomes, and accelerate the process of medical research.
    The use of AI in healthcare has also opened up new possibilities for patient care, such as telemedicine and predictive analytics. It can help doctors to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your personality or skills]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new opportunities to grow and learn. What's your favorite hobby or activity? I'm always looking for new challenges and opportunities to learn and grow. What's your favorite book or movie? I'm always looking for new experiences and opportunities to learn and grow. What's your favorite place to go?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the annual Eiffel Tower Festival. It is the largest city in France and the third largest in the world by population. Paris is also the birthplace of many famous French artists, writers, and composers. The city is home to the Louvre Museum, the Notre-Dame Cathedral, and the Champs-Élysées. It is also the seat of the French government and the headquarters of the French Foreign Ministry. Paris is a vibrant and diverse city with a rich cultural heritage and a strong sense of French identity. The city is known for its romantic atmosphere, historical
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead to more transparent and accountable AI systems that are designed to
    


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
    Generated text:  [Your Name], and I'm a [Your Job Title] at [Your Company], where I work diligently to [Your Company's Mission or Goal]. I'm passionate about [Your Profession or Career], and I'm always looking for ways to grow and improve in my field. What's your story, and what's the most memorable moment in your journey so far? Please include a short personal quote that reflects your character and your values. Welcome, [Name], to our team. I'm excited to meet you and learn all about you! Here's a little quote I came across that's reflective of my values and character: "The
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    1. The city is home to the Eiffel Tower and the Louvre Museum.
    2. It is the largest city in France by population.
    3. Paris is known for its rich history, art, and gastronomy. 
    
    Note: This statement is factual and accurate. However, it does not provide a vivid description or description of the city that would allow for a personal experience or reflection. Can you provide a more vivid description of Paris's historical significance and cultural impact? 
    For example, you could focus on the architectural landmarks and museums that make Paris unique, or the way its people interact with each other and the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving and there are several potential trends that are currently being explored and researched by experts, including:
    
    1. Advancements in Machine Learning: With the help of advanced algorithms, machine learning techniques will continue to improve, allowing for more accurate and complex predictions and decision-making.
    
    2. Increased Use of AI in Healthcare: AI is already being used in a wide range of healthcare applications, from diagnosing and treating diseases to optimizing drug development and predicting patient outcomes.
    
    3. Increased Integration of AI into Autonomous Vehicles: As autonomous vehicles become more advanced, there is a potential for AI to be integrated into their systems, allowing for more efficient and safer transportation


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

    ].

     I

    'm

     a

     [

    Person

    ality

    /

    Industry

    ]

     with

     [

    Number

     of

     years

     working

     in

     this

     field

    ].

     I

    've

     [

    Number

     of

     years

     of

     experience

    ]

     in

     this

     industry

     and

     have

     [

    Number

     of

     successful

     projects

    ]

     projects

     completed

    .

     I

    'm

     [

    Int

    imid

    ating

    /

    Co

    zy

    ]

     at

     work

     and

     enjoy

     [

    Major

     challenge

     you

    're

     currently

     facing

    ].

     I

    'm

     always

     eager

     to

     learn

     and

     expand

     my

     skills

    ,

     so

     I

    'm

     looking

     forward

     to

     [

    Int

    ention

     for

     future

     projects

    ].

     My

     motto

     is

     "

    Keep

     learning

    ,

     keep

     progressing

    ,

     keep

     achieving

    ,

     and

     keep

     moving

     forward

    ".

     Lastly

    ,

     I

    'm

     a

     [

    Person

    ality

    ]

     who

     always

     strive

     to

     be

     [

    Positive

    /

    End

    earing

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     Se

    ine

     River

     in

     the

     center

     of

     the

     country

    .

     It

     is

     the

     largest

     city

     and

     the

     largest

     metropolitan

     area

     in

     Europe

     and

     is

     one

     of

     the

     world

    's

     most

     popular

     tourist

     destinations

    .

     
    


    Paris

     has

     been

     a

     major

     center

     of

     culture

    ,

     politics

    ,

     and

     industry

     since

     the

     

    1

    2

    th

     century

    ,

     and

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

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Mont

    mart

    re

    .

     The

     city

     is

     home

     to

     many

     important

     historical

     and

     cultural

     landmarks

    ,

     including

     the

     Lou

    vre

    ,

     Notre

     Dame

     Cathedral

    ,

     the

     Palace

     of

     Vers

    ailles

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     also

     has

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     range

     of

     emerging

     trends

     and

     technologies

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     and

     responsible

     AI

    :

     As

     more

     and

     more

     AI

     systems

     become

     involved

     in

     decision

    -making

     processes

    ,

     there

     is

     a

     growing

     awareness

     of

     the

     potential

     impact

     of

     AI

     on

     society

    .

     As

     a

     result

    ,

     there

     may

     be

     an

     increased

     focus

     on

     ethical

     and

     responsible

     AI

     that

     ensures

     that

     AI

     systems

     are

     used

     in

     ways

     that

     benefit

     all

     stakeholders

    ,

     including

     people

    ,

     businesses

    ,

     and

     the

     environment

    .
    


    2

    .

     Integration

     with

     other

     technologies

    :

     The

     use

     of

     AI

     is

     likely

     to

     become

     more

     integrated

     with

     other

     technologies

    ,

     including

     machine

     learning

    ,

     robotics

    ,

     and

     blockchain

    ,

     as

    



```python
llm.shutdown()
```
