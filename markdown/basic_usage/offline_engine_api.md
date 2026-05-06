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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.06it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.05it/s]


    2026-05-06 14:26:23,077 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 14:26:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.92it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.92it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 23.92it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.01it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.01it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.01it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.01it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.01it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.01it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.01it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.01it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.61 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.61 GB):   3%|▎         | 2/58 [00:00<00:03, 18.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.60 GB):   3%|▎         | 2/58 [00:00<00:03, 18.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.60 GB):   3%|▎         | 2/58 [00:00<00:03, 18.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.60 GB):   3%|▎         | 2/58 [00:00<00:03, 18.39it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=53.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.59 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.19it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.19it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=53.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.19it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.56 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.56 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.56 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.55 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.55 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.55 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.53 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.51 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]

    Capturing num tokens (num_tokens=960 avail_mem=53.52 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s] Capturing num tokens (num_tokens=896 avail_mem=53.52 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=896 avail_mem=53.52 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=832 avail_mem=53.51 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=768 avail_mem=53.51 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=704 avail_mem=53.48 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=640 avail_mem=53.48 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=576 avail_mem=53.48 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=576 avail_mem=53.48 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.94it/s]Capturing num tokens (num_tokens=512 avail_mem=53.46 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.94it/s]Capturing num tokens (num_tokens=480 avail_mem=53.48 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.94it/s]

    Capturing num tokens (num_tokens=448 avail_mem=53.48 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.94it/s]Capturing num tokens (num_tokens=416 avail_mem=53.47 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.94it/s]Capturing num tokens (num_tokens=384 avail_mem=53.47 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.94it/s]Capturing num tokens (num_tokens=384 avail_mem=53.47 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.00it/s]Capturing num tokens (num_tokens=352 avail_mem=53.47 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.00it/s]Capturing num tokens (num_tokens=320 avail_mem=53.46 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.00it/s]Capturing num tokens (num_tokens=288 avail_mem=53.46 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.00it/s]Capturing num tokens (num_tokens=256 avail_mem=53.46 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.00it/s]Capturing num tokens (num_tokens=240 avail_mem=53.45 GB):  57%|█████▋    | 33/58 [00:01<00:00, 42.00it/s]Capturing num tokens (num_tokens=240 avail_mem=53.45 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=224 avail_mem=53.45 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=208 avail_mem=53.44 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.46it/s]

    Capturing num tokens (num_tokens=192 avail_mem=53.44 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=176 avail_mem=53.44 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=160 avail_mem=53.44 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=160 avail_mem=53.44 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=144 avail_mem=53.43 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=128 avail_mem=53.43 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=112 avail_mem=53.43 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=96 avail_mem=53.43 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.51it/s] Capturing num tokens (num_tokens=80 avail_mem=53.42 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=80 avail_mem=53.42 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=64 avail_mem=53.42 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=48 avail_mem=53.41 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.78it/s]

    Capturing num tokens (num_tokens=32 avail_mem=53.41 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=28 avail_mem=53.41 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=24 avail_mem=53.41 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=24 avail_mem=53.41 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=20 avail_mem=53.40 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=16 avail_mem=53.40 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=12 avail_mem=53.40 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=8 avail_mem=53.39 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.73it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=53.39 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.73it/s]Capturing num tokens (num_tokens=4 avail_mem=53.39 GB): 100%|██████████| 58/58 [00:01<00:00, 37.80it/s]Capturing num tokens (num_tokens=4 avail_mem=53.39 GB): 100%|██████████| 58/58 [00:01<00:00, 37.75it/s]


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
    Generated text:  Sergio Santos. I am a graduate student in the area of bioinformatics at the University of Illinois at Urbana-Champaign. I work on high throughput sequencing and bioinformatics related problems, and I currently mentor undergraduate students at the University of Illinois at Urbana-Champaign. I am also interested in applications of bioinformatics in healthcare, and in particular in developing and applying computational methods for developing new drugs and therapies. I have a PhD in Bioinformatics from the University of California, Berkeley.
    I am a member of the Bioinformatics Group at the University of Illinois at Urbana-Champaign, and I participate in several bioinformatics projects.
    My research
    ===============================
    Prompt: The president of the United States is
    Generated text:  two years older than the president of Brazil. The president of Brazil is 30 years younger than the president of the United States. If the president of the United States is currently 80 years old, what is the president of Brazil's current age?
    
    To determine the president of Brazil's current age, we need to work through the information given step by step.
    
    1. Identify the current age of the president of the United States.
       The president of the United States is currently 80 years old.
    
    2. Determine the president of Brazil's age based on the given information.
       The president of Brazil is 30 years
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Marseilles
    C. Nice
    D. Marseille
    Answer:
    A
    
    Which of the following is NOT a potential health issue for newborns?
    A. Weight loss
    B. Neonatal sepsis
    C. Congenital heart disease
    D. Lung infection
    Answer:
    C
    
    Which of the following is NOT a characteristic of acute toxicity?
    A. It affects only a few individuals
    B. It is short-lived
    C. It affects most individuals
    D. It is long-lasting
    E. It is irreversible
    Answer:
    A
    
    Which of the following is NOT a situation
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, with companies that invest heavily in research and development. But many companies still find it difficult to overcome the obstacles to making their ideas a reality. Below are three obstacles to being a successful AI company, and how they can be overcome. AI companies that successfully overcome these obstacles will be ahead of the curve in the next decade.
    1. Lack of Scale
    One of the biggest obstacles to AI companies being successful is not having the ability to scale their projects. At the moment, there are only a handful of companies that have taken advantage of the latest developments in AI, and they are all struggling to reach their full potential. If your company


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also known for its cuisine, including its famous French fries and its famous cheese, cheddar. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a city that is both ancient and modern, and it is a city that is constantly evolving.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced ethical considerations: As AI becomes more advanced, there will be increased scrutiny of its ethical implications, including issues such as bias, transparency, and accountability.
    
    3. Greater reliance on data: AI will become more dependent on large amounts of data, which will require more sophisticated data collection and analysis techniques.
    
    4. Increased use of AI in healthcare: AI is already being used in healthcare to
    


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
    Generated text:  [Name] and I'm a [Age] year-old [Gender] [Role]. I'm passionate about [Tell me about your hobby or interest]. I live [City, State] and I've always been [Describe your passion for your hobby or interest]. I'm incredibly [Describe your positive trait] and I always strive to [Tell me about a specific action or accomplishment you've accomplished]. I have a [Describe a specific skill or talent you excel at] and I'm always learning and growing. I'm [Describe your personality] and I have a lot of [Describe a hobby or interest you enjoy doing]. I'm [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historical and cultural center known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also the world's most populous city, with over 20 million residents, making it the largest city in Europe. Paris is a vibrant city known for its art, fashion, and cuisine, and is an important center for French culture and politics. Its name comes from the Latin word for "king," indicating its historical importance. Paris is home to numerous museums, theaters, and historical sites that contribute to the city's reputation as a cultural melting pot. Its status as the capital
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and evolving, and it is difficult to predict exactly where it will lead. However, some possible trends that are expected to shape the AI landscape in the coming years are:
    
    1. Increased development of AI ethics and guidelines: As AI becomes more integrated into our daily lives, it is important to develop and implement guidelines that guide its development and use. This will help to ensure that AI is used ethically and responsibly.
    
    2. Greater integration of AI with other technologies: AI is already being integrated into a wide range of technologies, from smart homes to autonomous vehicles. As these technologies continue to advance, we can expect AI to be even


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

     a

     [

    Your

     Role

    ]

     at

     [

    Your

     Company

    ].

     I

     recently

     graduated

     from

     [

    Your

     University

    ]

     with

     a

     [

    Your

     Degree

    ]

     and

     now

     I

    'm

     looking

     for

     a

     new

     challenge

    .

     As

     an

     AI

     assistant

    ,

     I

    'm

     always

     ready

     to

     help

     people

     with

     a

     wide

     range

     of

     problems

    .

     What

     can

     I

     do

     for

     you

     today

    ?

     [

    Your

     Name

    ]

     


    Repeat

     this

     set

    ence

    ,

     but

     with

     the

     correct

     capital

    ization

    .

     Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ],

     a

     [

    Your

     Role

    ]

     at

     [

    Your

     Company

    ].

     I

     recently

     graduated

     from

     [

    Your

     University

    ]

     with

     a

     [

    Your

     Degree

    ]

     and

     now

     I

    'm

     looking

     for

     a

     new

     challenge

    .

     As

     an

     AI

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    The

     city

    ,

     located

     in

     the

     south

     of

     the

     country

    ,

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     stunning

     views

     of

     the

     city

     and

     the

     surrounding

     countryside

    .

     Paris

     has

     been

     a

     major

     center

     of

     European

     and

     global

     culture

     for

     centuries

     and

     continues

     to

     be

     a

     hub

     for

     art

    ,

     literature

    ,

     and

     other

     forms

     of

     creative

     expression

    .

     The

     city

     is

     also

     home

     to

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    ,

     two

     of

     the

     most

     famous

     landmarks

     in

     the

     world

    .

     
    


    Paris

     is

     also

     home

     to

     numerous

     museums

    ,

     including

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     the

     Lou

    vre

    ,

     and

     the

     Mus

    ée

     de

     l

    '

    Or

    anger

    ie

    .

     It

     also

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     highly

     dynamic

    ,

     with

     a

     number

     of

     trends

     that

     are

     likely

     to

     shape

     its

     development

     in

     the

     coming

     years

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     The

     development

     of

     more

     powerful

     and

     flexible

     AI

     systems

    :

     As

     AI

     systems

     become

     more

     complex

     and

     capable

    ,

     they

     will

     be

     able

     to

     solve

     increasingly

     complex

     problems

     that

     have

     traditionally

     been

     tackled

     by

     human

     experts

    .

     This

     will

     require

     new

     algorithms

     and

     techniques

     to

     be

     developed

     to

     handle

     the

     increased

     complexity

    .
    


    2

    .

     The

     integration

     of

     AI

     into

     various

     industries

    :

     AI

     is

     already

     being

     used

     in

     many

     industries

    ,

     from

     healthcare

     and

     finance

     to

     manufacturing

     and

     transportation

    .

     As

     these

     industries

     continue

     to

     automate

     and

     digital

    ize

    ,

     they

     will

    



```python
llm.shutdown()
```
