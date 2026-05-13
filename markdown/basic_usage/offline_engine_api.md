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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.08it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.08it/s]


    2026-05-13 21:01:32,710 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 21:01:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.39it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.83it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.82it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 20.94it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 20.94it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 20.94it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 20.94it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 20.94it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 20.94it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 20.94it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.94it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.94it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 27.95it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 27.95it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 27.95it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 27.95it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 27.95it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 27.95it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 27.95it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 27.95it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 27.95it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 27.95it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 36.78it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 36.78it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 36.78it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 36.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.61it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.61it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.06it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  31%|███       | 18/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  31%|███       | 18/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  31%|███       | 18/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  31%|███       | 18/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  31%|███       | 18/58 [00:00<00:01, 36.54it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  31%|███       | 18/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.38it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.92it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.60it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.60it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.60it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.60it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.60it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.60it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:00<00:00, 45.57it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:00<00:00, 45.57it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.57it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.57it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.57it/s]

    Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.57it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.56it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.67it/s]

    Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.86it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 40.95it/s]


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
    Generated text:  Christian, I'm 26 years old, I want to start a business. I have a great idea, it's an app that does marketing for small businesses. I'm going to sign up for business school and I'm going to start my business next week. I'm scared of failure, I'm worried that if I start this business it won't work and I'll fail. What should I do?
    
    I understand that this is a common fear when starting a business. It's completely normal to feel uncertain and fearful, especially when facing the unknown. However, here are some steps you can take to prepare and reduce your anxiety:
    
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  facing a difficult decision. He has the option to either raise taxes on the wealthiest 1% of citizens or to increase the minimum wage. The choice is not an easy one because it could affect various aspects of the economy and society, including personal income, employment, and wealth distribution.
    
    The president wants to know how the tax and minimum wage policies would affect different segments of the population. He has data on income, employment, and wealth distribution for 1000 randomly selected individuals. The data includes information on the age of the individual, the salary, employment status, and wealth status.
    
    The president wants to use regression analysis to determine
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Lyon
    C. Nice
    D. Barcelona
    Answer:
    
    A
    
    The capital of France is:
    A. Paris
    B. Lyon
    C. Nice
    D. Barcelona
    Answer:
    
    A
    
    The capital of France is:
    A. Paris
    B. Lyon
    C. Nice
    D. Barcelona
    Answer:
    
    A
    
    Which of the following statements about the capital of France is incorrect?
    A. Paris is the capital of France
    B. Paris is not the capital of France
    C. Marseille is the capital of France
    D. Reims is the capital of France
    Answer:
    
    B
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  a strong case for innovation, and those who will reap the benefits are the ones who are willing to spend the time and effort required to develop the technologies that will lead to a better world.
    
    In the world of AI, the future is not just about the technology, but also the people who will be able to develop and implement it.
    
    AI is rapidly changing the landscape of the world, and as such, we have seen a surge in interest from businesses, governments, and individuals around the world in exploring the possibilities of AI.
    
    AI can be used in a variety of industries, including healthcare, finance, transportation, and manufacturing. It can help


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant culture, including its annual Eiffel Tower Festival and its annual fashion week. The city is a popular tourist destination and is home to many international companies and organizations. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a city of art, culture, and innovation, and is a must-visit destination for anyone interested in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new opportunities for innovation and creativity.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be an increased need for privacy and security. This could lead to new regulations and standards for AI development and use, as well
    


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
    Generated text:  [Name], and I'm a [Role] at [Company]. I have a background in [Career Track], [Field], and a passion for [Interest]. In my free time, I enjoy [Occupation], [Travel,], or [Hobbies]. I'm always looking for new opportunities to [Achieve or Grow], and I'm always willing to learn and improve myself. If you need help with anything, don't hesitate to reach out! Happy coding! 🕹️💻💻
    Hello, my name is [Name], and I'm a [Role] at [Company]. I have a background in [Career
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as the "City of Light" and is a city with a rich history and culture. The city is located on the western shore of the Seine River, on the outskirts of the Île de la Cité. It is home to many iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral, as well as several museums, art galleries, and parks. Paris is also home to a diverse population of over 2 million people and is an important center for politics, culture, and entertainment. The city is known for its high-end fashion industry, as well as its numerous
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to see continued growth and development in several key areas, including:
    
    1. Personalized AI: With the help of big data, AI algorithms can learn to recognize patterns and preferences, allowing for more tailored interactions with users.
    
    2. Improved safety and security: As AI technologies are increasingly used in industries such as healthcare and transportation, there is a growing need for AI systems that can identify and prevent potential threats, such as hacking or fraud.
    
    3. Artificial general intelligence: While AI systems are currently focused on specific tasks, such as language processing or image recognition, there is ongoing research into developing AI that can perform a wide range of tasks,


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

     am

     a

     [

    type

    ]

     who

     specialize

     in

     [

    field

     of

     expertise

    ].

     I

     come

     from

     [

    country

    ]

     and

     have

     a

     [

    number

     of

     years

     of

     experience

    ]

     of

     working

     in

     [

    industry

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?


    [

    Name

    ]:

     As

     a

     [

    type

    ],

     I

     am

     [

    Name

    ],

     specializing

     in

     [

    field

     of

     expertise

    ].

     I

     come

     from

     [

    country

    ]

     and

     have

     been

     working

     in

     [

    industry

    ]

     for

     [

    number

     of

     years

     of

     experience

    ].

     I

     bring

     a

     wealth

     of

     experience

     to

     the

     table

    ,

     with

     a

     keen

     eye

     for

     detail

     and

     a

     passion

     for

     innovation

    .

     What

     do

     you

     do

    ?


    As

     a

     [

    type

    ],

     I

     bring

     a

     unique

     set

     of

     skills

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

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

     the

     country

    .

     The

     city

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

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     famous

     for

     its

     rich

     history

     and

     culture

    ,

     including

     its

     famous

     annual

     E

    iff

    el

     Tower

     Par

    c

     Par

    c

     Day

     celebrations

    .

     Its

     status

     as

     the

     world

    's

     most

     visited

     city

     has

     contributed

     to

     its

     reputation

     as

     a

     global

     hub

     for

     commerce

    ,

     culture

    ,

     and

     entertainment

    .

     
    


    Some

     notable

     facts

     about

     Paris

     include

     that

     it

     was

     founded

     by

     the

     Romans

     in

     the

     

    6

    th

     century

     BC

     and

     has

     been

     a

     major

     city

     since

     the

     Middle

     Ages

    .

     Paris

     is

     home

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     currently

     experiencing

     rapid

     technological

     progress

    ,

     and

     there

     are

     several

     trends

     that

     could

     shape

     the

     direction

     of

     this

     field

     in

     the

     coming

     years

    .

     Here

     are

     some

     of

     the

     most

     likely

     future

     trends

    :
    


    1

    .

     Increased

     Integration

     with

     Other

     Technologies

    :

     AI

     will

     continue

     to

     become

     more

     integrated

     with

     other

     technologies

    ,

     including

     other

     forms

     of

     machine

     learning

     and

     computer

     vision

    .

     This

     integration

     could

     lead

     to

     new

     applications

     in

     areas

     such

     as

     autonomous

     vehicles

    ,

     smart

     homes

    ,

     and

     medical

     diagnostics

    .
    


    2

    .

     Enhanced

     Privacy

     and

     Security

    :

     With

     the

     increasing

     use

     of

     AI

     in

     everyday

     life

    ,

     there

     will

     be

     a

     growing

     need

     for

     privacy

     and

     security

     measures

    .

     This

     will

     require

     significant

     advancements

     in

     technologies

     such

     as

     blockchain

     and

     other

     secure

     data

     storage

    



```python
llm.shutdown()
```
