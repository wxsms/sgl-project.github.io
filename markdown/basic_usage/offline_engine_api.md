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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.25it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.24it/s]


    2026-05-06 13:28:21,450 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 13:28:21] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.25it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.25it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.25it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.25it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.25it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.25it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.25it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.25it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.25it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.25it/s]

    Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.25it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.56it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.56it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.56it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.56it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.56it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.56it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.56it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.56it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.56it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.73it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.33it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.33it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.33it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   7%|▋         | 4/58 [00:00<00:03, 17.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   7%|▋         | 4/58 [00:00<00:03, 17.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   7%|▋         | 4/58 [00:00<00:03, 17.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):  10%|█         | 6/58 [00:00<00:02, 17.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:02, 17.81it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:02, 17.81it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.90it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  31%|███       | 18/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  31%|███       | 18/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  31%|███       | 18/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  31%|███       | 18/58 [00:00<00:01, 26.34it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  31%|███       | 18/58 [00:00<00:01, 26.34it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.44it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.44it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.44it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.44it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:01, 28.44it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.14it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.14it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.14it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.14it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.14it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.14it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.73it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.73it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.73it/s]Capturing num tokens (num_tokens=352 avail_mem=74.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.73it/s]Capturing num tokens (num_tokens=320 avail_mem=74.93 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.73it/s]Capturing num tokens (num_tokens=288 avail_mem=74.77 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.73it/s]Capturing num tokens (num_tokens=288 avail_mem=74.77 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=256 avail_mem=63.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=240 avail_mem=61.65 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=224 avail_mem=61.64 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=208 avail_mem=61.64 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.28it/s]Capturing num tokens (num_tokens=192 avail_mem=61.64 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.28it/s]

    Capturing num tokens (num_tokens=192 avail_mem=61.64 GB):  71%|███████   | 41/58 [00:01<00:00, 38.90it/s]Capturing num tokens (num_tokens=176 avail_mem=61.64 GB):  71%|███████   | 41/58 [00:01<00:00, 38.90it/s]Capturing num tokens (num_tokens=160 avail_mem=61.63 GB):  71%|███████   | 41/58 [00:01<00:00, 38.90it/s]Capturing num tokens (num_tokens=144 avail_mem=61.63 GB):  71%|███████   | 41/58 [00:01<00:00, 38.90it/s]Capturing num tokens (num_tokens=128 avail_mem=61.63 GB):  71%|███████   | 41/58 [00:01<00:00, 38.90it/s]Capturing num tokens (num_tokens=112 avail_mem=61.63 GB):  71%|███████   | 41/58 [00:01<00:00, 38.90it/s]Capturing num tokens (num_tokens=112 avail_mem=61.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.56it/s]Capturing num tokens (num_tokens=96 avail_mem=61.62 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.56it/s] Capturing num tokens (num_tokens=80 avail_mem=61.62 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.56it/s]Capturing num tokens (num_tokens=64 avail_mem=61.61 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.56it/s]Capturing num tokens (num_tokens=48 avail_mem=61.61 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.56it/s]Capturing num tokens (num_tokens=32 avail_mem=61.61 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.56it/s]

    Capturing num tokens (num_tokens=32 avail_mem=61.61 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=28 avail_mem=61.60 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=24 avail_mem=61.60 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=20 avail_mem=61.60 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=16 avail_mem=61.60 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=12 avail_mem=61.59 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=12 avail_mem=61.59 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=8 avail_mem=61.59 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.69it/s] Capturing num tokens (num_tokens=4 avail_mem=61.58 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=4 avail_mem=61.58 GB): 100%|██████████| 58/58 [00:01<00:00, 33.14it/s]


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
    Generated text:  Niki and I am a C++ programmer. I have been working on a project that involves programming a simple game using C++ and my favorite programming language is C++. However, I'm having trouble understanding how to properly utilize the `std::cout` and `std::cin` classes in C++. Could you provide me with some examples and explanations on how to use these classes in C++?
    
    Sure, I'd be happy to help you understand how to use the `std::cout` and `std::cin` classes in C++. Let's start with the `std::cout` class.
    
    **Example 1: Using std
    ===============================
    Prompt: The president of the United States is
    Generated text:  a position that is held by one person. 
    A: Correct
    B: Incorrect
    C: 
    D: To determine the correct answer, let's analyze the statement step by step:
    
    1. **Identify the role**: The president of the United States is a position held by one person. This is a fundamental political fact.
    
    2. **Consider the definition**: The term "position" typically refers to a designated or assigned role or position. In this case, the position of president is one of the many roles and assignments that can be assigned to a person.
    
    3. **Evaluate the statement**: The statement claims that the president of
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Saint-Denis
    C. Lyon
    D. Marseille
    Answer:
    
    A
    
    Male, 26 years old, with a history of type 1 diabetes for 10 years, primarily treated with insulin therapy. The patient developed hypoglycemia due to increased blood sugar levels, and the blood glucose level is 2.0 mmol/L. The correct approach to managing this situation is
    A. Increase the dose of insulin to raise blood glucose
    B. Use a large dose of insulin to raise blood glucose
    C. Use a small dose of insulin to raise blood glucose
    
    ===============================
    Prompt: The future of AI is
    Generated text:  a rapidly evolving landscape with various applications that are rapidly evolving in different industries, each with its own unique characteristics. In the field of healthcare, AI has the potential to revolutionize the way we treat and diagnose diseases, but it also has the potential to pose a significant ethical and privacy concern.
    One of the main challenges that arise in the use of AI in healthcare is the need to ensure that patient data is protected and secure. This includes measures such as encryption, access controls, and regular audits to detect and mitigate any breaches or vulnerabilities.
    Another ethical concern is around the potential for AI to exacerbate existing inequalities in the healthcare system. For example


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast who loves to [Describe Your Hobby or Passion]. I'm always looking for new experiences and adventures, and I'm always eager to learn and grow. I'm a [Favorite Thing to Do] person who enjoys [Describe Your Favorite Thing to Do]. I'm a [Favorite Book or Movie] fan who loves to [Describe Your Favorite Book or Movie]. I'm a [Favorite Music Artist] lover who enjoys [Describe Your Favorite Music Artist]. I'm a [Favorite Sport] enthusiast who loves to [Describe Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Square. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. It is a popular tourist destination and a major economic center in Europe. The city is known for its cuisine, fashion, and art, and is home to many famous landmarks and museums. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. The city is also known for its annual festivals and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends that are expected to shape the development of AI in the coming years:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare in the coming years.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection and risk management. As AI technology continues to improve, we can
    


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
    Generated text:  [Name] and I am a [occupation]. I am an AI-powered personal assistant who specializes in [specific area of expertise], [Name], based in [city or location]. I am here to assist you with a wide variety of tasks, such as [list tasks that you can do with me]. I am available 24/7, and my hourly rate is [rate]. I have been trained on [number of algorithms] and [number of databases], which allow me to provide personalized responses and support. I am excited to help you with any questions you may have and to look forward to helping you achieve your goals. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also a city of innovation, with major companies such as Apple and Google operating from its headquarters. The city is known for its diverse cultural scene, including the world-renowned Louvre Museum and the annual Eiffel Tower walk. Paris is a popular tourist destination, known for its fashion, art, and cuisine, attracting millions of visitors each year. Overall, Paris is a city of contrasts, culture, and history. The city of Paris has become a global cultural and economic hub, and continues
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly promising, with potential applications in various fields including healthcare, finance, education, transportation, and more. Here are some possible future trends in AI:
    
    1. Increased personalization: AI will allow machines to learn from user data and adapt to their needs, resulting in more personalized and efficient products and services.
    
    2. Improved decision-making: AI will continue to improve its ability to analyze and predict outcomes, leading to more informed and ethical decision-making in various industries.
    
    3. Autonomous vehicles: The integration of AI in autonomous vehicles could revolutionize transportation by reducing accidents and improving safety.
    
    4. Improved health care: AI will help doctors and researchers analyze


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

     [

    job

     title

    ]

     at

     [

    company

     name

    ].

     I

     have

     been

     in

     this

     field

     for

     [

    number

     of

     years

    ]

     years

     and

     have

     a

     passion

     for

     [

    reason

     for

     interest

     in

     the

     field

    ].


    I

     hope

     you

     find

     this

     introduction

     neutral

    ,

     and

     that

     you

    'll

     enjoy

     learning

     more

     about

     me

     as

     an

     individual

    .

     Feel

     free

     to

     ask

     any

     questions

     you

     may

     have

    .

     I

    'm

     [

    name

    ]

    !

     

    🌍

    ✈

    ️

     #

    self

    int

    roduction

    


    [

    Number

     of

     Years

    ]

     years

     of

     experience

     in

     the

     field

     of

     [

    job

     title

    ].

     

    📜

     #

    Experience

    


    My

     passion

     for

     [

    reason

     for

     interest

     in

     the

     field

    ]

     has

     driven

     me

     to

     continuously

     improve

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    How

     is

     the

     capital

     of

     France

     different

     from

     other

     capitals

    ?

     
    


    Choose

     your

     answer

     from

    :

     A

    ).

     The

     capital

     of

     France

     has

     the

     same

     number

     of

     inhabitants

     as

     all

     the

     other

     capitals

    .

     B

    ).

     The

     capital

     of

     France

     is

     closer

     to

     the

     equ

    ator

    .

     C

    ).

     The

     capital

     of

     France

     is

     more

     developed

    .

     D

    ).

     The

     capital

     of

     France

     is

     more

     expensive

    .

     


    I

     chose

    :

     B

    ).

     The

     capital

     of

     France

     is

     closer

     to

     the

     equ

    ator

    .
    


    The

     capital

     of

     France

     is

     closer

     to

     the

     equ

    ator

    .


    You

     are

     an

     AI

     assistant

    .

     Provide

     a

     detailed

     answer

     so

     user

     don

    ’t

     need

     to

     search

     outside

     to

     understand

     the

     answer

    .

     Your

     answer

     should

     include

     an

     explanation

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     several

     trends

     that

     could

     significantly

     impact

     the

     field

    .

     Here

     are

     some

     possible

     trends

     that

     may

     emerge

    :
    


    1

    .

     Increased

     automation

    :

     AI

     will

     continue

     to

     evolve

     and

     become

     increasingly

     capable

     of

     performing

     tasks

     that

     were

     previously

     thought

     to

     be

     impossible

    ,

     such

     as

     image

     recognition

    ,

     natural

     language

     processing

    ,

     and

     decision

    -making

    .
    


    2

    .

     More

     integration

     with

     everyday

     objects

    :

     AI

     will

     continue

     to

     become

     more

     integrated

     into

     our

     lives

    ,

     with

     more

     devices

     and

     sensors

     being

     able

     to

     process

     and

     analyze

     data

     in

     real

    -time

    .

     This

     could

     lead

     to

     a

     greater

     ability

     to

     predict

     and

     anticipate

     human

     behavior

     and

     needs

    .
    


    3

    .

     AI

     will

     continue

     to

     evolve

    :

     AI

     will

     continue

     to

     be

     refined

     and

     improved

    ,

     with

    



```python
llm.shutdown()
```
