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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.98it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.98it/s]


    2026-05-03 14:22:25,588 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-03 14:22:25] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:05,  5.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:05,  5.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:05,  5.35s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:05,  5.35s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:05,  5.35s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:14,  3.32it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:14,  3.32it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:14,  3.32it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:14,  3.32it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:14,  3.32it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:14,  3.32it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:14,  3.32it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:14,  3.32it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:14,  3.32it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:14,  3.32it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:05,  7.45it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:05,  7.45it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:05,  7.45it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:05,  7.45it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:05,  7.45it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:05,  7.45it/s]

    Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:05,  7.45it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:05,  7.45it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 11.39it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 11.39it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 11.39it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 11.39it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:02, 11.39it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:02, 11.39it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:02, 11.39it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:02, 11.39it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:05<00:02, 11.39it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:05<00:02, 11.39it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 17.83it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 17.83it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 17.83it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 17.83it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 17.83it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 17.83it/s]

    Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 17.83it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 17.83it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 17.83it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:00, 24.22it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:00, 24.22it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 24.22it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:00, 24.22it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:06<00:00, 24.22it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:06<00:00, 24.22it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:06<00:00, 24.22it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:06<00:00, 24.22it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:06<00:00, 24.22it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:06<00:00, 24.22it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:06<00:00, 32.45it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:06<00:00, 32.45it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:06<00:00, 32.45it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:06<00:00, 32.45it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:06<00:00, 32.45it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:06<00:00, 32.45it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.94it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.94it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.94it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.31it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.31it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.12it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.12it/s]

    Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.24it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.24it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.24it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.24it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.24it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.40it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.40it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.40it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.40it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.40it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.79it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.79it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.79it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.79it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.79it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.79it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.63it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.63it/s] Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  81%|████████  | 47/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 36.83it/s]

    Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.41it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.41it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.41it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 34.65it/s]


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
    Generated text:  Mark, I am from the United States and I am 18 years old. I am studying at the University of the Philippines. I am an English major and I hope to one day study in the United States. 
    
    As a person who has visited many countries in my life, I know that it is important to be able to communicate with people from other cultures. Having been exposed to different cultures, I know that it is not always easy to communicate with people from a particular country or country group. Therefore, I feel that I will have to learn different languages and understand the communication patterns of each group of people in order to be able
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. Therefore, ____
    A. All presidents of the United States are people
    B. Some presidents of the United States are people
    C. All presidents of the United States are people
    D. Some presidents of the United States are not people
    
    To solve this logical reasoning problem, let's break down the given information and analyze each option step by step.
    
    1. **Identify the type of logical reasoning involved:**
       This problem is an example of identifying the correct classification or type of statement. We need to determine what the statement asserts about the president of the United States.
    
    2. **Analyze each option:**
    
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Please complete the following sentence:
    
    A) In the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of the heart of
    ===============================
    Prompt: The future of AI is
    Generated text:  so bright. We have seen the speed of the development in recent years, and we can expect that more and more AI will be employed in industries, and in the end it will be able to drive the development of society. In the first place, AI is a computer system, and its main task is to learn and adapt to the environment. In other words, it needs to learn from the environment to perform its tasks. In the future, it will not only need to learn from humans, but also need to be able to adapt to the environment.
    A. It will not only need to learn from humans, but also need to be


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Museum. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also known for its fashion industry and is home to many famous fashion designers. The city is also known for its cuisine, with its famous dishes such as croissants, boudin, and escargot. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI, there is likely to be a greater emphasis on ethical AI. This could mean stricter regulations on AI development and deployment, greater transparency in AI decision-making, and increased investment in research and development that focuses on ethical considerations.
    
    2. AI will become more integrated with other technologies: As AI becomes more integrated with other technologies, such as machine learning and blockchain, there is likely to
    


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
    Generated text:  [Name], and I'm a [career field or background] with [number of years of experience]. I'm always seeking to learn more about [job title] and what's possible. If you're interested, feel free to ask me anything! How about you, [Friend/Family/Group/Other]? Let's chat about it! Best, [Name]. Dear [Friend/Family/Group/Other],
    
    I'm [Name] and I've always found my passions in learning and growing as a person. I've been involved in various careers such as [career field], but my true interest lies in [career field]. My
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    I'm sorry, I cannot provide an answer as there is no factual statement about Paris. I don't have access to information about any city besides Paris. Could you please clarify your question or provide more context? Alternatively, you could ask about other cities in France or the United States. I'd be happy to help if you have a specific question or topic in mind. Let me know! 📞
    
    Thanks for the clarification. Is there anything else I can help you with? 😊
    
    Always happy to help! 📊olarity
    
    You're welcome! If you have any other questions or topics, feel free to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be dominated by four trends:
    
      1. Increased automation and precision: As AI becomes more advanced and capable, it is likely to automate a wider range of tasks, from mundane tasks to complex decision-making. This could lead to a decrease in the need for human workers and a rise in automation.
      2. Increased integration with other technologies: As AI becomes more integrated with other technologies, such as sensors, AI algorithms, and blockchain, it is likely to become even more powerful and capable. This could lead to a range of new applications and possibilities for AI, such as smart cities, personalized medicine, and autonomous vehicles.
    


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

    insert

     fictional

     character

    's

     name

    ].

     I

    'm

     a

     [

    insert

     fictional

     character

    's

     age

     or

     profession

    ]

     who

     has

     been

     exploring

     the

     world

     for

     [

    insert

     fictional

     character

    's

     age

     or

     profession

    ].

     I

    'm

     a

     [

    insert

     fictional

     character

    's

     profession

    ]

     with

     a

     passion

     for

     [

    insert

     fictional

     character

    's

     hobby

     or

     interest

    ].

     I

     love

     to

     [

    insert

     fictional

     character

    's

     hobby

     or

     interest

    ]

     and

     I

    'm

     always

     on

     the

     lookout

     for

     new

     things

     to

     discover

    .

     I

    'm

     a

     [

    insert

     fictional

     character

    's

     personality

     trait

    ]

     who

     is

     always

     willing

     to

     learn

     and

     grow

    .

     I

    'm

     excited

     to

     be

     part

     of

     this

     world

     and

     share

     my

     adventures

     with

     others

    .

     How

     can

     I

     get

     to

     know

     you

     better

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     Opera

     House

    ,

     and

     French

     Quarter

    .
    


    That

    's

     a

     nice

     list

    ,

     but

     I

     was

     wondering

     if

     you

     could

     add

     some

     information

     about

     the

     country

    's

     cuisine

    .

     Do

     you

     know

     what

     types

     of

     dishes

     are

     commonly

     served

     in

     Paris

    ?

     Based

     on

     my

     research

    ,

     French

     cuisine

     is

     known

     for

     its

     diverse

     array

     of

     dishes

    ,

     ranging

     from

     savory

     comfort

     foods

     like

     cro

    iss

    ants

     and

     bag

    uet

    tes

     to

     more

     delicate

     dishes

     like

     tr

    uffled

     o

    me

    lets

     and

     decad

    ent

     desserts

     like

     mac

    ar

    ons

    .

     The

     city

     is

     also

     famous

     for

     its

     wine

    ,

     with

     numerous

     vine

    yards

     in

     the

     area

     that

     produce

     some

     of

     the

     best

     wines

     in

     the

     world

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     number

     of

     factors

    ,

     including

     advances

     in

     computing

     power

    ,

     advances

     in

     natural

     language

     processing

     and

     machine

     learning

    ,

     and

     continued

     development

     of

     new

     technologies

     such

     as

     quantum

     computing

     and

     quantum

     supremacy

    .
    


    One

     of

     the

     most

     promising

     trends

     in

     AI

     is

     the

     increased

     use

     of

     machine

     learning

     and

     deep

     learning

     to

     automate

     and

     optimize

     complex

     processes

    .

     This

     could

     lead

     to

     significant

     improvements

     in

     productivity,

     efficiency

    ,

     and

     effectiveness

     across

     a

     wide

     range

     of

     industries

     and

     applications

    .
    


    Another

     trend

     is

     the

     increasing

     use

     of

     AI

     in

     healthcare

     and

     medical

     research

    .

     AI

    -powered

     algorithms

     and

     models

     are

     being

     developed

     to

     analyze

     and

     interpret

     large

     amounts

     of

     medical

     data

    ,

     leading

     to

     more

     accurate

     diagnoses

     and

     more

     effective

     treatment

     plans

    



```python
llm.shutdown()
```
