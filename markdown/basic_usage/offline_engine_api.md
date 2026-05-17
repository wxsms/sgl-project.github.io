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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.12it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.11it/s]


    2026-05-17 08:06:02,109 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-17 08:06:02] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.49it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 22.03it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 22.03it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 22.03it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 22.03it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 22.03it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 22.03it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 22.03it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 22.03it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 22.03it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 22.03it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:04<00:00, 22.03it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 31.59it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 31.59it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 31.59it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 31.59it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 31.59it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 31.59it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 31.59it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:04<00:00, 31.59it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 31.59it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 31.59it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 31.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.97 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.93 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.93 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.93 GB):   9%|▊         | 5/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.93 GB):   9%|▊         | 5/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.92 GB):   9%|▊         | 5/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.91 GB):   9%|▊         | 5/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.91 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.91 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.91 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.91 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.90 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.24it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.90 GB):  21%|██        | 12/58 [00:00<00:01, 30.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.90 GB):  21%|██        | 12/58 [00:00<00:01, 30.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.90 GB):  21%|██        | 12/58 [00:00<00:01, 30.10it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.89 GB):  21%|██        | 12/58 [00:00<00:01, 30.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.89 GB):  21%|██        | 12/58 [00:00<00:01, 30.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.89 GB):  21%|██        | 12/58 [00:00<00:01, 30.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.89 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.88 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.88 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.88 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.83it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.86 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.83it/s]Capturing num tokens (num_tokens=960 avail_mem=72.87 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.83it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=72.87 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.68it/s]Capturing num tokens (num_tokens=896 avail_mem=72.87 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.68it/s]Capturing num tokens (num_tokens=832 avail_mem=72.87 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.68it/s]Capturing num tokens (num_tokens=768 avail_mem=72.86 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.68it/s]Capturing num tokens (num_tokens=704 avail_mem=72.86 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.68it/s]Capturing num tokens (num_tokens=640 avail_mem=72.86 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.68it/s]Capturing num tokens (num_tokens=640 avail_mem=72.86 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.93it/s]Capturing num tokens (num_tokens=576 avail_mem=72.86 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.93it/s]Capturing num tokens (num_tokens=512 avail_mem=72.84 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.93it/s]Capturing num tokens (num_tokens=480 avail_mem=72.86 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.93it/s]Capturing num tokens (num_tokens=448 avail_mem=72.86 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.93it/s]Capturing num tokens (num_tokens=416 avail_mem=72.85 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.93it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.85 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.01it/s]Capturing num tokens (num_tokens=384 avail_mem=72.85 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.01it/s]Capturing num tokens (num_tokens=352 avail_mem=72.85 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.01it/s]Capturing num tokens (num_tokens=320 avail_mem=72.84 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.01it/s]Capturing num tokens (num_tokens=288 avail_mem=72.84 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.01it/s]Capturing num tokens (num_tokens=256 avail_mem=72.84 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.01it/s]Capturing num tokens (num_tokens=256 avail_mem=72.84 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=240 avail_mem=72.83 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.47it/s]Capturing num tokens (num_tokens=224 avail_mem=72.83 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.47it/s]Capturing num tokens (num_tokens=208 avail_mem=72.83 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.47it/s]Capturing num tokens (num_tokens=192 avail_mem=72.83 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.47it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.82 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.47it/s]Capturing num tokens (num_tokens=176 avail_mem=72.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.98it/s]Capturing num tokens (num_tokens=160 avail_mem=72.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.98it/s]Capturing num tokens (num_tokens=144 avail_mem=72.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.98it/s]Capturing num tokens (num_tokens=128 avail_mem=72.81 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.98it/s]Capturing num tokens (num_tokens=112 avail_mem=72.81 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.98it/s]Capturing num tokens (num_tokens=96 avail_mem=72.81 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.98it/s] Capturing num tokens (num_tokens=96 avail_mem=72.81 GB):  81%|████████  | 47/58 [00:01<00:00, 44.46it/s]Capturing num tokens (num_tokens=80 avail_mem=72.80 GB):  81%|████████  | 47/58 [00:01<00:00, 44.46it/s]Capturing num tokens (num_tokens=64 avail_mem=72.80 GB):  81%|████████  | 47/58 [00:01<00:00, 44.46it/s]Capturing num tokens (num_tokens=48 avail_mem=72.80 GB):  81%|████████  | 47/58 [00:01<00:00, 44.46it/s]Capturing num tokens (num_tokens=32 avail_mem=72.79 GB):  81%|████████  | 47/58 [00:01<00:00, 44.46it/s]

    Capturing num tokens (num_tokens=28 avail_mem=72.79 GB):  81%|████████  | 47/58 [00:01<00:00, 44.46it/s]Capturing num tokens (num_tokens=28 avail_mem=72.79 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=24 avail_mem=72.79 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=20 avail_mem=72.78 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=16 avail_mem=72.78 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=12 avail_mem=72.78 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.05it/s]Capturing num tokens (num_tokens=8 avail_mem=72.77 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.05it/s] Capturing num tokens (num_tokens=8 avail_mem=72.77 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=4 avail_mem=72.77 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=4 avail_mem=72.77 GB): 100%|██████████| 58/58 [00:01<00:00, 40.38it/s]


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
    Generated text:  Denis. I'm a young guy who likes to write science fiction stories. My story would include elements of a science fiction fantasy. My aim is to write a book. What's the first step to make a book? I think the answer is to start writing. But what does that actually mean? How to start? The answer is to start a journal. Writing is a very natural way of expressing your thoughts and ideas. When you write, you get a feeling in your soul as if you are in your own mind and you have a very clear vision of what you want to tell your story. Writing is a great way to get your
    ===============================
    Prompt: The president of the United States is
    Generated text:  32 years older than the president of Brazil, and the president of Brazil is 32 years younger than the president of France. If the president of France is currently 20 years old, how old will the president of Brazil be in 10 years?
    
    To determine the age of the president of Brazil in 10 years, we need to follow the information given and work backwards step by step.
    
    1. We know the president of France is currently 20 years old.
    2. The president of France is 32 years younger than the president of Brazil. Therefore, the president of Brazil is:
       \
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. London B. Paris C. Moscow D. Paris
    
    Paris
    
    The capital of the United States is ______. A. New York B. Washington D. C. C. New York
    
    Washington D. C.
    
    According to the passage, the idea of "the new" is a way to ______. A. show how old the country is B. show how progressive the country is C. show the present day country D. show the future of the country
    
    show the future of the country
    
    What is the significance of the "new" in the passage? A. It is a way to show how old the
    ===============================
    Prompt: The future of AI is
    Generated text:  fast approaching, and it's a topic that's been making headlines and prompting debate for years. But it's not clear yet what AI's future will be like. So what do we know for sure about the future of AI? Here are some of the most common AI questions being asked.
    AI is the future of artificial intelligence.
    AI is the future of artificial intelligence.
    AI is the future of artificial intelligence.
    AI is the future of artificial intelligence.
    AI is the future of artificial intelligence.
    Is AI becoming even more advanced than we thought?
    Is AI becoming even more advanced than we thought?
    Is AI becoming even more advanced than we thought?
    


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


    Generated text:  Paris, also known as the City of Light, and is located in the south of the country. It is the largest city in Europe and is home to many of the world’s most famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its rich cultural heritage, including its art, music, and cuisine. The city is home to many of the world’s most famous museums, including the Louvre and the Musée d'Orsay, and is a major center for the arts and entertainment industry. Paris is a vibrant and dynamic city with a rich history and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation: As AI becomes more advanced, it is likely to become more and more integrated into our daily lives. This will lead to an increase in automation, where machines will be able to perform tasks that were previously done by humans. This will have a significant impact on the job market, as many jobs that are currently performed by humans will be replaced by machines.
    
    2. Improved privacy and security: As AI
    


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
    Generated text:  [Your Name], and I'm a [Your occupation or profession] who loves [Your favorite hobby or activity]. I have a passion for [Your favorite genre of literature], and I am always looking for new ways to discover and express myself. Whether I'm writing stories, painting, or dancing, I'm always trying to bring something new to the world. I'm excited to meet you! 😊
    
    Hey there! I'm [Your Name], a [Your occupation or profession] who loves [Your favorite hobby or activity] and is always up for trying new things. I'm excited to meet you! 😊
    
    What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    (A) Paris is the second-largest city in France.
    (B) Paris is the largest city in France.
    (C) Paris is the capital of France.
    (D) Paris is the economic capital of France. 
    (C) Paris is the capital of France. 
    
    The capital of France is Paris. 
    (A) is incorrect because Paris is not the second-largest city in France. 
    (B) is incorrect because Paris is not the largest city in France. 
    (D) is incorrect because Paris is not the economic capital of France. 
    (C) is correct because Paris is indeed the capital of France.
    The capital of France is Paris.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating, with many potential trends that could shape the way we interact with technology and solve complex problems. Here are some potential future trends in AI:
    
    1. Increased focus on ethical and responsible AI: There's a growing recognition of the importance of AI in society, and there's a push to develop AI that's ethical, transparent, and accountable. This could mean increased regulation of AI development and greater emphasis on ethical considerations from the outset of project.
    
    2. AI-powered health care: AI could play a major role in developing smarter, more accurate medical treatments. For example, AI could be used to analyze medical images, predict disease progression, and


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

    name

    ],

     and

     I

     am

     an

     [

    age

    ],

     [

    gender

    ],

     [

    occupation

    ].

     I

     am

     a

     [

    career

    ],

     [

    role

    ],

     [

    s

    or

    cery

    ],

     and

     [

    profession

    ],

     all

     of

     which

     I

     have

     learned

     from

     my

     master

    ,

     [

    s

    or

    cer

    or

    's

     name

    ].

     I

     use

     this

     knowledge

     to

     create

     enchant

    ments

     that

     can

     protect

     and

     enhance

     the

     lives

     of

     those

     I

     care

     for

    .

     I

     also

     enjoy

     listening

     to

     my

     master

    's

     stories

    ,

     and

     I

     am

     eager

     to

     learn

     new

     spells

    ,

     techniques

    ,

     and

     adventures

     that

     will

     help

     me

     become

     even

     more

     powerful

    .

     My

     name

     is

     [

    name

    ],

     and

     I

     am

     an

     [

    age

    ],

     [

    gender

    ],

     [

    occupation

    ].

     I

     am

     a

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     one

     of

     the

     largest

     in

     the

     world

    .

     It

     is

     home

     to

     many

     important

     museums

    ,

     galleries

    ,

     and

     historical

     landmarks

    .

     It

     has

     a

     rich

     history

     and

     is

     known

     for

     its

     beautiful

     architecture

    ,

     including

     the

     iconic

     E

    iff

    el

     Tower

    .

     Paris

     is

     also

     a

     center

     for

     music

    ,

     art

    ,

     and

     culture

    ,

     with

     many

     prestigious

     universities

     and

     art

     museums

     serving

     the

     city

    's

     diverse

     population

    .

     Additionally

    ,

     it

     is

     known

     for

     its

     French

     language

    ,

     cuisine

    ,

     and

     cuisine

    .

     Overall

    ,

     Paris

     is

     a

     city

     of

     contrasts

     and

     cultural

     diversity

    ,

     with

     many

     landmarks

    ,

     museums

    ,

     and

     events

     that

     make

     it

     a

     popular

     tourist

     destination

    .

     Based

     on

     the

     passage

     above

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     and

     there

     are

     several

     trends

     that

     are

     expected

     to

     shape

     its

     direction

    :
    


    1

    .

     Increased

     sophistication

    :

     As

     AI

     becomes

     more

     capable

     of

     processing

     complex

     data

     and

     recognizing

     patterns

    ,

     it

     is

     expected

     to

     become

     increasingly

     sophisticated

     and

     capable

     of

     performing

     increasingly

     complex

     tasks

    .
    


    2

    .

     Personal

    ization

    :

     AI

     is

     expected

     to

     become

     more

     personalized

    ,

     with

     the

     ability

     to

     learn

     from

     individual

     users

    '

     behavior

     and

     preferences

     to

     provide

     more

     tailored

     recommendations

     and

     experiences

    .
    


    3

    .

     Eth

    ical

     considerations

    :

     As

     AI

     becomes

     more

     advanced

     and

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     increasing

     pressure

     to

     consider

     and

     address

     ethical

     concerns

    ,

     such

     as

     the

     use

     of

     AI

     for

     surveillance

    ,

     data

     privacy

    ,

     and

     the

     potential

     for

    



```python
llm.shutdown()
```
