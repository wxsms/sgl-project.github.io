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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.09it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.08it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.08it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.08it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.59it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.59it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.59it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.59it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.59it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.59it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.59it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.59it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:05<00:05,  7.59it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 19.65it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 19.65it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 19.65it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 19.65it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 19.65it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 19.65it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 19.65it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 19.65it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 19.65it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 19.65it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 27.62it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 37.22it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 37.22it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 37.22it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 37.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.88 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.85 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.85 GB):   3%|▎         | 2/58 [00:00<00:03, 18.48it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.85 GB):   3%|▎         | 2/58 [00:00<00:03, 18.48it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.85 GB):   3%|▎         | 2/58 [00:00<00:03, 18.48it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.85 GB):   3%|▎         | 2/58 [00:00<00:03, 18.48it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.85 GB):   9%|▊         | 5/58 [00:00<00:02, 20.82it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.84 GB):   9%|▊         | 5/58 [00:00<00:02, 20.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.83 GB):   9%|▊         | 5/58 [00:00<00:02, 20.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.83 GB):   9%|▊         | 5/58 [00:00<00:02, 20.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.83 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.83 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.82 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.82 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.82 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=71.82 GB):  21%|██        | 12/58 [00:00<00:01, 29.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.81 GB):  21%|██        | 12/58 [00:00<00:01, 29.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.81 GB):  21%|██        | 12/58 [00:00<00:01, 29.10it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.81 GB):  21%|██        | 12/58 [00:00<00:01, 29.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.80 GB):  21%|██        | 12/58 [00:00<00:01, 29.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.80 GB):  21%|██        | 12/58 [00:00<00:01, 29.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.80 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.80 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.79 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.79 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.77 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.07it/s]

    Capturing num tokens (num_tokens=960 avail_mem=71.79 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.07it/s] Capturing num tokens (num_tokens=960 avail_mem=71.79 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.44it/s]Capturing num tokens (num_tokens=896 avail_mem=71.78 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.44it/s]Capturing num tokens (num_tokens=832 avail_mem=71.78 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.44it/s]Capturing num tokens (num_tokens=768 avail_mem=71.78 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.44it/s]Capturing num tokens (num_tokens=704 avail_mem=71.77 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.44it/s]Capturing num tokens (num_tokens=704 avail_mem=71.77 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.47it/s]Capturing num tokens (num_tokens=640 avail_mem=71.77 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.47it/s]Capturing num tokens (num_tokens=576 avail_mem=71.77 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.47it/s]Capturing num tokens (num_tokens=512 avail_mem=71.75 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.47it/s]

    Capturing num tokens (num_tokens=480 avail_mem=71.77 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.47it/s]Capturing num tokens (num_tokens=480 avail_mem=71.77 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.62it/s]Capturing num tokens (num_tokens=448 avail_mem=71.77 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.62it/s]Capturing num tokens (num_tokens=416 avail_mem=71.77 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.62it/s]Capturing num tokens (num_tokens=384 avail_mem=71.76 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.62it/s]Capturing num tokens (num_tokens=352 avail_mem=71.76 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.62it/s]Capturing num tokens (num_tokens=352 avail_mem=71.76 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.07it/s]Capturing num tokens (num_tokens=320 avail_mem=71.75 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.07it/s]Capturing num tokens (num_tokens=288 avail_mem=71.75 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.07it/s]

    Capturing num tokens (num_tokens=256 avail_mem=71.75 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.07it/s]Capturing num tokens (num_tokens=240 avail_mem=71.74 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.07it/s]Capturing num tokens (num_tokens=224 avail_mem=71.74 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.07it/s]Capturing num tokens (num_tokens=224 avail_mem=71.74 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.76it/s]Capturing num tokens (num_tokens=208 avail_mem=71.74 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.76it/s]Capturing num tokens (num_tokens=192 avail_mem=71.74 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.76it/s]Capturing num tokens (num_tokens=176 avail_mem=71.73 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.76it/s]Capturing num tokens (num_tokens=160 avail_mem=71.73 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.76it/s]Capturing num tokens (num_tokens=160 avail_mem=71.73 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=144 avail_mem=71.73 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.23it/s]

    Capturing num tokens (num_tokens=128 avail_mem=71.73 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=112 avail_mem=71.72 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.23it/s]Capturing num tokens (num_tokens=96 avail_mem=71.72 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.23it/s] Capturing num tokens (num_tokens=96 avail_mem=71.72 GB):  81%|████████  | 47/58 [00:01<00:00, 28.65it/s]Capturing num tokens (num_tokens=80 avail_mem=71.71 GB):  81%|████████  | 47/58 [00:01<00:00, 28.65it/s]Capturing num tokens (num_tokens=64 avail_mem=71.71 GB):  81%|████████  | 47/58 [00:01<00:00, 28.65it/s]

    Capturing num tokens (num_tokens=48 avail_mem=71.71 GB):  81%|████████  | 47/58 [00:01<00:00, 28.65it/s]Capturing num tokens (num_tokens=32 avail_mem=71.70 GB):  81%|████████  | 47/58 [00:01<00:00, 28.65it/s]Capturing num tokens (num_tokens=32 avail_mem=71.70 GB):  88%|████████▊ | 51/58 [00:01<00:00, 28.55it/s]Capturing num tokens (num_tokens=28 avail_mem=71.70 GB):  88%|████████▊ | 51/58 [00:01<00:00, 28.55it/s]Capturing num tokens (num_tokens=24 avail_mem=71.70 GB):  88%|████████▊ | 51/58 [00:01<00:00, 28.55it/s]Capturing num tokens (num_tokens=20 avail_mem=71.69 GB):  88%|████████▊ | 51/58 [00:01<00:00, 28.55it/s]Capturing num tokens (num_tokens=16 avail_mem=71.69 GB):  88%|████████▊ | 51/58 [00:01<00:00, 28.55it/s]Capturing num tokens (num_tokens=16 avail_mem=71.69 GB):  95%|█████████▍| 55/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=12 avail_mem=71.68 GB):  95%|█████████▍| 55/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=8 avail_mem=71.68 GB):  95%|█████████▍| 55/58 [00:01<00:00, 30.58it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=71.68 GB):  95%|█████████▍| 55/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=4 avail_mem=71.68 GB): 100%|██████████| 58/58 [00:01<00:00, 32.07it/s]


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
    Generated text:  Tessa and I was born in Sydney and moved to Los Angeles at the age of 17. I have a keen interest in natural history and natural sciences, with particular interest in marine biology, including the biology of plankton, and the effects of marine pollution on marine ecosystems.
    As a marine biologist, I am committed to tackling real-world problems. This is important to me because it's the responsibility of the next generation to help resolve the environmental and ecological issues of the future. I also have a background in marine education, and have taught marine science at the high school and university levels. My research has focused on the causes of marine
    ===============================
    Prompt: The president of the United States is
    Generated text:  inaugurated every four years. The inauguration takes place on a Thursday during the month of May. Assume that the 42nd president will be inaugurated on a Thursday. On what day of the week will the 134th president be inaugurated? To determine the day of the week for the 134th president of the United States, we need to calculate the number of weeks and additional days between the inauguration of the 42nd president and the inauguration of the 134th president.
    
    1. **Determine the number of weeks between the 42nd and 134th presidents
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    B) Marseille
    C) Lyon
    D) Marseille
    
    To determine the capital of France, let's review the information provided and analyze it step by step.
    
    1. Identify the capital of France:
       The capital of France is Paris.
    
    2. Verify the options:
       A) Paris
       B) Marseille
       C) Lyon
       D) Marseille
    
    Based on the information given, the correct answer is:
    
    \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  bleak, and it’s not just the popular predictions that aren’t true. It’s the situation that most people can see, but can’t easily understand. But what about you? Do you see AI as something that will completely change the world? Or is it something that will make everything much better for you?
    
    The good news is that AI is not as scary as it might seem. In fact, it’s an incredibly valuable tool and can transform the way we live. If you’re interested in how AI can change your life, you’re in luck. In this article, we’ll be looking at how AI is changing our lives, and


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill or Hobby] enthusiast and I'm always looking for new challenges and opportunities to learn and grow. I'm passionate about [What I'm passionate about] and I'm always eager to share my knowledge and experiences with others. I'm a [Favorite Activity] lover and I love to explore new places and try new things. I'm always looking for ways to improve my skills and knowledge, and I'm always eager to learn from others. I'm a [Favorite Book] lover and I love to read and learn from the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French Quarter. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major tourist destination and a popular destination for art, music, and cuisine. The city is known for its vibrant nightlife, fashion, and food scene. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has been a center of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn and adapt in ways that are difficult for humans to do. This could lead to more efficient and effective AI systems that can perform tasks that are currently beyond the capabilities of humans.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for the development and use of AI, as well as greater transparency and accountability in the development and
    


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
    Generated text:  [Name], and I'm a [Age] year-old who has [Job Title] in a [Company/Location]. I'm known for my [Strength/Ability/Interest/ passion] and I enjoy [Experiance]. I'm an [Type of Person] who is [Nice to know] and always [Easy/Struggles with]. I'm [Personality]. And I'm [Overall Description]. If you could summarize my life in just a few words, what would you say? [Name] would respond with an introduction that highlights their unique qualities and personality traits, giving the reader an instant impression of who they
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France by population and the seat of government, culture, and art in the country. The city is also the largest city in the European Union and is home to the United Nations headquarters. France's capital is Paris, which is the largest city in France and the seat of government, culture, and art. The city has a population of over 2. 3 million and is also home to the United Nations headquarters. 
    
    Paris is known for its rich history, vibrant culture, and stunning architecture, and has been a major center of European and global culture since the 12th century. It
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid advancements and significant changes. Here are some potential trends that may occur in the field of artificial intelligence:
    
    1. Increased use of AI in healthcare: AI is increasingly being used to improve the accuracy and efficiency of medical diagnosis and treatment. AI-powered systems can analyze medical images, collect and analyze patient data, and predict potential health risks. This could lead to better patient outcomes and more effective treatments.
    
    2. Integration of AI into consumer electronics: AI-powered devices, such as smart home appliances and virtual assistants, are expected to become more widespread in the future. These devices will be able to learn and adapt to user behavior


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

     ______

    ___

     and

     I

     am

     a

     ______

    ___

    .
    


    Hello

    ,

     my

     name

     is

     [

    insert

     your

     name

    ]

     and

     I

     am

     a

     [

    insert

     your

     profession

    ,

     occupation

    ,

     or

     background

    ].

     If

     you

    're

     reading

     this

    ,

     I

     can

     tell

     that

     you

     have

     a

     natural

     inclination

     towards

     learning

     and

     education

    ,

     which

     is

     why

     I

     was

     drawn

     to

     pursue

     my

     passion

     for

     writing

    .

     I

     enjoy

     exploring

     new

     genres

     and

     learning

     new

     writing

     techniques

    ,

     and

     I

     find

     myself

     deeply

     engaged

     with

     the

     world

     of

     stories

    .

     My

     writing

     often

     takes

     me

     on

     exciting

     adventures

    ,

     from

     magical

     worlds

     to

     epic

     battles

    ,

     and

     I

    'm

     always

     seeking

     to

     push

     the

     boundaries

     of

     what

    's

     possible

     with

     my

     writing

    .

     So

    ,

     if

     you

    're

     reading

    
    
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

    .

     The

     city

     is

     located

     in

     the

     Î

    le

     de

     la

     C

    ité

    ,

     overlooking

     the

     Se

    ine

     River

    ,

     and

     is

     the

     fifth

    -largest

     city

     in

     France

     by

     population

    .

     The

     city

     is

     home

     to

     many

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

     and

     Notre

     Dame

     Cathedral

    .

     It

     is

     also

     known

     as

     the

     "

    City

     of

     Lights

    "

     for

     its

     warm

    ,

     sunny

     weather

     and

     vibrant

     culture

    .

     Paris

     has

     played

     a

     significant

     role

     in

     French

     history

     and

     culture

    ,

     and

     continues

     to

     be

     a

     major

     urban

     center

     and

     cultural

     center

     in

     the

     world

     today

    .

     

     The

     French

     are

     a

     proud

     and

     active

     society

    ,

     and

     are

     known

     for

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     speculative

    ,

     as

     it

     is

     constantly

     evolving

     and

     changing

    .

     However

    ,

     there

     are

     a

     few

     trends

     that

     are

     likely

     to

     continue

     or

     become

     more

     prevalent

     in

     the

     coming

     years

    .
    


    One

     of

     the

     most

     significant

     trends

     is

     the

     increasing

     integration

     of

     AI

     into

     various

     sectors

    ,

     including

     healthcare

    ,

     finance

    ,

     and

     manufacturing

    .

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     will

     likely

     allow

     for

     more

     accurate

     and

     personalized

     diagnoses

    ,

     better

     risk

     assessment

    ,

     and

     more

     efficient

     production

     processes

    .

     This

     could

     lead

     to

     significant

     improvements

     in

     healthcare

     outcomes

     and

     productivity

    .
    


    Another

     trend

     is

     the

     development

     of

     more

     robust

     and

     advanced

     AI

     models

    ,

     which

     will

     allow

     for

     more

     complex

     and

     nuanced

     decision

    -making

    .

     This

     could

     lead

     to

     improved

     customer

     experiences

    ,

     more

    



```python
llm.shutdown()
```
