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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.67it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.65it/s]


    2026-04-29 22:01:31,342 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 22:01:31] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:33,  4.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:33,  4.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:33,  4.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:33,  4.81s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:33,  4.81s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.35it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.35it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.26it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.26it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.26it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.78it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 15.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 15.55it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 15.55it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 15.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 20.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 20.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 20.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 20.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 20.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.36it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.36it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.36it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.36it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.28it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.28it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.28it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.28it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s] Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.29it/s]Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.29it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.29it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.29it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.29it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.29it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.29it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.22it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.68it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.68it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.68it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.68it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.68it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  59%|█████▊    | 34/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  59%|█████▊    | 34/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.00it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.00it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.00it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.00it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.00it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.00it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.00it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.14it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.14it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.14it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.14it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.14it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.14it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=16 avail_mem=72.46 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s] Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.80it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.80it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 41.18it/s]


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
    Generated text:  Jessie. I am a middle school student. The most important thing for me is to study well and take good care of my body. I also like to listen to music, especially country music, to relax myself. I often go to the library to find out information I need and I like to watch TV shows, especially science fiction. What are your favorite things to do? What would you like to be when you grow up? 1. What do you like to do when you go out? 2. What is your favorite thing to do? 3. What would you like to be when you grow up? 4.
    ===============================
    Prompt: The president of the United States is
    Generated text:  married to a woman who is not a citizen of the United States. Who is considered a foreign national?
    A) The president
    B) The woman
    C) Neither
    D) The president and the woman
    E) The president and the woman and both are considered foreign nationals
    
    To solve this problem, let's break down the information given and analyze each option step by step.
    
    1. **Identify the key information:**
       - The president of the United States is married to a woman who is not a citizen of the United States.
       - We need to determine who is considered a foreign national.
    
    2. **Consider the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of Russia is Moscow. Tokyo is in which country?
    A) Germany
    B) Japan
    C) United Kingdom
    D) United States
    The capital of Japan is Tokyo. Therefore, the correct answer is:
    B) Japan
    The capital of Japan is Tokyo.
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it’s not all sunshine and rainbows.
    To understand where AI is headed, you need to understand where it came from, what it does, and how it evolved. So let’s take a look at a few of the biggest AI technologies that have been around for a while.
    What’s the AI Market?
    There are several ways to measure the AI market.
      * By the number of firms
      * By the number of applications
      * By the number of jobs
      * By the number of patents
      * By the value of the AI market to the economy
    As of 2021,


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] with [Number of Wheels] wheels. I'm a [Favorite Food] lover, and I enjoy [Favorite Activity] with my friends. I'm a [Favorite Book] lover, and I read [Number of Books] books every year. I'm a [Favorite Movie] fan, and I watch [Number of Movies] movies every year. I'm a [Favorite Sport] enthusiast, and I play [Favorite Sport] with my [Favorite Team]. I'm a [Favorite Music] lover, and I listen
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French Parliament building. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the Middle Ages. The city is known for its cuisine, fashion, and music, and is a popular tourist destination. It is also home to many international organizations and institutions, including the European Parliament and the United Nations. Paris is a vibrant and dynamic city with a strong sense of community and a commitment to social justice and equality. Its status
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the potential for AI to be used for harmful purposes.
    
    2. Development of more advanced models: As AI technology continues to advance, there will be a greater focus on developing more advanced models that can better understand and interpret complex data. This will include models that can learn from
    


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
    Generated text:  [name]. I'm a [type of character] with a [occupation] background. I'm passionate about [why I'm passionate]. I have a knack for [how I'm skilled at something]. I love [why I love what I do]. What excites me is [what's exciting to me], and I'm always eager to learn and grow. I enjoy [how I spend my free time]. I'm [what I can expect to be someone else]. I'm a [what I'd like to become]. I look forward to [what I can look forward to]. What's your favorite hobby or activity to do on
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a city located in the center of the country. It is one of the most important cities in Europe and is known for its rich history, stunning architecture, and vibrant culture. The city is home to many iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also a popular tourist destination, hosting annual festivals, events, and cultural events throughout the year. It is a city that is renowned for its quality of life and is considered one of the world's top cities for work, education, and entertainment. According to the latest census data, Paris has a population of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  undoubtedly bright and full of opportunities, and here are some of the possible trends we are seeing:
    
    1. Increased efficiency: With the development of AI, there is a possibility of increasing efficiency in various industries. AI can help automate tasks, reduce errors, and improve productivity.
    
    2. Enhanced personalization: AI is enabling more personalized experiences for users. Personalization can be achieved through the use of big data, machine learning, and natural language processing.
    
    3. Greater transparency: AI is also making it easier for users to understand how AI systems work, and for companies to explain their decision-making processes.
    
    4. Autonomous systems: Autonomous systems like drones


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

    _

     and

     I

     specialize

     in

     ______

    _.

     My

     professional

     background

     is

     in

     ______

    _.

     Can

     you

     tell

     me

     a

     little

     bit

     about

     yourself

    ?

     I

    'm

     a

     ______

    _

     and

     I

    'm

     always

     looking

     for

     opportunities

     to

     grow

     and

     develop

    .

     What

     are

     your

     goals

     and

     what

     are

     you

     excited

     about

    ?

     In

     a

     nutshell

    ,

     I

    'm

     ______

    _

     and

     I

    'm

     always

     looking

     for

     ways

     to

     ______

    _.

     I

    'm

     always

     looking

     for

     ways

     to

     ______

    _

    .


    I

    'm

     a

     professional

     photographer

    ,

     with

     a

     passion

     for

     capturing

     the

     beauty

     of

     nature

    .

     I

    'm

     always

     eager

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     always

     looking

     for

     new

     opportunities

     to

     grow

     and

     develop

    .

     I

    'm

     a

     dedicated

     teacher

    ,

     and

     I

    'm

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     south

     of

     the

     country

     and

     known

     for

     its

     stunning

     architecture

    ,

     renowned

     cuisine

    ,

     and

     rich

     history

    .

     It

     is

     one

     of

     the

     world

    's

     most

     famous

     cities

    ,

     and

     Paris

     has

     played

     a

     significant

     role

     in

     shaping

     French

     culture

     and

     politics

     for

     centuries

    .

     It

     is

     also

     the

     world

    's

     most

     populous

     city

    ,

     with

     over

     

    2

    7

     million

     inhabitants

    .
    


    The

     city

     has

     a

     rich

     cultural

     heritage

     that

     dates

     back

     to

     ancient

     times

    ,

     including

     the

     Roman

     Empire

    ,

     the

     Renaissance

    ,

     and

     the

     French

     Revolution

    .

     It

     has

     hosted

     many

     famous

     events

     and

     attractions

     throughout

     its

     history

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     Today

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     and

     there

     are

     many

     possible

     trends

     shaping

     its

     direction

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

     Machine

     learning

     and

     deep

     learning

    :

     AI

     experts

     are

     already

     developing

     powerful

     machine

     learning

     algorithms

     and

     deep

     learning

     models

     that

     can

     recognize

     patterns

     in

     complex

     data

     sets

    ,

     predict

     outcomes

    ,

     and

     even

     create

     creative

     art

    .

     These

     algorithms

     are

     already

     being

     used

     in

     various

     applications

    ,

     including

     autonomous

     vehicles

    ,

     natural

     language

     processing

    ,

     and

     image

     recognition

    .
    


    2

    .

     Quantum

     computing

    :

     Quantum

     computers

     are

     currently

     difficult

     to

     build

     and

     operate

     due

     to

     the

     phenomenon

     of

     super

    position

    ,

     where

     particles

     can

     exist

     in

     multiple

     states

     simultaneously

    .

     However

    ,

     researchers

     are

     exploring

     new

     technologies

     that

     could

     revolution

    ize

     AI

     by

     enabling

     quantum

    



```python
llm.shutdown()
```
