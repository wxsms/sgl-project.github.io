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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.27it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.26it/s]


    2026-05-15 07:41:02,794 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 07:41:02] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:16,  4.50s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.89it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.89it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.89it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.89it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.89it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.89it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.89it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.89it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.89it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.89it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.71it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.71it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.71it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.71it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.71it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.71it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.71it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.71it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.71it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.71it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.67it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:01, 14.67it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.78it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 29.93it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 29.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 41.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.07it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.31it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.31it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.31it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.31it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  31%|███       | 18/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  31%|███       | 18/58 [00:00<00:01, 30.20it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 30.20it/s] Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.84it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.84it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.84it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.84it/s]

    Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.84it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.03it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.03it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.03it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.03it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.03it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.03it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.69it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.69it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.69it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.69it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.69it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.69it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  71%|███████   | 41/58 [00:01<00:00, 38.28it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 38.28it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 38.28it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 38.28it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 38.28it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.51it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.51it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.51it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.51it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.51it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.51it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.77it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.77it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.77it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.77it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.77it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.77it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.37it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.37it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.37it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.37it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 36.02it/s]


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
    Generated text:  Xander, a 16 year old male from the United States. I am an active member of the local Dungeons and Dragons community and I have been playing the game for several years now. I have played together with several players and have been able to learn a lot from them. I am a fan of both the game and the culture surrounding it.
    
    I have been asked to write an article about the history of Dungeons and Dragons, but I'm not sure where to start. Can you provide some tips for writing a good introduction and main body of an article on the history of Dungeons and Dragons?
    
    Sure, here are some tips to help
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to use the 2005 or 2012 version of the penny, which has a face value of $0.01. To help with his decision, he's looking at the inflation rates over a period of time. He wants to know how much more money would be worth in a year if he used the 2012 version instead of the 2005 version of the penny.
    
    To help with his decision, he's looking at the inflation rates over a period of time. He wants to know how much more money would be worth in a year if he used the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. New York
    D. Moscow
    A capital city is the largest and most populous city in a country, and in France, Paris is the capital. Therefore, the answer is A. Paris. However, the other options are not capitals of France: London is the capital of the United Kingdom, New York is the capital of the United States, and Moscow is the capital of Russia. But Paris is the capital of France.
    ===============================
    Prompt: The future of AI is
    Generated text:  highly uncertain, but it is a rapidly evolving field with immense potential for transforming industries and improving the lives of individuals. While the technology is not yet fully developed, AI has already been used in various ways to solve complex problems and improve human productivity and quality of life.
    AI has already been used in a wide range of industries, including healthcare, finance, transportation, and manufacturing. AI-powered solutions have been developed to help doctors diagnose diseases, find the best pricing strategies for goods and services, manage supply chains, and even detect fraudulent activities in financial transactions. AI has also been used to improve the efficiency of transportation systems, help businesses automate their processes


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


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. 
    
    (Note: The statement should be a single, clear sentence.) 
    
    Paris is the capital of France and is known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. 
    
    This statement encapsulates the key facts about Paris, including its capital status, notable landmarks, and cultural highlights. It provides a concise overview of the capital city's importance and attractions. 
    
    For a more detailed and comprehensive answer, you could expand on the Eiffel Tower, its history, and the city's role in French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be a greater need for privacy and security measures to protect user data. This could lead to the development of new technologies that are more secure and transparent
    


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
    Generated text:  [insert name], and I'm a/an [insert occupation or hobby] with [insert relevant experience or skills]. I'm currently [insert current location or career stage]. Let me know if you'd like to learn more about me, and I'll be happy to provide any information I can. [insert name] (write your name) [insert any name you choose] [insert any title or organization you'd like to include if applicable] (write your title) [insert any additional details you'd like to include] [insert name] (write your name) [insert any title or organization you'd like to include if applicable]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Paris is the largest city and the 11th largest city in the world. It is a metropolis located in the North of France. It is the seat of the French government and the most populous city of France with over 2 million inhabitants. Paris is well known for its historic landmarks, including the Eiffel Tower and Notre Dame Cathedral. The city is also famous for its vibrant cultural scene, including museums, theaters, and restaurants. Paris is a global city with a diverse population and a rich cultural heritage.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be rapidly evolving, with a number of trends shaping its development. One of the most significant trends is the increasing focus on developing more advanced and sophisticated AI models, especially those that can handle large and complex data sets. This will require the development of even more powerful hardware and software platforms, as well as new data collection and analysis techniques.
    
    Another trend is the increasing reliance on AI for decision-making, especially in industries such as healthcare, finance, and transportation. This will require the development of more sophisticated algorithms that can analyze large amounts of data and provide more accurate and reliable predictions and recommendations.
    
    AI will also continue to play a key role


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

    X

    ]

     programmer

    .

     I

     spend

     my

     days

     creating

     software

     and

     building

     systems

     that

     solve

     complex

     problems

    .

     I

    've

     been

     coding

     for

     [

    X

    ]

     years

     now

    ,

     and

     I

    'm

     constantly

     learning

     new

     languages

    ,

     frameworks

    ,

     and

     tools

    .

     I

    've

     always

     been

     passionate

     about

     technology

     and

     always

     strive

     to

     improve

     myself

     and

     my

     skills

    .

     I

     love

     coding

    ,

     solving

     problems

    ,

     and

     working

     with

     others

    .

     If

     you

     have

     any

     questions

     or

     need

     help

    ,

     feel

     free

     to

     reach

     out

    !

     

    😊

    👋

    🏼

    
    


    Hey

     there

    !

     I

    'm

     [

    Name

    ],

     a

     [

    X

    ]

     programmer

     with

     [

    X

    ]

     years

     of

     experience

    .

     I

    ’m

     always

     on

     the

     lookout

     for

     new

    
    
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

     the

     European

     Union

     and

     one

     of

     the

     largest

     cities

     in

     the

     world

    .

     Paris

     is

     known

     for

     its

     iconic

     architecture

    ,

     vibrant

     culture

    ,

     and

     delicious

     food

    .

     It

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     a

     popular

     tourist

     destination

    .

     The

     city

     is

     also

     the

     birth

    place

     of

     the

     French

     Revolution

     and

     the

     current

     President

    ,

     Emmanuel

     Macron

    .

     Paris

     is

     often

     referred

     to

     as

     the

     “

    City

     of

     Love

    ”

     and

     a

     symbol

     of

     French

     identity

     and

     excellence

    .

     The

     city

     is

     home

     to

     many

     famous

     museums

    ,

     galleries

    ,

     and

     theaters

    ,

     including

     the

     Lou

    vre

     and

     the

     Petit

     Pal

    ais

    .

     It

     is

     known

     for

     its

     cafes

    ,

     restaurants

    ,

     and

     nightlife

    ,

     and

     is

     home

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     vast

     and

     complex

    ,

     and

     it

     is

     constantly

     evolving

     with

     new

     advancements

     and

     innovations

     being

     made

     every

     day

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Autonomous

     Vehicles

    :

     The

     future

     of

     AI

     will

     likely

     see

     a

     significant

     shift

     towards

     autonomous

     vehicles

    ,

     with

     AI

    -powered

     self

    -driving

     technology

     becoming

     more

     common

     in

     the

     vehicle

     manufacturing

     and

     transportation

     industries

    .
    


    2

    .

     Quantum

     Computing

    :

     With

     the

     advancement

     of

     quantum

     computing

    ,

     AI

     could

     become

     even

     more

     powerful

     and

     capable

    .

     Quantum

     computers

     can

     process

     a

     vast

     amount

     of

     data

     much

     faster

     and

     with

     greater

     accuracy

     than

     traditional

     computers

    .
    


    3

    .

     Bi

    otechnology

    :

     AI

     is

     already

     playing

     a

     significant

     role

     in

     bi

    otechnology

    ,

     and

     it

     is

     expected

    



```python
llm.shutdown()
```
