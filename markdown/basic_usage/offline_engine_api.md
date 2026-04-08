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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.01it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.00it/s]


    2026-04-08 13:21:51,948 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 13:21:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.43it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.43it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.43it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.43it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.43it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.43it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.43it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.43it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.96it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.96it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.96it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.96it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.96it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.96it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.96it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.96it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.96it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.96it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.96it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.96it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.96it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.96it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 34.75it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 34.75it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 34.75it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 34.75it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 34.75it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 34.75it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 34.75it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.62it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.62it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  31%|███       | 18/58 [00:00<00:01, 34.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 34.62it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 34.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  31%|███       | 18/58 [00:00<00:01, 34.62it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  31%|███       | 18/58 [00:00<00:01, 34.62it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  31%|███       | 18/58 [00:00<00:01, 34.62it/s]Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.52it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.52it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.52it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.52it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.52it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.52it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.69it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.69it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.69it/s]

    Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.69it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.69it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.69it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.30it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.30it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.30it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.30it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.30it/s]

    Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.46it/s]

    Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.68it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.68it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.68it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.68it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.68it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.68it/s] Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  81%|████████  | 47/58 [00:01<00:00, 33.10it/s]Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  81%|████████  | 47/58 [00:01<00:00, 33.10it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  81%|████████  | 47/58 [00:01<00:00, 33.10it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  81%|████████  | 47/58 [00:01<00:00, 33.10it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  81%|████████  | 47/58 [00:01<00:00, 33.10it/s]

    Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  81%|████████  | 47/58 [00:01<00:00, 33.10it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.43it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.43it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.43it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.43it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.43it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.43it/s] Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.09it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.09it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 35.00it/s]


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
    Generated text:  Serena and I am a student at the University of Rochester. As a major in economics, I was interested in learning more about the role that leadership plays in business. I have been helping my brother with his assignment for a class called "Leadership and Management." As my brother's teacher, I have been very helpful in explaining complex concepts in a simple way that he can understand. I am currently reviewing an essay that I need to write for the assignment. I have been having a hard time putting together my thoughts. Can you provide me with advice on how to write a strong essay about leadership? Serena. Sure, I'd be happy to
    ===============================
    Prompt: The president of the United States is
    Generated text:  34 years older than the president of Australia. If the president of Australia was 32 years old 5 years ago, how old would the president of Australia be now? To determine the current age of the president of Australia, we need to follow these steps:
    
    1. **Identify the age of the president of the United States:**
       - We know the president of the United States is 34 years older than the president of Australia.
       - If the president of Australia was 32 years old 5 years ago, we can calculate his current age by adding 5 years to his past age.
    
    2
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It was named after the ancient capital of France, the City of the Sun, which was located on the island of Crete in the Aegean Sea. The ancient city of Phrygia, near modern-day Ankara, was capital of a Greek colony called Phrygia during the 6th century BC. Its name, "Phrygia," derived from the Greek name "Phryneia," meaning "sun" in Greek, and was the name given by the Greek colonists to their colony. The Phrygian language, similar to that of the Greeks, was spoken in Phrygia. The Ph
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but the current challenges have been well-documented. Here are some of the biggest challenges facing the field, as well as potential solutions and advancements being made in the next few years.
    
    1. Bias and fairness: One of the biggest challenges in AI is the issue of bias and fairness. AI systems often rely on pre-existing datasets that may not be representative of the population they are intended to serve. This can lead to biased outcomes, such as unfair treatment of minority groups or discrimination against certain groups. To address this, researchers and developers are exploring various techniques, including data augmentation, transfer learning, and fairness metrics such as F1-score


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Type of Vehicle] [Vehicle Name] and I have been driving for [Number of Years] years. I am a [Favorite Hobby] [Hobby Name] and I enjoy [Favorite Activity] [Activity Name]. I am [Favorite Food] [Food Name] and I love [Favorite Music] [Music Name]. I am [Favorite Book] [Book Title] and I have a [Favorite Movie] [Movie Title]. I am [Favorite Sport] [Sport Name] and I love [Favorite Sport] [Sport Name].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is known for its fashion, art, and cuisine, and is a popular destination for business and leisure. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a vibrant and diverse city with a rich cultural heritage and is a major tourist destination.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the development of AI in the coming years:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there is a growing need for ethical considerations to be taken into account. This will likely lead to increased focus on ethical guidelines and standards for AI development and deployment.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, including healthcare, finance, and transportation. As AI continues
    


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
    Generated text:  [Name], and I am a [job title] at [company name]. I have been with the company for [number of years] years, and I have been recognized for my contributions to the team. I am a [professional trait] in my field, and I am always willing to learn and grow. I am a [confident trait] person, and I am always up for a challenge. I am a [optimistic trait] person, and I am always ready to make a difference in the world. I am a [ambition] person, and I am always striving to improve myself and my work. I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a cultural and economic hub, hosting numerous festivals and events throughout the year, including the Eiffel Tower's annual iron maiden contest. It's often referred to as the "City of Light" for its vibrant nightlife and world-renowned museums and art galleries. Paris is a historic and modern city with a rich history, offering a diverse range of attractions and experiences for visitors.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  undoubtedly exciting and unpredictable. Here are some possible trends that will shape AI in the coming years:
    
    1. Increased AI Transparency: AI models will become more transparent and explainable, allowing people to understand the reasoning behind their decisions. This will make AI more ethical and responsible in decision-making.
    
    2. AI with Life: AI will become more like living beings, capable of learning, adapting, and evolving. This will enable AI to become more intelligent and adaptable to changing conditions.
    
    3. AI for Personalized Experiences: AI will become more personal, allowing for more accurate and efficient service delivery. This will enhance the user experience and provide better personalized


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

    ]

     and

     I

     am

     a

    /an

     [

    Job

     Title

    ]

     in

     a

    /an

     [

    Company

     Name

    ].

     I

     am

     excited

     to

     meet

     you

     and

     have

     been

     following

     you

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

     always

     wanted

     to

     learn

     more

     about

     you

     and

     your

     work

    ,

     so

     I

     decided

     to

     take

     the

     opportunity

     to

     meet

     you

    .
    


    I

     am

     confident

     that

     [

    Name

    ]

     is

     a

     very

     talented

     and

     driven

     individual

     with

     a

     passion

     for

     [

    Job

     Title

    ].

     I

     am

     looking

     forward

     to

     learning

     more

     about

     [

    Name

    ]

     and

     potentially

     working

     with

     them

     in

     the

     future

    .

     
    


    Thank

     you

     for

     taking

     the

     time

     to

     meet

     me

    .

     I

     am

     looking

     forward

     to

     the

     opportunity

     to

     learn

     more

     about

     you

     and

     our

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     with

     a

     population

     of

     over

     

    3

     million

    .
    


    I

    'm

     sorry

    ,

     I

     cannot

     provide

     an

     answer

     to

     that

     question

     as

     there

     is

     no

     such

     city

     as

     Paris

     in

     France

    .

     Paris

     is

     the

     capital

     city

     of

     France

     and

     is

     home

     to

     many

     famous

     landmarks

     and

     attractions

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

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     cultural

     and

     historical

     sites

    .

     However

    ,

     it

     is

     important

     to

     note

     that

     the

     city

     has

     undergone

     significant

     changes

     over

     time

    ,

     including

     the

     creation

     of

     the

     modern

     city

     of

     Paris

    ,

     which

     was

     established

     in

     

    1

    7

    9

    3

     as

     the

     capital

     of

     the

     French

     Empire

    .

     
    


    Paris

     is

     also

     home

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     development

     and

     adoption

     of

     new

     technologies

    .

     Here

     are

     some

     possible

     trends

     that

     are

     expected

     to

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     reliance

     on

     AI

    -powered

     automation

    :

     With

     the

     rapid

     advancement

     of

     AI

    ,

     we

     are

     likely

     to

     see

     an

     increasing

     use

     of

     AI

    -powered

     automation

     in

     various

     industries

    .

     For

     example

    ,

     in

     manufacturing

    ,

     AI

     can

     be

     used

     to

     reduce

     labor

     costs

    ,

     improve

     accuracy

     and

     reduce

     errors

    .
    


    2

    .

     More

     sophisticated

     applications

     of

     AI

    :

     AI

     is

     already

     being

     used

     in

     a

     wide

     range

     of

     applications

    ,

     but

     the

     possibilities

     are

     likely

     to

     expand

    .

     For

     example

    ,

     AI

    -powered

     chat

    bots

     can

     be

     used

     to

     provide

     customer

     support

    ,

     automate

     customer

     service

    ,

    



```python
llm.shutdown()
```
