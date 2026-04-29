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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.51it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.50it/s]


    2026-04-29 09:09:07,688 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 09:09:07] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:33,  4.80s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:33,  4.80s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:33,  4.80s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:33,  4.80s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:33,  4.80s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.06it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.09it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.09it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.09it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.09it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.09it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.09it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.09it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.09it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.09it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.09it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.79it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.67it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.67it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.67it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 28.49it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 28.49it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 38.07it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 38.07it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=117.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.07 GB):   3%|▎         | 2/58 [00:00<00:03, 18.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=117.06 GB):   3%|▎         | 2/58 [00:00<00:03, 18.21it/s]Capturing num tokens (num_tokens=6656 avail_mem=117.04 GB):   3%|▎         | 2/58 [00:00<00:03, 18.21it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=117.04 GB):   7%|▋         | 4/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=6144 avail_mem=117.04 GB):   7%|▋         | 4/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=5632 avail_mem=117.02 GB):   7%|▋         | 4/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.44 GB):   7%|▋         | 4/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.44 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.44 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.44 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.44 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.09it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=116.43 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.43 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.22it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.43 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.22it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.42 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.22it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.42 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.22it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.42 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.22it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.41 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.22it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.41 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.41 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.41 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.40 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.41it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=116.40 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.38 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.38 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=960 avail_mem=116.40 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.67it/s] Capturing num tokens (num_tokens=896 avail_mem=116.39 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=832 avail_mem=116.39 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=768 avail_mem=116.39 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=704 avail_mem=116.38 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=704 avail_mem=116.38 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.70it/s]Capturing num tokens (num_tokens=640 avail_mem=116.38 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.70it/s]Capturing num tokens (num_tokens=576 avail_mem=116.38 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.70it/s]

    Capturing num tokens (num_tokens=512 avail_mem=116.36 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.70it/s]Capturing num tokens (num_tokens=480 avail_mem=116.37 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.70it/s]Capturing num tokens (num_tokens=448 avail_mem=116.35 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.70it/s]Capturing num tokens (num_tokens=448 avail_mem=116.35 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=416 avail_mem=116.35 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=384 avail_mem=116.34 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.57it/s]Capturing num tokens (num_tokens=352 avail_mem=116.34 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.57it/s]Capturing num tokens (num_tokens=320 avail_mem=116.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.57it/s]Capturing num tokens (num_tokens=288 avail_mem=116.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.57it/s]Capturing num tokens (num_tokens=288 avail_mem=116.33 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=256 avail_mem=116.33 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.02it/s]

    Capturing num tokens (num_tokens=240 avail_mem=116.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=224 avail_mem=116.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=208 avail_mem=116.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=192 avail_mem=116.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=192 avail_mem=116.32 GB):  71%|███████   | 41/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=176 avail_mem=116.31 GB):  71%|███████   | 41/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=160 avail_mem=116.31 GB):  71%|███████   | 41/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=144 avail_mem=116.31 GB):  71%|███████   | 41/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=128 avail_mem=116.30 GB):  71%|███████   | 41/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=112 avail_mem=116.30 GB):  71%|███████   | 41/58 [00:01<00:00, 42.03it/s]

    Capturing num tokens (num_tokens=112 avail_mem=116.30 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=96 avail_mem=116.30 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.56it/s] Capturing num tokens (num_tokens=80 avail_mem=116.26 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=64 avail_mem=116.26 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=48 avail_mem=116.26 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=32 avail_mem=116.25 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=32 avail_mem=116.25 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=28 avail_mem=116.25 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=24 avail_mem=116.25 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=20 avail_mem=116.24 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=16 avail_mem=116.24 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.64it/s]

    Capturing num tokens (num_tokens=12 avail_mem=116.24 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=12 avail_mem=116.24 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.25it/s]Capturing num tokens (num_tokens=8 avail_mem=116.23 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.25it/s] Capturing num tokens (num_tokens=4 avail_mem=116.23 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.25it/s]Capturing num tokens (num_tokens=4 avail_mem=116.23 GB): 100%|██████████| 58/58 [00:01<00:00, 36.54it/s]


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
    Generated text:  Alan. I'm from USA. I'm a writer. But I've always wanted to write my own book. I'm a good writer, so I'm very excited to write a book of my own. First of all, I want to know what it is to be a writer. Then I want to write my book and make it a success. Then I want to write my own book as well. I love writing. And writing is my best friend. I like writing all the time. I think writing is the most important thing in my life. Then I want to make my book well known. I want my book to be
    ===============================
    Prompt: The president of the United States is
    Generated text:  a powerful person. He or she is the leader of the country and the head of the government. The president is the president of the United States and is the chief executive of the federal government. He or she is the highest-ranking elected official in the country. The president also serves as the commander-in-chief of the armed forces. In addition, the president is the primary religious and political figurehead in the United States. In the United States, the president is often referred to as "the boss." In the United States, the president is referred to as the boss. In the United States, the president is referred to as "the boss
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris was the capital of France from 1804 to 1870. The last capital of France was Nice.
    From 1789 until 1870, Paris was the capital of France. The last capital of France was Nice. 
    
    Are these two sentences paraphrases of each other?
    Choose from: [I] no [II] yes
    
    [I] no
    
    Explanation for the above reasoning: The first sentence states that Paris was the capital from 1804 to 1870. The second sentence states that Paris was the capital from 1789 until
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, and with that uncertainty comes the need for an open, transparent, and diverse community of developers, educators, researchers, and practitioners who can work together to solve the problems posed by this future. In this article, we discuss how a new community of developers, educators, and researchers can collaborate on the future of AI.
    AI is a complex and rapidly evolving field, and the success of any AI project depends on the ability of the community to share and collaborate with one another. Without a community of developers, educators, researchers, and practitioners, it will be difficult to overcome the challenges posed by the future of AI.
    In this article,


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] [Vehicle Name]. I'm currently [Current Location]. I'm [Current Activity]. I'm [Current Hobby or Passion]. I'm [Current Goal or Purpose]. I'm [Current Strengths or Weaknesses]. I'm [Current Interests or Interests]. I'm [Current Challenges or Challenges]. I'm [Current Education or Training]. I'm [Current Family]. I'm [Current Relationships]. I'm [Current Community]. I'm [Current Religion]. I'm [Current Language]. I'm [Current
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Middle Ages and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for art, culture, and cuisine, and is home to many world-renowned museums, theaters, and restaurants. The city is also known for its fashion industry, with many famous fashion designers and boutiques located in the city. Paris is a vibrant and dynamic city with a rich cultural heritage and a strong sense of community. It is a popular tourist destination
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential trends that are likely to shape the future of AI:
    
    1. Increased automation and artificial intelligence: As AI continues to advance, we are likely to see an increase in automation and artificial intelligence in various industries. This could lead to the creation of more efficient and cost-effective solutions to problems, as well as the creation of new jobs that require little or no human intervention.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be an increased need for privacy
    


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
    Generated text:  [insert name], and I'm [insert occupation]. I love to [insert something positive about my occupation] and I enjoy [insert something positive about my interests or hobbies]. I'm a [insert occupation] who is always looking for opportunities to [insert something positive about my abilities or skills]. I'm friendly, kind, and always eager to help others. I love [insert something positive about my personality or character]. I'm happy to meet new people and share my knowledge and experiences with them. I hope to work with you in the future! [insert name] [insert job title] [insert contact information]. [insert occupation]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its historical significance and cultural richness. It is the country's largest and most populous city, home to over a million people and serving as the administrative, economic, and cultural center of France. Paris is renowned for its skyline, including the Eiffel Tower, and its architecture, art, and food. It is known for its fashion and theater scenes, including the Notre-Dame Cathedral and the Louvre Museum. The city is a center of education, science, and arts, with many notable institutions and landmarks. Paris has been a hub for international diplomacy and is home to many influential organizations and institutions. It
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by significant advancements, innovation, and integration with new technologies. Some of the possible future trends in AI are:
    
    1. Personalization: AI will continue to become more personalized, with the ability to learn from user behavior and preferences to provide more relevant and tailored experiences.
    
    2. Automation: AI will continue to become more automated, with the ability to perform tasks that were previously performed by humans. This will include tasks such as data entry, production, and customer service.
    
    3. Artificial Intelligence in Healthcare: AI will continue to play a critical role in healthcare, with the ability to analyze medical data, detect trends, and provide


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

    ].

     I

     am

     a [

    insert

     profession

    ]

     who

     have

     always

     been

     passionate

     about

     [

    insert

     something

     that

     makes

     you

     unique

     or

     interesting

    ].

     I

    've

     always

     been

     driven

     to

     find

     solutions

     and

     try

     new

     things

     to

     push

     the

     boundaries

     of

     what

    's

     possible

    .

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     learning

     opportunities

     to

     grow

     and

     develop

     as

     a

     person

    .

     I

    'm

     a

     person

     who

     is

     always

     on

     the

     move

     and

     who

     is

     always

     looking

     for

     new

     experiences

     and

     opportunities

    .

     I

    'm

     a

     [

    insert

     your

     personality

     trait

     or

     interest

    ]

     that

     is

     always

     looking

     for

     the

     next

     big

     adventure

     and

     the

     next

     opportunity

     to

     achieve

     my

     full

     potential

    .

     Thank

     you

     for

     having

     me

    .

     [

    Your

     Name

    ]

     looks

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    To

     expand

     the

     scope

     of

     the

     statement

    :


    -

     Paris

     is

     the

     largest

     city

     in

     France

    .


    -

     It

     is

     known

     as

     the

     "

    City

     of

     Love

    "

     and

     is

     home

     to

     several

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

     the

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .


    -

     Paris

     has

     a

     rich

     history

     dating

     back

     to

     the

     ancient

     Roman

     Empire

     and

     has

     been

     a

     major

     hub

     of

     trade

     and

     culture

     for

     centuries

    .


    -

     It

     is

     also

     home

     to

     a

     large

     French

    -speaking

     minority

     and

     is

     considered

     a

     cultural

     melting

     pot

     of

     European

     countries

    .


    -

     The

     city

     is

     known

     for

     its

     diverse

     food

     and

     wine

     culture

    ,

     as

     well

     as

     its

     annual

     F

    ête

     de

     la

     France

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     quite

     fascinating

     and

     complex

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     its

     development

    :
    


    1

    .

     Adv

    ancements

     in

     Machine

     Learning

    :

     With

     the

     continued

     development

     of

     machine

     learning

     algorithms

    ,

     AI

     will

     become

     more

     complex

     and

     intelligent

    .

     Machine

     learning

     will

     become

     even

     more

     sophisticated

    ,

     enabling

     AI

     to

     learn

     from

     data

     and

     adapt

     to

     new

     situations

    .
    


    2

    .

     AI

     Integration

     with

     IoT

    :

     As

     IoT

     devices

     become

     more

     prevalent

    ,

     AI

     will

     be

     integrated

     into

     them

     to

     improve

     their

     functionality

     and

     efficiency

    .

     This

     will

     lead

     to

     the

     creation

     of

     more

     connected

     and

     intelligent

     systems

    .
    


    3

    .

     AI

     for

     Healthcare

    :

     AI

     will

     play

     a

     significant

     role

     in

     healthcare

    ,

     with

     the

     development

     of

     AI

    -powered

     medical

     robots

     and

    



```python
llm.shutdown()
```
