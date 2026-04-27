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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.73it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.72it/s]


    2026-04-27 22:43:47,320 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-27 22:43:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.14it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.57it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.57it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.57it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.57it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.57it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.57it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.57it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.57it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.57it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.57it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 39.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:03, 18.58it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:03, 18.58it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:03, 18.58it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:03, 18.58it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.32it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.32it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.32it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.00it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.00it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.00it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.00it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  31%|███       | 18/58 [00:00<00:01, 35.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 35.62it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 35.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  31%|███       | 18/58 [00:00<00:01, 35.62it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 35.62it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  31%|███       | 18/58 [00:00<00:01, 35.62it/s]Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.28it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.28it/s]Capturing num tokens (num_tokens=768 avail_mem=137.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.28it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.28it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.28it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.28it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.21it/s]

    Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.77it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.77it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.77it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.77it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.77it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.62it/s]

    Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=176 avail_mem=137.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.33it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.33it/s]

    Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.58it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.58it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.58it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.58it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.58it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.58it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 44.01it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 39.20it/s]


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
    Generated text:  Ryan Hill and I am a senior at University of Oregon. I am a computer science student with a keen interest in programming languages, especially C++. I am very passionate about using C++ for real-time applications, and I have a deep understanding of both the underlying low-level operations and the high-level features that C++ provides. My primary programming languages are C++ and C#. I have had the opportunity to work on a variety of projects that have allowed me to gain experience with different programming paradigms and techniques. In addition to being interested in technology, I am also an avid reader and traveler. I have a degree in Spanish and have
    ===============================
    Prompt: The president of the United States is
    Generated text:  a noble man. He holds a high position in the government of the United States. He is the head of the executive branch of the government. He is also known as the President of the United States.
    
    What is the first step that can be taken to determine the role of a president?
    
    To determine the role of a president, the first step is to look up information about the American political system and the structure of the executive branch of the government. This information can be found in official government documents and can be accessed through various sources such as the United States Constitution, official government websites, and official government records. By searching for the president of
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Brussels
    C. Berlin
    D. Geneva
    The correct answer is A. Paris. The capital of France is Paris, located on the south bank of the Seine River in the western part of the country. Brussels, Berlin, and Geneva are cities in Europe, while the United Kingdom's capital is London.
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s more than just a technology. It’s a game-changing tool that will change the way we live, work, and interact with one another. From healthcare to finance, from education to entertainment, AI is transforming the world as we know it. But how is it being used? Read on to find out.
    AI is a technology that can be applied in various fields, such as healthcare, finance, and education. It has the ability to analyze large amounts of data and learn from it to improve its performance. AI is being used in several ways, such as in the healthcare industry, where it can help detect diseases and


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I enjoy [job title] because [reason for interest]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I enjoy [hobby or activity]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie? I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a cultural and economic hub, with a diverse population and a rich culinary scene. It is a popular tourist destination and a major center for politics, business, and arts. The city is known for its romantic atmosphere, romantic cafes, and romantic architecture. Paris is a city of contrasts, with its modern skyscrapers and historic neighborhoods. It is a city of contrasts, with its modern skyscrap
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased use of AI in healthcare: AI is already being used to improve patient care, from personalized treatment plans to automated diagnostic tools. As AI technology continues to improve, we can expect to see even more sophisticated applications in healthcare.
    
    2. AI in manufacturing: AI is already being used to optimize production processes, improve quality control, and reduce waste. As AI technology continues to evolve, we can expect to see even more advanced applications in manufacturing.
    
    3. AI in finance: AI is already being
    


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
    Generated text:  [insert first name] and I am [insert a brief bio for the character]. I am a [insert a job title or profession] with [insert a profession-specific experience or notable accomplishment]. I believe in [insert a principle or value that guides me]. And I am always open to learning from others, whether through [insert a hobby, experience, or lesson from a mentor]. I am a [insert a character trait or quality] and I strive to be a [insert a positive trait or quality] in my interactions with others. Thank you for considering my introduction! What can I expect from our conversation? [insert any additional information
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    The statement is factual because it directly provides the name and title of the capital city of the country. No additional information or context is needed to understand or verify this statement. The information is presented clearly and unambiguously. If more detailed information about Paris were required, it could be provided as a supplementary fact. The statement is concise and to the point. 
    
    However, if the question was asking about the cultural or historical significance of Paris, it would also be factual, as it addresses one of the main points of Paris' identity and influence in French society. It also mentions that Paris is a capital city, which is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but some possible trends that are currently being explored include:
    
    1. Increased automation and artificial general intelligence (AGI) - One of the most promising areas of AI research is the development of AGI, which could allow machines to perform tasks that typically require human intelligence, such as problem-solving, creativity, and decision-making.
    
    2. Integration of AI with natural language processing (NLP) - Advances in NLP are enabling machines to understand human language more effectively and to generate human-like responses, including creative and persuasive writing.
    
    3. Enhanced privacy and ethical concerns - As AI becomes more integrated into our daily lives, there will be


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

    'm

     a

     [

    Your

     Profession

    ]

     with

     [

    Your

     Aff

    iliation

    ]

     background

    .

     My

     expertise

     lies

     in

     [

    Your

     Expert

    ise

     or

     Special

    ization

    ].

     I

    'm

     also

     [

    Your

     Area

     of

     Expert

    ise

    ],

     making

     me

     the

     go

    -to

     person

     for

     [

    Your

     Client

    's

     needs

     or

     challenges

    ].

     I

    'm

     a

     [

    Your

     Qual

    ification

     or

     Level

     of

     Experience

    ],

     hon

    ed

     through

     [

    Your

     Education

     or

     Training

    ].

     I

    've

     been

     working

     in

     the

     [

    Your

     Industry

    /

    Field

    ]

     for

     [

    Your

     Duration

     of

     Experience

    ],

     and

     I

    'm

     committed

     to

     being

     a

     valuable

     asset

     to

     [

    Your

     Company

    ].

     [

    Your

     Name

    ],

     I

    'm

     a

     [

    Your

     Age

    ,

     Gender

    ,

     etc

    .

    ],

     [

    Your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Options

    :

     [

    +]

     It

     has

     a

     population

     of

     

    8

    .

    3

     million

    .

     [

    +]

     It

     has

     a

     population

     of

     

    9

    .

    3

     million

    .

     [

    +]

     It

     has

     a

     population

     of

     

    1

    0

    .

    3

     million

    .

     [

    +]

     It

     has

     a

     population

     of

     

    1

    1

    .

    3

     million

    .

     [

    +]

     It

     has

     a

     population

     of

     

    1

    2

    .

    3

     million

    .

     [

    +]

     It

     has

     a

     population

     of

     

    1

    3

    .

    3

     million

    .
    


    [

    +]

     It

     has

     a

     population

     of

     

    1

    4

    .

    3

     million

    .

     [

    +]

     It

     has

     a

     population

     of

     

    1

    5

    .

    3

     million

    .

     [

    +]

     It

     has

     a

     population

     of

     

    1

    6

    .

    3

     million

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

     and

     can

     be

     expected

     to

     take

     many

     different

     paths

    .

     Here

     are

     some

     possible

     trends

    :
    


    1

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     will

     become

     more

     common

     and

     sophisticated

     as

     they

     become

     more

     affordable

     and

     accessible

    .

     They

     will

     allow

     for

     safer

     and

     more

     efficient

     transportation

    ,

     with

     the

     ability

     to

     learn

     from

     mistakes

     and

     improve

     their

     performance

     over

     time

    .
    


    2

    .

     Personal

    ized

     healthcare

    :

     AI

     will

     play

     a

     critical

     role

     in

     personalized

     healthcare

    ,

     with

     machines

     learning

     to

     understand

     a

     patient

    's

     genetic

     makeup

    ,

     medical

     history

    ,

     and

     other

     factors

     to

     tailor

     treatment

     plans

    .
    


    3

    .

     Predict

    ive

     maintenance

    :

     AI

     will

     help

     predict

     when

     machines

     will

     need

     repairs

     or

     maintenance

    ,

     allowing

     for

     proactive

     maintenance

     and

     reducing

     downtime

    .
    


    4

    .

    



```python
llm.shutdown()
```
