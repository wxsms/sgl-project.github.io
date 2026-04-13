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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.42it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.40it/s]


    2026-04-13 00:04:28,162 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 00:04:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.67it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.67it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.01it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.01it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.13it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.13it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.13it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.13it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.13it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.13it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.13it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.13it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]

    Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.91it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.91it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.91it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.91it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.91it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.91it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.91it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.34it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.34it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.34it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.34it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.34it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.34it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.34it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.64it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.64it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.64it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.64it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.64it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.64it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.64it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.64it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.64it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.74it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.29 GB):  31%|███       | 18/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.27 GB):  31%|███       | 18/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=960 avail_mem=120.29 GB):  31%|███       | 18/58 [00:00<00:01, 36.00it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=119.06 GB):  31%|███       | 18/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=896 avail_mem=119.06 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.69it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.69it/s]Capturing num tokens (num_tokens=768 avail_mem=118.96 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.69it/s]Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.69it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.69it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.69it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.88it/s]Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.88it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.88it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.88it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.88it/s]

    Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.88it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.65it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.65it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.65it/s]Capturing num tokens (num_tokens=288 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.65it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.65it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.65it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.68it/s]

    Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.68it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.61it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=64 avail_mem=118.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.92it/s]

    Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.25it/s] Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 45.99it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 40.16it/s]


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
    Generated text:  Mark and I was born in 1968. I graduated from the University of Manchester in 1991 and from the University of Edinburgh in 1995 with a BSc in Computer Science. I was the Chief Architect at SAP AG until 2009, during which time I worked on over 60 projects and had the privilege of being the Chief Technology Officer of SAP AG for two years.
    In 2010 I was appointed the Chief Executive Officer of HCL Technologies Group and in 2015 I was appointed CEO of SaaS company, SAP. I have a passion
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office with a term of eight years. For every year of service, the president receives an additional $1,000,000 in stipends. Additionally, every year, the president receives a bonus of $2,000,000. If the president is scheduled to serve for 9 years, how much total stipends and bonuses will the president receive?
    
    To determine the total stipends and bonuses the president will receive, we need to break down the calculation into the relevant components.
    
    1. **Stipends:**
       - The president receives $1,000,00
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is famous for its charming places such as the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. The history of Paris, from its founding to the present day, is fascinating. The city was founded around the year 800 AD by the ancient Romans, and the name Paris means "the highest place" or "the highest land". It is named after the Latin word Paris, meaning "the highest land". It was also the capital of the kingdom of Burgundy. After the Norman Conquest in 1066, the city was renamed Amiens, and Paris became the capital of
    ===============================
    Prompt: The future of AI is
    Generated text:  in human hands. We can be sure of that. But, how can we ensure AI is always safe and secure? This is a question that remains a concern for the government and the industry as they work towards building a future where AI is deployed safely and responsibly.
    Weighing the Pros and Cons
    In recent years, the debate over the safety and reliability of AI has become more pronounced. There are a number of factors to consider when it comes to ensuring that AI is safe and secure. One of these is the scope and impact of the technology, which can be influenced by a wide range of factors.
    The scope of AI technology can


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


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and is known for its rich history, art, and cuisine. It is also home to many important political and military institutions. The city is known for its vibrant nightlife and is a popular destination for tourists and locals alike. Paris is a city of contrasts, with its historical architecture and modern amenities. It is a city of art, culture, and history, and is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability. This will likely lead to more
    


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
    Generated text:  [Name], and I'm a [Job Title] with [Company Name]. I'm currently [Current Position] at [Company Name]. I'm passionate about [specific interest or career goal]. I'm known for [reasons for success], and I have a unique approach to [a specific aspect of the job], which allows me to excel in this role. I'm excited to learn more about your company and can't wait to discuss how I can contribute to your success. [Name]. 
    
    Feel free to customize this short self-introduction to better reflect the character and the role. What specific aspects of the job do you specialize in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. It is also home to many world-renowned museums, fashion houses, and traditional festivals. Paris is a bustling metropolis with a rich cultural history and a diverse population. Its importance as a major city is reflected in its status as one of the world's most visited cities, with millions of tourists annually visiting the city. The city has a strong French identity and is celebrated for its art, music, literature, and cuisine. Paris is a historic and modern city with a strong sense of community and tradition. Its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  shaped by many factors, including advances in computing power, advances in machine learning, and advancements in the development of hardware and software. Here are some potential future trends in AI:
    
    1. Increased automation and AI integration: As AI technology continues to advance, more companies will adopt AI to automate repetitive and labor-intensive tasks, increasing efficiency and productivity. This could lead to the development of new industries and new jobs.
    
    2. Personalization and AI: AI is becoming increasingly capable of analyzing and understanding data to provide personalized recommendations and experiences. This could lead to more personalized services, such as personalized medicine and customer service, and could also lead to the development


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

     [

    occupation

    ]

     in

     [

    field

    ].

     I

    've

     always

     had

     an

     interest

     in

     [

    interest

    ]

     and

     have

     been

     a

     [

    number

    ]

     times

     a

     [

    professional

     title

    ]

     for

     [

    duration

    ]

     years

    .

     I

     have

     a

     passion

     for

     [

    why

     I

     love

     what

     I

     do

    ],

     and

     I

    'm

     constantly

     learning

     and

     growing

     as

     a

     [

    field

    ]

     professional

    .

     I

    'm

     confident

     in

     [

    reason

    s

     for

     confidence

    ],

     and

     I

    'm

     excited

     to

     see

     where

     this

     journey

     will

     lead

     me

     next

    .


    [

    Name

    ]:

     A

     charismatic

     and

     experienced

     professional

     in

     the

     field

     of

     [

    field

    ],

     with

     a

     passion

     for

     [

    interest

    ].

     I

    'm

     a

     [

    number

    ]

     times

     a

     [

    professional

     title

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    What

     are

     the

     three

     main

     types

     of

     renewable

     energy

     sources

     in

     the

     United

     States

    ?
    


    The

     three

     main

     types

     of

     renewable

     energy

     sources

     in

     the

     United

     States

     are

     solar

    ,

     wind

    ,

     and

     hydro

    electric

    ity

    .
    


    Can

     you

     explain

     the

     concept

     of

     "

    green

     technology

    "

     and

     provide

     an

     example

     of

     a

     technology

     that

     falls

     under

     this

     category

    ?

     Green

     technology

     refers

     to

     innovative

     methods

     that

     reduce

     environmental

     impact

     by

     using

     energy

    ,

     materials

    ,

     and

     resources

     in

     a

     sustainable

     and

     environmentally

     responsible

     way

    .

     An

     example

     of

     green

     technology

     is

     the

     development

     of

     renewable

     energy

     sources

    ,

     such

     as

     solar

    ,

     wind

    ,

     and

     hydro

    electric

    ity

    ,

     which

     do

     not

     emit

     greenhouse

     gases

     or

     other

     pollutants

     when

     used

    .
    


    Please

     summarize

     the

     key

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     depends

     on

     a

     variety

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

     machine

     learning

     and

     natural

     language

     processing

    ,

     and

     advancements

     in

     robotics

     and

     automation

    .

     Some

     potential

     trends

     that

     are

     currently

     being

     explored

     include

    :
    


    1

    .

     Adv

    ancements

     in

     AI

     for

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     improve

     diagnosis

     and

     treatment

     outcomes

    ,

     and

     we

     may

     see

     even

     more

     breakthrough

    s

     in

     the

     coming

     years

    .

     For

     example

    ,

     researchers

     are

     working

     on

     developing

     AI

     algorithms

     that

     can

     analyze

     medical

     images

    ,

     detect

     abnormalities

    ,

     and

     provide

     early

     warning

     signs

     of

     disease

    .
    


    2

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     advanced

     and

     autonomous

    ,

     there

     is

     a

     growing

     need

     for

     ethical

     considerations

     to

    



```python
llm.shutdown()
```
