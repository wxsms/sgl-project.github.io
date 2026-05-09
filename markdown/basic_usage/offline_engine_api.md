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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.46it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.46it/s]


    2026-05-09 01:27:57,161 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 01:27:57] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.81it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.81it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.81it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 10.36it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 10.36it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 10.36it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 10.36it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 10.36it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 10.36it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 10.36it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 10.36it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:05<00:01, 14.80it/s]

    Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 21.94it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 29.96it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 29.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.35 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.35 GB):   3%|▎         | 2/58 [00:00<00:03, 17.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 17.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 17.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 17.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   9%|▊         | 5/58 [00:00<00:02, 20.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.33 GB):   9%|▊         | 5/58 [00:00<00:02, 20.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 20.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 20.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.32 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.32 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.31 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.22it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  21%|██        | 12/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.30 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.11it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.29 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.79it/s] Capturing num tokens (num_tokens=896 avail_mem=72.28 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=768 avail_mem=72.27 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.49it/s]

    Capturing num tokens (num_tokens=704 avail_mem=72.27 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.16it/s]Capturing num tokens (num_tokens=512 avail_mem=72.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.16it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.16it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.16it/s]Capturing num tokens (num_tokens=416 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.16it/s]Capturing num tokens (num_tokens=384 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.16it/s]Capturing num tokens (num_tokens=384 avail_mem=72.26 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.46it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.46it/s]

    Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.46it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.46it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.46it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.46it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.13it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.66it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.66it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.66it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.66it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.66it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.66it/s]Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.04it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.04it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.04it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.04it/s]

    Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.04it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.04it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.14it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.14it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.14it/s]

    Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.14it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.14it/s] Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  98%|█████████▊| 57/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  98%|█████████▊| 57/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 32.13it/s]


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
    Generated text:  Ani, a 21-year-old Ukrainian graduate and I am a passionate web developer.
    
    I have experience working on a variety of projects, such as a web application, mobile app, and a blog. My technical skills include HTML, CSS, JavaScript, Python, and Flask. I also have experience with web scraping and data analysis.
    
    My favorite hobby is learning new languages and expanding my knowledge. I also enjoy reading books and listening to music.
    
    My goal is to create visually appealing and easy-to-use websites that are intuitive to use and have a positive impact on people's lives. I am always looking for ways to improve my skills and
    ===============================
    Prompt: The president of the United States is
    Generated text:  now planning to visit six different countries. Each country has a different number of cities, ranging from 3 to 6. The president wants to visit the countries in such a way that the total number of cities visited is exactly 18. How many cities will the president visit in each country? To determine how many cities the president will visit in each country, we need to find a combination of numbers between 3 and 6 that sum up to 18. Let's go through the possible combinations step by step.
    
    1. **Summing up to 18 with a single number:**
       - If the president visits
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    
    B) Lyon
    
    C) Rennes
    
    D) Marseille
    
    To determine the capital of France, we can follow these steps:
    
    1. **Identify the capital cities of other countries**: First, we need to know the capital cities of other countries. France is a European country, so its capital is typically Lyon, which is the capital of France.
    
    2. **Verify the capital cities of other European countries**: Another European country that has its capital city as Lyon is Belgium. Belgium has its capital city as Brussels.
    
    Since the problem does not specify a particular country, we need to consider the most common capital cities
    ===============================
    Prompt: The future of AI is
    Generated text:  being shaped by a variety of factors and technologies, from quantum computing and machine learning to natural language processing and computer vision. In this article, we will explore the latest developments and applications of AI in various industries, including healthcare, finance, and transportation, and how they are shaping the future of technology and society. We will also examine the ethical considerations surrounding AI and its potential impact on human society. 
    Please summarize the key technologies and industries that are shaping the future of AI. Additionally, please provide examples of how AI is being used in healthcare, finance, and transportation, and how it is impacting these industries. Finally, please discuss the ethical


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


    Generated text:  [Name] and I'm a [occupation] with [number] years of experience in [field]. I'm a [type of person] who is always [positive trait]. I'm [gender] and I'm [age]. I'm [height] inches tall and [weight] pounds. I have [physical characteristic] and [physical characteristic]. I'm [gender] and I'm [age]. I'm [height] inches tall and [weight] pounds. I have [physical characteristic] and [physical characteristic]. I'm [gender] and I'm [age]. I'm [height] inches tall and [weight]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of the French Revolution and the seat of the French government. Paris is a cultural and historical center with a rich history dating back to ancient times. The city is known for its cuisine, fashion, and art scene, and is a major tourist destination. It is also home to many famous landmarks and attractions, including the Louvre, the Arc de Triomphe, and the Champs-Élysées. Paris is a vibrant and dynamic city that continues to be a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn from and adapt to the behavior and preferences of humans. This will enable more sophisticated and personalized AI systems that can better understand and respond to the needs of their users.
    
    2. Enhanced machine learning capabilities: AI systems will become even more capable of learning from data and making decisions based on that data. This will allow for more complex and sophisticated AI systems that can solve problems that were previously beyond the capabilities of human intelligence.
    
    3. Improved privacy and security: As AI
    


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
    Generated text:  [Name], and I'm a [career] with [number of years of experience]. I'm [insert your experience and background]. I'm a [insert your personal qualities or attributes]. And I'm always looking for new challenges and opportunities to grow and excel in my career. What are you looking forward to? Dear [Recipient's name], I'm glad to meet you, and I hope you're doing well. I'm excited to meet you and know that I can help you in any way I can. As a [insert your profession], I'm always looking for opportunities to learn and grow. I've been pursuing a career in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The statement is: The capital of France is Paris. 
    
    To elaborate, Paris is the largest city in France and the second-largest city in Europe, with a population of over 11 million people as of 2021. It is known for its iconic landmarks such as Notre-Dame Cathedral, the Louvre Museum, and the Eiffel Tower, as well as for its rich cultural and historical heritage, including its historical defenses and the influence of various architectural styles and art movements. Paris is also renowned for its fashion industry and the city of love, with the famous Eiffel Tower serving as a symbol of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very exciting and we can expect it to continue to grow and evolve in many ways. Here are a few possible trends in AI:
    
    1. Increased Integration with Other Technologies: AI is becoming more integrated with other technologies, including machine learning, computer vision, and natural language processing. This integration will enable AI to work more effectively with other systems, such as sensors, cameras, and other machines, to achieve more complex and accurate results.
    
    2. Advanced Machine Learning and Deep Learning: AI algorithms will become more complex and sophisticated, making it possible to train models that can learn and adapt to new data. This will lead to more advanced and accurate results


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

    First

     name

    ]

     and

     I

     am

     a

     [

    Last

     name

    ]

     (

    profession

    )

     [

    Brief

     biography

    ]

     (

    optional

    ).

     I

     am

     always

     ready

     to

     learn

     and

     improve

     my

     skills

    .

     I

     am

     a

     [

    Kind

     of

     individual

    ]

     [

    Occup

    ation

    ]

     (

    e

    .g

    .

     Writer

    ,

     Engineer

    ,

     Artist

    ,

     etc

    .)

     and

     I

     love

     to

     [

    Ex

    plain

     why

     you

     love

     your

     profession

    ].

     I

     am

     looking

     to

     build

     a

     [

    Career

     goal

    ]

     and

     I

     want

     to

     make

     the

     world

     a

     [

    Positive

     or

     Negative

    ]

     place

    .

     If

     you

     have

     any

     questions

     about

     my

     background

     or

     skills

    ,

     please

     feel

     free

     to

     ask

    .

     What

     is

     your

     name

    ?

     What

     is

     your

     profession

    ?

     What

     is

     your

     kind

     of

     individual

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     beautiful

     architecture

    ,

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     and

     rich

     cultural

     history

    .

     The

     city

     is

     also

     home

     to

     the

     French

     parliament

     and

     the

     seat

     of

     government

     for

     the

     country

    .

     Paris

     is

     a

     bustling

     and

     dynamic

     city

     with

     a

     diverse

     population

    ,

     and

     it

     is

     an

     important

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     France

    .

     The

     city

     has

     a

     rich

     history

     dating

     back

     thousands

     of

     years

     and

     is

     a

     UNESCO

     World

     Heritage

     site

    .

     Paris

     is

     also

     a

     popular

     tourist

     destination

    ,

     and

     it

     is

     a

     symbol

     of

     France

    's

     cultural

     and

     artistic

     heritage

    .

     The

     city

     is

     known

     for

     its

     delicious

     cuisine

    ,

     including

     French

     gastr

    onomy

    ,

     and

     for

     its

     charming

     streets

    
    
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

     potential

     trends

    .

     One

     of

     the

     most

     important

     trends

     is

     the

     rise

     of

     AI

     in

     industries

     such

     as

     healthcare

    ,

     finance

    ,

     and

     manufacturing

    .

     With

     the

     increasing

     demand

     for

     data

    -driven

     insights

    ,

     AI

     is

     being

     used

     to

     improve

     the

     accuracy

     of

     medical

     diagnoses

    ,

     optimize

     supply

     chains

    ,

     and

     enhance

     the

     efficiency

     of

     manufacturing

     processes

    .

     AI

     is

     also

     being

     used

     to

     automate

     tasks

    ,

     such

     as

     customer

     service

     and

     order

     fulfillment

    ,

     which

     can

     help

     reduce

     costs

     and

     improve

     customer

     satisfaction

    .

     Additionally

    ,

     AI

     is

     being

     used

     to

     develop

     new

     technologies

    ,

     such

     as

     autonomous

     vehicles

    ,

     that

     have

     the

     potential

     to

     revolution

    ize

     transportation

     and

     transport

    .
    


    Another

     trend

     that

     is

     likely

     to

     be

     prominent

    



```python
llm.shutdown()
```
