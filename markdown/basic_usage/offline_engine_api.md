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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.28it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.28it/s]


    2026-04-10 20:50:44,903 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 20:50:44] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:25,  2.56s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:25,  2.56s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:25,  2.56s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:39,  1.38it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:39,  1.38it/s]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:39,  1.38it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:20,  2.60it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:20,  2.60it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:20,  2.60it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:20,  2.60it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:02<00:10,  4.98it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:02<00:10,  4.98it/s]

    Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:02<00:10,  4.98it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:10,  4.98it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:06,  7.65it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:06,  7.65it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:06,  7.65it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:06,  7.65it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:06,  7.65it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:03, 11.95it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:03, 11.95it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:03, 11.95it/s]

    Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:03, 11.95it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:03, 11.95it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 16.12it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 16.12it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 16.12it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 16.12it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 16.12it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 16.12it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 21.61it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 21.61it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 21.61it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 21.61it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 21.61it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 24.68it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 24.68it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 24.68it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 24.68it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 24.68it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:00, 27.33it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:00, 27.33it/s]

    Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:00, 27.33it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:00, 27.33it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:00, 27.33it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:00, 27.33it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 31.11it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 31.11it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 31.11it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 31.11it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 31.11it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 31.11it/s]

    Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 34.36it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 35.27it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 35.27it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 35.27it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 35.27it/s]

    Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 36.82it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 36.82it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 36.82it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 36.82it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 36.82it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 36.82it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 36.82it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:04<00:00, 42.45it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:04<00:00, 42.45it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 13.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.39 GB):   3%|▎         | 2/58 [00:00<00:04, 11.73it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.39 GB):   3%|▎         | 2/58 [00:00<00:04, 11.73it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.39 GB):   3%|▎         | 2/58 [00:00<00:04, 11.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.39 GB):   7%|▋         | 4/58 [00:00<00:04, 13.40it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.40 GB):   7%|▋         | 4/58 [00:00<00:04, 13.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.41 GB):   7%|▋         | 4/58 [00:00<00:04, 13.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.41 GB):  10%|█         | 6/58 [00:00<00:03, 15.03it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.42 GB):  10%|█         | 6/58 [00:00<00:03, 15.03it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=43.41 GB):  10%|█         | 6/58 [00:00<00:03, 15.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.40 GB):  10%|█         | 6/58 [00:00<00:03, 15.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.40 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.64it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.39 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.38 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.64it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.37 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.64it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=43.37 GB):  21%|██        | 12/58 [00:00<00:02, 20.09it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.37 GB):  21%|██        | 12/58 [00:00<00:02, 20.09it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.36 GB):  21%|██        | 12/58 [00:00<00:02, 20.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.33 GB):  21%|██        | 12/58 [00:00<00:02, 20.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.35 GB):  21%|██        | 12/58 [00:00<00:02, 20.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.35 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.53it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.53it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.33 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.53it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.33 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.53it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=43.32 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.53it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.32 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.30 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=960 avail_mem=43.29 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.10it/s] Capturing num tokens (num_tokens=896 avail_mem=43.30 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=832 avail_mem=43.31 GB):  34%|███▍      | 20/58 [00:01<00:01, 26.10it/s]Capturing num tokens (num_tokens=832 avail_mem=43.31 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.01it/s]Capturing num tokens (num_tokens=768 avail_mem=43.30 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.01it/s]Capturing num tokens (num_tokens=704 avail_mem=43.30 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.01it/s]Capturing num tokens (num_tokens=640 avail_mem=43.29 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.01it/s]

    Capturing num tokens (num_tokens=576 avail_mem=43.28 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.01it/s]Capturing num tokens (num_tokens=576 avail_mem=43.28 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.71it/s]Capturing num tokens (num_tokens=512 avail_mem=43.27 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.71it/s]Capturing num tokens (num_tokens=480 avail_mem=43.28 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.71it/s]Capturing num tokens (num_tokens=448 avail_mem=43.28 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.71it/s]Capturing num tokens (num_tokens=416 avail_mem=43.27 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.71it/s]Capturing num tokens (num_tokens=416 avail_mem=43.27 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.01it/s]Capturing num tokens (num_tokens=384 avail_mem=43.27 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.01it/s]Capturing num tokens (num_tokens=352 avail_mem=43.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.01it/s]

    Capturing num tokens (num_tokens=320 avail_mem=43.25 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.01it/s]Capturing num tokens (num_tokens=288 avail_mem=43.24 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.01it/s]Capturing num tokens (num_tokens=288 avail_mem=43.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.71it/s]Capturing num tokens (num_tokens=256 avail_mem=43.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.71it/s]Capturing num tokens (num_tokens=240 avail_mem=43.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.71it/s]Capturing num tokens (num_tokens=224 avail_mem=43.22 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.71it/s]Capturing num tokens (num_tokens=208 avail_mem=43.22 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.71it/s]Capturing num tokens (num_tokens=192 avail_mem=43.22 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.71it/s]Capturing num tokens (num_tokens=192 avail_mem=43.22 GB):  71%|███████   | 41/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=176 avail_mem=43.22 GB):  71%|███████   | 41/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=160 avail_mem=43.21 GB):  71%|███████   | 41/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=144 avail_mem=43.21 GB):  71%|███████   | 41/58 [00:01<00:00, 37.12it/s]

    Capturing num tokens (num_tokens=128 avail_mem=43.21 GB):  71%|███████   | 41/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=112 avail_mem=43.20 GB):  71%|███████   | 41/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=96 avail_mem=43.20 GB):  71%|███████   | 41/58 [00:01<00:00, 37.12it/s] Capturing num tokens (num_tokens=96 avail_mem=43.20 GB):  81%|████████  | 47/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=80 avail_mem=43.20 GB):  81%|████████  | 47/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=64 avail_mem=43.19 GB):  81%|████████  | 47/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=48 avail_mem=43.19 GB):  81%|████████  | 47/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=32 avail_mem=43.19 GB):  81%|████████  | 47/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=28 avail_mem=43.18 GB):  81%|████████  | 47/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=24 avail_mem=43.18 GB):  81%|████████  | 47/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=24 avail_mem=43.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=20 avail_mem=43.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=16 avail_mem=43.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.03it/s]

    Capturing num tokens (num_tokens=12 avail_mem=43.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=8 avail_mem=43.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.03it/s] Capturing num tokens (num_tokens=4 avail_mem=43.16 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=4 avail_mem=43.16 GB): 100%|██████████| 58/58 [00:01<00:00, 31.90it/s]


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
    Generated text:  Shelia. I'm a 7th grader at a school in Texas. One day, my mom and dad were having a dinner. They were eating a big bowl of spaghetti with a lot of cheese and meat. After a while, they all were feeling hungry. My mom asked me, "Is it okay to have some pasta and cheese after that dinner? It's not the first time that's happened. I wonder what to do with it."
    
    What would be the best answer to the question: How should I respond to my mom's question?
    
    Responding to your mom's question can be a tricky situation. Here are some
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military tanks to buy. The tanks cost $250,000 each. However, there is a 50% chance that the government may need to return the tanks to the manufacturer before they are fully loaded. If the government decides to return the tanks, they will incur a $10,000 loss. To cover these unexpected costs, the president proposes to buy an additional 100 tanks and return them. The total cost of these additional tanks will be $200,000. If the government decides to keep the 100 tanks, the cost
    ===============================
    Prompt: The capital of France is
    Generated text:  _______. A. Paris B. London C. Paris D. Moscow
    
    The capital of France is Paris.
    
    Therefore, the correct answer is A. Paris. 
    
    I'll provide additional context: Paris is the capital city of France, situated on the Île de France (in the Seine River), in the Languedoc region, and the fourth-largest metropolitan area in Europe. It is known for its rich history, art, culture, and cuisine, and is a major tourist destination. 
    
    Note that some people might prefer to know the capital of Moscow or not include Moscow in the capital list. However, it's not typically
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of individuals and organizations like Google, whose CEO is Elon Musk and whose AI is akin to the first major breakthrough in computing. But it is all too easy to become complacent and neglect the importance of risk management when it comes to AI. The hyper-complex systems that are the underpinnings of the tech industry are not a topic that is always discussed, even when they become critical in times of crisis.
    For example, the North American Industry Association (NAIA) predicts that by 2025, AI will account for a significant portion of the industrial workforce in the US, which could have a profound impact


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is located in the south of the country. It is the largest city in Europe and the second-largest city in the world by population. Paris is known for its rich history, art, and culture, and is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also known for its fashion industry, with many famous fashion houses and designers operating in the area. Paris is a popular tourist destination and is a major economic center in France. It is also home to the French Parliament, the French Academy of Sciences,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential trends that could be expected in the future:
    
    1. Increased automation: As AI continues to advance, we can expect to see more automation in various industries, including manufacturing, transportation, and healthcare. This could lead to increased efficiency and productivity, but it could also lead to job displacement for some workers.
    
    2. Enhanced privacy and security: As AI becomes more advanced, we can expect to see more sophisticated ways of protecting privacy and security. This could include measures such as biometric authentication, encryption
    


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
    Generated text:  [Your Name], and I'm a [Your profession] with [Your background] experience. I'm passionate about [Your main area of interest or expertise]. I'm always looking for ways to help others, whether it's with computers, writing, or anything else that interests me. I believe in continuous learning and improvement, and I'm always eager to learn new things. Thank you for asking! How can I assist you today? As a fictional character, I'm just a general guide and don't have the ability to provide personal assistance. However, if you need help with a specific task or issue, feel free to ask and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city is the cultural and economic center of the country. It's known for its rich history, iconic landmarks, vibrant nightlife, and beautiful gardens. Paris has a global reputation for its cuisine and art. The city is also home to many famous museums, theaters, and parks. Its status as the world's most livable city and most affordable city has made it a popular tourist destination. Paris has also been the subject of many films, books, and music, drawing millions of visitors every year. Overall, Paris is a major hub of culture, commerce, and social life in France. Its impressive skyline, charming architecture, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be characterized by a number of trends that will shape the technology's evolution. Here are some potential future trends in AI:
    
    1. Increasing use of AI in healthcare: AI will continue to play a significant role in healthcare, especially in areas such as medical imaging, genetic testing, and personalized medicine. AI will be used to improve patient outcomes, reduce errors, and enhance the quality of care.
    
    2. Increased integration with everyday technology: AI will become more integrated into our daily lives, from our phones and cars to our homes and workplaces. This will enable AI to automate tasks, improve efficiency, and provide new services and entertainment.
    
    3


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

    Role

    ]

    !

     I

    'm

     currently

     [

    Current

     Location

    ]

     and

     I

     enjoy

     [

    My

     Hobby

    /

    Interest

    /

    Professional

     Activity

    /

    Other

     Passion

    ].

     How

     can

     I

     help

     you

     today

    ?

     I

    'm

     excited

     to

     learn

     more

     about

     you

     and

     the

     world

     around

     me

    .

     How

     about

     you

    ?

     Please

     share

     a

     little

     bit

     about

     yourself

     and

     what

     you

     do

     for

     a

     living

    .

     You

    're

     welcome

     to

     share

     your

     story

    ,

     and

     I

    'll

     be

     here

     to

     listen

    .

     What

     do

     you

     do

     for

     a

     living

    ?

     I

     look

     forward

     to

     hearing

     from

     you

    !

     

    📱

    🔍

    📝

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     located

     on

     the

     Se

    ine

     River

     in

     the

     Lo

    ire

     Valley

     and

     is

     the

     largest

     city

     in

     France

    ,

     the

     

    1

    4

    th

     largest

     in

     the

     world

    .

     Its

     population

     is

     over

     

    1

    .

    3

     million

     people

    .

     It

     is

     famous

     for

     its

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     other

     landmarks

    .

     Paris

     is

     also

     known

     for

     its

     rich

     history

    ,

     culture

    ,

     and

     delicious

     cuisine

    .

     The

     city

     is

     a

     significant

     global

     city

     and

     the

     country

    's

     economic

     and

     cultural

     center

    .

     Paris

     is

     home

     to

     many

     famous

     museums

    ,

     theaters

    ,

     and

     food

     festivals

    .

     With

     its

     beautiful

     architecture

    ,

     world

    -ren

    owned

     art

    ,

     and

     vibrant

     nightlife

    ,

     Paris

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     rapid

     and

     significant

     advancements

     in

     several

     key

     areas

    ,

     leading

     to

     a

     wide

     range

     of

     potential

     developments

    .

     Some

     of

     the

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     precision

     and

     accuracy

    :

     AI

     is

     already

     capable

     of

     performing

     complex

     tasks

     with

     a

     high

     level

     of

     accuracy

    .

     However

    ,

     it

    's

     expected

     that

     this

     level

     of

     precision

     will

     continue

     to

     improve

     as

     researchers

     and

     developers

     continue

     to

     fine

    -t

    une

     their

     models

    .
    


    2

    .

     Personal

    ization

    :

     AI

     is

     already

     able

     to

     learn

     from

     data

     and

     make

     personalized

     recommendations

     based

     on

     individual

     preferences

    .

     The

     future

     of

     AI

     is

     expected

     to

     see

     even

     more

     emphasis

     on

     personal

    izing

     the

     interactions

     between

     machines

     and

     humans

    ,

     leading

     to

     more

     effective

    



```python
llm.shutdown()
```
