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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.78it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.77it/s]


    2026-04-07 05:43:35,751 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 05:43:35] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.21it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.21it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.21it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.21it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.21it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:02<00:07,  6.21it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:02<00:07,  6.21it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:02<00:07,  6.21it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:02<00:07,  6.21it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:03, 12.58it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:03, 12.58it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.58it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.65it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.65it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.65it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.65it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.65it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.65it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.65it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.65it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.26it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.26it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.26it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.26it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.26it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.26it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.26it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.12it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.12it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.12it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.12it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.12it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.12it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.12it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 34.93it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 34.93it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 34.93it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 34.93it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 34.93it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 34.93it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 34.93it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 37.57it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 37.57it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 37.57it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 37.57it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 37.57it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 37.57it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 37.57it/s] 

    Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 37.95it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 37.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.46it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.88 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.88 GB):   2%|▏         | 1/58 [00:00<00:06,  8.62it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.85 GB):   2%|▏         | 1/58 [00:00<00:06,  8.62it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=131.85 GB):   2%|▏         | 1/58 [00:00<00:06,  8.62it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.85 GB):   5%|▌         | 3/58 [00:00<00:05, 10.89it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.85 GB):   5%|▌         | 3/58 [00:00<00:05, 10.89it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.85 GB):   5%|▌         | 3/58 [00:00<00:05, 10.89it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.85 GB):   9%|▊         | 5/58 [00:00<00:03, 13.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.84 GB):   9%|▊         | 5/58 [00:00<00:03, 13.59it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=131.84 GB):   9%|▊         | 5/58 [00:00<00:03, 13.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.84 GB):   9%|▊         | 5/58 [00:00<00:03, 13.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.84 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.84 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.83 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.83 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.45it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.82 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.45it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=131.82 GB):  21%|██        | 12/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=3072 avail_mem=131.82 GB):  21%|██        | 12/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.82 GB):  21%|██        | 12/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.81 GB):  21%|██        | 12/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.81 GB):  21%|██        | 12/58 [00:00<00:02, 22.37it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.81 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.90it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.81 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.80 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.80 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.90it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=131.79 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=131.79 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.78 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.12it/s]Capturing num tokens (num_tokens=960 avail_mem=131.79 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.12it/s] Capturing num tokens (num_tokens=896 avail_mem=131.79 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.12it/s]Capturing num tokens (num_tokens=832 avail_mem=131.78 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.12it/s]Capturing num tokens (num_tokens=832 avail_mem=131.78 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.72it/s]Capturing num tokens (num_tokens=768 avail_mem=131.78 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.72it/s]Capturing num tokens (num_tokens=704 avail_mem=131.78 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.72it/s]Capturing num tokens (num_tokens=640 avail_mem=131.77 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.72it/s]

    Capturing num tokens (num_tokens=576 avail_mem=131.77 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.72it/s]Capturing num tokens (num_tokens=576 avail_mem=131.77 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.05it/s]Capturing num tokens (num_tokens=512 avail_mem=131.76 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.05it/s]Capturing num tokens (num_tokens=480 avail_mem=131.78 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.05it/s]

    Capturing num tokens (num_tokens=448 avail_mem=131.78 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.05it/s]Capturing num tokens (num_tokens=416 avail_mem=131.77 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.05it/s]Capturing num tokens (num_tokens=384 avail_mem=131.77 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.05it/s]Capturing num tokens (num_tokens=384 avail_mem=131.77 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.36it/s]Capturing num tokens (num_tokens=352 avail_mem=131.76 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.36it/s]Capturing num tokens (num_tokens=320 avail_mem=131.19 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.36it/s]Capturing num tokens (num_tokens=288 avail_mem=131.73 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.36it/s]

    Capturing num tokens (num_tokens=256 avail_mem=131.26 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.36it/s]Capturing num tokens (num_tokens=256 avail_mem=131.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 23.39it/s]Capturing num tokens (num_tokens=240 avail_mem=131.72 GB):  64%|██████▍   | 37/58 [00:01<00:00, 23.39it/s]Capturing num tokens (num_tokens=224 avail_mem=131.76 GB):  64%|██████▍   | 37/58 [00:01<00:00, 23.39it/s]Capturing num tokens (num_tokens=208 avail_mem=131.71 GB):  64%|██████▍   | 37/58 [00:01<00:00, 23.39it/s]

    Capturing num tokens (num_tokens=208 avail_mem=131.71 GB):  69%|██████▉   | 40/58 [00:01<00:00, 21.81it/s]Capturing num tokens (num_tokens=192 avail_mem=131.76 GB):  69%|██████▉   | 40/58 [00:01<00:00, 21.81it/s]Capturing num tokens (num_tokens=176 avail_mem=131.71 GB):  69%|██████▉   | 40/58 [00:01<00:00, 21.81it/s]Capturing num tokens (num_tokens=160 avail_mem=131.71 GB):  69%|██████▉   | 40/58 [00:01<00:00, 21.81it/s]Capturing num tokens (num_tokens=160 avail_mem=131.71 GB):  74%|███████▍  | 43/58 [00:01<00:00, 21.56it/s]Capturing num tokens (num_tokens=144 avail_mem=131.36 GB):  74%|███████▍  | 43/58 [00:01<00:00, 21.56it/s]Capturing num tokens (num_tokens=128 avail_mem=131.70 GB):  74%|███████▍  | 43/58 [00:01<00:00, 21.56it/s]

    Capturing num tokens (num_tokens=112 avail_mem=131.73 GB):  74%|███████▍  | 43/58 [00:02<00:00, 21.56it/s]Capturing num tokens (num_tokens=112 avail_mem=131.73 GB):  79%|███████▉  | 46/58 [00:02<00:00, 21.62it/s]Capturing num tokens (num_tokens=96 avail_mem=131.41 GB):  79%|███████▉  | 46/58 [00:02<00:00, 21.62it/s] Capturing num tokens (num_tokens=80 avail_mem=131.69 GB):  79%|███████▉  | 46/58 [00:02<00:00, 21.62it/s]Capturing num tokens (num_tokens=64 avail_mem=131.69 GB):  79%|███████▉  | 46/58 [00:02<00:00, 21.62it/s]Capturing num tokens (num_tokens=64 avail_mem=131.69 GB):  84%|████████▍ | 49/58 [00:02<00:00, 21.27it/s]Capturing num tokens (num_tokens=48 avail_mem=131.68 GB):  84%|████████▍ | 49/58 [00:02<00:00, 21.27it/s]

    Capturing num tokens (num_tokens=32 avail_mem=131.45 GB):  84%|████████▍ | 49/58 [00:02<00:00, 21.27it/s]Capturing num tokens (num_tokens=28 avail_mem=131.52 GB):  84%|████████▍ | 49/58 [00:02<00:00, 21.27it/s]Capturing num tokens (num_tokens=28 avail_mem=131.52 GB):  90%|████████▉ | 52/58 [00:02<00:00, 22.29it/s]Capturing num tokens (num_tokens=24 avail_mem=131.67 GB):  90%|████████▉ | 52/58 [00:02<00:00, 22.29it/s]Capturing num tokens (num_tokens=20 avail_mem=131.47 GB):  90%|████████▉ | 52/58 [00:02<00:00, 22.29it/s]Capturing num tokens (num_tokens=16 avail_mem=131.47 GB):  90%|████████▉ | 52/58 [00:02<00:00, 22.29it/s]

    Capturing num tokens (num_tokens=16 avail_mem=131.47 GB):  95%|█████████▍| 55/58 [00:02<00:00, 21.89it/s]Capturing num tokens (num_tokens=12 avail_mem=131.65 GB):  95%|█████████▍| 55/58 [00:02<00:00, 21.89it/s]Capturing num tokens (num_tokens=8 avail_mem=131.52 GB):  95%|█████████▍| 55/58 [00:02<00:00, 21.89it/s] Capturing num tokens (num_tokens=4 avail_mem=131.64 GB):  95%|█████████▍| 55/58 [00:02<00:00, 21.89it/s]Capturing num tokens (num_tokens=4 avail_mem=131.64 GB): 100%|██████████| 58/58 [00:02<00:00, 22.98it/s]Capturing num tokens (num_tokens=4 avail_mem=131.64 GB): 100%|██████████| 58/58 [00:02<00:00, 22.52it/s]


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
    Generated text:  Kim and I am a marketing intern for Quality Partners. My job is to compile and analyze data to provide a report that can be used for marketing. I was assigned to create a report about the effectiveness of marketing campaigns for a specific industry. The industry is the healthcare industry. 
    
    I want to make sure the report includes the following sections:
    
    1. A brief overview of the industry
    2. A brief overview of the marketing campaigns
    3. The performance of the marketing campaigns
    4. The impact of the marketing campaigns on the target audience
    5. A recommendation for future marketing campaigns
    
    I need help with section 5 of the report
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to finalize a set of new policies to improve the country. In a hypothetical election, there are three candidates, each with a probability of winning the presidency. The probability that each candidate will win is as follows:
    
    - Candidate A has a 20% chance of winning.
    - Candidate B has a 30% chance of winning.
    - Candidate C has a 40% chance of winning.
    
    The president has a policy of never making a mistake, meaning each candidate has a 100% chance of winning. The president also has a policy of never letting the election results influence his decisions, meaning the policies do not
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a city in which continent?
    
      1. Europe
      2. North America
      3. South America
      4. Asia
    The answer is:
    
    To determine the correct continent for Paris, we need to consider the following information:
    
    1. Paris is located in the western part of France.
    2. The capital of France is Paris, which is located in the northwestern region of France.
    
    Given this information, we can conclude that Paris is in the northern hemisphere and is situated in the northern part of Europe.
    
    Therefore, the correct answer is:
    
    1. Europe
    
    This corresponds to option 
    ===============================
    Prompt: The future of AI is
    Generated text:  being shaped by the incredible advances in computing and machine learning that have transformed countless industries. Machine learning, a subset of artificial intelligence, has the potential to revolutionize the way we live, work, and communicate. As the technology continues to evolve, we're likely to see more sophisticated AI systems that are capable of handling complex tasks that we can't even imagine right now. One of the most promising areas for AI development is healthcare. Machine learning algorithms can analyze vast amounts of patient data to identify patterns and make personalized treatment recommendations. This could lead to faster and more accurate diagnoses, as well as better patient outcomes.
    In addition to healthcare, AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a major tourist destination. It is also a major economic center and a major player in the French government. The city is known for its fashion industry, art scene, and its role in the French language and culture. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a unique and fascinating city. It is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical AI: As more and more AI systems become involved in decision-making processes, there is a growing emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and fairness. As a result, there is likely to be a greater focus on developing AI systems that are designed to be transparent, accountable, and fair.
    
    2. Integration of AI with other technologies: AI is already being integrated into
    


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
    Generated text:  [Name] and I'm a [briefly describe your profession, experience, or role here]. I'm excited to meet you and discuss my journey so far. How about you?
    Come on in, you too! Let's get to know each other. I'm [Name], a [briefly describe your profession, experience, or role here]. My journey is filled with challenges and discoveries, but I'm eager to share my insights and learn more about you. Welcome to our interactive, engaging conversation.
    Let's get started! What brings you to this table and what's on your mind? Is there anything specific you'd like to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre Dame Cathedral, and many other famous structures. It is also known for its rich cultural heritage, including the French language and traditions, as well as its historical importance dating back to ancient times. Despite its long history, Paris remains a vibrant and diverse city with a rich cultural scene. It is the third-largest city in Europe by population, and a major hub of European culture and business. The city has a dynamic economy and is recognized as one of the world's most beautiful cities, with a wide range of attractions and amenities for visitors. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly diverse and rapidly evolving, with many different areas and technologies emerging and contributing to its development. Here are some possible future trends in artificial intelligence:
    
    1. Improved accuracy and precision: As AI technology continues to advance, the ability to process and analyze data more accurately and precisely will become increasingly important. This will enable AI to better understand and predict human behavior, which could lead to more effective and personalized AI applications.
    
    2. Deep learning and machine learning: As AI technology becomes more advanced, the focus will shift towards using more advanced algorithms and models. Deep learning, which involves using multiple layers of artificial neurons to learn complex patterns in


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

     am

     a

     [

    Age

    ]

     year

     old

     [

    Gender

    ]

     who

     currently

     resides

     in

     [

    Your

     Town

     or

     City

    ].

     I

     have

     a

     love

     for

     [

    Favorite

     Activity

    /

    Interest

    /

    Dis

    appointment

    ]

     which

     has

     made

     me

     decide

     to

     start

     a

     new

     career

     in

     [

    Your

     Field

    ].

     I

     believe

     that

     my

     unique

     background

     and

     experiences

     have

     made

     me

     an

     ideal

     candidate

     for

     this

     role

    ,

     and

     I

     am

     excited

     to

     share

     my

     passion

     for

     [

    Favorite

     Activity

    /

    Interest

    /

    Dis

    appointment

    ]

     with

     you

    .

     [

    Contact

     Information

     or

     Contact

     Us

    ]

     If

     you

     have

     any

     questions

    ,

     I

     am

     here

     to

     answer

     them

    .

     [

    Open

     Door

     Policy

    ]

     Thank

     you

     for

     considering

     my

     application

    .

     I

     look

     forward

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     and

     the

     capital

     of

     the

     country

    .

     It

     is

     also

     the

     world

    ’s

     most

     populous

     city

     and

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

    .

     It

     was

     founded

     in

     

    7

    8

    7

     and

     was

     named

     after

     the

     river

     Se

    ine

    ,

     which

     flows

     through

     the

     city

    .

     Paris

     has

     a

     diverse

     and

     colorful

     population

    ,

     which

     has

     been

     influenced

     by

     its

     history

    ,

     culture

    ,

     and

     the

     country

    ’s

     various

     languages

     and

     religions

    .

     The

     city

     has

     a

     rich

     artistic

     and

     literary

     heritage

    ,

     with

     many

     famous

     museums

    ,

     theaters

    ,

     and

     music

     venues

    .

     It

     has

     also

     been

     the

     location

     of

     many

     political

     and

     cultural

     events

     throughout

     its

     history

    ,

     including

     the

     Congress

     of

     Vienna

     in

     

    1

    8

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     trends

     and

     developments

     that

     are

     likely

     to

     shape

     the

     technology

    ,

     applications

    ,

     and

     ethics

     of

     AI

     in

     the

     coming

     decades

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

     Increased

     Intelligence

     and

     Learning

    :

     One

     of

     the

     most

     significant

     trends

     in

     AI

     is

     the

     increasing

     intelligence

     and

     learning

     capabilities

     of

     AI

     systems

    .

     With

     advancements

     in

     neural

     networks

    ,

     machine

     learning

    ,

     and

     deep

     learning

    ,

     AI

     models

     are

     becoming

     increasingly

     capable

     of

     learning

     from

     vast

     amounts

     of

     data

     and

     improving

     their

     performance

     over

     time

    .
    


    2

    .

     Personal

    ization

     and

     Adapt

    ability

    :

     AI

     systems

     are

     increasingly

     being

     used

     to

     provide

     personalized

     and

     adaptive

     services

     to

     users

    .

     This

     trend

     is

     likely

     to

     continue

     as

    



```python
llm.shutdown()
```
