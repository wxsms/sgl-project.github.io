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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.48it/s]


    2026-04-09 07:37:32,776 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 07:37:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.26it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  5.94it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.04it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.04it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.04it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.04it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.04it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.04it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.04it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.04it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 17.91it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 17.91it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 17.91it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 17.91it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 17.91it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 17.91it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 17.91it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 17.91it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]

    Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.32it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 28.83it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 28.83it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 28.83it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 28.83it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 28.83it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 28.83it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 28.83it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 32.62it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 32.62it/s]

    Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 32.62it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 32.62it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 32.62it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 32.62it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 32.62it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 35.75it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 35.75it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 35.75it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 35.75it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 35.75it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 35.75it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 35.75it/s] 

    Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 35.75it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 42.40it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   2%|▏         | 1/58 [00:00<00:05,  9.75it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:05,  9.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:05,  9.75it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:03, 13.79it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:03, 13.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:03, 13.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):   5%|▌         | 3/58 [00:00<00:03, 13.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 16.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 16.55it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  10%|█         | 6/58 [00:00<00:03, 16.55it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.76 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.76 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.01it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=118.75 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.74 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.74 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.89it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=118.74 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.84it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.84it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.73 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.73 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.73 GB):  33%|███▎      | 19/58 [00:00<00:01, 22.87it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 22.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.87it/s]Capturing num tokens (num_tokens=960 avail_mem=118.71 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.87it/s] Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.87it/s]

    Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.85it/s]Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.85it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.85it/s]Capturing num tokens (num_tokens=704 avail_mem=118.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.85it/s]Capturing num tokens (num_tokens=640 avail_mem=118.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.85it/s]Capturing num tokens (num_tokens=640 avail_mem=118.70 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.38it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.38it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.38it/s]Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.38it/s]Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.38it/s]Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.38it/s]

    Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.78it/s]Capturing num tokens (num_tokens=384 avail_mem=118.70 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.78it/s]Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.78it/s]Capturing num tokens (num_tokens=320 avail_mem=118.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.78it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.78it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.65it/s]Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.65it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.65it/s]Capturing num tokens (num_tokens=224 avail_mem=118.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.65it/s]

    Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.65it/s]Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.78it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.78it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.78it/s]Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.78it/s]Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.78it/s]Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.88it/s]Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.88it/s]Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.88it/s]

    Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.88it/s] Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.88it/s]Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.61it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.61it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.61it/s]Capturing num tokens (num_tokens=32 avail_mem=118.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.61it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.61it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.95it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.95it/s]

    Capturing num tokens (num_tokens=20 avail_mem=118.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.95it/s]Capturing num tokens (num_tokens=16 avail_mem=118.63 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.95it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.95it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  97%|█████████▋| 56/58 [00:02<00:00, 33.40it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  97%|█████████▋| 56/58 [00:02<00:00, 33.40it/s] Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  97%|█████████▋| 56/58 [00:02<00:00, 33.40it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 27.34it/s]


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
    Generated text:  Makeda and I’m a 15 year old patient with severe allergies. I don’t like to eat meat, dairy, eggs, eggs, and I also have a sensitive skin that is itchy, red, and irritated. I’m always very sensitive to any kind of food or cosmetics. I used to avoid eating meat and dairy, but I changed my diet to include organic fruits, vegetables and grains after I became a student in South Korea. 
    
    My allergy has been triggered by some specific foods, but I don’t know what they are and I’m not sure if I’m allergic to them because I’m not always sure
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to hold a new presidential term. The last two presidents held their term and were both re-elected. However, the president wants to make sure that they are not concerned about losing their term.
    
    To make sure that they are not concerned about losing their term, the president wants to know the maximum number of presidents that could be in the last 100 years of the American presidency that are not re-elected, assuming that the last two presidents were re-elected. The president wants to know the maximum number of presidents that are not re-elected in 100 years. 
    
    To help the president, what can the president
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, a very famous city. Paris is famous for its beauty. There are many parks in Paris and you can enjoy beautiful sights from the parks. You can also have fun with other things in the parks. However, the time when the parks are open is very important. It is a new idea in Paris. Paris usually only opens on weekends, but now it can be opened on any day. It is a very nice idea to make parks open on any day. You can open parks at a time that you want to. You can also open parks during the winter or the summer. Paris will have a lot of parks in the future
    ===============================
    Prompt: The future of AI is
    Generated text:  more connected than ever before. It has become easier for companies to collect data on individuals and the environment. While it’s also more than beneficial to have access to this data to help improve products and services, it can also lead to a plethora of problems. In this article, we will discuss 4 possible issues that can arise from AI connected to data.
    
      1. Privacy violations
    
    With AI connected to data, companies are having to collect more information from individuals. This means that they have the potential to access sensitive information about individuals or the environment. For example, companies might collect data from individuals to personalize their marketing campaigns or to use


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a few details about your personality, skills, or interests]. I'm always looking for new challenges and opportunities to grow and learn. What do you think makes you unique and what makes you stand out from the rest? I'm always looking for new ways to make a positive impact in the world and I'm always eager to learn and grow. What's your favorite hobby or activity? I'm always looking for new ways to challenge myself and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the country and the second-largest city in Europe. It is located in the south of France and is the seat of government, administration, and culture in the country. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. It is also home to many famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a vibrant and diverse city with a rich cultural heritage that has been shaped by its history and its role as a major European city. It is a city that
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there is a growing emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and the impact of AI on society as a whole.
    
    2. Development of more advanced AI systems: As AI technology continues to advance, there is a potential for more advanced AI systems to emerge that are capable of performing tasks that are currently beyond the
    


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
    Generated text:  [Name], and I'm a [role] for [Company]. My experience in [industry] is [amount of years] years, and I am known for [specific achievements, skills, or qualities]. As a [role], I bring a unique blend of [specific skill, experience, or trait] that sets me apart from others. I am driven by [reason for motivation], and I am always striving to [specific goal or achievement]. I am a [specific personality trait, such as honest, determined, or empathetic] person who is always looking out for the best interests of [specific group]. As a [specific role
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as the "City of Light" for its vibrant art, fashion, and nightlife scene. Its historic center, which includes the Arc de Triomphe, is also home to the Louvre Museum, the Eiffel Tower, and the Notre-Dame Cathedral. France's second-largest city, Lyon, is known for its industrial heritage and its charming, historic streets. Lastly, the River Seine flows through the heart of Paris, connecting the city to the Mediterranean and the rest of France. 
    
    Would you like me to provide any additional information about Paris or any other city? Please let me know! Paris, the City
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to grow and evolve rapidly, with many possibilities and areas of research that could lead to breakthroughs in this area. Here are some possible future trends in AI:
    
    1. Increased Personalization: As AI continues to improve, more personalized experiences are becoming possible. This could include personalized healthcare, advertising, and education.
    
    2. Autonomous and Self-Driving Cars: Autonomous cars are already being tested in various cities around the world, and it is likely that they will become more common in the future. Self-driving cars will continue to develop, with more features and capabilities being added.
    
    3. Artificial Intelligence in Healthcare: AI is already being


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

     Jane

    .

     I

    'm

     a

     self

    -employed

     writer

     with

     a

     passion

     for

     creative

     writing

     and

     emerging

     media

    .

     I

    'm

     always

     looking

     for

     new

     opportunities

     to

     share

     my

     words

     with

     readers

     and

     keep

     learning

     and

     improving

    .

     I

    'm

     excited

     about

     the

     potential

     of

     my

     writing

     to

     bring

     people

     together

     and

     create

     meaningful

     connections

    .

     I

     hope

     you

    'll

     have

     an

     enjoyable

     reading

     experience

    .

     
    


    What

    's

     your

     profession

     and

     what

     do

     you

     do

     for

     a

     living

    ?

     As

     an

     independent

     writer

    ,

     I

     create

     and

     publish

     original

     content

     in

     various

     formats

    .

     I

     love

     to

     write

     short

     stories

    ,

     poetry

    ,

     and

     short

     fiction

    .

     I

     also

     edit

     and

     revise

     my

     own

     work

     and

     collaborate

     with

     other

     writers

     on

     projects

    .

     I

    'm

     always

     looking

     for

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    .

     Yes

    


    B

    .

     No

    


    B

    .

     No

    


    France

    's

     capital

     city

    ,

     Paris

    ,

     is

     a

     bustling

     met

    ropolis

     known

     for

     its

     unique

     architecture

    ,

     rich

     history

    ,

     and

     vibrant

     culture

    .

     The

     city

     is

     home

     to

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

     iconic

     landmarks

    .

     Paris

     also

     has

     a

     diverse

     range

     of

     food

     and

     drink

     options

    ,

     including

     famous

     dishes

     like

     cro

    iss

    ants

    ,

     b

    oud

    in

    ,

     and

     esc

    arg

    ot

    .

     The

     French

     love

     to

     enjoy

     champagne

     and

     other

     fine

     wines

    ,

     and

     there

     are

     many

     wine

     tasting

     spots

     to

     enjoy

     the

     tasting

     experience

    .

     Paris

     is

     a

     city

     that

     has

     been

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     poised

     for

     significant

     growth

     and

     development

    ,

     with

     the

     potential

     to

     transform

     many

     aspects

     of

     society

     in

     the

     coming

     decades

    .

     Here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Autonomous

     vehicles

    :

     Self

    -driving

     cars

     and

     trucks

     are

     already

     becoming

     more

     common

    ,

     and

     AI

     is

     playing

     a

     crucial

     role

     in

     their

     development

    .

     Autonomous

     vehicles

     are

     expected

     to

     become

     more

     widespread

     and

     affordable

     in

     the

     coming

     years

    ,

     potentially

     reducing

     accidents

     and

     accidents

     caused

     by

     human

     error

    .
    


    2

    .

     Personal

    ized

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     medical

     diagnosis

     and

     treatment

    ,

     but

     it

     has

     the

     potential

     to

     greatly

     enhance

     the

     quality

     and

     efficiency

     of

     healthcare

    .

     AI

     can

     analyze

     medical

     data

     to

     identify

     patterns

    



```python
llm.shutdown()
```
