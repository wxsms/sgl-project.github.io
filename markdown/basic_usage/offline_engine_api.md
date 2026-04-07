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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.11it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.10it/s]


    2026-04-07 09:51:56,492 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 09:51:56] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:16,  3.19it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:16,  3.19it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:16,  3.19it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:16,  3.19it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:09,  5.44it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:09,  5.44it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:09,  5.44it/s]

    Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:09,  5.44it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:09,  5.44it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:03<00:04,  9.10it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:03<00:04,  9.10it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:04,  9.10it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:04,  9.10it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:03<00:04,  9.10it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:03<00:04,  9.10it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:02, 14.44it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:02, 14.44it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:02, 14.44it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:02, 14.44it/s]

    Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:02, 14.44it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:02, 14.44it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:02, 14.44it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 21.25it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 21.25it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 21.25it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 21.25it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 21.25it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:01, 21.25it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:03<00:01, 21.25it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:00, 28.10it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:00, 28.10it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:00, 28.10it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:00, 28.10it/s]

    Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:00, 28.10it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:00, 28.10it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:00, 28.10it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 37.79it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 37.79it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 37.79it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 37.79it/s]

    Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 37.79it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 37.79it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 40.66it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 40.66it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 40.66it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 40.66it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 40.66it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 40.66it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 40.66it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 43.99it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 43.99it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 43.99it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 43.99it/s]

    Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 43.99it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 43.99it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 43.99it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.42 GB):   2%|▏         | 1/58 [00:00<00:06,  9.43it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.43 GB):   2%|▏         | 1/58 [00:00<00:06,  9.43it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.44 GB):   2%|▏         | 1/58 [00:00<00:06,  9.43it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=118.44 GB):   5%|▌         | 3/58 [00:00<00:04, 11.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.52 GB):   5%|▌         | 3/58 [00:00<00:04, 11.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.51 GB):   5%|▌         | 3/58 [00:00<00:04, 11.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.51 GB):   9%|▊         | 5/58 [00:00<00:04, 13.03it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.54 GB):   9%|▊         | 5/58 [00:00<00:04, 13.03it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.50 GB):   9%|▊         | 5/58 [00:00<00:04, 13.03it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=118.50 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.85it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.56 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.85it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.55 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.53 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.53 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.53 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.53 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.01it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=118.63 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.63 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.63 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.62 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.57 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.31it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=118.57 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.55 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.55 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.55 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.55 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.55 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.55 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.67it/s]Capturing num tokens (num_tokens=960 avail_mem=118.56 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.67it/s] Capturing num tokens (num_tokens=896 avail_mem=118.55 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.67it/s]Capturing num tokens (num_tokens=832 avail_mem=118.54 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.67it/s]

    Capturing num tokens (num_tokens=832 avail_mem=118.54 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.91it/s]Capturing num tokens (num_tokens=768 avail_mem=118.54 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.91it/s]Capturing num tokens (num_tokens=704 avail_mem=118.53 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.91it/s]Capturing num tokens (num_tokens=640 avail_mem=118.52 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.91it/s]Capturing num tokens (num_tokens=576 avail_mem=118.54 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.91it/s]Capturing num tokens (num_tokens=576 avail_mem=118.54 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.55it/s]Capturing num tokens (num_tokens=512 avail_mem=118.52 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.55it/s]Capturing num tokens (num_tokens=480 avail_mem=118.53 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.55it/s]Capturing num tokens (num_tokens=448 avail_mem=118.53 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.55it/s]Capturing num tokens (num_tokens=416 avail_mem=118.52 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.55it/s]

    Capturing num tokens (num_tokens=416 avail_mem=118.52 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.40it/s]Capturing num tokens (num_tokens=384 avail_mem=118.52 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.40it/s]Capturing num tokens (num_tokens=352 avail_mem=118.51 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.40it/s]Capturing num tokens (num_tokens=320 avail_mem=118.50 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.40it/s]Capturing num tokens (num_tokens=288 avail_mem=118.49 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.40it/s]Capturing num tokens (num_tokens=288 avail_mem=118.49 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.89it/s]Capturing num tokens (num_tokens=256 avail_mem=118.49 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.89it/s]Capturing num tokens (num_tokens=240 avail_mem=118.48 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.89it/s]Capturing num tokens (num_tokens=224 avail_mem=118.48 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.89it/s]Capturing num tokens (num_tokens=208 avail_mem=118.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.89it/s]

    Capturing num tokens (num_tokens=208 avail_mem=118.47 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.75it/s]Capturing num tokens (num_tokens=192 avail_mem=118.47 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.75it/s]Capturing num tokens (num_tokens=176 avail_mem=118.46 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.75it/s]Capturing num tokens (num_tokens=160 avail_mem=118.45 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.75it/s]Capturing num tokens (num_tokens=144 avail_mem=118.47 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.75it/s]Capturing num tokens (num_tokens=144 avail_mem=118.47 GB):  76%|███████▌  | 44/58 [00:01<00:00, 35.33it/s]Capturing num tokens (num_tokens=128 avail_mem=118.44 GB):  76%|███████▌  | 44/58 [00:01<00:00, 35.33it/s]Capturing num tokens (num_tokens=112 avail_mem=118.44 GB):  76%|███████▌  | 44/58 [00:01<00:00, 35.33it/s]Capturing num tokens (num_tokens=96 avail_mem=118.43 GB):  76%|███████▌  | 44/58 [00:01<00:00, 35.33it/s] Capturing num tokens (num_tokens=80 avail_mem=118.42 GB):  76%|███████▌  | 44/58 [00:01<00:00, 35.33it/s]

    Capturing num tokens (num_tokens=80 avail_mem=118.42 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.78it/s]Capturing num tokens (num_tokens=64 avail_mem=118.41 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.78it/s]Capturing num tokens (num_tokens=48 avail_mem=118.43 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.78it/s]Capturing num tokens (num_tokens=32 avail_mem=118.42 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.78it/s]Capturing num tokens (num_tokens=28 avail_mem=118.41 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.78it/s]Capturing num tokens (num_tokens=28 avail_mem=118.41 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.65it/s]Capturing num tokens (num_tokens=24 avail_mem=118.41 GB):  90%|████████▉ | 52/58 [00:02<00:00, 36.65it/s]Capturing num tokens (num_tokens=20 avail_mem=118.40 GB):  90%|████████▉ | 52/58 [00:02<00:00, 36.65it/s]Capturing num tokens (num_tokens=16 avail_mem=118.40 GB):  90%|████████▉ | 52/58 [00:02<00:00, 36.65it/s]Capturing num tokens (num_tokens=12 avail_mem=118.39 GB):  90%|████████▉ | 52/58 [00:02<00:00, 36.65it/s]

    Capturing num tokens (num_tokens=12 avail_mem=118.39 GB):  97%|█████████▋| 56/58 [00:02<00:00, 37.58it/s]Capturing num tokens (num_tokens=8 avail_mem=118.38 GB):  97%|█████████▋| 56/58 [00:02<00:00, 37.58it/s] Capturing num tokens (num_tokens=4 avail_mem=118.38 GB):  97%|█████████▋| 56/58 [00:02<00:00, 37.58it/s]Capturing num tokens (num_tokens=4 avail_mem=118.38 GB): 100%|██████████| 58/58 [00:02<00:00, 26.88it/s]


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
    Generated text:  byron and I am 21 years old. I have a bachelor's degree in history from the University of Florida and I have been working for 3 years in a legal department. As a lawyer, I work with people who have been in prison and are seeking representation. In my free time, I am an avid reader and enjoy exploring new books. How can I make my work as a lawyer more impactful and memorable for those who have been in prison? Developing a more impactful and memorable career as a lawyer for individuals in prison involves several strategies:
    
    1. **Customer-Centric Approach**: Understand the needs and challenges of your clients.
    ===============================
    Prompt: The president of the United States is
    Generated text:  from the state of New York. The president of the state of New York is an officer of the _____. ( )
    A. U. S. House of Representatives
    B. U. S. Senate
    C. President of the United States
    D. US. Supreme Court
    E. US. Supreme Court (A) U. S. House of Representatives is not an officer of the United States, so this option is incorrect.  
    B. U. S. Senate is not an officer of the United States, so this option is incorrect.  
    C. President of the United States is an officer of the United States, so
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It has a population of 2.3 million. The majority of Parisians are of French ancestry. The official language of the city is French.
    
    When it comes to the population of Paris, there is a significant discrepancy between the official figure and the actual figure of 2.3 million. This discrepancy is explained by the fact that Paris does not allow self-immigrants. The French government has attempted to implement a policy of voluntary immigration to Paris to increase the population. However, the policy has not been successful.
    
    Now, imagine that Paris is a city with a population of 100,000. If you
    ===============================
    Prompt: The future of AI is
    Generated text:  forecasted to continue to grow with a wide range of applications from healthcare to manufacturing, education, and entertainment. In this article, we will explore the potential benefits and drawbacks of using AI in these areas.
    
    AI in Healthcare:
    
    AI in healthcare is expected to revolutionize the industry by providing new tools and technologies to improve patient care, reduce errors, and improve the overall quality of care. Here are some potential benefits and drawbacks of using AI in healthcare:
    
    Benefits:
    
      1. Improved patient care: AI can help doctors and nurses analyze data more quickly and accurately, leading to better patient outcomes.
      2. Increased efficiency: AI can


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a unique skill or personality trait here]. And what can you tell me about your work at [company name]? I'm always looking for new opportunities to grow and learn, and I'm eager to contribute to your team. What do you think makes you a good fit for this role? I'm confident in my ability to contribute to your team and make a positive impact, and I'm excited to bring my unique skills and experience to your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also a popular tourist destination and home to numerous museums, theaters, and restaurants. Paris is a cultural and economic center that plays a significant role in French society and politics. It is also a major hub for international trade and diplomacy. The city is known for its rich history, art, and cuisine, and is a popular destination for tourists from around the world. Paris is a city of contrasts and beauty, and is a must-visit destination for anyone interested in French culture and history. 
    
    This statement provides
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into many aspects of our lives, from self-driving cars to personalized medicine. As AI becomes more integrated into our daily routines, we can expect to see even more widespread adoption of AI in our daily lives.
    
    2. AI becoming more autonomous: As AI technology continues to advance, we can expect to see more autonomous vehicles on the roads, drones delivering packages, and other forms of AI-driven automation becoming more common.
    
    3. AI
    


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
    Generated text:  [Your Name], and I'm a [role] with a passion for [interest or hobby]. I enjoy writing, reading, and exploring the world. I love to learn new things and challenge myself to grow as a writer. I'm a lifelong learner, always seeking new ideas and perspectives to enrich my writing. I'm excited to meet new people and share my love for writing with them.
    
    Your message is urgent. You need to write a detailed self-introduction for a fictional character. 
    
    Write me a self-introduction for a fictional character. I would like you to include information about the character's role, their interests, and their
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the capital of France and the largest city in the European Union. It is known for its historical significance, art, architecture, and food. It has over 3 million inhabitants and is home to many famous landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. Paris is also a major tourist destination, drawing millions of visitors annually. The city is known for its vibrant culture, music, and fashion industry. The city is home to many cultural and artistic institutions, including the Museum of Fine Arts, the Opera House, and the Musée d'Orsay. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several key trends:
    
    1. More autonomous robots and drones: As AI technology advances, we can expect to see more autonomous robots and drones that can perform tasks independently, reducing the need for human intervention.
    
    2. AI in healthcare: AI is already being used in healthcare to analyze medical images, predict disease outbreaks, and identify potential treatment options. The future of AI in healthcare could be even more transformative, with AI helping to improve patient outcomes and reduce healthcare costs.
    
    3. AI in finance: AI is already being used to analyze financial data and predict market trends. As AI technology continues to advance, we can expect to


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

     ____

    _.

     I

    'm

     a

    /an

     (

    fill

     in

     the

     blank

     with

     your

     full

     name

    ).

     I

    'm

     a

    /an

     (

    fill

     in

     the

     blank

     with

     your

     profession

     or

     occupation

    ).

     I

    'm

     (

    fill

     in

     the

     blank

     with

     some

     unique

     or

     interesting

     personality

     traits

    ).

     I

    'm

     (

    fill

     in

     the

     blank

     with

     some

     positive

     qualities

    ).

     I

    'm

     (

    fill

     in

     the

     blank

     with

     any

     other

     personal

     details

     or

     qualities

     that

     make

     me

     unique

    ).

     I

    'm

     passionate

     about

     (

    fill

     in

     the

     blank

     with

     some

     hobby

     or

     interest

    ).

     I

     love

     (

    fill

     in

     the

     blank

     with

     something

     that

     defines

     me

    ).

     I

    'm

     (

    fill

     in

     the

     blank

     with

     any

     accol

    ades

    ,

     awards

    ,

     or

     other

     recogn

    itions

    ).

     I

    'm

     a

    /an

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     summarizes

     the

     key

     facts

     about

     Paris

    ,

     including

    :
    


    1

    .

     Its

     capital

     status

    


    2

    .

     The

     city

    's

     name

    


    3

    .

     Its

     location

     within

     France

    


    4

    .

     The

     French

     language

     spoken

     there

    


    5

    .

     The

     cultural

     and

     historical

     importance

     of

     the

     city

    


    6

    .

     Its

     status

     as

     a

     major

     financial

     and

     tourist

     center

     in

     Europe

    
    


    The

     statement

     is

     concise

    ,

     providing

     the

     essential

     information

     in

     a

     single

     sentence

    .

     It

     allows

     readers

     to

     quickly

     grasp

     the

     central

     aspects

     of

     Paris

     and

     its

     role

     in

     French

     society

     and

     culture

    .

     Additionally

    ,

     it

     avoids

     any

     unnecessary

     details

    ,

     ensuring

     a

     straightforward

     and

     informative

     response

    .

     
    


    This

     answer

     effectively

     communicates

     the

     key

     facts

     about

     Paris

    ,

     making

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     full

     of

     possibilities

    ,

     but

     it

     is

     not

     clear

     what

     exactly

     it

     will

     look

     like

    .

     Some

     of

     the

     most

     common

     trends

     we

     can

     expect

     to

     see

     in

     AI

     over

     the

     next

     few

     years

     are

    :
    


    1

    .

     Improved

     accuracy

     and

     reliability

    :

     As

     AI

     continues

     to

     become

     more

     sophisticated

    ,

     we

     may

     see

     a

     shift

     towards

     more

     accurate

     and

     reliable

     predictions

    ,

     decisions

    ,

     and

     recommendations

    .
    


    2

    .

     Growth

     of

     ethical

     considerations

    :

     As

     more

     AI

     systems

     become

     complex

     and

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     an

     increasing

     focus

     on

     ethical

     considerations

     and

     the

     responsible

     use

     of

     AI

    .
    


    3

    .

     Expansion

     of

     AI

     into

     new

     domains

    :

     AI

     will

     likely

     continue

     to

     expand

     into

     new

     areas

    ,

     such

     as

     medical

    



```python
llm.shutdown()
```
