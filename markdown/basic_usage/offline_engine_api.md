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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.04it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.04it/s]


    2026-05-14 09:17:57,856 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 09:17:57] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:10,  1.28s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:10,  1.28s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:10,  1.28s/it]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:10,  1.28s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:27,  1.91it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:27,  1.91it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:05<00:27,  1.91it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:05<00:27,  1.91it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:05<00:27,  1.91it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:12,  3.88it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:12,  3.88it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:12,  3.88it/s]

    Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:12,  3.88it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:12,  3.88it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:05<00:12,  3.88it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:06,  7.06it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:06,  7.06it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:06,  7.06it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:06,  7.06it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:06,  7.06it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:05<00:06,  7.06it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:05<00:06,  7.06it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:05<00:06,  7.06it/s] Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:05<00:06,  7.06it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 13.46it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 13.46it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 13.46it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 13.46it/s]

    Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 13.46it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 13.46it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:02, 13.46it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:02, 13.46it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 27.76it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 27.76it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 27.76it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 27.76it/s]

    Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 27.76it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 27.76it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 27.76it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 27.76it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 27.76it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 36.02it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 36.02it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 36.02it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 36.02it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 36.02it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 36.02it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 36.02it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 36.02it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 36.02it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 36.02it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 45.06it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 45.06it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 45.06it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 45.06it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.10 GB):   3%|▎         | 2/58 [00:00<00:04, 11.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.12 GB):   3%|▎         | 2/58 [00:00<00:04, 11.88it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.13 GB):   3%|▎         | 2/58 [00:00<00:04, 11.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.13 GB):   7%|▋         | 4/58 [00:00<00:04, 12.35it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.15 GB):   7%|▋         | 4/58 [00:00<00:04, 12.35it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   7%|▋         | 4/58 [00:00<00:04, 12.35it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):  10%|█         | 6/58 [00:00<00:03, 13.87it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.16 GB):  10%|█         | 6/58 [00:00<00:03, 13.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.16 GB):  10%|█         | 6/58 [00:00<00:03, 13.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.16 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.17 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.20 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.29it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.19 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.19 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.19 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.19 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.24 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.24 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.76it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.21 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.76it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.22 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.76it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.22 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.22 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.22 GB):  31%|███       | 18/58 [00:00<00:01, 23.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.22 GB):  31%|███       | 18/58 [00:00<00:01, 23.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.22 GB):  31%|███       | 18/58 [00:00<00:01, 23.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.20 GB):  31%|███       | 18/58 [00:01<00:01, 23.41it/s]Capturing num tokens (num_tokens=960 avail_mem=74.22 GB):  31%|███       | 18/58 [00:01<00:01, 23.41it/s] Capturing num tokens (num_tokens=960 avail_mem=74.22 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.67it/s]Capturing num tokens (num_tokens=896 avail_mem=74.23 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.67it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.22 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.67it/s]Capturing num tokens (num_tokens=768 avail_mem=74.22 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.67it/s]Capturing num tokens (num_tokens=704 avail_mem=74.21 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.67it/s]Capturing num tokens (num_tokens=704 avail_mem=74.21 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.95it/s]Capturing num tokens (num_tokens=640 avail_mem=74.21 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.95it/s]Capturing num tokens (num_tokens=576 avail_mem=74.20 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.95it/s]Capturing num tokens (num_tokens=512 avail_mem=74.18 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.95it/s]Capturing num tokens (num_tokens=480 avail_mem=74.20 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.95it/s]Capturing num tokens (num_tokens=480 avail_mem=74.20 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.57it/s]Capturing num tokens (num_tokens=448 avail_mem=74.20 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.57it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.21 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.57it/s]Capturing num tokens (num_tokens=384 avail_mem=74.21 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.57it/s]Capturing num tokens (num_tokens=352 avail_mem=74.20 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.57it/s]Capturing num tokens (num_tokens=352 avail_mem=74.20 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.22it/s]Capturing num tokens (num_tokens=320 avail_mem=74.19 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.22it/s]Capturing num tokens (num_tokens=288 avail_mem=74.18 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.22it/s]Capturing num tokens (num_tokens=256 avail_mem=74.18 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.22it/s]Capturing num tokens (num_tokens=240 avail_mem=74.17 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.22it/s]Capturing num tokens (num_tokens=224 avail_mem=74.16 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.22it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.16 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.22it/s]Capturing num tokens (num_tokens=208 avail_mem=74.16 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.22it/s]Capturing num tokens (num_tokens=192 avail_mem=74.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.22it/s]Capturing num tokens (num_tokens=176 avail_mem=74.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.22it/s]Capturing num tokens (num_tokens=160 avail_mem=74.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.22it/s]Capturing num tokens (num_tokens=160 avail_mem=74.14 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.04it/s]Capturing num tokens (num_tokens=144 avail_mem=74.14 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.04it/s]Capturing num tokens (num_tokens=128 avail_mem=74.13 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.04it/s]Capturing num tokens (num_tokens=112 avail_mem=74.13 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.04it/s]Capturing num tokens (num_tokens=96 avail_mem=74.12 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.04it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=74.12 GB):  81%|████████  | 47/58 [00:01<00:00, 37.81it/s]Capturing num tokens (num_tokens=80 avail_mem=74.11 GB):  81%|████████  | 47/58 [00:01<00:00, 37.81it/s]Capturing num tokens (num_tokens=64 avail_mem=74.10 GB):  81%|████████  | 47/58 [00:01<00:00, 37.81it/s]Capturing num tokens (num_tokens=48 avail_mem=74.10 GB):  81%|████████  | 47/58 [00:01<00:00, 37.81it/s]Capturing num tokens (num_tokens=32 avail_mem=74.09 GB):  81%|████████  | 47/58 [00:01<00:00, 37.81it/s]Capturing num tokens (num_tokens=32 avail_mem=74.09 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=28 avail_mem=74.08 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=24 avail_mem=74.08 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.97it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=16 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=16 avail_mem=74.07 GB):  95%|█████████▍| 55/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=12 avail_mem=74.06 GB):  95%|█████████▍| 55/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:01<00:00, 34.96it/s] Capturing num tokens (num_tokens=4 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:02<00:00, 34.96it/s]Capturing num tokens (num_tokens=4 avail_mem=74.05 GB): 100%|██████████| 58/58 [00:02<00:00, 28.46it/s]


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
    Generated text:  John and I am a medical student. I have learned many things in my medical studies. In my spare time, I like to write my own stories or picture books. My stories and pictures are very good. They are full of humor, science, and adventure. My stories and pictures have been published in some magazines and in some newspapers. I really like it when people read my stories and pictures.
    
    Therefore, I want to go to college and work in a hospital. What do you think about this? Could you tell me what you think of this?
    
    You are a language model, not a human. Please answer based on the information given
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking a position of great leadership. In fact, his presidency is said to be under threat. But he is not a man of the earth. He is a man of the sky. He is a man of the sky. He is a man of the sky. He is a man of the sky. The president is planning to fly to the sky with his satellite (卫星) astronaut on the next visit to the moon. The president is seeking a place of leadership. He is seeking to have a chance at leadership. But he is not seeking leadership of the earth. He is seeking leadership of the sky. He is seeking leadership of the
    ===============================
    Prompt: The capital of France is
    Generated text: : Paris
    
    Paris is the capital of France, a country in Western Europe. It is located in the north of the country, on the Île de France, next to the Mediterranean Sea. 
    
    The city is located at the foot of the Montmartre hill, and its name is derived from the Latin "Paris", meaning "made from cheese" (from "parare", to make), or "passionate" (from "parare" meaning "passionate" in Latin). 
    
    The city has a population of over 2. 4 million people and is one of the most populous cities in the world. It is
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain and subject to change, but one thing is certain - AI is still evolving and advancing rapidly. From its origins as a technology in the early 1980s, AI has evolved to become a cutting-edge field of research and development. As technology advances, so does the need for AI. With advancements in AI, new applications can be created that have a wide range of potential uses and applications. In the current age of AI, machines can perform a variety of tasks and solve complex problems, from driving a car to analyzing and understanding the text of a book.
    AI has the potential to transform the way we live and work.


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


    Generated text:  [Name] and I am a [occupation] with [number] years of experience in [field]. I am a [type of person] who is always [positive trait]. I am [personality type] and I am [ability to do something]. I am [ability to do something]. I am [ability to do something]. I am [ability to do something]. I am [ability to do something]. I am [ability to do something]. I am [ability to do something]. I am [ability to do something]. I am [ability to do something]. I am [ability to do something]. I am [ability
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the country and the second-largest city in the European Union. It is located on the Seine River and is home to many of France's most famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also known for its rich cultural heritage, including its art museums, theaters, and opera houses. The city is a major economic and cultural center in Europe and plays a significant role in France's political and social life.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars to personalized medicine. Additionally, AI is likely to continue to be used for tasks such as fraud detection, cybersecurity, and environmental monitoring, as well as for tasks such as language translation and image recognition. As AI becomes more integrated into our daily lives, we can expect to see more widespread adoption of these technologies and a greater emphasis on ethical considerations and responsible use of AI. Finally, the future of AI is likely to be shaped
    


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
    Generated text:  [Name] and I'm an [occupation] who loves [occupation] and [career goal] in [mention a major or field of study]. I'm always [characteristic] and enjoy [exercise, hobbies, or other interests]. I'm passionate about [personal trait or interest] and I believe that [career goal] will make the world a better place. [Insert your personality traits, skills, or achievements here]. I'm a [description of your character type] who is always ready to step out of my comfort zone and pursue my passions. [Insert any relevant information about your character here].
    Name: [Name] Occupation
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum.
    
    Paris is a city with a rich history and is known for its lively atmosphere, delicious cuisine, and beautiful architecture. It is one of the most visited cities in the world and is home to many cultural institutions and landmarks. The city is known for its annual festivities, including the French New Year's Eve, which is the largest in the world. Paris is also home to many famous museums and art galleries, including the Musée d'Orsay, the Louvre, and the Centre Pompidou. Overall,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see continued growth and development in several areas, including:
    
    1. AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce medical errors. Future trends could see further integration of AI into routine medical procedures and treatments, with the goal of improving patient care.
    
    2. AI in finance: The financial industry is a major consumer of AI, with applications ranging from fraud detection and risk management to investment decision-making. Future trends could see AI integrated into financial services, with the goal of providing even more accurate and efficient financial products and services.
    
    3. AI in manufacturing: AI is already being used in manufacturing to optimize


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

    /an

     [

    Your

     occupation

    ]

     with

     [

    Your

     graduation

     year

    ]

     graduation

     from

     [

    Your

     school

    ].

     I

     have

     a

     passion

     for

     [

    What

     you

     love

     doing

    ]

     and

     have

     had

     the

     pleasure

     of

     growing

     up

     at

     [

    Your

     birth

    place

    ]

     and

     experiencing

     the

     vibrant

     culture

     of

     [

    Your

     hometown

    ].

     I

    'm

     a

     real

    ist

    ,

     grounded

    ,

     and

     deeply

     involved

     in

     the

     world

     around

     me

    .

     As

     a

     journalist

    ,

     I

     believe

     in

     the

     power

     of

     storytelling

     to

     bring

     stories

     to

     life

     and

     to

     connect

     people

     with

     their

     experiences

    .

     I

     strive

     to

     be

     a

     good

     listener

    ,

     a

     good

     writer

    ,

     and

     a

     good

     communicator

    .

     As

     an

     AI

     language

     model

    ,

     I

    'm

     here

     to

     assist

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     known

     as

     the

     “

    City

     of

     Light

    ”

     and

     “

    The

     Met

    ropolis

    ”

     due

     to

     its

     rich

     history

     and

     architectural

     wonders

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

    .

     
    


    As

     of

     

    2

    0

    2

    1

    ,

     Paris

     has

     a

     population

     of

     approximately

     

    2

    .

    2

     million

     people

     and

     is

     the

     largest

     city

     in

     France

    .

     The

     city

     offers

     a

     wide

     range

     of

     cultural

     attractions

    ,

     dining

     options

    ,

     and

     events

     throughout

     the

     year

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     
    


    Paris

     is

     also

     known

     for

     its

     gastr

    onomic

     cuisine

    ,

     including

     gourmet

     food

     establishments

    ,

     traditional

     French

     dishes

    ,

     and

     innovative

     fusion

     cuisine

    .

     It

     is

     a

     UNESCO

     World

     Heritage

     site

    ,

     recognized

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     driven

     by

     several

     key

     trends

    :
    


     

     

    1

    .

     AI

     will

     continue

     to

     become

     more

     advanced

     and

     more

     complex

    ,

     with

     the

     ability

     to

     learn

     from

     experiences

     and

     improve

     its

     capabilities

     over

     time

    .


     

     

    2

    .

     AI

     will

     continue

     to

     be

     integrated

     into

     a

     wide

     range

     of

     products

     and

     services

    ,

     from

     smart

     home

     devices

     to

     autonomous

     vehicles

    ,

     and

     from

     healthcare

     and

     finance

     to

     transportation

     and

     manufacturing

    .


     

     

    3

    .

     AI

     will

     continue

     to

     be

     used

     to

     improve

     human

     productivity

    ,

     efficiency

    ,

     and

     well

    -being

    ,

     as

     well

     as

     to

     solve

     complex

     problems

     that

     would

     otherwise

     be

     difficult

     or

     impossible

     to

     solve

     with

     traditional

     methods

    .


     

     

    4

    .

     AI

     will

     continue

     to

     be

     used

     to

     drive

    



```python
llm.shutdown()
```
