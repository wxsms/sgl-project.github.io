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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.88it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.87it/s]


    2026-04-09 19:54:43,005 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 19:54:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.27it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:07,  5.99it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:07,  5.99it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:07,  5.99it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:07,  5.99it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  5.99it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  5.99it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  5.99it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  5.99it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  5.99it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.16it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.16it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.16it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.16it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.16it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.16it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.16it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.16it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.11it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.11it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.11it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.11it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.11it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.11it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.11it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 23.47it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 23.47it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 23.47it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 23.47it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 23.47it/s]

    Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 23.47it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 23.47it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 28.67it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 28.67it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 28.67it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 28.67it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 28.67it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 28.67it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 28.67it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.12it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.12it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.12it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.12it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.12it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.12it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.12it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 37.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.02 GB):   2%|▏         | 1/58 [00:00<00:07,  7.60it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.99 GB):   2%|▏         | 1/58 [00:00<00:07,  7.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.99 GB):   2%|▏         | 1/58 [00:00<00:07,  7.60it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.98 GB):   2%|▏         | 1/58 [00:00<00:07,  7.60it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.98 GB):   7%|▋         | 4/58 [00:00<00:03, 15.93it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.99 GB):   7%|▋         | 4/58 [00:00<00:03, 15.93it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.98 GB):   7%|▋         | 4/58 [00:00<00:03, 15.93it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.98 GB):   7%|▋         | 4/58 [00:00<00:03, 15.93it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.98 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.98 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.98 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.85it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=118.97 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.97 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.85it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.97 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.97 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.69it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.96 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.96 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.96 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.96 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.96 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.78it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.78it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.78it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=118.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.91 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.89 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.89 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.94it/s]Capturing num tokens (num_tokens=960 avail_mem=118.90 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.94it/s] Capturing num tokens (num_tokens=896 avail_mem=118.90 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.94it/s]Capturing num tokens (num_tokens=832 avail_mem=118.89 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.94it/s]Capturing num tokens (num_tokens=768 avail_mem=118.89 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.94it/s]Capturing num tokens (num_tokens=704 avail_mem=118.89 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.94it/s]Capturing num tokens (num_tokens=704 avail_mem=118.89 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=640 avail_mem=118.89 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=576 avail_mem=118.89 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.91it/s]

    Capturing num tokens (num_tokens=512 avail_mem=118.87 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=480 avail_mem=118.89 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=448 avail_mem=118.89 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=448 avail_mem=118.89 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.97it/s]Capturing num tokens (num_tokens=416 avail_mem=118.89 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.97it/s]Capturing num tokens (num_tokens=384 avail_mem=118.88 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.97it/s]Capturing num tokens (num_tokens=352 avail_mem=118.88 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.97it/s]Capturing num tokens (num_tokens=320 avail_mem=118.87 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.97it/s]Capturing num tokens (num_tokens=288 avail_mem=118.87 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.97it/s]

    Capturing num tokens (num_tokens=288 avail_mem=118.87 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.45it/s]Capturing num tokens (num_tokens=256 avail_mem=118.87 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.45it/s]Capturing num tokens (num_tokens=240 avail_mem=118.86 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.45it/s]Capturing num tokens (num_tokens=224 avail_mem=118.86 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.45it/s]Capturing num tokens (num_tokens=208 avail_mem=118.86 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.45it/s]

    Capturing num tokens (num_tokens=208 avail_mem=118.86 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.40it/s]Capturing num tokens (num_tokens=192 avail_mem=118.86 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.40it/s]Capturing num tokens (num_tokens=176 avail_mem=118.85 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.40it/s]Capturing num tokens (num_tokens=160 avail_mem=118.85 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.40it/s]Capturing num tokens (num_tokens=144 avail_mem=118.84 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.40it/s]Capturing num tokens (num_tokens=144 avail_mem=118.84 GB):  76%|███████▌  | 44/58 [00:01<00:00, 24.35it/s]Capturing num tokens (num_tokens=128 avail_mem=118.84 GB):  76%|███████▌  | 44/58 [00:01<00:00, 24.35it/s]

    Capturing num tokens (num_tokens=112 avail_mem=118.84 GB):  76%|███████▌  | 44/58 [00:01<00:00, 24.35it/s]Capturing num tokens (num_tokens=96 avail_mem=118.83 GB):  76%|███████▌  | 44/58 [00:01<00:00, 24.35it/s] Capturing num tokens (num_tokens=80 avail_mem=118.83 GB):  76%|███████▌  | 44/58 [00:01<00:00, 24.35it/s]Capturing num tokens (num_tokens=80 avail_mem=118.83 GB):  83%|████████▎ | 48/58 [00:01<00:00, 27.08it/s]Capturing num tokens (num_tokens=64 avail_mem=118.83 GB):  83%|████████▎ | 48/58 [00:01<00:00, 27.08it/s]Capturing num tokens (num_tokens=48 avail_mem=118.83 GB):  83%|████████▎ | 48/58 [00:01<00:00, 27.08it/s]

    Capturing num tokens (num_tokens=32 avail_mem=118.82 GB):  83%|████████▎ | 48/58 [00:01<00:00, 27.08it/s]Capturing num tokens (num_tokens=28 avail_mem=118.82 GB):  83%|████████▎ | 48/58 [00:01<00:00, 27.08it/s]Capturing num tokens (num_tokens=28 avail_mem=118.82 GB):  90%|████████▉ | 52/58 [00:01<00:00, 25.36it/s]Capturing num tokens (num_tokens=24 avail_mem=118.81 GB):  90%|████████▉ | 52/58 [00:01<00:00, 25.36it/s]Capturing num tokens (num_tokens=20 avail_mem=118.81 GB):  90%|████████▉ | 52/58 [00:01<00:00, 25.36it/s]Capturing num tokens (num_tokens=16 avail_mem=118.81 GB):  90%|████████▉ | 52/58 [00:01<00:00, 25.36it/s]

    Capturing num tokens (num_tokens=16 avail_mem=118.81 GB):  95%|█████████▍| 55/58 [00:02<00:00, 23.59it/s]Capturing num tokens (num_tokens=12 avail_mem=118.80 GB):  95%|█████████▍| 55/58 [00:02<00:00, 23.59it/s]Capturing num tokens (num_tokens=8 avail_mem=118.80 GB):  95%|█████████▍| 55/58 [00:02<00:00, 23.59it/s] Capturing num tokens (num_tokens=4 avail_mem=118.80 GB):  95%|█████████▍| 55/58 [00:02<00:00, 23.59it/s]Capturing num tokens (num_tokens=4 avail_mem=118.80 GB): 100%|██████████| 58/58 [00:02<00:00, 27.58it/s]


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
    Generated text:  Lisa. I live in a large city. I enjoy going to the big restaurant every day, because the food is very delicious and people there are very friendly. At the same time, I also like the tall building that I can see the river every day. I don't know what to do with my free time, so I often stay at home to read books. I never go to the cinema or the theater because I think they are boring. I often watch TV. I don't want to be late for school, so I like to stay up late. I'm afraid that I will be caught by the police. I like reading
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking advice on how to successfully lead a country on the path of self-governance. What are the key elements that the president should consider in their leadership approach?
    1. Leadership Style: The president should consider their leadership style to ensure that they effectively communicate with their citizens and guide them towards self-governance. They should be adaptable and willing to listen to diverse opinions to achieve common ground.
    
    2. Policy: The president should consider their policy approach to ensure that they are making decisions that are in the best interest of the country. They should be transparent, accountable, and consult with stakeholders to ensure that their policies are implemented effectively.
    
    3
    ===============================
    Prompt: The capital of France is
    Generated text:  located in which country?
    The capital of France is Paris. Paris is the most populous city in France and is located in the heart of the country. It is situated in the northwestern region of France and is also known as the "City of Love" due to its romantic atmosphere. Paris is home to many famous landmarks such as Notre-Dame Cathedral, the Louvre Museum, the Eiffel Tower, and the Champs-Élysées. It is a cultural and historical center that has a diverse population of over 2 million people. Paris is also known for its fashion, art, and gastronomy, making it one of
    ===============================
    Prompt: The future of AI is
    Generated text:  not in a video game. It is in the lives of people. That’s the argument of the new book by Dr. Khaled Hosseini, author of the novel The Kitahari Trilogy. Hosseini is known as the “father of Iranian literature” and has written many other novels including the celebrated The Shah of Khashoggi. The Kitahari Trilogy is the first of his two novels about the history of the United States.
    The Kitahari Trilogy is a series of novels about a family from Iran’s history. After the death of King Gholamreza, the youngest of his children died, and


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


    Generated text:  Paris, the city known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. It is also the birthplace of many famous French artists, writers, and composers. Paris is a bustling metropolis with a rich history and a vibrant cultural scene that attracts millions of visitors each year. Its status as the world's most populous city is due to its large population and the city's importance in French culture and politics. The city is also known for its cuisine, fashion, and music scene, making it a popular destination for tourists and locals alike. Paris is a city that has been a center of power and culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased automation: As AI continues to become more advanced, it is likely to become more efficient and capable of performing tasks that were previously done by humans. This could lead to a greater reliance on automation in various industries, including manufacturing, transportation, and healthcare.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be a greater need for privacy and security measures to protect personal data.
    


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
    Generated text:  [Name] and I'm [Age] years old. I'm originally from [Your Country]. I'm passionate about [Your Passion]. I enjoy [Your Hobby/Interest]. I'm always looking for [Your Goal/Challenge]. I have [X] years of experience in [X]. My professional career has taken me to [X], [X], and [X], and I believe that my experience has allowed me to [X]. If you're interested in learning more about my background, my skills, and my achievements, please feel free to ask me anything. [Name] Hello, my name is [Name] and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement is factually correct as it provides the name of the largest city in France, specifically Paris. Other notable cities in France include Paris, Lyon, Marseille, and Strasbourg, with the capital city being Paris. The capital of France has a rich history dating back to the Roman times, and is known for its cultural, political, and economic importance in Europe. The city is also renowned for its landmarks such as the Eiffel Tower, the Louvre Museum, and Notre Dame Cathedral. Paris is often considered the "Paris of the City" due to its wide variety of cultural and entertainment options. Overall, Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and promising, with many possible trends to consider. Here are some potential areas where AI is expected to continue to evolve and advance:
    
    1. Enhanced Machine Learning: As we continue to develop machine learning models, we will see new methods and techniques being used to improve their performance. This includes the development of more complex models, more effective ways of training models, and better ways of handling the "small world problem" that can arise when dealing with large amounts of data.
    
    2. Autonomous Robots: Autonomous robots are already starting to emerge as a significant part of AI. These robots can be used in a variety of applications, from manufacturing to healthcare


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

    ],

     and

     I

     am

     [

    Age

    ],

     a

     [

    Occup

    ation

    ].

     I

     am

     passionate

     about

     [

    your

     interest

     or

     hobby

    ]

     and

     I

     have

     always

     been

     driven

     to

     [

    why

     you

     are

     passionate

     about

     this

     topic

    ].

     I

     am

     always

     learning

     and

     seeking

     to

     grow

    ,

     and

     I

     am

     always

     looking

     for

     new

     experiences

     and

     ways

     to

     make

     the

     world

     a

     better

     place

    .

     I

     believe

     that

     my

     experiences

     have

     given

     me

     a

     unique

     perspective

     and

     have

     helped

     me

     grow

     as

     a

     person

    .

     I

     love

     to

     be

     outdoors

     and

     travel

    ,

     and

     I

     am

     always

     exploring

     new

     places

     and

     trying

     new

     things

    .

     I

     am

     a

     loyal

     and

     passionate

     member

     of

     my

     community

     and

     always

     look

     out

     for

     the

     good

     of

     all

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Explanation

    :

     Paris

     is

     the

     largest

     and

     most

     populous

     city

     in

     France

    .

     It

     is

     located

     on

     the

     left

     bank

     of

     the

     Se

    ine

     River

    ,

     near

     the

     Mediterranean

     Sea

    ,

     and

     is

     the

     heart

     of

     the

     French

     economy

     and

     culture

    .

     Paris

     is

     known

     for

     its

     iconic

     landmarks

    ,

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    ,

     as

     well

     as

     for

     its

     gastr

    onomic

     and

     cultural

     attractions

    .

     The

     city

     is

     also

     home

     to

     many

     museums

    ,

     theaters

    ,

     and

     other

     cultural

     institutions

    ,

     and

     is

     a

     popular

     tourist

     destination

     for

     residents

     and

     visitors

     alike

    .

     Despite

     being

     the

     capital

    ,

     Paris

     is

     a

     diverse

     and

     culturally

     rich

     city

     with

     a

     rich

     history

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     continue

     to

     evolve

     rapidly

    ,

     with

     a

     wide

     range

     of

     potential

     trends

     and

     developments

    .

     Here

     are

     some

     of

     the

     most

     likely

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     AI

     Transparency

    :

     AI

     will

     become

     more

     transparent

    ,

     allowing

     people

     to

     understand

     how

     and

     why

     AI

     systems

     are

     making

     decisions

    .

     This

     will

     help

     to

     increase

     trust

     and

     confidence

     in

     AI

     systems

    ,

     and

     reduce

     the

     risk

     of

     bias

     in

     AI

     models

    .
    


    2

    .

     Personal

    ized

     AI

    :

     As

     AI

     technology

     advances

    ,

     there

     will

     be

     an

     increasing

     focus

     on

     creating

     AI

     that

     is

     more

     personalized

     and

     tailored

     to

     individual

     needs

    .

     This

     will

     allow

     AI

     systems

     to

     learn

     from

     user

     data

     and

     make

     more

     accurate

     predictions

     and

     recommendations

    .
    


    3

    .

     Eth

    ical

    



```python
llm.shutdown()
```
