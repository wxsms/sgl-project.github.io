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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.50it/s]


    2026-04-14 01:37:35,013 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 01:37:35] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]

    Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  4.95it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  4.95it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:09,  4.95it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:09,  4.95it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:09,  4.95it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:03<00:09,  4.95it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:03<00:09,  4.95it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:03<00:09,  4.95it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:04, 10.38it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:04, 10.38it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:04, 10.38it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:04, 10.38it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:04, 10.38it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:03<00:04, 10.38it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:03<00:04, 10.38it/s] 

    Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:03<00:04, 10.38it/s]Compiling num tokens (num_tokens=832):  28%|██▊       | 16/58 [00:03<00:04, 10.38it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:03<00:01, 17.71it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 24.46it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 24.46it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 24.46it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 24.46it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 24.46it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 24.46it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 24.46it/s]

    Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:03<00:01, 24.46it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 31.35it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 31.35it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 31.35it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 31.35it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 31.35it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 31.35it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 31.35it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 31.35it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 37.81it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 37.81it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 37.81it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 37.81it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 37.81it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 37.81it/s]

    Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 37.81it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 37.81it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 44.28it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 44.28it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 44.28it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 44.28it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 44.28it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 44.28it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 44.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.92 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.89 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.89 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.89 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.88 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.89 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.89 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.88 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.88 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.88 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.88 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.88 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.88it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.87 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.88it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.87 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.88it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.87 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.88it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.86 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.88it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.86 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.86 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.86 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.85 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.85 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.87it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=72.85 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.85 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.84 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.84 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.82 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.82 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.62it/s]Capturing num tokens (num_tokens=960 avail_mem=72.83 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.62it/s] Capturing num tokens (num_tokens=896 avail_mem=72.83 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.62it/s]Capturing num tokens (num_tokens=832 avail_mem=72.41 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.62it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.41 GB):  41%|████▏     | 24/58 [00:00<00:01, 26.07it/s]Capturing num tokens (num_tokens=768 avail_mem=72.28 GB):  41%|████▏     | 24/58 [00:00<00:01, 26.07it/s]Capturing num tokens (num_tokens=704 avail_mem=72.27 GB):  41%|████▏     | 24/58 [00:00<00:01, 26.07it/s]Capturing num tokens (num_tokens=640 avail_mem=72.27 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.07it/s]

    Capturing num tokens (num_tokens=640 avail_mem=72.27 GB):  47%|████▋     | 27/58 [00:01<00:01, 20.16it/s]Capturing num tokens (num_tokens=576 avail_mem=72.27 GB):  47%|████▋     | 27/58 [00:01<00:01, 20.16it/s]Capturing num tokens (num_tokens=512 avail_mem=72.26 GB):  47%|████▋     | 27/58 [00:01<00:01, 20.16it/s]

    Capturing num tokens (num_tokens=480 avail_mem=72.27 GB):  47%|████▋     | 27/58 [00:01<00:01, 20.16it/s]Capturing num tokens (num_tokens=480 avail_mem=72.27 GB):  52%|█████▏    | 30/58 [00:01<00:01, 15.43it/s]Capturing num tokens (num_tokens=448 avail_mem=72.27 GB):  52%|█████▏    | 30/58 [00:01<00:01, 15.43it/s]Capturing num tokens (num_tokens=416 avail_mem=72.27 GB):  52%|█████▏    | 30/58 [00:01<00:01, 15.43it/s]Capturing num tokens (num_tokens=384 avail_mem=72.27 GB):  52%|█████▏    | 30/58 [00:01<00:01, 15.43it/s]Capturing num tokens (num_tokens=352 avail_mem=72.26 GB):  52%|█████▏    | 30/58 [00:01<00:01, 15.43it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.26 GB):  59%|█████▊    | 34/58 [00:01<00:01, 18.65it/s]Capturing num tokens (num_tokens=320 avail_mem=72.25 GB):  59%|█████▊    | 34/58 [00:01<00:01, 18.65it/s]Capturing num tokens (num_tokens=288 avail_mem=72.25 GB):  59%|█████▊    | 34/58 [00:01<00:01, 18.65it/s]Capturing num tokens (num_tokens=256 avail_mem=72.25 GB):  59%|█████▊    | 34/58 [00:01<00:01, 18.65it/s]Capturing num tokens (num_tokens=240 avail_mem=72.25 GB):  59%|█████▊    | 34/58 [00:01<00:01, 18.65it/s]Capturing num tokens (num_tokens=240 avail_mem=72.25 GB):  66%|██████▌   | 38/58 [00:01<00:00, 21.85it/s]Capturing num tokens (num_tokens=224 avail_mem=72.24 GB):  66%|██████▌   | 38/58 [00:01<00:00, 21.85it/s]Capturing num tokens (num_tokens=208 avail_mem=72.24 GB):  66%|██████▌   | 38/58 [00:01<00:00, 21.85it/s]Capturing num tokens (num_tokens=192 avail_mem=72.24 GB):  66%|██████▌   | 38/58 [00:01<00:00, 21.85it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 21.85it/s]Capturing num tokens (num_tokens=176 avail_mem=72.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.44it/s]Capturing num tokens (num_tokens=160 avail_mem=72.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.44it/s]Capturing num tokens (num_tokens=144 avail_mem=72.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.44it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.44it/s]Capturing num tokens (num_tokens=112 avail_mem=72.22 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.44it/s]Capturing num tokens (num_tokens=96 avail_mem=72.22 GB):  72%|███████▏  | 42/58 [00:01<00:00, 24.44it/s] Capturing num tokens (num_tokens=96 avail_mem=72.22 GB):  81%|████████  | 47/58 [00:01<00:00, 29.42it/s]Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  81%|████████  | 47/58 [00:01<00:00, 29.42it/s]Capturing num tokens (num_tokens=64 avail_mem=72.21 GB):  81%|████████  | 47/58 [00:01<00:00, 29.42it/s]

    Capturing num tokens (num_tokens=48 avail_mem=72.21 GB):  81%|████████  | 47/58 [00:02<00:00, 29.42it/s]Capturing num tokens (num_tokens=32 avail_mem=72.21 GB):  81%|████████  | 47/58 [00:02<00:00, 29.42it/s]Capturing num tokens (num_tokens=32 avail_mem=72.21 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.24it/s]Capturing num tokens (num_tokens=28 avail_mem=72.20 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.24it/s]Capturing num tokens (num_tokens=24 avail_mem=72.13 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.24it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.54 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.24it/s]Capturing num tokens (num_tokens=20 avail_mem=72.54 GB):  93%|█████████▎| 54/58 [00:02<00:00, 18.28it/s]Capturing num tokens (num_tokens=16 avail_mem=72.16 GB):  93%|█████████▎| 54/58 [00:02<00:00, 18.28it/s]Capturing num tokens (num_tokens=12 avail_mem=71.79 GB):  93%|█████████▎| 54/58 [00:02<00:00, 18.28it/s]Capturing num tokens (num_tokens=8 avail_mem=72.12 GB):  93%|█████████▎| 54/58 [00:02<00:00, 18.28it/s] Capturing num tokens (num_tokens=8 avail_mem=72.12 GB):  98%|█████████▊| 57/58 [00:02<00:00, 18.20it/s]Capturing num tokens (num_tokens=4 avail_mem=72.15 GB):  98%|█████████▊| 57/58 [00:02<00:00, 18.20it/s]

    Capturing num tokens (num_tokens=4 avail_mem=72.15 GB): 100%|██████████| 58/58 [00:02<00:00, 21.57it/s]


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
    Generated text:  Jason. I'm 24 years old and a college student. I live in the city and go to school from home. I use the internet and watch movies. I read newspapers and I read books. The last time I went to the movies, I saw a film called "What's the Time of Your Life" which was really bad. There was an episode that I didn't like at all and I think it's too short. I haven't seen anything that I like. I have a good friend who is in the same city as me. She's also a college student and she went to school from home. She uses
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide between two mutually exclusive projects. The first project will produce $75,000 in annual revenue and $10,000 in annual expenses. The second project will produce $60,000 in annual revenue and $12,000 in annual expenses. The president is interested in the time value of money. Assuming the discount rate is 5%, which project should he choose? To determine which project the president should choose, we need to calculate the Net Present Value (NPV) for each project. The NPV is a measure of the present value of future cash flows,
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. Lyon
    C. Lyon
    D. Vienna
    Answer:
    
    A
    
    Mr. Wang, a construction project manager for a certain company, spent a considerable amount of time studying the relevant laws, regulations, and standards related to construction projects. Mr. Wang's actions reflect the professional competence in which area of the professional competence assessment?
    A. Knowledge
    B. Competence
    C. Willingness to Fulfill
    D. Integrity
    Answer:
    
    A
    
    Who is the author of the novel "The American Dream"? ____
    A. Jack London
    B. Washington Irving
    C. Washington Irving
    ===============================
    Prompt: The future of AI is
    Generated text:  far from a singularly bright future; it’s a very bright future, and one of the many bright things about AI is that it can be used for the common good. The vision of ubiquitous AI and smart cities is the ultimate good that we are all seeking. This vision is the same as the vision of the common good: a vision of good for all people.
    The vision of ubiquitous AI and smart cities is an increasingly important one in the twenty-first century. As AI becomes more advanced, we will find that the benefits of ubiquitous AI and smart cities will extend far beyond the existing ones. Although the benefits of AI and smart cities are


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working here for [number of years] years and I am currently [current position]. I am passionate about [reason why I love my job] and I am always looking for ways to improve my skills and knowledge. I am a [type of person] and I am always willing to learn and grow. I am a [character trait] and I am always ready to help others and make a positive impact. I am a [character trait] and I am always willing to go above and beyond to help others. I am a [character trait
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of French literature and cuisine. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a city of contrasts, with its historical architecture and modern fashion, and is a popular
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Enhanced ethical considerations: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical implications. This could lead to new regulations and standards to ensure that AI is used in a responsible and ethical manner
    


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
    Generated text:  [Name], and I'm a [Occupation/Field/Role] with over [Number] years of experience in [Your Profession or Role]. I bring a blend of [Any relevant skills or personality traits], and a [Any relevant background, education, or education level] to every project I undertake. I'm always looking for challenges and opportunities to learn from and grow. My work ethic is impeccable, and I'm constantly striving to improve my skills and knowledge. I believe in [Any relevant philosophy or ethical principles] and am committed to [Any relevant philanthropic, humanitarian, or social causes]. I am always looking for ways to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A French citizen visiting Paris will likely experience beautiful architecture, Parisian cuisine, and the famous Eiffel Tower. The city is known for its vibrant culture, including the Louvre Museum and many other museums. Paris also has a rich history and has been a major center for science and art for centuries. The city is known for its music and fashion scene, with many famous artists and designers living and working there. 
    
    The French capital is located in the south of the country, near the Mediterranean Sea. The city has a diverse population of about 22 million people. Paris is known for its long history and culture,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a significant shift towards more natural language processing and machine learning. This will likely lead to the development of more sophisticated and intelligent conversational interfaces, as well as the creation of more powerful and capable AI systems that can handle a wider range of tasks. Additionally, there will be an increased focus on developing ethical and responsible AI systems that can be used for good, rather than causing harm. Overall, the future of AI is likely to be one of progress, improvement, and growth, as more people and organizations become increasingly dependent on AI technology for their daily lives and work. However, it is also important to be mindful of the potential


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

    name

    ],

     and

     I

     am

     a

    /an

     [

    occupation

    ].

     I

     have

     always

     been

     passionate

     about

     [

    reason

     for

     interest

    ].

     I

     love

     [

    something

     that

     makes

     me

     happy

    ],

     and

     I

     am

     always

     looking

     for

     new

     ways

     to

     [

    something

     that

     makes

     me

     grow

    ].

     What

    's

     your

     favorite

     hobby

    ?

     What

    's

     something

     you

    'd

     like

     to

     achieve

     in

     the

     future

    ?

     How

     do

     you

     approach

     problem

    -solving

    ?


    Hello

    ,

     my

     name

     is

     [

    name

    ]

     and

     I

     am

     a

    /an

     [

    occupation

    ].

     I

     have

     always

     been

     passionate

     about

     [

    reason

     for

     interest

    ].

     I

     love

     [

    something

     that

     makes

     me

     happy

    ],

     and

     I

     am

     always

     looking

     for

     new

     ways

     to

     [

    something

     that

     makes

     me

     grow

    ].

     What

    's

     your

     favorite

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     in

     France

    ,

     home

     to

     the

     French

     parliament

    ,

     the

     Palace

     of

     Vers

    ailles

    ,

     and

     the

     iconic

     E

    iff

    el

     Tower

    .

     The

     city

     is

     also

     a

     significant

     cultural

    ,

     economic

    ,

     and

     political

     center

    ,

     known

     for

     its

     rich

     history

    ,

     diverse

     heritage

    ,

     and

     vibrant

     street

     life

    .

     Paris

    's

     cuisine

    ,

     art

    ,

     and

     music

     industries

     are

     also

     highly

     regarded

    .

     Its

     annual

     Carn

    aval

     and

     annual

     "

    F

    ête

     de

     la

     May

    enne

    "

     are

     major

     cultural

     events

     that

     draw

     millions

     of

     visitors

     each

     year

    .

     Additionally

    ,

     Paris

     is

     the

     country

    's

     economic

     heart

     and

     one

     of

     the

     most

     visited

     tourist

     destinations

     in

     the

     world

    .

     The

     city

     is

     known

     for

     its

     gastr

    onomy

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     involve

     a

     number

     of

     new

     technologies

     and

     advancements

     that

     will

     further

     expand

     its

     capabilities

     and

     impact

     on

     our

     lives

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

     Deep

     learning

    :

     Deep

     learning

     is

     a

     subset

     of

     machine

     learning

     that

     focuses

     on

     training

     artificial

     neural

     networks

     to

     solve

     complex

     tasks

    .

     It

     is

     expected

     to

     become

     more

     powerful

     and

     efficient

     as

     more

     data

     is

     available

     to

     train

     these

     networks

    ,

     leading

     to

     new

     breakthrough

    s

     in

     areas

     such

     as

     image

     and

     speech

     recognition

    .
    


    2

    .

     Explain

    ability

    :

     AI

     systems

     that

     are

     designed

     to

     be

     explain

    able

     will

     become

     more

     prominent

     in

     the

     future

    .

     This

     will

     require

     greater

     transparency

     in

     how

     these

     systems

     work

     and

     what

     decisions

     they

     make

    ,

     which

    



```python
llm.shutdown()
```
