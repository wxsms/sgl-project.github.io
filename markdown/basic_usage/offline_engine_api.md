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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.24it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.22it/s]


    2026-04-08 19:01:47,366 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 19:01:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.74it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.74it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:30,  1.74it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:30,  1.74it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:30,  1.74it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:30,  1.74it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:30,  1.74it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.38it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.38it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.38it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.38it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.38it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.38it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.38it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.38it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.38it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.40it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.40it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.40it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.40it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.40it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.40it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.40it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.40it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.21it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.21it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.21it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.21it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.21it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.21it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.21it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.21it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 23.65it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 23.65it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 23.65it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 23.65it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 23.65it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 23.65it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 23.65it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 28.55it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 28.55it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 28.55it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 28.55it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 28.55it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 28.55it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 28.55it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 33.81it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 33.81it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 33.81it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 33.81it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 33.81it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 33.81it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 33.81it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.09it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.09it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.09it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.09it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.09it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.09it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.09it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.09it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 37.09it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 46.07it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:06,  8.78it/s]Capturing num tokens (num_tokens=7168 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:06,  8.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:06,  8.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=119.02 GB):   7%|▋         | 4/58 [00:00<00:04, 12.67it/s]Capturing num tokens (num_tokens=6144 avail_mem=119.02 GB):   7%|▋         | 4/58 [00:00<00:04, 12.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.01 GB):   7%|▋         | 4/58 [00:00<00:04, 12.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.01 GB):   7%|▋         | 4/58 [00:00<00:04, 12.67it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=119.01 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.01 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.01 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.00 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.03it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.00 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.03it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.00 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.00 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.35it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.99 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.35it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.99 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.35it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.99 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.35it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=118.98 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.35it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.18it/s]Capturing num tokens (num_tokens=960 avail_mem=118.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.18it/s] Capturing num tokens (num_tokens=896 avail_mem=118.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.18it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.18it/s]

    Capturing num tokens (num_tokens=768 avail_mem=118.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.18it/s]Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.18it/s]Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.09it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.09it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.09it/s]Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.09it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.09it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.09it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=384 avail_mem=118.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.13it/s]

    Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=288 avail_mem=118.93 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=288 avail_mem=118.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.38it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.38it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.38it/s]Capturing num tokens (num_tokens=224 avail_mem=118.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.38it/s]Capturing num tokens (num_tokens=208 avail_mem=118.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.38it/s]

    Capturing num tokens (num_tokens=192 avail_mem=118.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.38it/s]Capturing num tokens (num_tokens=192 avail_mem=118.62 GB):  71%|███████   | 41/58 [00:01<00:00, 35.75it/s]Capturing num tokens (num_tokens=176 avail_mem=118.62 GB):  71%|███████   | 41/58 [00:01<00:00, 35.75it/s]Capturing num tokens (num_tokens=160 avail_mem=118.62 GB):  71%|███████   | 41/58 [00:01<00:00, 35.75it/s]Capturing num tokens (num_tokens=144 avail_mem=118.61 GB):  71%|███████   | 41/58 [00:01<00:00, 35.75it/s]Capturing num tokens (num_tokens=128 avail_mem=118.61 GB):  71%|███████   | 41/58 [00:01<00:00, 35.75it/s]

    Capturing num tokens (num_tokens=128 avail_mem=118.61 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.00it/s]Capturing num tokens (num_tokens=112 avail_mem=118.61 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.00it/s]Capturing num tokens (num_tokens=96 avail_mem=118.60 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.00it/s] Capturing num tokens (num_tokens=80 avail_mem=118.60 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.00it/s]Capturing num tokens (num_tokens=64 avail_mem=118.60 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.00it/s]Capturing num tokens (num_tokens=64 avail_mem=118.60 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.02it/s]Capturing num tokens (num_tokens=48 avail_mem=118.60 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.02it/s]Capturing num tokens (num_tokens=32 avail_mem=118.59 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.02it/s]Capturing num tokens (num_tokens=28 avail_mem=118.59 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.02it/s]

    Capturing num tokens (num_tokens=24 avail_mem=118.58 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.02it/s]Capturing num tokens (num_tokens=24 avail_mem=118.58 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.86it/s]Capturing num tokens (num_tokens=20 avail_mem=118.58 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.86it/s]Capturing num tokens (num_tokens=16 avail_mem=118.58 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.86it/s]Capturing num tokens (num_tokens=12 avail_mem=118.57 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.86it/s]Capturing num tokens (num_tokens=8 avail_mem=118.57 GB):  91%|█████████▏| 53/58 [00:01<00:00, 31.86it/s] Capturing num tokens (num_tokens=8 avail_mem=118.57 GB):  98%|█████████▊| 57/58 [00:01<00:00, 33.45it/s]Capturing num tokens (num_tokens=4 avail_mem=118.56 GB):  98%|█████████▊| 57/58 [00:01<00:00, 33.45it/s]Capturing num tokens (num_tokens=4 avail_mem=118.56 GB): 100%|██████████| 58/58 [00:01<00:00, 30.87it/s]


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
    Generated text:  Jingyuan Wang. I'm a 2nd-year PhD student in the Department of Materials Science and Engineering at the University of Toronto. My research group at the University of Toronto focuses on the development of hybrid nanofluidic structures for the study of superfluids and quantum vortices. My research areas are nanofluidics, fluid mechanics, and quantum physics.
    I obtained my B.S. in Mechanical Engineering from Tongji University, Shanghai, China, in 2018. I also studied Nano Science and Engineering at Zhejiang University, Hangzhou, China, with a specialization in nanofl
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office with a term of four years. After 1974, this term was extended to 10 years. Since 1980, the president of the United States has been elected for a term of 5 years. The president of the United States was inaugurated on January 20, 1981.
    
    Given this information, how many years will it be before the next president of the United States is inaugurated? To determine how many years it will be before the next president of the United States is inaugurated, we need to calculate the total number of years from the inauguration of the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the capital city of the country of France.
    The capital of India is New Delhi. New Delhi is the capital city of the country of India.
    Which of the following is the correct order of the capital cities of France and India?  Answer Choices in numbers.
    To determine the correct order of the capital cities of France and India, we need to carefully compare the information provided in both statements and then identify the logical sequence.
    
    First, let's summarize the information given:
    1. Paris is the capital of France.
    2. New Delhi is the capital of India.
    
    Now, let's consider the logical order:
    1. Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but with the rapid adoption of AI in various industries, there are also concerns about the ethical implications of AI systems. The well-being of individuals and society as a whole is at stake. So, what can we do to ensure the ethical implementation of AI? In this article, I will discuss the ethical implications of AI systems and how we can ensure their ethical implementation. Here are some ways to ensure the ethical implementation of AI systems:
    
    1. Create clear and consistent ethical guidelines and policies: AI systems should be designed with clear and consistent ethical guidelines and policies. These guidelines and policies should be transparent and accessible to all stakeholders.
    
    2.


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


    Generated text:  Paris, also known as "La Ville de Paris" and "La Ville de la Rose". It is the largest city in France and the third largest city in the world by population. The city is located on the Seine River and is home to many of France's most famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its rich cultural heritage, including the Notre-Dame Cathedral, the Louvre Museum, and the Opéra Garnier. The city is a major economic and cultural center in France and is home to many of the country's most famous museums
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we can expect to see more automation and artificial intelligence in our daily lives. This could include the development of robots and other machines that can perform tasks that were previously done by humans, such as manufacturing, healthcare, and transportation.
    
    2. Improved privacy and security: As AI technology becomes more advanced, we can expect to see more privacy and security concerns. This could include the development of more secure
    


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
    Generated text:  [Name]. I am [Age] years old, but I have always been [Lover]. I enjoy [Reason for Enjoying]. My love for [Topic] is [Aspect]. I am a [Gender] and I believe that [Purpose of Life]. I am [Image]. I am [Career]. I am [Hobby]. I am [Addressing Audience]. My name is [Name], and I am an [Appearance]. I am [Age], and I am [Gender]. I am [Relationship Status]. I am [Occupation]. I am [Hobbies]. I am [Addressing Audience]. My name is
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its rich history, cultural diversity, and iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral. Paris is also known for its culinary traditions, especially in its famous food districts such as the Marais and Montmartre. The city has a diverse population of around 2.7 million people and is the third-largest city in the European Union by population. Paris was founded in 787 as a Roman colony and is now a multicultural and cosmopolitan city with a strong focus on arts and culture. The French capital is home to many museums, parks, and tourist attractions, and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising and has the potential to transform nearly every industry and sector. Here are some possible future trends in AI:
    
    1. Enhanced cognitive abilities: AI is being trained to learn and adapt in ways that were previously impossible. This includes the ability to process complex information and make decisions that are beyond human comprehension.
    
    2. AI-powered healthcare: AI is being used to improve the accuracy and speed of medical diagnosis, treatment planning, and patient care. AI-powered medical devices, such as mobile devices with AI capabilities, are already being used in hospitals.
    
    3. Autonomous vehicles: The use of AI is expected to increase in the future, with autonomous vehicles


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

    'm

     a

     [

    job

     title

    ]

     with

     over

     [

    number

    ]

     years

     of

     experience

    .

     I

    'm

     passionate

     about

     [

    personal

     interest

     or

     hobby

    ].

     I

     enjoy

     [

    occupation

    ],

     and

     I

     strive

     to

     be

     [

    desired

     qualities

    ].

     I

     am

     always

     looking

     for

     opportunities

     to

     [

    desired

     skill

     or

     skill

    set

    ],

     and

     I

    'm

     always

     eager

     to

     learn

     and

     grow

    .

     I

    'm

     a

     [

    personal

     trait

     or

     characteristic

    ]

     that

     makes

     me

     unique

     and

     can

     be

     used

     in

     many

     different

     situations

    .

     I

     believe

     in

     [

    char

    isma

     or

     personality

     trait

    ]

     and

     am

     always

     willing

     to

     contribute

     my

     best

     self

     to

     [

    overall

     goal

     or

     purpose

    ].

     I

     believe

     that

     my

     experience

     and

     skills

     can

     be

     a

     valuable

     asset

    
    
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

     and

     the

     capital

     of

     the

     country

    .

     It

     is

     a

     bustling

     met

    ropolis

     with

     many

     attractions

     and

     events

     throughout

     the

     year

    ,

     including

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

     and

     the

     Luxembourg

     Gardens

    .

     The

     city

     also

     hosts

     numerous

     cultural

     and

     artistic

     events

     throughout

     the

     year

    ,

     such

     as

     the

     Mar

    ais

     Festival

     and

     the

     Jazz

     Festival

    .

     Paris

     is

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    ,

     known

     for

     its

     stunning

     architecture

    ,

     delicious

     cuisine

    ,

     and

     vibrant

     nightlife

    .

     The

     city

     is

     a

     fascinating

     blend

     of

     history

    ,

     culture

    ,

     and

     modern

    ity

    ,

     making

     it

     a

     must

    -

    visit

     destination

     for

     anyone

     interested

     in

     France

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     here

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

     technology

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

    -powered

     healthcare

     systems

     will

     be

     more

     accurate

     and

     efficient

     in

     diagn

    osing

     diseases

    ,

     providing

     personalized

     treatment

     plans

    ,

     and

     improving

     patient

     outcomes

    .
    


    2

    .

     AI

     in

     finance

    :

     AI

     will

     become

     more

     integrated

     into

     the

     financial

     industry

    ,

     autom

    ating

     complex

     tasks

     like

     fraud

     detection

    ,

     risk

     assessment

    ,

     and

     personalized

     investment

     strategies

    .
    


    3

    .

     AI

     in

     transportation

    :

     Autonomous

     vehicles

     and

     self

    -driving

     cars

     will

     become

     more

     common

    ,

     and

     AI

     will

     be

     used

     to

     optimize

     traffic

     flow

     and

     reduce

     congestion

    .
    


    4

    .

     AI

     in

     entertainment

    :

     AI

     will

     be

     used

     to

    



```python
llm.shutdown()
```
