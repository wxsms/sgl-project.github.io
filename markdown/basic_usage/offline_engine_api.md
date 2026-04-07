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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.19it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.18it/s]


    2026-04-07 08:51:51,825 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 08:51:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.23it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.23it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.23it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.23it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:07,  5.89it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:07,  5.89it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:07,  5.89it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:07,  5.89it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  5.89it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  5.89it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  5.89it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  5.89it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  5.89it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 11.98it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 11.98it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 11.98it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 11.98it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 11.98it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 11.98it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 11.98it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 11.98it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 17.84it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 17.84it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 17.84it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 17.84it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 17.84it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 17.84it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 17.84it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 17.84it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.17it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.17it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.12it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.12it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.12it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.12it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.12it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.12it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.12it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 33.95it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 33.95it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 33.95it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 33.95it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 33.95it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 33.95it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 33.95it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.12it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.21it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=132.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=132.42 GB):   2%|▏         | 1/58 [00:00<00:29,  1.92it/s]Capturing num tokens (num_tokens=7680 avail_mem=132.39 GB):   2%|▏         | 1/58 [00:00<00:29,  1.92it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=132.39 GB):   3%|▎         | 2/58 [00:00<00:21,  2.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=132.38 GB):   3%|▎         | 2/58 [00:00<00:21,  2.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=132.38 GB):   5%|▌         | 3/58 [00:00<00:15,  3.54it/s]Capturing num tokens (num_tokens=6656 avail_mem=132.38 GB):   5%|▌         | 3/58 [00:00<00:15,  3.54it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=132.38 GB):   7%|▋         | 4/58 [00:01<00:12,  4.19it/s]Capturing num tokens (num_tokens=6144 avail_mem=132.38 GB):   7%|▋         | 4/58 [00:01<00:12,  4.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=132.38 GB):   7%|▋         | 4/58 [00:01<00:12,  4.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=132.38 GB):  10%|█         | 6/58 [00:01<00:07,  6.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=132.38 GB):  10%|█         | 6/58 [00:01<00:07,  6.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=132.37 GB):  10%|█         | 6/58 [00:01<00:07,  6.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=132.37 GB):  10%|█         | 6/58 [00:01<00:07,  6.90it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=132.37 GB):  16%|█▌        | 9/58 [00:01<00:04, 11.68it/s]Capturing num tokens (num_tokens=3840 avail_mem=132.37 GB):  16%|█▌        | 9/58 [00:01<00:04, 11.68it/s]Capturing num tokens (num_tokens=3584 avail_mem=132.37 GB):  16%|█▌        | 9/58 [00:01<00:04, 11.68it/s]Capturing num tokens (num_tokens=3328 avail_mem=132.36 GB):  16%|█▌        | 9/58 [00:01<00:04, 11.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=132.36 GB):  16%|█▌        | 9/58 [00:01<00:04, 11.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=132.36 GB):  22%|██▏       | 13/58 [00:01<00:02, 17.94it/s]Capturing num tokens (num_tokens=2816 avail_mem=132.36 GB):  22%|██▏       | 13/58 [00:01<00:02, 17.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=132.35 GB):  22%|██▏       | 13/58 [00:01<00:02, 17.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=132.35 GB):  22%|██▏       | 13/58 [00:01<00:02, 17.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=132.35 GB):  22%|██▏       | 13/58 [00:01<00:02, 17.94it/s]Capturing num tokens (num_tokens=1792 avail_mem=132.34 GB):  22%|██▏       | 13/58 [00:01<00:02, 17.94it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=132.34 GB):  31%|███       | 18/58 [00:01<00:01, 24.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=132.34 GB):  31%|███       | 18/58 [00:01<00:01, 24.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=132.34 GB):  31%|███       | 18/58 [00:01<00:01, 24.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=132.32 GB):  31%|███       | 18/58 [00:01<00:01, 24.93it/s]Capturing num tokens (num_tokens=960 avail_mem=132.33 GB):  31%|███       | 18/58 [00:01<00:01, 24.93it/s] Capturing num tokens (num_tokens=896 avail_mem=132.33 GB):  31%|███       | 18/58 [00:01<00:01, 24.93it/s]Capturing num tokens (num_tokens=896 avail_mem=132.33 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.40it/s]Capturing num tokens (num_tokens=832 avail_mem=132.32 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.40it/s]Capturing num tokens (num_tokens=768 avail_mem=132.32 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.40it/s]Capturing num tokens (num_tokens=704 avail_mem=132.32 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.40it/s]Capturing num tokens (num_tokens=640 avail_mem=132.31 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.40it/s]Capturing num tokens (num_tokens=576 avail_mem=132.31 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.40it/s]

    Capturing num tokens (num_tokens=576 avail_mem=132.31 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=512 avail_mem=132.30 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=480 avail_mem=132.32 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=448 avail_mem=132.32 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=416 avail_mem=132.31 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=384 avail_mem=132.31 GB):  48%|████▊     | 28/58 [00:01<00:00, 34.51it/s]Capturing num tokens (num_tokens=384 avail_mem=132.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=352 avail_mem=132.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=320 avail_mem=132.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=288 avail_mem=132.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=256 avail_mem=132.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=240 avail_mem=132.29 GB):  57%|█████▋    | 33/58 [00:02<00:00, 37.78it/s]

    Capturing num tokens (num_tokens=240 avail_mem=132.29 GB):  66%|██████▌   | 38/58 [00:02<00:00, 39.85it/s]Capturing num tokens (num_tokens=224 avail_mem=132.29 GB):  66%|██████▌   | 38/58 [00:02<00:00, 39.85it/s]Capturing num tokens (num_tokens=208 avail_mem=132.29 GB):  66%|██████▌   | 38/58 [00:02<00:00, 39.85it/s]Capturing num tokens (num_tokens=192 avail_mem=132.29 GB):  66%|██████▌   | 38/58 [00:02<00:00, 39.85it/s]Capturing num tokens (num_tokens=176 avail_mem=132.28 GB):  66%|██████▌   | 38/58 [00:02<00:00, 39.85it/s]Capturing num tokens (num_tokens=160 avail_mem=132.28 GB):  66%|██████▌   | 38/58 [00:02<00:00, 39.85it/s]Capturing num tokens (num_tokens=160 avail_mem=132.28 GB):  74%|███████▍  | 43/58 [00:02<00:00, 41.70it/s]Capturing num tokens (num_tokens=144 avail_mem=132.28 GB):  74%|███████▍  | 43/58 [00:02<00:00, 41.70it/s]Capturing num tokens (num_tokens=128 avail_mem=132.27 GB):  74%|███████▍  | 43/58 [00:02<00:00, 41.70it/s]Capturing num tokens (num_tokens=112 avail_mem=132.27 GB):  74%|███████▍  | 43/58 [00:02<00:00, 41.70it/s]Capturing num tokens (num_tokens=96 avail_mem=132.27 GB):  74%|███████▍  | 43/58 [00:02<00:00, 41.70it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=132.26 GB):  74%|███████▍  | 43/58 [00:02<00:00, 41.70it/s]Capturing num tokens (num_tokens=80 avail_mem=132.26 GB):  83%|████████▎ | 48/58 [00:02<00:00, 41.55it/s]Capturing num tokens (num_tokens=64 avail_mem=132.26 GB):  83%|████████▎ | 48/58 [00:02<00:00, 41.55it/s]Capturing num tokens (num_tokens=48 avail_mem=132.26 GB):  83%|████████▎ | 48/58 [00:02<00:00, 41.55it/s]Capturing num tokens (num_tokens=32 avail_mem=132.25 GB):  83%|████████▎ | 48/58 [00:02<00:00, 41.55it/s]Capturing num tokens (num_tokens=28 avail_mem=132.25 GB):  83%|████████▎ | 48/58 [00:02<00:00, 41.55it/s]Capturing num tokens (num_tokens=24 avail_mem=132.25 GB):  83%|████████▎ | 48/58 [00:02<00:00, 41.55it/s]Capturing num tokens (num_tokens=24 avail_mem=132.25 GB):  91%|█████████▏| 53/58 [00:02<00:00, 42.73it/s]Capturing num tokens (num_tokens=20 avail_mem=132.24 GB):  91%|█████████▏| 53/58 [00:02<00:00, 42.73it/s]Capturing num tokens (num_tokens=16 avail_mem=132.24 GB):  91%|█████████▏| 53/58 [00:02<00:00, 42.73it/s]Capturing num tokens (num_tokens=12 avail_mem=132.24 GB):  91%|█████████▏| 53/58 [00:02<00:00, 42.73it/s]Capturing num tokens (num_tokens=8 avail_mem=132.23 GB):  91%|█████████▏| 53/58 [00:02<00:00, 42.73it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=132.23 GB):  91%|█████████▏| 53/58 [00:02<00:00, 42.73it/s]Capturing num tokens (num_tokens=4 avail_mem=132.23 GB): 100%|██████████| 58/58 [00:02<00:00, 43.98it/s]Capturing num tokens (num_tokens=4 avail_mem=132.23 GB): 100%|██████████| 58/58 [00:02<00:00, 23.46it/s]


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
    Generated text:  Aniket. I am 22 years old and I have been programming for about 1 year now. I have been working on a project that aims to create a dictionary app. The app should have functionality for adding new words, deleting existing words, and showing the user's current word. I have been following a tutorial on Udemy and I have started using Google's App Engine, so I have a good understanding of how to work with the backend. I have also been using the Jython library on my local machine to run the application. My code is all in Python and I am trying to implement Google's PubSub library
    ===============================
    Prompt: The president of the United States is
    Generated text:  inaugurated on March 4th and 5th of each year. The last day he will be inaugurated is the last day of the ________.
    This is a logical reasoning problem related to the historical context of the inauguration ceremony of the U. S. president. Here's a step-by-step breakdown of how we can solve this:
    
    1. Identify the key information: The inauguration ceremony of the U. S. president is held on March 4th and 5th each year.
    2. Understand the time frames: The inauguration ceremony typically occurs on the last day of each month.
    3. Determine the last day: Since
    ===============================
    Prompt: The capital of France is
    Generated text:  in what location? To determine the capital of France, we can follow these steps:
    
    1. Identify the official capital of France.
    2. Confirm the official capital by providing a reference.
    
    The official capital of France is Paris. Therefore, the answer is:
    
    \boxed{Paris}
    ===============================
    Prompt: The future of AI is
    Generated text:  what AI is today. The technology is making huge leaps forward in a matter of years, but there’s still a lot of work to be done to ensure that the future of AI is as equitable and accessible as the present.
    
    But what exactly is AI and what is it meant to be? While the term has been around for a long time, the current definition is a blend of cyber and biological sciences. Although the word "AI" is often thought of as an all-encompassing term, it actually refers to the ability of machines to think and reason like humans.
    
    Just like the brain of a living organism, AI is made up of


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Character] who has always been [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits] and I'm [Positive Traits]. I'm [Positive Traits
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" or "La Ville Blanche" (White City). It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, art, and culture, and is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major transportation hub, with many major highways and rail lines connecting the city to other parts of France and the world. Paris is a popular tourist destination and is home to many museums, theaters, and other cultural institutions. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations more effectively. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability. AI systems will need to be designed and implemented in a way that is fair and equitable, and that respects the privacy and autonomy of individuals
    


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
    Generated text:  Emily, and I am a science fiction writer who has been working for over a decade. I am always on the lookout for the next great adventure, and I love to explore the unknown and see the world in new and exciting ways. I am always up for a challenge, and I love to work hard and try new things. If you have a story in mind, I would love to hear it. I'm a free agent, and I'm always looking for the next writing assignment.
    Your writing assignment is a bit of a surprise for me. I have a deadline for this story, but I'm up for a challenge and ready to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    The statement provided accurately represents the capital city of France. It is widely recognized as the political, economic, and cultural center of the country. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and the Palace of Versailles. The city is also renowned for its rich history, art, fashion, and cuisine. As one of the most visited cities in the world, Paris plays a significant role in France's identity and its global significance. The statement succinctly encapsulates the core importance of Paris in French culture and the French way of life. 
    
    To provide
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be marked by both positive and negative developments, as it continues to advance at a rapid pace. Here are some possible future trends in AI:
    
    1. Increased AI Integration: With the increasing availability of data, AI is likely to become more integrated into various sectors, such as healthcare, finance, transportation, and manufacturing. This integration will enable AI to learn from and adapt to the data it receives, leading to more accurate and personalized results.
    
    2. Enhanced AI Ethics and Transparency: As AI becomes more sophisticated, there will be a need for a more ethical and transparent approach to its development and deployment. The development of AI systems should consider


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

     Name

    ]

     and

     I

    'm

     [

    Last

     Name

    ]

    !

     I

    'm

     a

     [

    Describe

     your

     profession

     or

     role

    ]

     and

     I

     have

     [

    Your

     experience

     or

     skills

    ].

     I

    've

     been

     working

     for

     [

    Company

     Name

    ]

     for

     [

    Number

    ]

     years

    ,

     and

     I

     specialize

     in

     [

    Describe

     your

     specialty

     or

     area

     of

     expertise

    ].

     I

     enjoy

     [

    Your

     passion

     or

     interest

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    Express

     a

     new

     skill

     or

     accomplishment

    ].

     I

    'm

     always

     eager

     to

     learn

     and

     grow

    ,

     and

     I

     strive

     to

     make

     a

     positive

     impact

     on

     [

    Describe

     the

     impact

     or

     effect

     you

     want

     to

     have

     on

     the

     world

     or

     community

    ].

     So

    ,

     what

    's

     your

     experience

     and

     what

    's

     your

     passion

    
    
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

     second

     largest

     city

     in

     Europe

     by

     population

    .

     It

     is

     a

     historic

     and

     cultural

     center

     with

     many

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     a

     world

    -ren

    owned

     food

     and

     wine

     capital

    ,

     with

     famous

     dishes

     like

     cro

    iss

    ants

    ,

     o

    me

    lets

    ,

     and

     steak

     au

     bacon

    .

     Paris

     has

     a

     rich

     history

     and

     is

     home

     to

     many

     museums

    ,

     theaters

    ,

     and

     art

     galleries

    .

     It

     has

     been

     a

     cultural

     and

     economic

     center

     for

     centuries

    ,

     and

     its

     name

     has

     become

     synonymous

     with

     sophistication

     and

     glamour

    .

     Today

    ,

     Paris

     remains

     one

     of

     the

     most

     popular

     tourist

     destinations

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     likely

     to

     be

     shaped

     by

     a

     variety

     of

     trends

     that

     are

     currently

     being

     developed

     or

     under

     development

    .

     Some

     of

     the

     most

     significant

     trends

     currently

     influencing

     the

     AI

     landscape

     include

    :
    


    1

    .

     Increased

     Use

     of AI

     for

     Healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     assist

     doctors

     in

     diagn

    osing

     and

     treating

     illnesses

    .

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     will

     be

     able

     to

     analyze

     medical

     images

     and

     data

     more

     accurately

    ,

     leading

     to

     more

     accurate

     diagnoses

     and

     treatment

     outcomes

    .
    


    2

    .

     Increased

     Use

     of

     AI

     in

     Agriculture

    :

     AI

     is

     also

     being

     used

     in

     agriculture

     to

     improve

     crop

     yields

     and

     reduce

     the

     amount

     of

     pesticides

     and

     other

     chemicals

     needed

    .

     AI

     algorithms

     can

     predict

     plant

     growth

     and

     disease

     patterns

    ,

     allowing

    



```python
llm.shutdown()
```
