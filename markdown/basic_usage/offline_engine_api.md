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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.25it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.24it/s]


    2026-04-08 18:49:50,841 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 18:49:50] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:57,  3.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:57,  3.12s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<02:57,  3.12s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<02:57,  3.12s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.61it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.61it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:33,  1.61it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:33,  1.61it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:33,  1.61it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:12,  3.86it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:12,  3.86it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:12,  3.86it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:12,  3.86it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:03<00:12,  3.86it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:03<00:12,  3.86it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:03<00:12,  3.86it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:05,  8.19it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:05,  8.19it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:05,  8.19it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:05,  8.19it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:05,  8.19it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:03<00:05,  8.19it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:03<00:05,  8.19it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:03<00:05,  8.19it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 14.24it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 14.24it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 14.24it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 14.24it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 14.24it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 14.24it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 14.24it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 19.86it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 19.86it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 19.86it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 19.86it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 19.86it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 19.86it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 19.86it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.67it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.67it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.67it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.67it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.67it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.67it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.67it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.96it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.96it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.96it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.96it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.96it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.96it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.96it/s]

    Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 36.43it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 36.43it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 36.43it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 36.43it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:04<00:00, 36.43it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:04<00:00, 36.43it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:04<00:00, 36.43it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 39.71it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 39.71it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 39.71it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 39.71it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 39.71it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 39.71it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 39.71it/s] 

    Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 39.71it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 13.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 17.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 17.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.68 GB):   3%|▎         | 2/58 [00:00<00:03, 17.88it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.68 GB):   3%|▎         | 2/58 [00:00<00:03, 17.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 20.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 20.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 20.37it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 20.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.47it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.64it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.24it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.60it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.60it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.60it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.60it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.18it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.18it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.18it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.18it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:01<00:01, 21.76it/s]Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:01<00:01, 21.76it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:01<00:01, 21.76it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:01<00:01, 21.76it/s]

    Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:01<00:01, 21.03it/s]Capturing num tokens (num_tokens=576 avail_mem=74.61 GB):  47%|████▋     | 27/58 [00:01<00:01, 21.03it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  47%|████▋     | 27/58 [00:01<00:01, 21.03it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:01<00:01, 21.03it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:01<00:01, 21.03it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  47%|████▋     | 27/58 [00:01<00:01, 21.03it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:01<00:01, 20.62it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:01<00:01, 20.62it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:01<00:01, 20.62it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:01<00:01, 20.62it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:01, 18.08it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:01, 18.08it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:01, 18.08it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:01, 18.08it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:01<00:01, 18.08it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 21.14it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 21.14it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 21.14it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.14it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:02<00:00, 21.14it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.83it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.83it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.83it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.83it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.83it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.83it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  83%|████████▎ | 48/58 [00:02<00:00, 29.14it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:02<00:00, 29.14it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:02<00:00, 29.14it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:02<00:00, 29.14it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:02<00:00, 29.14it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:02<00:00, 31.44it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:02<00:00, 31.44it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:02<00:00, 31.44it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:02<00:00, 31.44it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:02<00:00, 31.44it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:02<00:00, 31.44it/s] Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  98%|█████████▊| 57/58 [00:02<00:00, 35.12it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:02<00:00, 35.12it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:02<00:00, 23.94it/s]


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
    Generated text:  Isabella and I am a full-stack developer. I love Python and React. I started this website called MyPronunciation so I can take notes, practice speaking, and learn more about pronunciation.
    I also want to learn more about SEO.
    I am a self-taught developer who wants to become a good developer.
    How can I learn more about SEO?
    And what are some resources you would recommend for this?
    I am so excited to learn more about SEO! What would you recommend starting with?
    To learn more about SEO, I would recommend starting with the following resources:
    1. "Search Engine Optimization for Dummies" by Mark
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 3 inches tall. The vice president is 3 feet 1 inch tall. How many feet taller is the vice president than the president?
    To determine how many feet taller the vice president is than the president, we need to convert both heights to the same unit and then find the difference.
    
    First, let's convert the heights from feet and inches to just inches. The president's height is 5 feet 3 inches, and the vice president's height is 3 feet 1 inch.
    
    1. Convert the president's height to inches:
       \[
       5 \text{ feet} \times 1
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the ____.
    A. Center of the Atlantic Ocean
    B. River Seine
    C. River Seine and the Jura Mountains
    D. River Seine and the Alps
    Answer:
    C
    
    When selecting new materials for building components, which of the following should NOT be used as the basis for selection? 
    A. The current cost of the material
    B. The current price of similar materials
    C. The current price of similar components
    D. The current value of new materials
    Answer:
    D
    
    The patient is a 48-year-old female. She has had a high fever for 2 days and
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly evolving, and in a decentralized ecosystem, there is an urgent need to address the issues of ethical and equitable AI. In this article, we will explore how to overcome these challenges and create a fair, transparent, and secure AI ecosystem.
    
    ## Introduction to AI and the Future of Ethics
    
    AI refers to the ability of machines to learn and adapt, from the emergence of computer programs to the development of artificial intelligence technologies like machine learning and deep learning. This technology is rapidly evolving, and the field of AI is expected to continue to grow in the coming years.
    
    However, the ethical implications of AI are also growing. As AI becomes more integrated


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [job title] at [company name], and I've been working in the [industry] for [number of years] years. I'm always looking for ways to improve my skills and stay up-to-date with the latest trends in my field. What's your favorite hobby or activity to do in your free time? I enjoy [mention a hobby or activity]. What's your favorite book or movie to watch? I love [mention a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and restaurants. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, art, and cuisine, and is a major center of politics, science, and culture in France. Paris is often referred to as the "City of Light" and is considered one of the most beautiful cities in the world. The city is also home to many famous landmarks and attractions
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial general intelligence: As AI technology continues to advance, we can expect to see more automation and the development of AI that can perform tasks that were previously done by humans. This could lead to a more efficient and cost-effective way of doing things, but it could also lead to job displacement for some workers.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, we can expect to see increased concerns about privacy and security. This
    


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
    Generated text:  [Name] and I'm a/an [Background]. I love [特长/爱好/成就/等等]. I'm [Hobby/Interest/Professional Preference/等等]. I believe in [Your Personal Values/Professionalism]. I'm always ready to learn from my mistakes and work on my strengths. I always strive to do my best and I believe in the power of hard work. What do you think makes you stand out from the crowd? What is your favorite [food/Drink/Activity/ etc.]. I'm looking forward to meeting you! [Name]. [Name] is the [Name] of your character. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and is the capital of France. It is also the seat of the country's government, the Council of State, and the President of the Senate. Paris is a world-renowned cultural and artistic center, with many famous museums, art galleries, and historical sites. It is also a major tourist destination, attracting millions of visitors each year. Paris is a blend of historical heritage and modernity, known for its elegant architecture, cafes, and restaurants. Its status as the world's most populous city has also led to a thriving food and beverage industry. Paris is a popular destination for business and government
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be full of exciting changes, and many possibilities for where the technology is headed. Here are a few potential future trends that AI might have:
    
    1. Increased Use in Healthcare: AI has the potential to revolutionize healthcare by improving diagnostic accuracy, reducing medical errors, and enabling new treatments. AI-powered diagnostic tools could be integrated into existing medical practices, and new AI-assisted procedures could be developed to improve patient outcomes.
    
    2. Autonomous Driving: Autonomous vehicles could be the future of transportation, with AI being used to optimize routes, reduce traffic congestion, and prevent accidents. AI-driven self-driving cars could also be used for other tasks,


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

     [

    company

     name

    ].

     I

    've

     been

     working

     in

     this

     field

     since

     [

    insert

     your

     starting

     year

    ],

     and

     I

    've

     always

     been

     passionate

     about

     [

    your

     field

     of

     interest

    ].

     I

     enjoy

     taking

     on

     new

     challenges

     and

     learning

     new

     things

    ,

     and

     I

    'm

     always

     eager

     to

     grow

     and

     develop

     my

     skills

    .

     I

     have

     a

     strong

     work

     ethic

     and

     am

     always

     looking

     for

     ways

     to

     improve

     myself

     and

     my

     abilities

    .

     I

     hope

     that

     my

     experience

     and

     skills

     can

     be

     of

     help

     to

     others

    ,

     and

     I

    'm

     excited

     to

     see

     what

     kind

     of

     impact

     I

     can

     make

     in

     the

     future

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     with

     the

     most

     populous

     of

     the

     country

    's

     

    6

    6

    4

     cities

     and

     the

     third

     largest

     city

     in

     the

     world

    .

     It

     is

     located

     on

     the

     River

     Se

    ine

     and

     is

     home

     to

     many

     famous

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     Lou

    vre

     Museum

    ,

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     The

     city

     is

     also

     known

     for

     its

     diverse

     cultural

     scene

    ,

     including

     art

    ,

     music

    ,

     and

     gastr

    onomy

    ,

     as

     well

     as

     its

     proximity

     to

     neighboring

     countries

    .

     Paris

     is

     an

     iconic

     city

     that

     attracts

     millions

     of

     visitors

     each

     year

    .

     Its

     history

    ,

     from

     Roman

     times

     to

     the

     present

     day

    ,

     has

     played

     a

     significant

     role

     in

     shaping

     France

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     is

     likely

     to

     continue

     to

     evolve

     rapidly

    .

     Some

     possible

     trends

     that

     are

     likely

     to

     drive

     AI

     in

     the

     future

     include

    :
    


    1

    .

     Increased

     collaboration

     between

     AI

     and

     other

     fields

    :

     As

     AI

     becomes

     more

     integrated

     into

     various

     industries

    ,

     there

     will

     be

     increased

     collaboration

     between

     AI

     researchers

     and

     industry

     experts

    .

     This

     can

     lead

     to

     new

     breakthrough

    s

     and

     applications

     for

     AI

    .
    


    2

    .

     Faster

     development

     of

     AI

     technologies

    :

     AI

     is

     advancing

     at

     a

     rapid

     pace

    ,

     and

     this

     will

     likely

     continue

     in

     the

     future

    .

     New

     algorithms

    ,

     models

    ,

     and

     techniques

     will

     be

     developed

     and

     refined

     at

     an

     ever

    -in

    creasing

     pace

    .
    


    3

    .

     AI

     will

     become

     more

     integrated

     with

     the

     human

     mind

    :

     With

     the

     increasing

     understanding

     of

    



```python
llm.shutdown()
```
