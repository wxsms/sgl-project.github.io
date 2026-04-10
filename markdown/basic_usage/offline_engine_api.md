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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.85it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.85it/s]


    2026-04-10 13:05:24,038 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 13:05:24] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.36it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:02<00:07,  6.20it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.56it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.56it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.56it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.56it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.56it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.56it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.56it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.56it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.68it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.68it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.68it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.68it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.68it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.68it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.68it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.68it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]

    Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.35it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.43it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.43it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.43it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.43it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.43it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.43it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.43it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 35.38it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 35.38it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 35.38it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 35.38it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 35.38it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 35.38it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 35.38it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 39.52it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 39.52it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 39.52it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 39.52it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 39.52it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 39.52it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 39.52it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 39.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   2%|▏         | 1/58 [00:00<00:07,  7.44it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   2%|▏         | 1/58 [00:00<00:07,  7.44it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.64 GB):   2%|▏         | 1/58 [00:00<00:07,  7.44it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=131.64 GB):   5%|▌         | 3/58 [00:00<00:03, 13.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.63 GB):   5%|▌         | 3/58 [00:00<00:03, 13.91it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   5%|▌         | 3/58 [00:00<00:03, 13.91it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.63 GB):   5%|▌         | 3/58 [00:00<00:03, 13.91it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.63 GB):  10%|█         | 6/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.63 GB):  10%|█         | 6/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):  10%|█         | 6/58 [00:00<00:02, 19.19it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.62 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.56it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.56it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.56it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  21%|██        | 12/58 [00:00<00:03, 12.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=131.61 GB):  21%|██        | 12/58 [00:00<00:03, 12.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.61 GB):  21%|██        | 12/58 [00:00<00:03, 12.74it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=131.61 GB):  24%|██▍       | 14/58 [00:00<00:03, 13.22it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.61 GB):  24%|██▍       | 14/58 [00:00<00:03, 13.22it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.60 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  31%|███       | 18/58 [00:01<00:02, 18.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  31%|███       | 18/58 [00:01<00:02, 18.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=131.59 GB):  31%|███       | 18/58 [00:01<00:02, 18.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  31%|███       | 18/58 [00:01<00:02, 18.54it/s]Capturing num tokens (num_tokens=960 avail_mem=131.58 GB):  31%|███       | 18/58 [00:01<00:02, 18.54it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=131.58 GB):  31%|███       | 18/58 [00:01<00:02, 18.54it/s]Capturing num tokens (num_tokens=896 avail_mem=131.58 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.18it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.18it/s]Capturing num tokens (num_tokens=768 avail_mem=131.57 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.18it/s]Capturing num tokens (num_tokens=704 avail_mem=131.57 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.18it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.18it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.18it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.34it/s]Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.34it/s]Capturing num tokens (num_tokens=480 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.34it/s]Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.34it/s]

    Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.34it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.34it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.67it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.67it/s]Capturing num tokens (num_tokens=320 avail_mem=131.56 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.67it/s]Capturing num tokens (num_tokens=288 avail_mem=131.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.67it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.67it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.67it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=224 avail_mem=131.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=208 avail_mem=131.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.78it/s]

    Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=160 avail_mem=131.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.78it/s]Capturing num tokens (num_tokens=160 avail_mem=131.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=128 avail_mem=131.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.24it/s] Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.24it/s]Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=64 avail_mem=131.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=48 avail_mem=131.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.60it/s]

    Capturing num tokens (num_tokens=28 avail_mem=131.50 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.05it/s]Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.05it/s]Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.05it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.05it/s]Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.05it/s] Capturing num tokens (num_tokens=4 avail_mem=131.48 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.05it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 29.43it/s]


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
    Generated text:  Xander and I'm a social media strategist. In addition to my work, I like to explore new areas of learning and growth on a regular basis. I am an avid reader and enjoy reading books and reading blogs by writers such as Kristin Hannah. I am also interested in learning about the latest trends in technology and I am always on the lookout for new tools and platforms that will help me reach new audiences.
    As you can see, my experience spans a wide range of topics, from social media management to technology trends, to even the latest trends in marketing. I have a strong interest in learning and staying up-to-date on these topics
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He or she helps run the country by making decisions. The president also helps the country protect the country from other countries that want to attack the country. The president also helps the country learn new things from other countries. The president is usually from the same family as the president. They're very close to each other. How does a president become president? A president needs to get a very good education. They must go to school for 10 years. They also have to work hard to get a good job. The president has to be very good at thinking. They also have to be very good
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located on the left bank of the Seine river in the center of France, on the Île de France, about north of the Atlantic Ocean. Paris is the capital of France, and the second largest city of France in terms of population.
    Where is Paris located?
    The capital of France is located in the center of France. Paris is a city and a country in the Île de France in the center of the country. It is located about north of the Atlantic Ocean, on the right bank of the Seine river, on the left bank of the Seine river. Paris is also the largest city of France
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, and in the future, more and more people will be using AI. In the past few years, the development of AI has gone from being a brave new world to being a reality, and even a step too far in the wrong direction. However, the future of AI is bright, and the real question is whether we will be able to protect it from those who will be harmed by it.
    
    If AI is seen as a tool that people should use for the good of humanity, but it can be used to harm humanity in the wrong way, then we will be stuck in the wrong direction and the future of AI will be bleak


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


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the world by population. The city is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a popular tourist destination, attracting millions of visitors each year. Paris is a cultural and intellectual center of the world and is considered one of the most beautiful cities in the world. The city is also home to many important institutions such as the French Academy of Sciences and the French National Library
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into the production process, from manufacturing to healthcare. This will lead to the automation of repetitive tasks, freeing up workers to focus on more complex and creative work.
    
    2. AI ethics: As AI becomes more advanced, there will be a need for ethical guidelines and regulations to ensure that AI is used in a responsible and beneficial way. This will involve developing AI that is transparent, accountable, and respects human rights.
    
    
    


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
    Generated text:  [Name] and I'm a [Job Title] with over [number] years of experience in [Industry]. I'm constantly learning and improving, and I'm proud to have been a part of [Company Name] for [number] years. I have a natural ability to work with people and I love getting to know people on a deeper level. I'm a great communicator and always strive to be a positive influence in our workplace. I believe in team work and always want to make sure everyone feels valued and included. My favorite hobby is hiking and I'm always up for a new adventure. Thank you for taking the time to learn
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Its name derives from the Latin word "parvis" which means "gate" or "entrance". It is the largest and most populous city in Europe. Paris is known for its Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and countless museums. It is also the seat of the French government, capital city, and is the official language of France. Paris is a cosmopolitan city with a diverse population and is often referred to as the "Parisite". The city has been visited by countless famous and famous people, including both male and female celebrities, politicians, and athletes. Paris has a rich cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are some possible trends that are likely to shape the technology's direction:
    
    1. Increased AI ethics and privacy concerns: As more AI systems are developed, there will likely be a rise in ethical concerns and privacy issues. This will require that AI systems are designed and regulated with more advanced ethical standards.
    
    2. Integration of AI with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence. This could lead to the development of new AI algorithms that can learn from human behavior and adapt accordingly.
    
    3. AI-driven automation: The potential for AI to automate routine tasks will likely lead to


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

    ]

     and

     I

    'm

     a

     [

    field

     or

     profession

    ]

     with

     [

    number

     or

     degree

    ]

     in

     [

    major

     or

     specialty

    ].

     I

     have

     always

     been

     passionate

     about

     [

    subject

     or

     area

     of

     interest

    ].

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     learn

     and

     grow

    .

     I

    'm

     a

     [

    positive

     or

     cautious

    ]

     person

     with

     [

    details

     about

     your

     personality

    ],

     and

     I

     enjoy

     [

    reason

     for

     your

     personality

    ,

     such

     as

     hard

     work

    ,

     kindness

    ,

     or

     uniqueness

    ].

     I

    'm

     eager

     to

     make

     a

     difference

     and

     bring

     about

     positive

     change

     in

     the

     world

     through

     [

    your

     specific

     goal

     or

     purpose

    ].

     So

    ,

     if

     you

    'd

     like

     to

     meet

     me

    ,

     I

    'd

     love

     to

     know

     more

     about

     you

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

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

    .
    


    What

     is

     the

     capital

     of

     France

    ?

     Paris

     is

     the

     capital

     of

     France

    .

     It

    's

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

    .

     
    


    Therefore

    ,

     the

     answer

     is

    :

     Paris

    .

     
    


    A

     capital

     city

     is

     the

     central

     government

     of

     a

     country

    ,

     and

     Paris

     is

     indeed

     the

     capital

     of

     France

    .

     It

    's

     the

     seat

     of

     the

     French

     government

     and

     is

     known

     for

     its

     elegant

     architecture

    ,

     vibrant

     culture

    ,

     and

     rich

     history

    .

     Despite

     its

     size

    ,

     Paris

     has

     a

     small

     population

     and

     is

     a

     major

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     every

     year

    .

     The

     city

     is

     home

     to

     many

     important

     institutions

     and

     landmarks

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     quite

     promising

    ,

     and

     there

     are

     several

     potential

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

    .

     Here

     are

     some

     of

     the

     most

     significant

     trends

     that

     could

     be

     expected

     in

     AI

     in

     the

     next

     few

     years

    :
    


    1

    .

     Increased

     reliance

     on

     machine

     learning

     and

     deep

     learning

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     will

     see

     an

     increasing

     reliance

     on

     machine

     learning

     and

     deep

     learning

     techniques

    .

     These

     techniques

     will

     enable

     AI

     to

     learn

     from

     large

     amounts

     of

     data

     and

     make

     predictions

     and

     decisions

     with

     greater

     accuracy

     and

     efficiency

     than

     ever

     before

    .
    


    2

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     we

     become

     more

     aware

     of

     the

     potential

     risks

     of

     AI

    ,

     there

     will

     be

     a

     greater

     push

    



```python
llm.shutdown()
```
