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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.52it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.51it/s]


    2026-04-11 03:30:41,639 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-11 03:30:41] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.81it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.57it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.57it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.57it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.57it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.57it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.57it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.57it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.57it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.57it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.81it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.82it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.82it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.82it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.82it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.82it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.82it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.82it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.82it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.34it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.34it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.34it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.34it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.34it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.34it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.34it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.35it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.73it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.73it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.73it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.73it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.73it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.73it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.73it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.92it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.92it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.92it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.92it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.92it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.92it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.92it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.92it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 37.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.91 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.88 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.88 GB):   3%|▎         | 2/58 [00:00<00:03, 15.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.88 GB):   3%|▎         | 2/58 [00:00<00:03, 15.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.88 GB):   3%|▎         | 2/58 [00:00<00:03, 15.10it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.88 GB):   7%|▋         | 4/58 [00:00<00:03, 16.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.83 GB):   7%|▋         | 4/58 [00:00<00:03, 16.10it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.80 GB):   7%|▋         | 4/58 [00:00<00:03, 16.10it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.81 GB):   7%|▋         | 4/58 [00:00<00:03, 16.10it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.81 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.80 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.80 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.88it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.80 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.88it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=118.79 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.88it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.79 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.51it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.79 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.79 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.51it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.79 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.51it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.76 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.51it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.76 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.74 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.03it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=118.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.73 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.62it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.62it/s]Capturing num tokens (num_tokens=960 avail_mem=118.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.62it/s] Capturing num tokens (num_tokens=960 avail_mem=118.71 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.39it/s]Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.39it/s]

    Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.39it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.39it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:00<00:01, 29.14it/s]Capturing num tokens (num_tokens=704 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:00<00:01, 29.14it/s]Capturing num tokens (num_tokens=640 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:00<00:01, 29.14it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.14it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.15it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.15it/s]

    Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.15it/s]Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.15it/s]Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.03it/s]Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.03it/s]Capturing num tokens (num_tokens=384 avail_mem=118.70 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.03it/s]Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.03it/s]Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.19it/s]Capturing num tokens (num_tokens=320 avail_mem=118.69 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.19it/s]

    Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.19it/s]Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.19it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.19it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.60it/s]Capturing num tokens (num_tokens=224 avail_mem=118.68 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.60it/s]Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.60it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.60it/s]

    Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  71%|███████   | 41/58 [00:01<00:00, 29.56it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  71%|███████   | 41/58 [00:01<00:00, 29.56it/s]Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  71%|███████   | 41/58 [00:01<00:00, 29.56it/s]Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  71%|███████   | 41/58 [00:01<00:00, 29.56it/s]Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  71%|███████   | 41/58 [00:01<00:00, 29.56it/s]Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.32it/s]Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.32it/s]Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.32it/s] Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.32it/s]

    Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.32it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 29.92it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 29.92it/s]Capturing num tokens (num_tokens=32 avail_mem=118.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 29.92it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 29.92it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 29.92it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.68it/s]Capturing num tokens (num_tokens=20 avail_mem=118.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.68it/s]

    Capturing num tokens (num_tokens=16 avail_mem=118.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.68it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.68it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.68it/s] Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  98%|█████████▊| 57/58 [00:02<00:00, 29.94it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  98%|█████████▊| 57/58 [00:02<00:00, 29.94it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 28.28it/s]


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
    Generated text:  Daniel and I am a young person from New York City. I am currently a senior student at Union College and I have been an avid reader for a long time. I love to be exposed to new ideas and think about interesting topics that engage my curiosity and creativity. My favorite hobby is cooking up a storm. I make my own food and love trying new recipes. I have also always been fascinated by the idea of entrepreneurship and how to turn ideas into successful companies. My hobbies are also very important to me and I enjoy spending time with my family and making new friends. I have been active in sports and organized sports teams as well. I
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office with a rotating term of two years. During the past year, he served 3 years as the president, 2 years in office as the president, and 2 years as the president before that. What is the president's total term length if he had previously served 1 year as the president before that? To determine the president's total term length, we need to sum the years he served as the president throughout his tenure. According to the problem, he served 3 years as the president, 2 years in office, and 2 years as the president before that. The president had previously served 1 year
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. New York
    D. Berlin
    Answer:
    
    A
    
    According to the conversation, what do we know about this man? He is:
    A. A teacher.
    B. A pilot.
    C. An instructor.
    D. A guide.
    Answer:
    
    A
    
    Which of the following best describes a situation where a government organization is likely to consider a technology decision to be too risky?
    A. A small government agency with limited resources
    B. A small company or individual with limited resources
    C. A large government agency with limited resources
    D. A small company or individual with limited resources
    
    ===============================
    Prompt: The future of AI is
    Generated text:  vast, with a wide range of possibilities and applications. However, the implementation of AI technology has its own challenges, including privacy, security, and ethical concerns. As a result, many people are skeptical about the future of AI. This is because the future of AI is not fully clear, and it is still in the process of development.
    In this post, we will explore some of the key factors that can shape the future of AI, including emerging technologies, regulatory changes, and the potential impact of AI on society. By understanding these factors, we can gain a better understanding of the future of AI and make informed decisions about how to navigate


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your profession or role]. I enjoy [insert a brief description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a hobby or activity you enjoy]. I'm always looking for ways to improve myself and make the world a better place. What's your favorite book or movie? I love [insert a favorite book
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many famous museums, including the Musée d'Orsay and the Musée Rodin. Paris is a bustling city with a rich history and a diverse population, and is a popular tourist destination. It is also home to many international organizations and institutions, including the European Parliament and the United Nations. The city is known for its cuisine, including its famous croissants and its traditional French wine. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, including in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection, risk management, and trading algorithms.
    


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
    Generated text:  [Name], and I am [Age] years old. I am a [Occupation or Profession] with a [Related Skill or Qualification] and [Unique Attribute or Strength]. I am a [Favorite Subject or Hobby] enthusiast, and I enjoy [Favorite Activity or Sport]. I am a [Occupation or Profession] who values [Benefit or Benefit of Pursuing Your Hobby]. I am a [Physical Description or Appearance] person, and I wear my [Clothing or Accessories]. I am [Preferred Language or Write Language] fluent, and I have a [Interest or Passion] in [What My Hobby or Interest is
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its iconic landmarks and rich history. It is the largest city in Europe, and is home to many of the world’s most famous museums, art galleries, and historical sites. Paris is also known for its delicious cuisine, fashion, and vibrant nightlife. The city has a population of around 10 million, making it one of the largest urban areas in the world. Paris is a bustling, diverse, and culturally rich city that is widely regarded as one of the world’s greatest cities. Paris is a city of contrasts, with its modern architecture, historic neighborhoods, and vibrant entertainment options making it a unique and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  undeniably promising and exciting, with many potential trends shaping how we live, work, and interact with technology. Here are some of the key trends that are likely to shape the AI landscape in the coming years:
    
    1. Personalization: One of the biggest trends in AI is personalization, where machines can learn to recognize patterns and preferences in large amounts of data and tailor their responses accordingly. This could lead to more efficient and effective communication, personalized products and services, and even more intelligent assistants and companions.
    
    2. Artificial intelligence in healthcare: With the increasing use of AI in healthcare, we are likely to see more intelligent and accurate medical


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

    Age

    ]

     year

     old

    .

     I

    've

     been

     [

    Job

     Title

    ]

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

     was

     born

     in

     [

    Year

    ]

     and

     have

     lived

     in

     [

    City

    /

    State

    /

    Province

    ]

     since

     then

    .

     I

    'm

     [

    gender

    ]

     and

     have

     [

    height

    ,

     weight

    ,

     or

     other

     physical

     attributes

    ].

     I

    'm

     passionate

     about

     [

    occupation

    ]

     and

     strive

     to

     [

    im

    agination

    /

    amb

    ition

    /s

    uccess

     goal

    ],

     always

     striving

     to

     be

     my

     best

     self

    .

     I

     enjoy

     [

    sports

    /

    activities

    /

    rel

    igion

    /etc

    .

    ].

     And

    ,

     I

     have

     a

     great

     sense

     of

     humor

     and

     love

     to

     make

     people

     laugh

    .

     How

     do

     you

     usually

     spend

     your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Ville

     Bl

    anche

    ,"

     the

     "

    City

     of

     Light

    ,"

     and

     the

     "

    City

     of

     Music

    ."


    This

     statement

     encaps

    ulates

     Paris

    '

     key

     cultural

    ,

     historical

    ,

     and

     artistic

     attributes

    ,

     highlighting

     its

     status

     as

     the

     cultural

     and

     artistic

     capital

     of

     France

    .

     The

     term

     "

    City

     of

     Light

    "

     refers

     to

     its

     historical

     and

     cultural

     importance

     in

     the

     development

     of

     modern

     art

     and

     architecture

    ,

     while

     the

     "

    City

     of

     Music

    "

     reflects

     its

     role

     in

     the

     development

     of

     classical

     music

     and

     opera

    .

     Paris

     is

     a

     vibrant

     and

     cosm

    opolitan

     city

     with

     a

     rich

     history

     and

     a

     diverse

     array

     of

     attractions

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     With

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     significant

     growth

     and

     development

    ,

     and

     there

     are

     several

     possible

     trends

     that

     could

     impact

     the

     field

     in

     the

     coming

     years

    .

     Some

     potential

     areas

     of

     focus

     include

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     different

     industries

    :

     As

     AI

     continues

     to

     advance

    ,

     its

     applications

     in

     various

     sectors

     will

     likely

     become

     more

     widespread

    ,

     including

     healthcare

    ,

     transportation

    ,

     manufacturing

    ,

     and

     more

    .
    


    2

    .

     Greater

     focus

     on

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     prevalent

     in

     our

     daily

     lives

    ,

     there

     will

     likely

     be

     increased

     scrutiny

     of

     its

     use

     and

     impact

     on

     society

    .

     There

     may

     be

     increased

     focus

     on

     ethical

     considerations

     such

     as

     fairness

    ,

     transparency

    ,

     and

     accountability

    .
    


    3

    .

     Development

     of

     more

     sophisticated

     algorithms

    :

     AI

    



```python
llm.shutdown()
```
