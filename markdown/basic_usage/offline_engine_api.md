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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.56it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.55it/s]


    2026-05-01 14:00:16,613 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-01 14:00:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:49,  1.09it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:49,  1.09it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:49,  1.09it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:49,  1.09it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:22,  2.23it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:22,  2.23it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:22,  2.23it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:22,  2.23it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:05<00:22,  2.23it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:11,  4.19it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:11,  4.19it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:11,  4.19it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:11,  4.19it/s]

    Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:11,  4.19it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:11,  4.19it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:05,  7.33it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:05,  7.33it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:05,  7.33it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:05<00:05,  7.33it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:05<00:05,  7.33it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:05<00:05,  7.33it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:03, 11.15it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:03, 11.15it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:03, 11.15it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 18.76it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 18.76it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 18.76it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 18.76it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 18.76it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 18.76it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 18.76it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 24.38it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 24.38it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 24.38it/s]

    Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 24.38it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 24.38it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 24.38it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 24.38it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 29.12it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 29.12it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 29.12it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 29.12it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 29.12it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 29.12it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 29.12it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 33.87it/s]

    Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 33.87it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 38.86it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 38.86it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 38.86it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 38.86it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 38.86it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 38.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.97 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.89 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.89 GB):   3%|▎         | 2/58 [00:00<00:05, 10.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.90 GB):   3%|▎         | 2/58 [00:00<00:05, 10.20it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.90 GB):   3%|▎         | 2/58 [00:00<00:05, 10.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.90 GB):   7%|▋         | 4/58 [00:00<00:04, 11.14it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.89 GB):   7%|▋         | 4/58 [00:00<00:04, 11.14it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.86 GB):   7%|▋         | 4/58 [00:00<00:04, 11.14it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=71.86 GB):  10%|█         | 6/58 [00:00<00:04, 12.24it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.85 GB):  10%|█         | 6/58 [00:00<00:04, 12.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.84 GB):  10%|█         | 6/58 [00:00<00:04, 12.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.84 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.83 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.82 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.80it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.82 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.06it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.81 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.80 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.06it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.79 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.06it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.79 GB):  22%|██▏       | 13/58 [00:00<00:02, 17.21it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.70 GB):  22%|██▏       | 13/58 [00:00<00:02, 17.21it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.68 GB):  22%|██▏       | 13/58 [00:00<00:02, 17.21it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=71.77 GB):  22%|██▏       | 13/58 [00:00<00:02, 17.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.77 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.47it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.76 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.47it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.69 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.47it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.74 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.47it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.74 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.99it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.73 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.99it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.99it/s]

    Capturing num tokens (num_tokens=960 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.99it/s] Capturing num tokens (num_tokens=896 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.99it/s]Capturing num tokens (num_tokens=896 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.25it/s]Capturing num tokens (num_tokens=832 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.25it/s]Capturing num tokens (num_tokens=768 avail_mem=71.68 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.25it/s]Capturing num tokens (num_tokens=704 avail_mem=71.68 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.25it/s]Capturing num tokens (num_tokens=640 avail_mem=71.67 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.25it/s]Capturing num tokens (num_tokens=640 avail_mem=71.67 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.01it/s]Capturing num tokens (num_tokens=576 avail_mem=71.66 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.01it/s]

    Capturing num tokens (num_tokens=512 avail_mem=71.62 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.01it/s]Capturing num tokens (num_tokens=480 avail_mem=71.64 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.01it/s]Capturing num tokens (num_tokens=480 avail_mem=71.64 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.50it/s]Capturing num tokens (num_tokens=448 avail_mem=71.65 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.50it/s]Capturing num tokens (num_tokens=416 avail_mem=71.62 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.50it/s]Capturing num tokens (num_tokens=384 avail_mem=71.64 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.50it/s]Capturing num tokens (num_tokens=352 avail_mem=71.61 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.50it/s]Capturing num tokens (num_tokens=352 avail_mem=71.61 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.44it/s]Capturing num tokens (num_tokens=320 avail_mem=71.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.44it/s]

    Capturing num tokens (num_tokens=288 avail_mem=71.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.44it/s]Capturing num tokens (num_tokens=256 avail_mem=71.61 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.44it/s]Capturing num tokens (num_tokens=240 avail_mem=71.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 31.44it/s]Capturing num tokens (num_tokens=240 avail_mem=71.60 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.82it/s]Capturing num tokens (num_tokens=224 avail_mem=71.60 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.82it/s]Capturing num tokens (num_tokens=208 avail_mem=71.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.82it/s]Capturing num tokens (num_tokens=192 avail_mem=71.57 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.82it/s]Capturing num tokens (num_tokens=176 avail_mem=71.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.82it/s]Capturing num tokens (num_tokens=176 avail_mem=71.56 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.64it/s]Capturing num tokens (num_tokens=160 avail_mem=71.56 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.64it/s]

    Capturing num tokens (num_tokens=144 avail_mem=71.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.64it/s]Capturing num tokens (num_tokens=128 avail_mem=71.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.64it/s]Capturing num tokens (num_tokens=112 avail_mem=71.56 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.64it/s]Capturing num tokens (num_tokens=112 avail_mem=71.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.52it/s]Capturing num tokens (num_tokens=96 avail_mem=71.55 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.52it/s] Capturing num tokens (num_tokens=80 avail_mem=71.53 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.52it/s]Capturing num tokens (num_tokens=64 avail_mem=71.54 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.52it/s]Capturing num tokens (num_tokens=48 avail_mem=71.51 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.52it/s]Capturing num tokens (num_tokens=48 avail_mem=71.51 GB):  86%|████████▌ | 50/58 [00:02<00:00, 33.88it/s]Capturing num tokens (num_tokens=32 avail_mem=71.52 GB):  86%|████████▌ | 50/58 [00:02<00:00, 33.88it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.52 GB):  86%|████████▌ | 50/58 [00:02<00:00, 33.88it/s]Capturing num tokens (num_tokens=24 avail_mem=71.51 GB):  86%|████████▌ | 50/58 [00:02<00:00, 33.88it/s]Capturing num tokens (num_tokens=20 avail_mem=71.50 GB):  86%|████████▌ | 50/58 [00:02<00:00, 33.88it/s]Capturing num tokens (num_tokens=20 avail_mem=71.50 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.24it/s]Capturing num tokens (num_tokens=16 avail_mem=71.50 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.24it/s]Capturing num tokens (num_tokens=12 avail_mem=71.49 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.24it/s]Capturing num tokens (num_tokens=8 avail_mem=71.48 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.24it/s] Capturing num tokens (num_tokens=4 avail_mem=71.48 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.24it/s]Capturing num tokens (num_tokens=4 avail_mem=71.48 GB): 100%|██████████| 58/58 [00:02<00:00, 26.04it/s]


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
    Generated text:  Ruth. 1. Could you provide a list of the 50 most commonly used words in the English language?
    2. I have a group of students and I want to create a coding project. Can you provide me with some basic coding projects that would teach students basic programming skills? 3. What are some of the most difficult programming problems that students have had to solve in the past? 4. I have a client who needs a website. How would you suggest creating the design and code of a website for them? 5. How do I make a website stand out in a crowded market? 6. How can
    ===============================
    Prompt: The president of the United States is
    Generated text:  a powerful figure, but he is only a cabinet member. The American president is a government official who is the head of government and is charged with the responsibility of leading and managing the government. The president is the highest-ranking official in the federal government. The president is elected by the American people and is supposed to be a perfect person and a good leader. The president has to be elected or appointed by the president, the United States Congress, and the states. The president is the commander in chief of the military and must also be able to make key decisions on the day-to-day operations of the government. The president is supposed to be able
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population of Paris is 2 million. 10% of the city’s population is from another country. There is a certain percentage of the population who speak French as their first language. If 2500 people do not speak French, how many people speak French as their first language? First, we need to find out how many people from another country live in Paris. Since 10% of the city's population is from another country, we calculate:
    
    10% of 2 million = 0.10 * 2,000,000 = 200,
    ===============================
    Prompt: The future of AI is
    Generated text:  where the world is now
    
    It’s been a while since I’ve posted here. We’ve had a few small updates on my blog, but lately the feed from my team of students on twitter has been hectic, and I haven’t been able to write for quite a while. Today I want to talk about a topic that has been getting the most discussion on the internet recently: The future of AI. If you’re not familiar with the term, it’s the future of technology, in general, which refers to the future state of the way technology is shaped and utilized. For me, I see the future of AI as being very much


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower, Notre-Dame Cathedral, and the annual Eiffel Tower Parc de la Villette festival. 
    
    This statement encapsulates the key elements of Paris' cultural and historical significance, including its iconic landmarks, architectural landmarks, and annual events. It provides a brief overview of Paris' importance as the capital of France, highlighting its cultural and historical landmarks. The statement is concise and informative, providing a quick overview of Paris' capital city. 
    
    This statement is suitable for use in a variety of contexts, such as news articles, travel guides, or brief descriptions of French cities.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced machine learning capabilities: AI is likely to become more capable of learning from large amounts of data and making more accurate predictions and decisions.
    
    3. Increased focus on ethical considerations: As AI becomes more integrated with human intelligence, there will be increased focus on ethical considerations and the responsible development and use of AI.
    
    4. Development of new AI technologies: AI is likely to continue to develop new technologies and applications,
    


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
    Generated text:  [Name] and I am a/an [occupation or role] with [number] years of experience in [occupation or role]. In my [number] years of experience, I have successfully [describe a successful outcome or accomplishment] and have always been [describe a trait or quality]. I am excited to learn more about your [occupation or role] and I am looking forward to [describe a pleasant surprise or opportunity]! [Name]. [Name]. Your experience as a/an [occupation or role] with [number] years of experience, my name is [Name], is more than just a job. I have successfully achieved [describe
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known as the "City of Love" for its iconic Eiffel Tower and romantic architecture. It is the seat of government, culture, and government for 6 million people and is home to the annual Louvre Museum, the Notre-Dame Cathedral, and the Louvre Palace. Paris is a cultural and historical center with a rich culinary scene, a vibrant nightlife, and many museums, theaters, and parks. It is also home to a number of international organizations and conferences. The city is also known for its fashion and art scene, with Paris Fashion Week and the French Film Festival being held there. It is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to evolve and expand, with the following potential trends:
    
    1. Autonomous robots and drones: The development of autonomous robots and drones is likely to continue, with more advanced technology in areas such as navigation, decision-making, and communication.
    
    2. Speech and language recognition: There will be continued advancements in speech and language recognition technology, with more accurate speech-to-text and text-to-speech conversion.
    
    3. Brain-computer interfaces: With the development of deep learning and neural networks, there will be more sophisticated brain-computer interfaces, allowing for more advanced control of physical devices.
    
    4. Personalized AI: With the increasing amount of


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

    insert

     first

     name

    ],

     and

     I

    'm

     an

     [

    insert

     profession

     or

     role

    ]

     with

     a

     deep

     passion

     for

     [

    insert

     a

     hobby

     or

     skill

    ]

     and

     a

     character

    -driven

     approach

     to

     life

    .

     I

    'm

     always

     looking

     for

     new

     adventures

     and

     ideas

     for

     my

     projects

    ,

     and

     I

    'm

     always

     up

     for

     a

     challenge

    .

     I

     love

     making

     people

     laugh

     and

     I

     thrive

     on

     problem

    -solving

    ,

     so

     I

     strive

     to

     always

     push

     the

     boundaries

     of

     what

    's

     possible

    .

     Whether

     it

    's

     writing

    ,

     cooking

    ,

     painting

    ,

     or

     anything

     else

    ,

     I

    'm

     always

     open

     to

     learning

     and

     discovering

     new

     ways

     to

     express

     myself

    .

     I

    'm

     confident

     in

     my

     abilities

     and

     excited

     about

     the

     opportunities

     that

     lie

     ahead

     for

     me

    .

     If

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     city

     that

     is

     renowned

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     cultural

     richness

    .

     The

     city

     has

     an

     urban

     spraw

    l

     of

     over

     

    6

    0

    0

     kilometers

    ,

     making

     it

     the

     largest

     urban

     area

     in

     the

     European

     Union

    .

     Paris

     has

     a

     population

     of

     over

     

    7

     million

    ,

     and

     it

     is

     one

     of

     the

     world

    's

     leading

     cities

     in

     terms

     of

     education

    ,

     entertainment

    ,

     and

     fashion

    .

     Additionally

    ,

     the

     city

     is

     home

     to

     many

     iconic

     landmarks

     and

     cultural

     institutions

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

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     known

     for

     its

     international

     music

    ,

     theater

    ,

     and

     film

     industries

    ,

     and

     has

     been

     a

     center

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     wide

     range

     of

     potential

     developments

     and

     changes

    .

     Here

     are

     some

     possible

     trends

     that

     could

     emerge

     in

     the

     near

     and

     long

     term

    :
    


    1

    .

     Increased

     automation

    :

     AI

     systems

     will

     continue

     to

     become

     more

     intelligent

     and

     efficient

    ,

     leading

     to

     automation

     of

     many

     tasks

    .

     This

     could

     result

     in

     more

     efficient

     and

     cost

    -effective

     ways

     of

     working

    ,

     such

     as

     in

     manufacturing

     and

     logistics

    .
    


    2

    .

     Enhanced

     privacy

    :

     There

     will

     be

     increased

     emphasis

     on

     privacy

     in

     AI

     systems

    ,

     with

     greater

     scrutiny

     of

     how

     data

     is

     collected

    ,

     stored

    ,

     and

     used

    .

     This

     could

     lead

     to

     new

     privacy

     protections

     and

     regulations

    .
    


    3

    .

     AI

     will

     become

     more

     human

    -like

    :

     AI

     will

     continue

     to

     be

     able

     to

     learn

     and

    



```python
llm.shutdown()
```
