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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]


    2026-05-12 03:44:09,857 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 03:44:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.43it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.43it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.43it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.43it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.43it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.43it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.43it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:04,  8.43it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:04<00:02, 13.02it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:04<00:02, 13.02it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:04<00:02, 13.02it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:04<00:02, 13.02it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:04<00:02, 13.02it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:04<00:02, 13.02it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:04<00:02, 13.02it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:04<00:02, 13.02it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:04<00:02, 13.02it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:04<00:01, 19.45it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:04<00:01, 19.45it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:04<00:01, 19.45it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:04<00:01, 19.45it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:04<00:01, 19.45it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:04<00:01, 19.45it/s]

    Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:04<00:01, 19.45it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:04<00:01, 19.45it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:04<00:01, 19.45it/s]Compiling num tokens (num_tokens=160):  59%|█████▊    | 34/58 [00:04<00:01, 19.45it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:04<00:00, 27.89it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:04<00:00, 27.89it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:04<00:00, 27.89it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:04<00:00, 27.89it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 27.89it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 27.89it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 27.89it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:05<00:00, 27.89it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:05<00:00, 27.89it/s]Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:05<00:00, 27.89it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 36.78it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 36.78it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 36.78it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 36.78it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 36.78it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 36.78it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 36.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 14.17it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 14.17it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 14.17it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.63it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.63it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.63it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):  10%|█         | 6/58 [00:00<00:03, 15.14it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  10%|█         | 6/58 [00:00<00:03, 15.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  10%|█         | 6/58 [00:00<00:03, 15.14it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.31it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.31it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.87it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.30it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.30it/s]

    Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.76it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.76it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.76it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.76it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 31.76it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:01<00:00, 34.15it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:01<00:00, 34.15it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:01<00:00, 34.15it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:01<00:00, 34.15it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:01<00:00, 34.15it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 36.01it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 36.01it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 36.01it/s]

    Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 36.01it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 36.01it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.51it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.51it/s]Capturing num tokens (num_tokens=192 avail_mem=76.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.51it/s]Capturing num tokens (num_tokens=176 avail_mem=76.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.51it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.51it/s]Capturing num tokens (num_tokens=160 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.02it/s]Capturing num tokens (num_tokens=144 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.02it/s]Capturing num tokens (num_tokens=128 avail_mem=76.63 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.02it/s]Capturing num tokens (num_tokens=112 avail_mem=76.14 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.02it/s]Capturing num tokens (num_tokens=96 avail_mem=76.14 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.02it/s] Capturing num tokens (num_tokens=96 avail_mem=76.14 GB):  81%|████████  | 47/58 [00:01<00:00, 28.27it/s]Capturing num tokens (num_tokens=80 avail_mem=76.04 GB):  81%|████████  | 47/58 [00:01<00:00, 28.27it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 28.27it/s]

    Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 28.27it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 28.27it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.54it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.54it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.54it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.54it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.54it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  95%|█████████▍| 55/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 32.51it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:02<00:00, 32.51it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:02<00:00, 28.46it/s]


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
    Generated text:  Timmy, and I live in the city of Codeville, where the roads are numbered from 1 to 10. I have a treasure map, but I'm having trouble reading it. Can you help me understand the directions in the map? Here's the map:
    
    ```
      10
      |  |
      7
      |  |
      4
      |  |
      9
      |  |
      5
      |  |
      6
      |  |
      3
      |  |
      8
    ```
    
    The first row represents the north-s
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political leader who represents the people of the United States in the world. One of the most significant functions of the president is to serve as the head of the executive branch of the United States government. The president also has the power to appoint other high-ranking government officials.
    The president is responsible for crafting and implementing policies that affect the United States, which includes national security, foreign policy, and economic policies. The president also has the power to issue executive orders and sign executive actions, which are actions taken by the president without the need to seek a vote in Congress.
    In addition to these official roles, the president also has the authority to med
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is located in the centre of the Mediterranean Sea. Since it is a large city, it is an important center of business and finance and is the largest city of France. It is the most important French city and is the capital of the north of France.
    Paris is a very large city with a population of over 2.5 million people. It is a vibrant and multicultural city, with people from over 80 nationalities living in Paris. Paris is also home to many of the world's major museums and cultural institutions. It also has a long history of being a center of art, science, and architecture.
    Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  upon us. The field has advanced so rapidly that it’s now increasingly important to consider how we design and build the AI systems that will power the future. The human race has been working hard for centuries to develop the technologies to change the world and revolutionize the way we live and work. And the future is looking increasingly uncertain, with new technologies constantly emerging and causing waves of change in the process.
    AI is the future of innovation and is poised to bring about sweeping changes to how we live and work. It’s already changed the way we interact with the world around us, and the technology is advancing rapidly. AI has the potential to change


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm passionate about [job title] and I love [job title] because [reason why you love your job]. I'm always looking for ways to [something]. I'm always eager to learn and grow, and I'm always willing to help others. I'm a [job title] who is always ready to [something]. I'm a [job title] who is always [something]. I'm a [job title] who is always [something]. I'm a [job title] who is always [something]. I'm a [job title
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to ancient times. Paris is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. The city is also known for its food scene, with many famous restaurants and cafes serving traditional French cuisine. Paris is a popular tourist destination, with millions of visitors annually. It is also home to many important historical sites, including the Palace of Versailles and the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is likely to become more prevalent in various industries, including manufacturing, transportation, and healthcare. Automation will likely lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be a growing concern about privacy and security. AI systems will need to be designed and implemented with the same level of care and attention to privacy as traditional systems
    


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
    Generated text:  [name], and I'm a [occupation]. I'm a... [describe your occupation, such as a scientist, a teacher, a doctor, etc.] who specialize in... [specify the area of expertise or expertise that you have]. My main job is to... [state the most important aspect of your job here]. I work in [city or state], and I'm currently [job title or current job] at [company name]. My favorite [occupation] is... [describe your favorite aspect or hobby here, such as learning a new language, cooking a favorite meal, etc.]. I enjoy [reason why I like
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement encapsulates the essential information about Paris: its capital city, the French capital, and its role as the country's political and cultural center. It's concise and provides a clear and accurate factual statement about the capital city's importance in French society and politics. 
    
    To break it down:
    1. "Paris" is the key noun and subject of the statement.
    2. It refers to the capital city of France.
    3. The "French capital" is implied in the statement.
    4. It serves as a political and cultural center in France.
    5. It is a major city in France with a rich history and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and exciting, with many possible trends shaping its development and impact. Here are some of the most notable trends:
    
    1. Increased AI ethics and responsibility: With the growing awareness of AI's potential risks and negative consequences, there is a growing emphasis on ensuring that AI is used in ways that are ethical and responsible. This includes developing guidelines for AI development, decision-making, and deployment, as well as promoting transparency and accountability in the use of AI.
    
    2. AI in healthcare: AI is already being used to improve healthcare outcomes, from personalized medicine to diagnostic and treatment recommendations. As the technology advances, we may see even more sophisticated use of


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

    Name],

     and

     I

    'm

     a

     [

    job

     title

    ]

     at

     [

    company

     name

    ].

     I

    'm

     passionate

     about

     [

    reason

     for

     job

     title

    ],

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     [

    reason

     for

     job

     title

    ]

     and

     help

     my

     colleagues

     achieve

     their

     goals

    .

     I

     have

     a

     deep

     love

     for

     [

    personal

     interest

     or

     hobby

    ]

     and

     I

    'm

     always

     looking

     to

     learn

     and

     grow

    .

     Thank

     you

     for

     having

     me

    ,

     [

    Name

    ].

     I

    'm

     eager

     to

     learn

     more

     about

     your

     company

     and

     the

     role

     you

    're

     filling

    ,

     and

     I

     look

     forward

     to

     hearing

     from

     you

    .

     [

    Name

    ]

     [

    Title

    ]

     at

     [

    Company

     Name

    ]

     I

    'm

     excited

     to

     meet

     you

     and

     learn

     more

     about

     you

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Here

    's

     the

     same

     information

    ,

     formatted

     as

     bullet

     points

    :
    


    -

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    ,

     is

     the

     capital

     of

     France

    


    -

     It

     is

     the

     largest

     city

     in

     both

     the

     European

     Union

     and

     the

     United

     Nations

    


    -

     Known

     for

     its

     historical

     landmarks

    ,

     fashion

     industry

    ,

     and

     world

    -ren

    owned

     art

     and

     architecture

    ,

     Paris

     is

     a

     major

     economic

     center

     and

     a

     cultural

     hub

    


    -

     Founded

     in

     the

     

    1

    2

    th

     century

    ,

     Paris

     has

     a

     rich

     history

     dating

     back

     over

     

    1

    ,

    0

    0

    0

     years

    


    -

     Paris

     is

     the

     second

     largest

     city

     by

     population

     in

     the

     European

     Union

     and

     the

     

    1

    5

    th

     most

     populous

     city

     in

     the

     world

    


    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

     significant

     advancements

     in

     several

     areas

    ,

     including

    :
    


    1

    .

     Increased

     intelligence

     and

     learning

     capabilities

    :

     As

     AI

     technology

     advances

    ,

     it

     will

     be

     able

     to

     learn

     and

     improve

     on

     tasks

     more

     quickly

     and

     accurately

     than

     human

     experts

    .

     This

     will

     enable

     AI

     systems

     to

     become

     more

     versatile

     and

     able

     to

     solve

     complex

     problems

     in

     new

     and

     unexpected

     ways

    .
    


    2

    .

     Integration

     of

     AI

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     be

     integrated

     with

     other

     technologies

    ,

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     and

     computer

     vision

    ,

     to

     create

     new

     and

     more

     powerful

     tools

     and

     systems

    .
    


    3

    .

     Expansion

     of

     AI

     applications

    :

     As

     AI

     technology

     becomes

     more

     advanced

    ,

     it

     will

     be

     able

     to

     perform

     a

     wider

     range

    



```python
llm.shutdown()
```
