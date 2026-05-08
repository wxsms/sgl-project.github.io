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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.78it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.78it/s]


    2026-05-08 22:10:20,901 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 22:10:20] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:32,  4.78s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:19,  2.61it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:19,  2.61it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:19,  2.61it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:19,  2.61it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:19,  2.61it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:05<00:19,  2.61it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:05<00:19,  2.61it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:05<00:19,  2.61it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:06,  6.19it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:06,  6.19it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:06,  6.19it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:06,  6.19it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:06,  6.19it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:05<00:06,  6.19it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:05<00:06,  6.19it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:05<00:06,  6.19it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:05<00:06,  6.19it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03, 11.35it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03, 11.35it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03, 11.35it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03, 11.35it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03, 11.35it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03, 11.35it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03, 11.35it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03, 11.35it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:03, 11.35it/s]Compiling num tokens (num_tokens=416):  40%|███▉      | 23/58 [00:05<00:03, 11.35it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 18.38it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 18.38it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 18.38it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 18.38it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 18.38it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 18.38it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 18.38it/s]

    Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 18.38it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:05<00:01, 18.38it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 25.41it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 25.41it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 25.41it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 25.41it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 25.41it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 25.41it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 25.41it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 25.41it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 25.41it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 25.41it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 34.35it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 34.35it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 34.35it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 34.35it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 34.35it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 34.35it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 34.35it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 34.35it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 34.35it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 41.68it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 41.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.33 GB):   3%|▎         | 2/58 [00:00<00:04, 13.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.32 GB):   3%|▎         | 2/58 [00:00<00:04, 13.88it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.32 GB):   3%|▎         | 2/58 [00:00<00:04, 13.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.32 GB):   7%|▋         | 4/58 [00:00<00:03, 15.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.30 GB):   7%|▋         | 4/58 [00:00<00:03, 15.97it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.28 GB):   7%|▋         | 4/58 [00:00<00:03, 15.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.29 GB):   7%|▋         | 4/58 [00:00<00:03, 15.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.28 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.78it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=74.28 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.27 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.27 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.26 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.25 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.26 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.26 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.26 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.02it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.25 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.02it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.25 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.21 GB):  31%|███       | 18/58 [00:00<00:01, 29.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.22 GB):  31%|███       | 18/58 [00:00<00:01, 29.71it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.22 GB):  31%|███       | 18/58 [00:00<00:01, 29.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.20 GB):  31%|███       | 18/58 [00:00<00:01, 29.71it/s]Capturing num tokens (num_tokens=960 avail_mem=74.21 GB):  31%|███       | 18/58 [00:00<00:01, 29.71it/s] Capturing num tokens (num_tokens=960 avail_mem=74.21 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=896 avail_mem=74.20 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.65it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.20 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=768 avail_mem=74.19 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=704 avail_mem=74.19 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=704 avail_mem=74.19 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.41it/s]Capturing num tokens (num_tokens=640 avail_mem=74.18 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.41it/s]Capturing num tokens (num_tokens=576 avail_mem=74.17 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.41it/s]Capturing num tokens (num_tokens=512 avail_mem=74.16 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.41it/s]Capturing num tokens (num_tokens=480 avail_mem=74.17 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.41it/s]Capturing num tokens (num_tokens=448 avail_mem=74.16 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.41it/s]Capturing num tokens (num_tokens=448 avail_mem=74.16 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=416 avail_mem=74.16 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.69it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.15 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=352 avail_mem=74.14 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=320 avail_mem=74.14 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=288 avail_mem=74.13 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.69it/s]Capturing num tokens (num_tokens=288 avail_mem=74.13 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.24it/s]Capturing num tokens (num_tokens=256 avail_mem=74.12 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.24it/s]Capturing num tokens (num_tokens=240 avail_mem=74.12 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.24it/s]Capturing num tokens (num_tokens=224 avail_mem=74.11 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.24it/s]Capturing num tokens (num_tokens=208 avail_mem=74.10 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.24it/s]Capturing num tokens (num_tokens=192 avail_mem=74.10 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.24it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.10 GB):  71%|███████   | 41/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=176 avail_mem=74.09 GB):  71%|███████   | 41/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=160 avail_mem=74.09 GB):  71%|███████   | 41/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=144 avail_mem=74.08 GB):  71%|███████   | 41/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  71%|███████   | 41/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  71%|███████   | 41/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.60it/s] Capturing num tokens (num_tokens=80 avail_mem=74.07 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=48 avail_mem=74.06 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=32 avail_mem=74.06 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.60it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=24 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=12 avail_mem=74.04 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=12 avail_mem=74.04 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.93it/s]Capturing num tokens (num_tokens=8 avail_mem=74.04 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.93it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.93it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 34.99it/s]


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
    Generated text:  Olga. I am a senior at the University of Toronto and I am from the city of Toronto. My family moved to Toronto from the Ukraine in 2000. I was born and raised there. I grew up with my family at a time when we were still very poor. We lived in a very small apartment and the neighborhood was very noisy. But despite these difficult circumstances, we had a great time. My parents worked hard to provide for our family and our dreams of attending university and studying at the University of Toronto. They were very proud of me. I also have a brother, and a sister. My brothers are
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. The president has very important job, one of which is to make the Constitution. The Constitution is the first written constitution that the United States has ever had. It is very important because it's made to help the country work together to form a new country. The Constitution was made in 1787. The president and the congress do not agree on anything about the Constitution. However, the president has made it very difficult for the congress to change the Constitution. The president also has the power to veto any bill that is not passed by the congress. The president also has the power to dissolve the congress. The
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. London C. Shanghai D. Rome
    A: Paris
    B: London
    C: Shanghai
    D: Rome
    To determine the capital of France, let's consider the following options:
    
    A. Paris
    B. London
    C. Shanghai
    D. Rome
    
    France is known for its beautiful cities and architectural design. Paris is the capital of France, which is the largest and most populated city in the country. It is also the cultural, economic, and financial hub of France.
    
    Let's analyze each option:
    
    A. Paris - This is the correct capital of France.
    B. London - While
    ===============================
    Prompt: The future of AI is
    Generated text:  bright but we should be cautious
    
    Researchers at the University of Toronto have developed a "fast and cheap" algorithm that can quickly identify possible vulnerabilities in various systems.
    
    It's a "fast and cheap" algorithm because it's able to predict whether a given system is vulnerable to a specific type of attack, such as a SQL injection attack, with very little computational effort, making it potentially very useful in identifying vulnerabilities in a wide variety of systems.
    
    In a paper published this week in the journal BMC Research Notes, a team led by professor Marc Liggett at the University of Toronto, along with researchers from the University of Toronto, the University of Southern


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a brief description of your character or personality]. And what's your background? I have a [insert a brief description of your background or education]. And what's your favorite hobby or activity? I love [insert a brief description of your favorite hobby or activity]. And what's your greatest achievement? I've achieved [insert a brief description of your greatest achievement]. And what's your dream job? I dream of [insert a brief
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a bustling city with a rich cultural heritage and is a major tourist destination. It is the capital of France and the largest city in the European Union. It is also the birthplace of the French Revolution and is home to many famous French artists, writers, and musicians. The city is known for its cuisine, including its famous Parisian dishes such as croissants, escargot, and escargot. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from feedback and improve their performance over time.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well
    


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
    Generated text:  [Name] and I'm a [role in the story]. I'm always looking for a new challenge and always eager to learn more about the world around me. I enjoy trying new foods, trying new places, and exploring the world's cultural traditions. I'm a natural problem solver and always have a good solution to any problem I encounter. I'm an insatiable reader and love to dive into new genres and genres to broaden my horizons. In short, I'm an open-minded, curious, and adventurous person who is always on the lookout for new experiences and adventures. I'm excited to get started with this project and learn
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is a historic city famous for its iconic Eiffel Tower, Le Marais, and historic landmarks like Notre-Dame Cathedral. It's the world's most populous city, home to 22 million people, making it the largest city by population in the world according to the United Nations. Paris's cultural and political center, it's also known for its rich history, world-renowned art, and vibrant nightlife. Paris is a cultural melting pot, home to countless museums, theaters, and literary and artistic organizations, making it a vibrant hub of global culture. With its stunning architecture, vibrant life, and world-renowned
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, and it is likely to continue to evolve in exciting ways. Here are some possible trends in the AI field in the next few years:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare, from identifying disease patterns in medical imaging to predicting patient outcomes. In the future, we may see even more personalized treatment approaches, such as using AI to identify the best combination of treatments for a patient with a specific condition.
    
    2. AI in manufacturing: AI is being used in manufacturing to optimize production processes, improve quality control, and even predict maintenance needs. In the future, we may see even more intelligent manufacturing


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

     am

     a

     [

    Your

     Profession

    ]

     at

     [

    Your

     Company

    ].

     I

     am

     a

     [

    Your

     Age

    ],

     [

    Your

     Height

    ],

     [

    Your

     Weight

    ]

     inches

     tall

    ,

     and

     [

    Your

     Gender

    ]

     with

     [

    Your

     Hair

     Color

    ]

     and

     [

    Your

     Eyes

     Color

    ].

     I

     have

     a

     [

    Your

     Job

     Title

    ]

     at

     [

    Your

     Company

    ],

     and

     I

     have

     been

     working

     here

     for

     [

    Your

     Duration

     of

     Time

    ]

     years

    .

     I

     strive

     to

     be

     a

     [

    Your

     Goal

    /

    Value

    ]

     employee

     at

     [

    Your

     Company

    ].

     I

     am

     passionate

     about

     [

    Your

     Passion

    ],

     and

     I

     am

     always

     looking

     to

     learn

     and

     grow

    .

     I

     enjoy

     [

    Your

     Hobby

    /

    Inter

    ests

    /

    Att

    ire

    /

     etc

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     that

     serves

     as

     the

     political

    ,

     economic

    ,

     and

     cultural

     center

     of

     the

     country

    .

     It

     is

     known

     for

     its

     iconic

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

     Dame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    ,

     as

     well

     as

     its

     romantic

     and

     historic

     atmosphere

    .

     Paris

     is

     a

     bustling

     city

     with

     a

     diverse

     population

     and

     a

     rich

     history

    ,

     and

     it

     is

     home

     to

     many

     famous

     French

     artists

    ,

     writers

    ,

     and

     musicians

    .

     The

     city

     is

     also

     known

     for

     its

     world

    -class

     museums

     and

     theaters

    ,

     and

     it

     is

     a

     popular

     tourist

     destination

     for

     visitors

     from

     all

     over

     the

     world

    .

     As

     the

     oldest

     capital

     city

     in

     the

     world

    ,

     Paris

     has

     been

     inhabited

     for

     over

     

    5

    0

    0

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     with

     a

     range

     of

     potential

     trends

     and

     developments

     that

     could

     shape

     the

     technology

     and

     applications

     of

     AI

     in

     the

     coming

     years

    .

     Some

     potential

     areas

     of

     future

     research

     and

     development

     include

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

    :

     As

     AI

     systems

     become

     more

     complex

     and

     powerful

    ,

     there

     will

     be

     an

     increasing

     focus

     on

     developing

     more

     sophisticated

     and

     efficient

     machine

     learning

     algorithms

     that

     can

     handle

     more

     complex

     tasks

     and

     provide

     more

     accurate

     predictions

    .
    


    2

    .

     Integration

     of

     AI

     into

     healthcare

    :

     AI

     has

     the

     potential

     to

     revolution

    ize

     healthcare

     by

     improving

     diagnosis

     accuracy

    ,

     developing

     new

     treatments

    ,

     and

     personal

    izing

     patient

     care

    .

     However

    ,

     there

     are

     also

     ethical

     concerns

     about

     the

     use

     of

     AI

     in

     medical

     procedures

     and

     the

     potential

    



```python
llm.shutdown()
```
