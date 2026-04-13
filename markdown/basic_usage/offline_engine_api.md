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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.96it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.96it/s]


    2026-04-13 03:22:33,719 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 03:22:33] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.74it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.74it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:30,  1.74it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:30,  1.74it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:30,  1.74it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:30,  1.74it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:30,  1.74it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.45it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.45it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.45it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.45it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.45it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.45it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.45it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.45it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.56it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.20it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.20it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.20it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.20it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.20it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.20it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.20it/s]

    Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:01, 24.20it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.80it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.80it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.80it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.80it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.80it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.80it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.80it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.80it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 37.12it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 37.12it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 37.12it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 37.12it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 37.12it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 37.12it/s]

    Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 37.12it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 37.12it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 37.12it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 44.92it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 44.92it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 44.92it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 44.92it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 44.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.96it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 16.96it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 16.96it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:05,  9.58it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:05,  9.58it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:05,  9.58it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):  10%|█         | 6/58 [00:00<00:07,  6.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  10%|█         | 6/58 [00:00<00:07,  6.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  10%|█         | 6/58 [00:00<00:07,  6.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.48it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.48it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.48it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.68it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.68it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.24it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:01<00:02, 16.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:01<00:02, 16.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:01<00:02, 16.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:01<00:02, 16.81it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:01<00:02, 16.81it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  31%|███       | 18/58 [00:01<00:02, 16.81it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.26it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.26it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.26it/s]

    Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.26it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.26it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.26it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.58it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.58it/s]

    Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.58it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.58it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.58it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.58it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.88it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.88it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 37.88it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:02<00:00, 37.88it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:02<00:00, 37.88it/s]

    Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:02<00:00, 37.88it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  76%|███████▌  | 44/58 [00:02<00:00, 34.90it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:02<00:00, 34.90it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:02<00:00, 34.90it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:02<00:00, 34.90it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:02<00:00, 34.90it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:02<00:00, 34.90it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:02<00:00, 37.77it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:02<00:00, 37.77it/s]Capturing num tokens (num_tokens=32 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:02<00:00, 37.77it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:02<00:00, 37.77it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:02<00:00, 37.77it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:02<00:00, 37.77it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:02<00:00, 40.46it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:02<00:00, 40.46it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:02<00:00, 40.46it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:02<00:00, 40.46it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:02<00:00, 40.46it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:02<00:00, 23.85it/s]


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
    Generated text:  Alex. I'm an IT professional. I have a keen interest in computer science and computing. My main area of expertise is web development.
    Currently, I'm working as a software developer in the cloud, so I'll be able to talk about web development trends.
    Can you provide some general tips on how to improve my skills in web development? Sure, here are some general tips on how to improve your skills in web development:
    
    1. Read and learn: Web development is a fast-paced field with a lot of new technologies and frameworks. Reading and learning about these technologies is essential to stay up-to-date with the latest trends and to understand
    ===============================
    Prompt: The president of the United States is
    Generated text:  30 years older than the president of Brazil, and the president of Brazil is 2 times older than the president of Mexico. If the president of the United States is currently 46 years old, how old would the president of Mexico be in 5 years?
    To determine the current age of the president of Mexico, we start by defining the ages of the presidents of the United States and Brazil based on the given relationships.
    
    Let:
    - \( U \) be the current age of the president of the United States.
    - \( B \) be the current age of the president of Brazil.
    - \( M \) be the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a big city in the southwest of France, where the Mediterranean Sea meets the English Channel. It is one of the most important cities in Europe and one of the most important cities in the world.
    
    What is the main point of the passage?
    
    A) The capital of France
    B) The capital of France and its history
    C) The capital of France and the people who live there
    D) The capital of France and its geographical location
    D) The capital of France and its geographical location
    
    The passage clearly states that the capital of France is Paris, and it provides information about its location, including its position relative
    ===============================
    Prompt: The future of AI is
    Generated text:  in demand, and companies across the globe are making AI a high priority for their business strategies and growth plans. AI is used in almost every aspect of business, from marketing, customer service, product design, and finance to everything from supply chain management to cybersecurity. However, the need for AI to provide more value than just a technical tool has only grown in recent years.
    In the past, AI was used to automate repetitive tasks and improve efficiency, but now it is being used to create artificial intelligence that can understand human emotions, learn from data, and make decisions that have a profound impact on the human world. AI has the potential to dramatically


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your profession or role]. I enjoy [insert a short description of your hobbies or interests]. What brings you to [company name] and what do you hope to achieve with your work? I'm excited to learn more about your company and how it can benefit me. What do you hope to achieve with your work? I'm looking forward to meeting you and learning more about your company. What do you hope to achieve
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" and "La Ville de la Rose". It is the largest city in France and the third-largest city in the world by population. The city is located on the Seine River and is home to many of France's most famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its rich cultural heritage, including the Opéra Garnier, the Musée d'Orsay, and the Musée Rodin. The city is a major transportation hub and is home to many of France's major cities, including Lyon
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as
    


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
    Generated text:  [name], and I'm a [job title] at [company name]. My expertise lies in [specific skill or area], and I've been working here for [number] years. I'm a [status] employee. What's your name and what brings you here today? [Your Name]... (This can be the same as your full name or initials only.) [Your Name]... (Again, this can be the same as your full name or initials only.) It's a pleasure to meet you. I look forward to learning more about you and my role at [company name]. What's the next project that you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, a historic city known for its iconic landmarks and rich cultural heritage. Paris is a global financial center and a popular tourist destination, known for its rich history, vibrant arts scene, and diverse culinary offerings. The city is home to the Louvre Museum, Eiffel Tower, and Notre-Dame Cathedral, among other landmarks. Paris is a thriving metropolis with a vibrant social and cultural scene, and has been a major player in French politics and society for centuries. The city is also home to the French Parliament, the headquarters of France's government. The French capital is considered to be a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and unpredictable. Here are some potential trends that could shape the technology:
    
    1. Increased automation: As AI systems become more advanced and powerful, we can expect to see more automation in various industries. This could lead to the development of more efficient and automated processes, which could potentially improve productivity and reduce human errors.
    
    2. Integration with natural language processing: As AI systems become more sophisticated, they are likely to be integrated with natural language processing technology. This could result in more intelligent language understanding and speech recognition systems that can better understand and respond to human language.
    
    3. Greater personalization: AI systems are becoming increasingly personalized, and this trend


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

     __

    ________

    _

     and

     I

    'm

     a

    /an

     ______

    __

    _.

     I

    'm

     a

    /an

     ______

    __

    _.

     I

    'm

     passionate

     about

     __

    ________

    _.

     I

    'm

     an

     artist

     who

     uses

     ______

    _

     to

     convey

     my

     message

    .

     I

     love

     to

     ______

    _

     and

     ______

    _.

     I

    'm

     a

    /an

     __

    ________

    _

     who

     has

     been

     training

     for

     _____

     years

    .

     I

    'm

     always

     ready

     to

     learn

     and

     grow

    ,

     and

     I

     believe

     that

     being

     a

    /an

     __

    ________

    _

     means

     that

     I

     can

     make

     a

     big

     difference

     in

     the

     world

    .

     I

    'm

     excited

     to

     be

     here

     today

     and

     to

     work

     with

     you

     all

    .
    


    ---
    


    **

    Note

    :**

     This

     self

    -int

    roduction

     is

     written

     in

     a

     neutral

     and

     casual

     style

    ,

     appropriate

     for

     a

     fictional

     character

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     with

     the

     iconic

     E

    iff

    el

     Tower

    .

     
    


    France

    's

     capital

    ,

     Paris

    ,

     is

     the

     largest

     city

     in

     the

     European

     Union

    ,

     known

     for

     its

     rich

     history

     and

     vibrant

     culture

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     which

     was

     originally

     designed

     for

     military

     purposes

    .

     
    


    Paris

     also

     has

     a

     rich

     and

     diverse

     culture

    ,

     with

     numerous

     museums

    ,

     theaters

    ,

     and

     neighborhoods

     that

     offer

     a

     mix

     of

     art

    ,

     food

    ,

     and

     entertainment

    .

     The

     city

     is

     home

     to

     numerous

     famous

     landmarks

    ,

     including

     the

     Lou

    vre

    ,

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     
    


    The

     French

     capital

     is

     a

     bustling

     met

    ropolis

     with

     a

     population

     of

     over

     

    2

     million

     people

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     exciting

     and

     may

     include

     a

     wide

     range

     of

     possibilities

     and

     potential

     developments

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Enhanced

     human

     AI

    :

     AI

     is

     likely

     to

     become

     even

     more

     human

    -like

    ,

     with

     a

     greater

     ability

     to

     understand

     and

     respond

     to

     human

     emotions

     and

     social

     cues

    .

     This

     could

     lead

     to

     more

     empath

    etic

     and

     effective

     AI

     systems

     that

     can

     better

     understand

     and

     interact

     with

     humans

    .
    


    2

    .

     Virtual

     assistants

    :

     AI

    -powered

     virtual

     assistants

     may

     become

     even

     more

     advanced

     and

     complex

    ,

     with

     the

     ability

     to

     understand

     and

     respond

     to

     a

     wider

     range

     of

     human

     needs

     and

     preferences

    .
    


    3

    .

     Autonomous

     vehicles

    :

     The

     development

     of

     autonomous

     vehicles

     could

     revolution

    ize

     transportation

    ,

     reducing

     accidents

    ,

     reducing

     carbon

    



```python
llm.shutdown()
```
