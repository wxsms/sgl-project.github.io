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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.58it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.57it/s]


    2026-04-28 22:11:48,850 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 22:11:48] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.89it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.89it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.72it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.72it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.72it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.72it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.72it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.72it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.72it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.72it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.72it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.72it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.71it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.71it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.71it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.71it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.71it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.71it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.71it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.71it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.71it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.71it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.43it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.43it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.43it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.43it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.43it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.43it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.43it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.43it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.43it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.43it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 37.95it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 37.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.21 GB):   2%|▏         | 1/58 [00:00<00:07,  7.17it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.18 GB):   2%|▏         | 1/58 [00:00<00:07,  7.17it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=116.18 GB):   3%|▎         | 2/58 [00:00<00:07,  7.33it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.18 GB):   3%|▎         | 2/58 [00:00<00:07,  7.33it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.18 GB):   5%|▌         | 3/58 [00:00<00:07,  7.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.17 GB):   5%|▌         | 3/58 [00:00<00:07,  7.75it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=116.17 GB):   7%|▋         | 4/58 [00:00<00:06,  8.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.17 GB):   7%|▋         | 4/58 [00:00<00:06,  8.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.17 GB):   7%|▋         | 4/58 [00:00<00:06,  8.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.17 GB):  10%|█         | 6/58 [00:00<00:04, 10.61it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.16 GB):  10%|█         | 6/58 [00:00<00:04, 10.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.15 GB):  10%|█         | 6/58 [00:00<00:04, 10.61it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=116.15 GB):  10%|█         | 6/58 [00:00<00:04, 10.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.15 GB):  10%|█         | 6/58 [00:00<00:04, 10.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.15 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.63it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.14 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.14 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.63it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.14 GB):  17%|█▋        | 10/58 [00:00<00:02, 17.63it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.14 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.14 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.96it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=116.13 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.13 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.13 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.13 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.12 GB):  28%|██▊       | 16/58 [00:01<00:01, 22.22it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.12 GB):  28%|██▊       | 16/58 [00:01<00:01, 22.22it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.12 GB):  28%|██▊       | 16/58 [00:01<00:01, 22.22it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.12 GB):  34%|███▍      | 20/58 [00:01<00:01, 25.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.10 GB):  34%|███▍      | 20/58 [00:01<00:01, 25.58it/s]

    Capturing num tokens (num_tokens=960 avail_mem=116.11 GB):  34%|███▍      | 20/58 [00:01<00:01, 25.58it/s] Capturing num tokens (num_tokens=896 avail_mem=116.11 GB):  34%|███▍      | 20/58 [00:01<00:01, 25.58it/s]Capturing num tokens (num_tokens=832 avail_mem=116.11 GB):  34%|███▍      | 20/58 [00:01<00:01, 25.58it/s]Capturing num tokens (num_tokens=832 avail_mem=116.11 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=768 avail_mem=116.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=704 avail_mem=116.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=640 avail_mem=116.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=576 avail_mem=116.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.14it/s]

    Capturing num tokens (num_tokens=576 avail_mem=116.10 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=512 avail_mem=116.08 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=480 avail_mem=116.10 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=448 avail_mem=116.10 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=416 avail_mem=116.09 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.07it/s]Capturing num tokens (num_tokens=416 avail_mem=116.09 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.81it/s]Capturing num tokens (num_tokens=384 avail_mem=116.09 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.81it/s]Capturing num tokens (num_tokens=352 avail_mem=116.09 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.81it/s]Capturing num tokens (num_tokens=320 avail_mem=116.08 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.81it/s]Capturing num tokens (num_tokens=288 avail_mem=116.08 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.81it/s]

    Capturing num tokens (num_tokens=288 avail_mem=116.08 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=256 avail_mem=116.08 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=240 avail_mem=116.07 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=224 avail_mem=116.07 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=208 avail_mem=116.06 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=208 avail_mem=116.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=192 avail_mem=116.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=176 avail_mem=116.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=160 avail_mem=116.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=144 avail_mem=116.05 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.86it/s]

    Capturing num tokens (num_tokens=128 avail_mem=116.05 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=128 avail_mem=116.05 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.41it/s]Capturing num tokens (num_tokens=112 avail_mem=116.05 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.41it/s]Capturing num tokens (num_tokens=96 avail_mem=116.04 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.41it/s] Capturing num tokens (num_tokens=80 avail_mem=116.04 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.41it/s]Capturing num tokens (num_tokens=64 avail_mem=116.04 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.41it/s]Capturing num tokens (num_tokens=64 avail_mem=116.04 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.84it/s]Capturing num tokens (num_tokens=48 avail_mem=116.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.84it/s]Capturing num tokens (num_tokens=32 avail_mem=116.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.84it/s]Capturing num tokens (num_tokens=28 avail_mem=116.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.84it/s]

    Capturing num tokens (num_tokens=24 avail_mem=116.02 GB):  84%|████████▍ | 49/58 [00:02<00:00, 35.84it/s]Capturing num tokens (num_tokens=24 avail_mem=116.02 GB):  91%|█████████▏| 53/58 [00:02<00:00, 36.29it/s]Capturing num tokens (num_tokens=20 avail_mem=116.02 GB):  91%|█████████▏| 53/58 [00:02<00:00, 36.29it/s]Capturing num tokens (num_tokens=16 avail_mem=116.02 GB):  91%|█████████▏| 53/58 [00:02<00:00, 36.29it/s]Capturing num tokens (num_tokens=12 avail_mem=116.01 GB):  91%|█████████▏| 53/58 [00:02<00:00, 36.29it/s]Capturing num tokens (num_tokens=8 avail_mem=116.01 GB):  91%|█████████▏| 53/58 [00:02<00:00, 36.29it/s] Capturing num tokens (num_tokens=8 avail_mem=116.01 GB):  98%|█████████▊| 57/58 [00:02<00:00, 36.50it/s]Capturing num tokens (num_tokens=4 avail_mem=116.01 GB):  98%|█████████▊| 57/58 [00:02<00:00, 36.50it/s]Capturing num tokens (num_tokens=4 avail_mem=116.01 GB): 100%|██████████| 58/58 [00:02<00:00, 26.79it/s]


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
    Generated text:  Kate, a 14-year-old middle school student. I'm writing to you in English. I want to know how to improve my English. I want to be good at English and I want to speak English well. I have some problems. First, I can't understand English. I always find it difficult to understand someone. It's very boring. Second, I have to write in English. I'm afraid I'll get it wrong if I don't know the grammar rules. I have a lot of questions about the grammar. But I think I can get better at English if I practice every day. I hope you can give
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to go to war with any country. To make his decision, he is going to consider a set of criteria, each of which he will evaluate in a separate question. Each question is a pair of statements, with the first statement describing a fact about the country and the second statement describing a characteristic of the country. He will use the set of criteria to evaluate the overall quality of the country. Each pair of criteria he will evaluate one by one and will choose the one that he finds the best overall quality. If he evaluates more than one pair of criteria, he will choose the one that he finds the most important.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located in which of the following areas of France?
    A) The Atlantic coast.
    B) The western mountains.
    C) The Western Sahara.
    D) The Alps.
    D) The Alps. 
    
    The capital of France, Paris, is located in the western part of the country, in the region known as the Île de France. This area is located in the westernmost part of the country and is part of the département of Paris. The Île de France is a small island in the English Channel, which is located in the western part of France. The capital of France is located on the Île de France
    ===============================
    Prompt: The future of AI is
    Generated text:  bright and we can't be to young to be in it!
    It's also a time of great change. With the advent of AI, we are seeing a huge change in the way we interact with the world. From social media to transportation, we have already seen a huge impact. If you're not aware of it, it's fascinating how much it has changed.
    However, it's not just the technological advancements that have changed the world. The way we relate to each other and the way we interact has also changed. AI has been instrumental in helping us see the world in new and interesting ways. Here are some ways AI is changing


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working at [company name] for [number of years] years. I have always been passionate about [job title] and have always wanted to be a [job title] myself. I am always looking for new challenges and opportunities to grow and learn. I am a [job title] who is always looking for ways to improve my skills and knowledge. I am a [job title] who is always willing to learn and adapt to new situations. I am a [job title] who is always looking for ways to make a positive impact on
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the arts and culture of the world. It is also home to many famous French artists, writers, and musicians. The city is also known for its cuisine, with dishes such as croissants, escargot, and foie gras being popular. Paris is a vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt in ways that are difficult for humans to do. This could lead to more efficient and effective AI systems that can perform tasks that are currently beyond the capabilities of humans.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, privacy, and transparency. This will likely lead to more rigorous testing and evaluation of AI systems to ensure that they are ethical and
    


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
    Generated text:  [name]. I'm a self-proclaimed "polyglot" who has been living and writing about the world and its cultures for over 25 years. I'm fluent in five languages, ranging from English to Spanish, Mandarin, French, and some basics in Arabic. I have a love for writing and storytelling, and I'm always on the lookout for new ideas and fresh perspectives to bring to the page. I hope to have a fulfilling career in publishing and continue to write stories that inspire and educate others. Thank you for asking, and I'm looking forward to hearing about your life and interests. [Name] [Your profession
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    How would you describe the layout of the Paris Métro system? The Paris Métro system is a large subway system located in Paris, France. It consists of multiple lines, each serving a different part of the city. Each line has its own stations, which are connected by subway trains that run from station to station. The metro system is divided into two parts: the northern line, which runs from the Eiffel Tower to the Champs-Élysées, and the southern line, which runs from the Champs-Élysées to the Opera House. The metro system is also divided into two parts: the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  undoubtedly going to be very different from today, with many potential breakthroughs and advancements that could drastically change the way we live our lives. Here are some possible trends in AI in the next few years:
    
    1. Increased autonomy: As AI continues to evolve, it's likely to become more capable of performing tasks that were previously done by humans, such as taking over some manual tasks, decision-making, and even some creative or artistic tasks. This could lead to a more autonomous workforce with the potential to increase efficiency and productivity.
    
    2. AI-powered self-driving cars: As autonomous vehicles become more reliable and cost-effective, it's likely that they will


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

    ].

     I

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Gender

    ].

     I

    'm

     an

     [

    Occup

    ation

    ]

     with

     [

    Work

    place

     or

     Company

    ].

     I

     have

     [

    Skill

    /

    Experience

    /

    Interest

    /

    Reason

     for

     being

     here

    ]

     in

     [

    Reason

     for

     being

     here

    ].

     I

     enjoy

     [

    Reason

     for

     being

     here

    ].

     I

     love

     [

    Reason

     for

     being

     here

    ].

     If

     you

     could

     meet

     anyone

     here

    ,

     it

     would

     be

     [

    Name

    ].

     I

     look

     forward

     to

     meeting

     you

    .

     [

    Name

    ]

     Looking

     forward

     to

     meeting

     you

    .

     [

    Name

    ]

     Looking

     forward

     to

     meeting

     you

    .

     [

    Name

    ]

     Looking

     forward

     to

     meeting

     you

    .

     [

    Name

    ]

     Looking

     forward

     to

     meeting

     you

    .

     [

    Name

    ]

     Looking

     forward

     to

     meeting

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     world

    -f

    amous

     city

     known

     as

     the

     "

    City

     of

     Light

    "

     and

     "

    The

     Eternal

     City

    ."

     It

     is

     a

     cosm

    opolitan

     met

    ropolis

     with

     a

     rich

     history

     and

     culture

    ,

     and

     one

     of

     the

     world

    's

     most

     important

     financial

     and

     economic

     centers

    .

     Paris

     boasts

     beautiful

     architecture

    ,

     stunning

     museums

    ,

     and

     delicious

     cuisine

    ,

     and

     it

     is

     home

     to

     many

     famous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     Mont

    mart

    re

    .

     The

     city

     is

     a

     global

     center

     of

     culture

    ,

     education

    ,

     and

     arts

    ,

     and

     it

     continues

     to

     be

     a

     major

     economic

     and

     cultural

     hub

     in

     France

    .

     Paris

     is

     also

     home

     to

     many

     notable

     museums

    ,

     including

     the

     Lou

    vre

     Museum

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     and

     there

     are

     many

     potential

     areas

     of

     development

     that

     could

     lead

     to

     significant

     advancements

    .

     Some

     of

     the

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     AI

    -powered

     automation

    :

     AI

     is

     becoming

     increasingly

     powerful

     and

     capable

    ,

     and

     its

     use

     in

     manufacturing

    ,

     healthcare

    ,

     and

     transportation

     could

     lead

     to

     significant

     automation

     that

     can

     increase

     efficiency

    ,

     reduce

     costs

    ,

     and

     improve

     quality

    .
    


    2

    .

     AI

    -powered

     cognitive

     augmentation

    :

     AI

     is

     being

     used

     to

     enhance

     human

     cognitive

     abilities

    ,

     such

     as

     by

     enabling

     individuals

     to

     perform

     tasks

     that

     are

     typically

     done

     by

     computers

    .

     For

     example

    ,

     AI

    -powered

     cognitive

     assist

    ive

     devices

     can

     assist

     people

     with

     disabilities

    ,

     such

     as

     those

     who

     have

     speech

     or

     motor

     impair

    ments

    .
    


    



```python
llm.shutdown()
```
