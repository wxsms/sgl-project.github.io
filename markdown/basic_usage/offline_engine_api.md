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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-21 04:51:12] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.96it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.95it/s]


    2026-04-21 04:51:17,444 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-21 04:51:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.74it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.74it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.74it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.74it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.74it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.74it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.74it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.74it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.74it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.74it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.77it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.77it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.77it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.77it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.77it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.77it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.77it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.77it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.77it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.77it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.85it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.85it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.85it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.85it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.85it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.85it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.85it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.85it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.85it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.85it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.68it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.68it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.68it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.68it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.68it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.68it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.68it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.68it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.68it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.68it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.47it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.47it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.47it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.47it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.47it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.47it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.47it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.47it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.47it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.47it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.47it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.91it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.55it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.55it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.55it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.55it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.55it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.08it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.08it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.80it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.80it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.80it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.80it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.80it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  50%|█████     | 29/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 39.77it/s]

    Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  50%|█████     | 29/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  50%|█████     | 29/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 40.93it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 40.93it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 40.93it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 40.93it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.93it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.93it/s]

    Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.27it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.27it/s]

    Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.68it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.68it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.68it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.68it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.68it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 38.66it/s]


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
    Generated text:  Kim and I'm a software engineer. My job involves using my skills to develop web and mobile applications. I'm also really good at learning and growing my knowledge about technology and programming. How can I improve my career prospects in the tech industry? Improving your career prospects in the tech industry can be achieved through several strategies. Here are some key approaches you can take:
    
    1. **Continuous Learning and Adaptability**: The tech industry is constantly evolving, and it's essential to stay updated with the latest technologies, programming languages, and methodologies. Regularly attend conferences, participate in online courses, and engage with community forums can help you keep up
    ===============================
    Prompt: The president of the United States is
    Generated text:  a significant position in American politics and policy. They are elected by the people for a term of office. Some of the key responsibilities of the president include:
    1. Representing the country on the international stage
    2. Providing leadership and direction for the country's government and executive branch
    3. Drafting and enforcing laws and policies
    4. Managing the federal bureaucracy and making important decisions
    5. Securing and protecting the country's interests and resources
    6. Serving as the leader of the country's electoral process
    7. Stopping criminal or illegal activities within the country
    8. Representing the country on the global stage through diplomatic
    ===============================
    Prompt: The capital of France is
    Generated text:  called:
    A. Paris
    B. Bordeaux
    C. Lyon
    D. Avignon
    
    A. Paris is the capital of France. Therefore, the correct answer is:
    
    A. Paris
    
    To elaborate, Paris is the largest city in France, located on the island of France. It is the country's capital and the seat of the government, as well as its largest city. The other options are not capitals of France:
    
    B. Bordeaux is the capital of France, located in southwestern France and known for its wine industry.
    
    C. Lyon is the capital of France, located in the northeastern region of France and known for its industrial heritage
    ===============================
    Prompt: The future of AI is
    Generated text:  shaping up to be a transformative force, but for many, it will be a long journey to get there.
    
    A new study published in the journal Nature by researchers from the University of California, San Francisco (UCSF) and the University of Edinburgh (University of Edinburgh) reveals that the rapid development of machine learning is more than merely a matter of innovation in algorithms or a new tool for tackling specific problems. It is more about the wider landscape of artificial intelligence and the search for solutions that can help people find what they are looking for. AI is currently being applied in a wide range of sectors including health care, finance, business, and law


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a brief description of your job or experience here]. I enjoy [insert a short description of your hobbies or interests here]. I'm always looking for new challenges and opportunities to grow and learn. What do you like to do in your free time? I enjoy [insert a short description of your hobbies or interests here]. I'm always looking for new experiences and opportunities to expand my knowledge and skills. What's your favorite hobby or activity
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other cultural institutions. Paris is a popular tourist destination, known for its rich history, art, and cuisine. It is also home to many famous French artists, writers, and musicians. The city is known for its vibrant nightlife and has a diverse range of restaurants, bars, and cafes. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a city that
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and context-aware AI, as well as more efficient and effective use of AI in various applications.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead to more transparent and accountable AI
    


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
    Generated text:  [Name], I'm a [age] year old girl. I'm a [occupation] at [job title] [location]. I'm an [occupation] at [job title] [location]. I'm an [occupation] at [job title] [location]. I'm an [occupation] at [job title] [location]. I'm an [occupation] at [job title] [location]. I'm an [occupation] at [job title] [location]. I'm an [occupation] at [job title] [location]. I'm an [occupation] at [job title] [location]. I'm an
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest city and its most populous city in the European Union. 
    
    In summary, Paris is the political, cultural, and economic center of France and the world's most populous city, known for its diverse culture, architecture, art, food, fashion, and music. 
    
    - The city is located on the western bank of the Seine River, on the Île de la Cité, and is situated at 42°47'00"N 2°54'00"E, with a population of around 2.2 million. 
    
    - The city is home to the Eiffel Tower
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, with new technologies and applications emerging constantly. Here are some possible trends in AI that are currently shaping the landscape:
    
    1. **Deep Learning and Explainable AI**: With the increasing availability of large datasets and the emergence of new architectures like AlphaFold, deep learning is becoming more efficient and capable. Explainable AI (XAI) will allow AI systems to be transparent in their decision-making processes, making it easier to understand and trust. This will be particularly important in industries like healthcare and finance where trust in AI is crucial.
    
    2. **Quantum Computing**: As quantum computing continues to advance, AI systems could leverage quantum computing capabilities


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

    ],

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

    why

     you

    're

     passionate

    ]

     and

     I

    'm

     constantly

     learning

     and

     growing

     as

     a

     person

    .

     Whether

     it

    's

     [

    job

     title

    ]

     or

     not

    ,

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     myself

     and

     make

     the

     world

     a

     better

     place

    .

     I

    'm

     excited

     to

     meet

     you

     and

     share

     my

     knowledge

     and

     experiences

     with

     you

    .

     Thanks

     for

     taking

     the

     time

     to

     meet

     me

    ,

     and

     look

     forward

     to

     our

     conversation

    !

     

    🌟

    ✨

    ✨

    ✨

    
    


    What

     is

     the

     most

     important

     lesson

     you

     learned

     while

     working

     on

     your

     project

    ?

     As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     personal

     experiences

     like

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

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

     the

     Lou

    vre

     Museum

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

     capital

     of

     France

     is

     Paris

    ,

     an

     historic

     city

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

     the

     Lou

    vre

     Museum

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

     city

     is

     also

     home

     to

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

     Arc

     de

     Tri

    omp

    he

    ,

     which

     have

     become

     symbols

     of

     France

    's

     rich

     cultural

     heritage

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     major

     financial

     and

     cultural

     hub

    .

     Its

     cultural

     and

     artistic

     heritage

     includes

     the

     Notre

    -D

    ame

     Cathedral

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     trends

    ,

     including

    :
    


    1

    .

     Increased

     reliance

     on

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     diagnose

     diseases

     and

     treat

     patients

    ,

     and

     as

     AI

     technology

     continues

     to

     evolve

    ,

     we

     may

     see

     a

     growing

     reliance

     on

     AI

     in

     healthcare

    .
    


    2

    .

     Improved

     safety

     and

     security

    :

     AI

     is

     already

     being

     used

     in

     security

     systems

    ,

     and

     as

     AI

     technology

     advances

    ,

     we

     may

     see

     even

     more

     advanced

     and

     sophisticated

     security

     measures

    .
    


    3

    .

     Increased

     use

     of

     AI

     in

     manufacturing

    :

     AI

     is

     already

     being

     used

     in

     manufacturing

    ,

     and

     as

     AI

     technology

     continues

     to

     evolve

    ,

     we

     may

     see

     even

     more

     advanced

     and

     sophisticated

     manufacturing

     technologies

    .
    


    4

    .

     Growth

     of

     the

     AI

     workforce

    :

     As

    



```python
llm.shutdown()
```
