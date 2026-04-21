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
    [2026-04-21 02:32:02] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.70it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.69it/s]


    2026-04-21 02:32:07,362 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-21 02:32:07] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 13.91it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 13.91it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 13.91it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 13.91it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 22.25it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 22.25it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 22.25it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 22.25it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 22.25it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 22.25it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 22.25it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 22.25it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 22.25it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 22.25it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:03<00:01, 22.25it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 32.56it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 32.56it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 43.22it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 43.22it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 43.22it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 43.22it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 43.22it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 43.22it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 43.22it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 43.22it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 43.22it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 43.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.89it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.89it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.89it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.58it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.58it/s]Capturing num tokens (num_tokens=960 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.58it/s] Capturing num tokens (num_tokens=896 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.58it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.58it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.12it/s]Capturing num tokens (num_tokens=768 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.12it/s]Capturing num tokens (num_tokens=704 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.12it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.12it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.12it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.12it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:00<00:00, 40.11it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=288 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=224 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.59it/s]Capturing num tokens (num_tokens=224 avail_mem=74.07 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.74it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=64 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=64 avail_mem=74.04 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.61it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.59it/s] Capturing num tokens (num_tokens=4 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=4 avail_mem=74.01 GB): 100%|██████████| 58/58 [00:01<00:00, 40.44it/s]


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
    Generated text:  Jonnie and I was born in 1968. My mother was a school teacher, and my father, a seafarer. I was the youngest child of my parents, and I loved my dad very much. I was always the youngest child, the son of the other. I was an only child, and the only child. I lived with my parents in the family home in London for a few years until I was 13. I started school at the age of 15, and later went to college. I began my first job at the age of 18. I became a successful writer and
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. [ ]
    A. True
    B. False
    C. Uncertain
    Answer:
    
    A
    
    The origin of the Chinese language is in the _____ of the Chinese nation.
    A. History
    B. Ethnology
    C. Culture
    D. Mythology
    Answer:
    
    C
    
    If the difference between the first and second terms of a geometric sequence is 18, and the product of the first and third terms is 6, then the common ratio of this geometric sequence is ____
    A. 2
    B. 3
    C. 6
    D. 12
    Answer:
    
    B
    
    Given
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. London
    C. Tokyo
    D. Rome
    Answer: A
    
    The European Constitution adopted at the 1959 European Parliament was ____
    A. The European Constitution
    B. The Lisbon Treaty
    C. The Treaty of Lisbon
    D. The Treaty of Amsterdam
    Answer: B
    
    In the 2009 UK general election, what was the second-choice option?
    A. Remain
    B. Leave
    C. NP
    D. V
    Answer: D
    
    The largest city in the United States is ____
    A. Houston
    B. New York
    C. Philadelphia
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain as it appears that the technology is bound to become more advanced, which could lead to its use for things that are now a thing of the past, such as in artificially created skin. Additionally, the ability to perceive or understand a person’s emotions and thoughts is something that would be very useful for a person in the future. These are some of the major trends that are likely to shape the future of AI.
    Some of the potential impacts of AI on society are employment, education, and health care. In the employment sector, AI is likely to transform the way that we work, leading to the creation of new jobs and the replacement


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Monument. Paris is a bustling metropolis with a diverse population and is a major center for art, culture, and politics in France. It is also home to many famous French restaurants and cafes. Paris is a popular tourist destination and is known for its fashion, cuisine, and music. The city is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the impact of AI on society as a whole.
    
    2. Greater integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This will allow for
    


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
    Generated text:  [insert character's name] and I'm a [insert job title or profession] at [insert company name]. I'm enthusiastic, hardworking, and always ready to go above and beyond for the best possible experience. My passion for [insert a favorite hobby, project, or activity] drives me to deliver results, while my approachable personality helps me connect with clients on a personal level. My goal is to help them achieve their goals and succeed in whatever field they are pursuing. I am always eager to learn and adapt to new situations, and I'm confident that I can help you achieve success. #excellent #Professional #Friendly
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as “la Champs-Elysées” in French. It is the world's most populous city and the largest city by area and population in the European Union. It is the second most populous city in the world after Delhi, and the second most populous city in the United States. Paris was founded in 869 by the Duke of Aquitaine. The city is home to the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and Musée d'Orsay. The city is known for its rich cultural heritage, including the Notre-Dame Cathedral, which is a symbol of France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be influenced by a variety of factors, including advances in computing power and data storage, continued improvements in machine learning algorithms, and a growing interest in ethical considerations. Some possible trends in AI include:
    
    1. Increased AI in healthcare: AI is already being used to analyze medical images and predict patient outcomes, but more advanced algorithms may be able to identify patterns and predict disease progression more accurately.
    
    2. AI in transportation: Self-driving cars and other autonomous vehicles are already being tested, but more advanced AI could make them safer and more efficient.
    
    3. AI in finance: AI is already being used to detect fraud and insider trading, but


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

     character

    's

     name

    ].

     I

    'm

     a

     professional

     data

     analyst

     with

     

    1

    0

     years

     of

     experience

     in

     the

     industry

    ,

     and

     I

     thrive

     on

     using

     my

     skills

     to

     help

     solve

     complex

     problems

     and

     generate

     actionable

     insights

     for

     organizations

    .

     Whether

     it

    's

     using

     SQL

    ,

     Python

    ,

     or

     Excel

    ,

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     my

     abilities

     and

     keep

     up

     with

     industry

     trends

    .

     I

     enjoy

     working

     with

     a

     team

     and

     being

     involved

     in

     the

     broader

     picture

     of

     the

     company

    ,

     and

     I

    'm

     always

     looking

     for

     opportunities

     to

     learn

     new

     things

     and

     grow

     as

     a

     professional

    .

     Thank

     you

     for

     considering

     me

     for

     a

     role

    .

     [

    Insert

     character

    's

     name

    ]

     and

     enjoy

     working

     with

     you

    !

     

    🚀

    
    
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

    -largest

     city

     in

     Europe

    .

     It

     is

     located

     on

     the

     northern

     bank

     of

     the

     Se

    ine

     River

    ,

     in

     the

     region

     of

     the

     same

     name

    ,

     and

     is

     the

     capital

     of

     the

     Paris

     region

    ,

     which

     includes

     the

     Î

    le

    -de

    -F

    rance

    ,

     the

     south

     of

     France

    ,

     and

     the

     metropolitan

     region

     of

     Paris

    .

     It

     is

     an

     international

     met

    ropolis

     with

     a

     population

     of

     around

     

    1

    .

    3

     million

     inhabitants

    .

     The

     city

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     has

     a

     long

     and

     illust

    rious

     history

    .

     In

     

    2

    0

    1

    7

    ,

     it

     was

     declared

     a

     European

     Capital

     of

     Culture

     by

     the

     European

     Union

    .

     The

     city

     is

     also

     home

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     it

     is

     likely

     to

     continue

     evolving

     rapidly

    .

     Some

     potential

     future

     trends

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     medical

     diagnosis

     and

     treatment

    :

     AI

    -powered

     diagnostic

     tools

     and

     algorithms

     are

     likely

     to

     become

     more

     sophisticated

     in

     the

     coming

     years

    ,

     enabling

     better

     diagnosis

     and

     treatment

     of

     diseases

    .

     AI

     may

     also

     be

     used

     to

     improve

     patient

     outcomes

     in

     fields

     such

     as

     mental

     health

     and

     ger

    iatrics

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     automation

     and

     robotics

    :

     AI

     is

     already

     being

     used

     in

     many

     industries

    ,

     and

     it

     is

     likely

     to

     become

     more

     ubiquitous

     in

     the

     future

    .

     Robots

     may

     be

     used

     in

     manufacturing

    ,

     delivery

    ,

     and

     other

     industries

     to

     perform

     tasks

     that

     were

     previously

     done

     by

     humans

    .
    


    



```python
llm.shutdown()
```
