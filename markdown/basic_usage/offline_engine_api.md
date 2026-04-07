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


    2026-04-07 22:14:57.886 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 22:14:57] Persistent cache disabled, using in-memory JIT cache
    2026-04-07 22:14:57.886 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 22:14:57] Persistent cache disabled, using in-memory JIT cache
    2026-04-07 22:14:57.886 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 22:14:57] Persistent cache disabled, using in-memory JIT cache
    2026-04-07 22:14:57.886 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 22:14:57] Persistent cache disabled, using in-memory JIT cache
    2026-04-07 22:14:57.886 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 22:14:57] Persistent cache disabled, using in-memory JIT cache


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.14it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.13it/s]


    2026-04-07 22:15:00,337 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 22:15:00] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.10it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  6.10it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  6.10it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.10it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.10it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.34it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.34it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.34it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.34it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.34it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.34it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.34it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.34it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.32it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.32it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.32it/s]

    Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:06<00:01, 18.32it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:06<00:01, 18.32it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:06<00:01, 18.32it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:06<00:01, 18.32it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:06<00:05,  4.76it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:06<00:05,  4.76it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:06<00:05,  4.76it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:06<00:05,  4.76it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:06<00:05,  4.76it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:06<00:05,  4.76it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:06<00:05,  4.76it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:06<00:02,  6.73it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:06<00:02,  6.73it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:06<00:02,  6.73it/s]

    Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:06<00:02,  6.73it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:06<00:02,  6.73it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:06<00:02,  6.73it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:06<00:02,  6.73it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:01,  9.33it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:01,  9.33it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:01,  9.33it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:01,  9.33it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:06<00:01,  9.33it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:06<00:01,  9.33it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:06<00:00, 11.94it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:06<00:00, 11.94it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:06<00:00, 11.94it/s]

    Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:06<00:00, 11.94it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:06<00:00, 11.94it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:06<00:00, 11.94it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:06<00:00, 11.94it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:06<00:00, 11.94it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:06<00:00, 11.94it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:06<00:00, 17.61it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:06<00:00, 17.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=134.85 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=134.82 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=134.82 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=134.82 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=134.81 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=6144 avail_mem=134.82 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=134.82 GB):   9%|▊         | 5/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=5632 avail_mem=134.81 GB):   9%|▊         | 5/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=5120 avail_mem=134.81 GB):   9%|▊         | 5/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=4608 avail_mem=134.81 GB):   9%|▊         | 5/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=4608 avail_mem=134.81 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=134.81 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=134.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.83it/s]Capturing num tokens (num_tokens=3584 avail_mem=134.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.83it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=134.80 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=134.79 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.35it/s]Capturing num tokens (num_tokens=3072 avail_mem=134.79 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.35it/s]Capturing num tokens (num_tokens=2816 avail_mem=134.79 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.35it/s]Capturing num tokens (num_tokens=2560 avail_mem=134.78 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.35it/s]Capturing num tokens (num_tokens=2304 avail_mem=134.78 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.35it/s]Capturing num tokens (num_tokens=2304 avail_mem=134.78 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=134.78 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.48it/s]Capturing num tokens (num_tokens=1792 avail_mem=134.78 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.48it/s]Capturing num tokens (num_tokens=1536 avail_mem=134.77 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.48it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=134.77 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=134.77 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=134.75 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.66it/s]Capturing num tokens (num_tokens=960 avail_mem=134.76 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.66it/s] Capturing num tokens (num_tokens=896 avail_mem=134.72 GB):  34%|███▍      | 20/58 [00:00<00:01, 22.66it/s]Capturing num tokens (num_tokens=832 avail_mem=134.72 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.66it/s]

    Capturing num tokens (num_tokens=832 avail_mem=134.72 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=768 avail_mem=134.72 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=704 avail_mem=134.71 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=640 avail_mem=134.71 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=576 avail_mem=134.71 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=512 avail_mem=134.70 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.28it/s]Capturing num tokens (num_tokens=512 avail_mem=134.70 GB):  50%|█████     | 29/58 [00:01<00:00, 30.18it/s]Capturing num tokens (num_tokens=480 avail_mem=134.71 GB):  50%|█████     | 29/58 [00:01<00:00, 30.18it/s]Capturing num tokens (num_tokens=448 avail_mem=134.71 GB):  50%|█████     | 29/58 [00:01<00:00, 30.18it/s]Capturing num tokens (num_tokens=416 avail_mem=134.71 GB):  50%|█████     | 29/58 [00:01<00:00, 30.18it/s]Capturing num tokens (num_tokens=384 avail_mem=134.71 GB):  50%|█████     | 29/58 [00:01<00:00, 30.18it/s]

    Capturing num tokens (num_tokens=352 avail_mem=134.70 GB):  50%|█████     | 29/58 [00:01<00:00, 30.18it/s]Capturing num tokens (num_tokens=352 avail_mem=134.70 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=320 avail_mem=134.70 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=288 avail_mem=134.70 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=256 avail_mem=134.69 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=240 avail_mem=134.69 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=224 avail_mem=134.69 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.04it/s]Capturing num tokens (num_tokens=224 avail_mem=134.69 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.87it/s]Capturing num tokens (num_tokens=208 avail_mem=134.68 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.87it/s]Capturing num tokens (num_tokens=192 avail_mem=134.68 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.87it/s]Capturing num tokens (num_tokens=176 avail_mem=134.68 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.87it/s]

    Capturing num tokens (num_tokens=160 avail_mem=134.68 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.87it/s]Capturing num tokens (num_tokens=144 avail_mem=134.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.87it/s]Capturing num tokens (num_tokens=144 avail_mem=134.67 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.05it/s]Capturing num tokens (num_tokens=128 avail_mem=134.67 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.05it/s]Capturing num tokens (num_tokens=112 avail_mem=134.67 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.05it/s]Capturing num tokens (num_tokens=96 avail_mem=134.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.05it/s] Capturing num tokens (num_tokens=80 avail_mem=134.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.05it/s]Capturing num tokens (num_tokens=64 avail_mem=134.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.05it/s]Capturing num tokens (num_tokens=64 avail_mem=134.66 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=48 avail_mem=134.65 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=32 avail_mem=134.65 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.39it/s]

    Capturing num tokens (num_tokens=28 avail_mem=134.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=24 avail_mem=134.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=20 avail_mem=134.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.39it/s]Capturing num tokens (num_tokens=20 avail_mem=134.64 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.82it/s]Capturing num tokens (num_tokens=16 avail_mem=134.64 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.82it/s]Capturing num tokens (num_tokens=12 avail_mem=134.63 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.82it/s]Capturing num tokens (num_tokens=8 avail_mem=134.63 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.82it/s] Capturing num tokens (num_tokens=4 avail_mem=134.63 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.82it/s]Capturing num tokens (num_tokens=4 avail_mem=134.63 GB): 100%|██████████| 58/58 [00:01<00:00, 31.95it/s]


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
    Generated text:  Anthony and I'm a developer, and I specialize in building and customizing mobile, web, and desktop applications using the Java programming language. I'm a self-taught developer who has been developing applications for over 10 years, and I enjoy learning new technologies and using them in my work. I'm open to trying new things and exploring new platforms, and I believe in being a problem-solver and constantly learning new skills to meet the evolving needs of my clients. My goal is to provide high-quality, custom-developed solutions that meet the specific requirements of my clients. 
    I have a passion for creating intuitive, user-friendly
    ===============================
    Prompt: The president of the United States is
    Generated text:  doing a good job of managing the nation's affairs. However, some people are questioning whether he is doing a good job at all. Many of them believe that the president's problems lie in his constant actions and lack of action. Others think that the president's flaws lie in his language and his willingness to make compromises. In the first place, we can use an analogy to explain this question. The president is like a master chef, and his actions are like the preparation of a dish. The president's actions are necessary for his effectiveness, and without them, his dish would be insufficient. However, the president is also like a chef who
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, with a population of approximately 2.9 million. The metropolitan area is also known as Paris Métropole, with a population of about 13 million.
    
    A. 正确 B. 错误
    
    To determine whether the statement is correct or incorrect, I will follow these steps:
    
    1. Verify the capital of France.
    2. Verify the metropolitan area of Paris.
    3. Verify the population of Paris.
    4. Verify the population of the metropolitan area.
    5. Compare the results.
    
    Step 1: The capital of France is Paris.
    
    Step 2: The metropolitan area of Paris is also known as Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it's heading directly to the core of the way we live our lives. It has already transformed the way we travel, work, communicate, and even keep track of our health. Now, AI is entering a new phase where it is going to completely rewire the way we are thinking about and using artificial intelligence.
    We must shift from a "hard" AI to a "soft" AI. The term "soft" refers to AI that can think and learn without being constrained by physical limits, and can be used for many different purposes like health and agriculture.
    The "hard" AI refers to AI systems that are constrained by


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. How can I assist you today? [Name] [Job Title] [Company Name] [Company Address] [City, State, Zip Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on modern French culture and politics. Paris is also home to many famous French artists, writers, and musicians. The city is known for its cuisine, including its famous French cuisine, and is a popular tourist
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI, there will be a greater emphasis on ethical AI. This will likely lead to more rigorous testing and validation of AI systems, as well as greater transparency and accountability in their development and deployment.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, from smart homes to self-driving cars. As more companies and governments invest in AI, it is likely
    


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
    Generated text:  [insert character's name], and I'm a [insert fictional profession or role] who has always had a natural affinity for [insert a detail about your profession, such as "fishing" or "aerospace engineering."]. I believe that my background in [insert a detail about the profession you've pursued, such as "fishing" or "aerospace engineering" or "programming" or "sports" or "traveling" or "singing" or "traveling" or "photography" or "teaching" or "acting" or "singing" or "playing the guitar" or "playing
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the capital city of France and was founded in the 6th century by the Romans. It is located in the northeastern part of France and is the seat of government, culture, and commerce of France. It is home to many important historical sites and landmarks, including the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and the Champs-Elysées. Paris is also famous for its fashion and food scene, which attract millions of visitors each year. It is also home to numerous museums, including the Louvre, the Musée d'Orsay, and the Musée de l'Hom
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but there are several possible trends that could play a role in shaping the landscape of artificial intelligence over the next decade.
    
    1. Autonomous vehicles: As the technology continues to advance, autonomous vehicles could become a reality. This could have a significant impact on the transportation industry and could potentially lead to a decrease in traffic accidents.
    
    2. AI in healthcare: AI could be used to improve the diagnosis and treatment of diseases, allowing for earlier and more effective treatment. This could lead to a reduction in the cost of healthcare and a decrease in medical errors.
    
    3. AI in finance: AI could be used to improve risk management and investment strategies


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

    Name

    ].

     I

    'm

     a

     [

    Job

     Title

    ]

     at

     [

    Company

     Name

    ].

     I

     love

     [

    Anything

     you

     enjoy

    ].

     I

    'm

     always

     looking

     for

     new

     ways

     to

     [

    Describe

     how

     you

     like

     to

     stay

     updated

     on

     the

     latest

     trends

    ,

     f

    ash

    ions

    ,

     etc

    .

    ].

     I

    'm

     passionate

     about

     [

    What

    's

     your

     area

     of

     expertise

     or

     interests

    ?

    ].

     I

     can

     be

     found

     [

    Where

     you

    're

     most

     known

     for

    ,

     like

     having

     a

     passion

     for

     cooking

    ,

     playing

     sports

    ,

     etc

    .

    ].

     Thank

     you

     for

     asking

     me

     out

    .


    You

    're

     quite

     charming

    !

     I

     love

     any

     kind

     of

     art

    ,

     reading

    ,

     writing

    ,

     and

     dancing

    .

     I

    'm

     always

     looking

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     often

     referred

     to

     as

     "

    The

     City

     of

     Light

    "

     due

     to

     its

     vibrant

     culture

     and

     lively

     atmosphere

    .

     The

     city

     is

     located

     in

     the

     northern

     part

     of

     France

     and

     is

     home

     to

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     France

    's

     capital

     city

     is

     the

     largest

     city

     in

     the

     European

     Union

     and

     is

     a

     major

     economic

    ,

     cultural

    ,

     and

     transportation

     hub

    .

     Paris

     is

     also

     a

     symbol

     of

     French

     culture

     and

     history

    ,

     and

     it

     has

     played

     an

     important

     role

     in

     French

     politics

    ,

     culture

    ,

     and

     society

     for

     centuries

    .

     The

     city

     is

     known

     for

     its

     sophisticated

     cuisine

    ,

     iconic

     architecture

    ,

     and

     annual

     cultural

     events

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     extremely

     promising

    ,

     and

     there

     are

     a

     number

     of

     trends

     that

     are

     expected

     to

     shape

     the

     field

     in

     the

     years

     ahead

    .

     Here

     are

     a

     few

     potential

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     accuracy

     and

     efficiency

    :

     One

     of

     the

     most

     significant

     trends

     in

     AI

     is

     the

     need

     for

     improved

     accuracy

     and

     efficiency

    .

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     will

     be

     able

     to

     perform

     tasks

     more

     quickly

     and

     accurately

    ,

     which

     can

     lead

     to

     increased

     efficiency

     and

     reduced

     costs

    .

     This

     will

     be

     especially

     important

     as

     the

     world

     becomes

     increasingly

     data

    -driven

    ,

     and

     AI

     systems

     will

     be

     required

     to

     make

     decisions

     in

     a

     wide

     range

     of

     fields

    .
    


    2

    .

     Enhanced

     personal

    ization

    :

     AI

     will

     also

     play

     a

     major

     role

    



```python
llm.shutdown()
```
