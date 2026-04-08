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


    2026-04-08 03:04:39.752 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 03:04:39] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 03:04:39.752 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 03:04:39] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 03:04:39.752 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 03:04:39] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 03:04:39.752 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 03:04:39] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 03:04:39.752 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 03:04:39] Persistent cache disabled, using in-memory JIT cache


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.42it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.41it/s]


    2026-04-08 03:04:42,753 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 03:04:42] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  4.92it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  4.92it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:09,  4.92it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:09,  4.92it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:09,  4.92it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:03<00:09,  4.92it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:03<00:09,  4.92it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:03<00:09,  4.92it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:04, 10.37it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:04, 10.37it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:04, 10.37it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:04, 10.37it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:04, 10.37it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:03<00:04, 10.37it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:03<00:04, 10.37it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:03<00:04, 10.37it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:02, 16.68it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:02, 16.68it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:02, 16.68it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:02, 16.68it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:02, 16.68it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:03<00:02, 16.68it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:03<00:02, 16.68it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:03<00:02, 16.68it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 23.60it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 23.60it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 23.60it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 23.60it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 23.60it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 23.60it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 23.60it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 23.60it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 30.48it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 37.49it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 37.49it/s]

    Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:03<00:00, 37.49it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 42.90it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 42.90it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 42.90it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 42.90it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 42.90it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 42.90it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 42.90it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 42.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.64it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.68 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.68 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.08it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.08it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.08it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.56it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.56it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.56it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.56it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  31%|███       | 18/58 [00:00<00:01, 36.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 36.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 36.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  31%|███       | 18/58 [00:00<00:01, 36.39it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  31%|███       | 18/58 [00:00<00:01, 36.39it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  31%|███       | 18/58 [00:00<00:01, 36.39it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  31%|███       | 18/58 [00:00<00:01, 36.39it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=576 avail_mem=74.61 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.32it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  50%|█████     | 29/58 [00:00<00:00, 43.47it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 43.47it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 43.47it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 43.47it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 43.47it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 43.47it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.58it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.58it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.58it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.58it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.58it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.58it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.31it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.31it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.31it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.31it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.31it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.31it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.82it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.82it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.82it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.82it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.82it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.82it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.75it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.86it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.86it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.86it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.86it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.86it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 40.35it/s]


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
    Generated text:  Dave and I am a Personal Trainer. I am here to share information about myself, my fitness journey, and my beliefs and values, as well as to answer any questions you may have. I am a certified personal trainer with a degree in kinesiology. I am also a certified massage therapist and I have helped clients recover from injuries and improve their performance and well-being.
    I am a board certified personal trainer and I specialize in helping people improve their fitness level and recovery. I offer personal training, group classes, and one-on-one sessions. My personal training classes are fully customizable to accommodate everyone's level of fitness. I work with my
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person, a corporation, a club, or a government? – Cogito, Ergo Propter Hoc\nThe president of the United States is a person, a corporation, a club, or a government? – Cogito, Ergo Propter Hoc\nWhat’s the answer?\n5 votes\nanswered Jan 21, 2023 by Lawrence001\nanswered Jan 21, 2023 by James 000\nGustavus Adolphus II (1736–1818) was the King of Denmark and the Grand
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, France. It has a population of about 2.2 million and a population density of about 13,800 people per km². The country is located in the north-central part of the continent, near the Mediterranean Sea. The capital city is Paris. It has been a center of French culture, and many of the country’s historic buildings and landmarks are located in Paris. It is the home of the French Riviera, where one can visit the beaches, luxury resorts, and vineyards. The capital is the birthplace of the French monarchy, the country’s largest economy, the country’s most important cultural
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the developers. The developers who create algorithms and neural networks that drive our intelligence and technology. But they also have the responsibility to make sure that the data they collect and use is accurate, complete, and up-to-date. This is where security comes in.
    
    Security plays a crucial role in AI by protecting the integrity, confidentiality, and availability of the data being used. When security is not taken seriously, it can lead to significant data breaches, loss of user trust, and even legal consequences. Therefore, developers must prioritize security when creating and using AI systems.
    
    One way to ensure security in AI is to implement strong authentication and


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


    Generated text:  [Name] and I'm a [occupation] with [number of years] years of experience. I'm a [occupation] with [number of years] years of experience. I'm a [occupation] with [number of years] years of experience. I'm a [occupation] with [number of years] years of experience. I'm a [occupation] with [number of years] years of experience. I'm a [occupation] with [number of years] years of experience. I'm a [occupation] with [number of years] years of experience. I'm a [occupation] with [number of years]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is located in the south of the country. It is the largest city in Europe and the second-largest city in the world by population. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a cultural and artistic center, and is home to many famous museums, theaters, and art galleries. It is also a major transportation hub, with many major highways and rail lines connecting it to other parts of France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve risk management, fraud detection, and trading strategies.
    


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
    Generated text:  [insert character's name], and I'm a [insert the character's occupation or profession]!
    
    I'm an [insert the character's age], [insert the character's occupation or profession], and I have [insert a positive trait or characteristic of the character, such as "brilliant" or "brave"] that I strive to [insert the specific trait or behavior that the character exhibits, such as "take initiative", "keep an eye on things", etc.]. I'm always [insert a positive trait or characteristic of the character, such as "dedicated", "cautious", "intelligent", etc.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France, located in the south of the country and is the largest and most populous city. It is home to many of the country's most renowned landmarks and historical sites. The city is known for its rich history, fashion, cuisine, and music scene, and has been an international hub for diplomacy, commerce, and culture for centuries. Today, Paris remains an important cultural and economic center of the world, attracting millions of visitors annually and hosting the Eiffel Tower, Louvre Museum, and other iconic landmarks. Despite the city's size and population, Paris is known for its relaxed, diverse culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by a number of key trends, including:
    
      1. Increased integration with other technologies: As AI becomes more integrated with other technologies, such as machine learning, natural language processing, and computer vision, the potential for new applications and capabilities is likely to grow.
      2. Advancements in hardware and software: As AI technology advances, so too will the hardware and software that are required to support it. This could lead to increased efficiency and cost savings for businesses and individuals.
      3. Increase in ethical considerations: As AI becomes more advanced and integrated into our lives, there is a growing need for ethical


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

    ],

     and

     I

    'm

     a

     [

    occupation

    ]

     who

     has

     always

     been

     passionate

     about

     exploring

     the

     world

     and

     learning

     about

     new

     cultures

    .

     I

     love

     traveling

    ,

     trying

     new

     foods

    ,

     and

     exploring

     unfamiliar

     places

    .

     I

    'm

     always

     up

     for

     a

     challenge

     and

     am

     always

     looking

     for

     new

     adventures

     to

     try

    .

     I

     love

     the

     freedom

     of

     the

     outdoors

     and

     the

     thrill

     of

     new

     experiences

    .

     I

     have

     a

     natural

     talent

     for

     writing

    ,

     and

     I

     love

     to

     share

     my

     experiences

     and

     learn

     about

     different

     cultures

     through

     my

     writing

    .

     I

    'm

     excited

     to

     meet

     you

     and

     get

     to

     know

     you

     better

    .

     #

    self

    -int

    roduction

     #

    travel

    er

     #

    writer

     #

    out

    doors

     #

    c

    ultural

     exploration

    


    Hello

    !

     I

    'm

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     modern

    ,

     cosm

    opolitan

     city

     with

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

    .

     The

     city

     is

     known

     for

     its

     vibrant

     art

     scene

    ,

     extensive

     cultural

     institutions

    ,

     and

     diverse

     culinary

     offerings

    .

     It

     is

     also

     home

     to

     some

     of

     the

     world

    ’s

     most

     famous

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Metropolitan

     Museum

     of

     Art

    .

     Paris

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

     

    1

    0

     million

     people

    .

     Its

     status

     as

     the

     largest

     and

     most

     populous

     city

     in

     the

     European

     Union

     is

     recognized

     worldwide

    .

     It

     is

     a

     cultural

     and

     economic

     hub

     that

     plays

     a

     significant

     role

     in

     France

    's

     foreign

     policy

     and

     political

     landscape

    .

     Paris

     is

     also

     known

     for

     its

     

    5

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     full

     of

     possibilities

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     transparency

    :

     With

     the

     growing

     awareness

     of

     AI

     biases

     and

     privacy

     concerns

    ,

     it

    's

     likely

     that

     we

    'll

     see

     more

     transparency

     in

     AI

     models

    .

     This

     means

     that

     users

     can

     understand

     how

     a

     model

     makes

     its

     decisions

     and

     that

     we

     can

     trust

     the

     data

     used

     to

     train

     the

     model

    .
    


    2

    .

     Self

    -driving

     cars

    :

     Self

    -driving

     cars

     are

     likely

     to

     become

     more

     prevalent

     in

     the

     future

    ,

     thanks

     to

     advancements

     in

     AI

    .

     The

     technology

     will

     continue

     to

     improve

     and

     become

     more

     efficient

    ,

     making

     it

     easier

     for

     humans

     to

     trust

     autonomous

     vehicles

    .
    


    3

    .

     AI

     for

     medical

     diagnosis

    :

     AI

    -powered

     medical

     diagnosis

     tools

     will

    



```python
llm.shutdown()
```
