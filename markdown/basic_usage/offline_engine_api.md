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
    [2026-04-26 08:49:00] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.13it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.12it/s]


    2026-04-26 08:49:05,123 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 08:49:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.14it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.78it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.58it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.58it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.58it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.58it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.58it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.58it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.58it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.58it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.58it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.58it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.55it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.55it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.34it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.34it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 39.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.67it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.65 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=121.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=121.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=121.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=121.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=121.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=121.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.02it/s]Capturing num tokens (num_tokens=2816 avail_mem=121.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.02it/s]Capturing num tokens (num_tokens=2560 avail_mem=121.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.02it/s]Capturing num tokens (num_tokens=2304 avail_mem=121.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=121.59 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=121.59 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=121.59 GB):  31%|███       | 18/58 [00:00<00:01, 35.77it/s]Capturing num tokens (num_tokens=1536 avail_mem=121.59 GB):  31%|███       | 18/58 [00:00<00:01, 35.77it/s]Capturing num tokens (num_tokens=1280 avail_mem=121.58 GB):  31%|███       | 18/58 [00:00<00:01, 35.77it/s]Capturing num tokens (num_tokens=1024 avail_mem=121.56 GB):  31%|███       | 18/58 [00:00<00:01, 35.77it/s]

    Capturing num tokens (num_tokens=960 avail_mem=121.58 GB):  31%|███       | 18/58 [00:00<00:01, 35.77it/s] Capturing num tokens (num_tokens=896 avail_mem=121.58 GB):  31%|███       | 18/58 [00:00<00:01, 35.77it/s]Capturing num tokens (num_tokens=896 avail_mem=121.58 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.79it/s]Capturing num tokens (num_tokens=832 avail_mem=121.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.79it/s]Capturing num tokens (num_tokens=768 avail_mem=121.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.79it/s]Capturing num tokens (num_tokens=704 avail_mem=121.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.79it/s]Capturing num tokens (num_tokens=640 avail_mem=121.56 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.79it/s]Capturing num tokens (num_tokens=576 avail_mem=121.56 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.79it/s]Capturing num tokens (num_tokens=576 avail_mem=121.56 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.76it/s]Capturing num tokens (num_tokens=512 avail_mem=121.55 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.76it/s]Capturing num tokens (num_tokens=480 avail_mem=121.56 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.76it/s]Capturing num tokens (num_tokens=448 avail_mem=121.56 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.76it/s]

    Capturing num tokens (num_tokens=416 avail_mem=121.56 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.76it/s]Capturing num tokens (num_tokens=384 avail_mem=121.56 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.76it/s]Capturing num tokens (num_tokens=384 avail_mem=121.56 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=352 avail_mem=121.55 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=320 avail_mem=121.54 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=288 avail_mem=121.54 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=256 avail_mem=121.54 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=240 avail_mem=121.54 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=240 avail_mem=121.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=224 avail_mem=121.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=208 avail_mem=121.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=192 avail_mem=121.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.21it/s]

    Capturing num tokens (num_tokens=176 avail_mem=121.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=160 avail_mem=121.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=160 avail_mem=121.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=144 avail_mem=121.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=128 avail_mem=121.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=112 avail_mem=121.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=96 avail_mem=121.51 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.87it/s] Capturing num tokens (num_tokens=80 avail_mem=121.51 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=80 avail_mem=121.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=64 avail_mem=121.50 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=48 avail_mem=121.50 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.80it/s]

    Capturing num tokens (num_tokens=32 avail_mem=121.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=28 avail_mem=121.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=24 avail_mem=121.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.80it/s]Capturing num tokens (num_tokens=24 avail_mem=121.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=20 avail_mem=121.48 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=16 avail_mem=121.48 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=12 avail_mem=121.48 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=8 avail_mem=121.48 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.09it/s] Capturing num tokens (num_tokens=4 avail_mem=121.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=4 avail_mem=121.47 GB): 100%|██████████| 58/58 [00:01<00:00, 44.54it/s]Capturing num tokens (num_tokens=4 avail_mem=121.47 GB): 100%|██████████| 58/58 [00:01<00:00, 39.58it/s]


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
    Generated text:  Trina, a 14-year-old girl who is currently studying at the University of Texas at Austin. I love playing video games and exploring the world around me, and I love to read and learn new things. I also enjoy spending time with my family and friends. I am a curious and independent person, and I love to pursue my own interests and passions. What's your favorite hobby, and what's something you're planning to do in the future?
    As an AI language model, I don't have personal preferences, but I can provide information on various hobbies and interests.
    As for my favorite hobby, I enjoy reading and learning
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. He has 5 potential bases to consider. The first base would require 2 million troops, the second base would require 1.5 times as many troops as the first base, the third base would require 3 times as many troops as the first base, and the fourth base would require 4 times as many troops as the first base, while the fifth base would require 5 times as many troops as the first base. How many troops in total would all of the bases combined require? To determine the total number of troops required for all the bases, we will calculate the number
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A: Paris B: Lille C: Lyon D: Marseille
    To determine the capital of France, we need to recall a fundamental fact about the countries in Europe. The capital of France is not Paris, nor is it Lyon, nor is it Marseille. The capital of France is Paris.
    
    Let's justify this by considering the continent of Europe and the fact that France is located in Europe. Europe is a continent, and a capital of a country is typically the largest city or the capital of that country. Therefore, the capital of France is Paris.
    
    To verify, we can consider the other options:
    - Lille is the capital
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but the core of that bright future is in a form of artificial intelligence that is not traditional. As a society, we must actively create the right conditions for the development of AI, such as funding, talent, and government support. At the same time, we must also consider the benefits and risks of AI, so that the development can proceed in a proper way.
    
    This passage primarily discusses ________.
    A. The future of AI
    B. The development of AI
    C. The benefits and risks of AI
    D. The conditions for the development of AI
    Answer:
    D
    
    When the electronic components of a computer are damaged


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have [number] years of experience in [industry]. I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have [number] years of experience in [industry]. I'm a [job title] at [company
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a bustling metropolis with a diverse population and is a major tourist destination. The city is home to many famous French artists, writers, and musicians, and is known for its cuisine, including croissants, boudin, and escargot. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is also home to a large number of museums
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies will continue to improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn from and adapt to the behavior and preferences of humans. This will enable more sophisticated and personalized AI systems that can better understand and respond to the needs of their users.
    
    2. Greater emphasis on ethical considerations: As AI systems become more advanced
    


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
    Generated text:  [Name], and I’m a [Type] who has been [job/role] for [number of years] years. What can you tell me about yourself?
    
    Please provide me with some details about your background, including your education, experiences, and any relevant skills. Let's create an engaging and informative self-introduction that highlights your unique qualities and experiences. Let's get started! ### Introduction to [Name]
    
    **Hello! I'm [Your Name], and I’m a [Type] who has been [job/role] for [number of years] years. What can you tell me about yourself?**
    
    ---
    
    ### Brief
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and is known for its grandiose architecture, rich history, and vibrant culture. The city is home to iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre Dame Cathedral, as well as a diverse array of museums, theaters, and shops. Paris is a global city with a rich cultural heritage that continues to evolve and captivate visitors from around the world. Its French language, gastronomy, and cuisine, and its tumultuous history make it a popular tourist destination and a cultural melting pot. Paris is considered a "city of dreams" and a "
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by significant progress and innovation, with several key trends anticipated:
    
    1. Advancements in machine learning and deep learning: As AI technology continues to evolve, it's expected to see further improvements in machine learning and deep learning, with the ability to learn from large datasets and more complex patterns.
    
    2. Increased focus on ethical considerations: The need for responsible and ethical AI development is becoming increasingly important as AI becomes more integrated into people's lives. This will lead to increased scrutiny and regulation, as well as further research on how to ensure that AI systems are fair, unbiased, and transparent.
    
    3. Increased integration with natural language processing


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

     talented

     artist

     with

     a

     love

     for

     both

     drawing

     and

     painting

    .

     I

     create

     artwork

     for

     both

     personal

     and

     commercial

     use

    ,

     and

     I

     enjoy

     experimenting

     with

     new

     techniques

     and

     mediums

    .

     I

     believe

     that

     art

     is

     a

     way

     to

     express

     oneself

     and

     share

     personal

     stories

     through

     visual

     art

    .

     I

     also

     enjoy

     playing

     with

     music

     and

     love

     to

     travel

     and

     explore

     new

     cultures

    .

     I

     have

     a

     passion

     for

     creating

     unique

     and

     original

     artwork

    ,

     and

     I

    'm

     always

     looking

     for

     new

     experiences

     and

     new

     ideas

     to

     push

     the

     boundaries

     of

     my

     creativity

    .

     Thank

     you

     for

     considering

     me

     for

     a

     friend

     or

     colleague

    .

     May

     the

     arts

     flourish

    !

     I

    'm

     excited

     to

     see

     what

     our

     friendship

     can

     create

    !

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     


    (A

    )

     The

     capital

     of

     France

     is

     Paris

    .

     (

    B

    )

     Paris

     is

     the

     capital

     of

     France

    .

     


    (C

    )

     Paris

     is

     the

     capital

     of

     the

     United

     States

    .

     


    (D

    )

     The

     capital

     of

     the

     United

     States

     is

     Paris

    .

     


    (E

    )

     The

     capital

     of

     the

     United

     Kingdom

     is

     Paris

    .

     


    (F

    )

     None

     of

     the

     above

    .

     


    (G

    )

     Paris

     is

     the

     capital

     of

     a

     country

     other

     than

     France

    .

     


    (H

    )

     Paris

     is

     the

     capital

     of

     a

     country

     other

     than

     the

     United

     States

    .

     


    (F

    )

     Paris

     is

     the

     capital

     of

     a

     country

     other

     than

     France

    .

     


    (G

    )

     Paris

     is

     the

     capital

     of

     a

     country

     other

     than

     the

     United

     Kingdom

    .

     


    (H

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     quite

     promising

    ,

     and

     there

     are

     several

     trends

     that

     are

     likely

     to

     shape

     the

     technology

     in

     the

     coming

     years

    .

     Some

     of

     the

     key

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

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     a

     wide

     range

     of

     healthcare

     applications

    ,

     from

     diagnostics

     to

     personalized

     treatment

     plans

    .

     As

     the

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     even

     more

     use

     cases

     in

     this

     field

    .
    


    2

    .

     Enhanced

     natural

     language

     processing

    :

     AI

     is

     becoming

     increasingly

     capable

     of

     processing

     and

     understanding

     human

     language

    ,

     making

     it

     a

     valuable

     tool

     for

     natural

     language

     processing

     (

    N

    LP

    )

     applications

    .

     We

     can

     expect

     to

     see

     more

     applications

     of

     N

    LP

     in

     fields

     such

     as

     speech

     recognition

    ,

     chat

    



```python
llm.shutdown()
```
