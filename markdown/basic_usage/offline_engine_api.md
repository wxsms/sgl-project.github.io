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
    [2026-04-27 05:06:14] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.85it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.84it/s]


    2026-04-27 05:06:18,936 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-27 05:06:18] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:29,  4.73s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.12it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.12it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.12it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.74it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.46it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.21it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.21it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.21it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.21it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.21it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.21it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.21it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.21it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.21it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.21it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 39.08it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.65 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.64 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.64 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.64 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=121.64 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.63 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=5120 avail_mem=121.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.05it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3584 avail_mem=121.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=3328 avail_mem=121.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=121.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=121.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=121.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=2560 avail_mem=121.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=121.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=121.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.96it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=121.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=121.59 GB):  31%|███       | 18/58 [00:00<00:01, 27.76it/s]Capturing num tokens (num_tokens=1536 avail_mem=121.59 GB):  31%|███       | 18/58 [00:00<00:01, 27.76it/s]Capturing num tokens (num_tokens=1280 avail_mem=121.59 GB):  31%|███       | 18/58 [00:00<00:01, 27.76it/s]Capturing num tokens (num_tokens=1024 avail_mem=121.57 GB):  31%|███       | 18/58 [00:00<00:01, 27.76it/s]Capturing num tokens (num_tokens=960 avail_mem=121.58 GB):  31%|███       | 18/58 [00:00<00:01, 27.76it/s] Capturing num tokens (num_tokens=896 avail_mem=121.58 GB):  31%|███       | 18/58 [00:00<00:01, 27.76it/s]Capturing num tokens (num_tokens=896 avail_mem=121.58 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.45it/s]Capturing num tokens (num_tokens=832 avail_mem=121.57 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.45it/s]Capturing num tokens (num_tokens=768 avail_mem=121.57 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.45it/s]Capturing num tokens (num_tokens=704 avail_mem=121.57 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.45it/s]

    Capturing num tokens (num_tokens=640 avail_mem=121.56 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.45it/s]Capturing num tokens (num_tokens=576 avail_mem=121.56 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.45it/s]Capturing num tokens (num_tokens=576 avail_mem=121.56 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.00it/s]Capturing num tokens (num_tokens=512 avail_mem=121.55 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.00it/s]Capturing num tokens (num_tokens=480 avail_mem=121.56 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.00it/s]Capturing num tokens (num_tokens=448 avail_mem=121.56 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.00it/s]Capturing num tokens (num_tokens=416 avail_mem=121.56 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.00it/s]Capturing num tokens (num_tokens=384 avail_mem=121.56 GB):  48%|████▊     | 28/58 [00:00<00:00, 36.00it/s]Capturing num tokens (num_tokens=384 avail_mem=121.56 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=352 avail_mem=121.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=320 avail_mem=121.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.66it/s]

    Capturing num tokens (num_tokens=288 avail_mem=121.54 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=256 avail_mem=121.54 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=240 avail_mem=121.54 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=240 avail_mem=121.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=224 avail_mem=121.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=208 avail_mem=121.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=192 avail_mem=121.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=176 avail_mem=121.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=160 avail_mem=121.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.52it/s]Capturing num tokens (num_tokens=160 avail_mem=121.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=144 avail_mem=121.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=128 avail_mem=121.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.91it/s]

    Capturing num tokens (num_tokens=112 avail_mem=121.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=96 avail_mem=121.51 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.91it/s] Capturing num tokens (num_tokens=80 avail_mem=121.51 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=80 avail_mem=121.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.36it/s]Capturing num tokens (num_tokens=64 avail_mem=121.50 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.36it/s]Capturing num tokens (num_tokens=48 avail_mem=121.50 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.36it/s]Capturing num tokens (num_tokens=32 avail_mem=121.50 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.36it/s]Capturing num tokens (num_tokens=28 avail_mem=121.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.36it/s]Capturing num tokens (num_tokens=24 avail_mem=121.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.36it/s]Capturing num tokens (num_tokens=24 avail_mem=121.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.06it/s]Capturing num tokens (num_tokens=20 avail_mem=121.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.06it/s]

    Capturing num tokens (num_tokens=16 avail_mem=121.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.06it/s]Capturing num tokens (num_tokens=12 avail_mem=121.48 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.06it/s]Capturing num tokens (num_tokens=8 avail_mem=121.48 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.06it/s] Capturing num tokens (num_tokens=4 avail_mem=121.48 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.06it/s]Capturing num tokens (num_tokens=4 avail_mem=121.48 GB): 100%|██████████| 58/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=4 avail_mem=121.48 GB): 100%|██████████| 58/58 [00:01<00:00, 36.78it/s]


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
    Generated text:  Jonathan and I'm a new student at the University of Florida.
    I love to stay active and socialize. I love to go to the beach and spend my weekends lounging in the sun. I love the feeling of independence and freedom. I also enjoy the outdoors and love exploring all the different places and people in Florida. I love to cook and enjoy trying new recipes, whether it’s for my own enjoyment or as a way to teach my friends. I'm always looking for new experiences and adventures to take me around the world. I enjoy going to different places such as Mexico and Argentina. I love to do all sorts of different activities
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in our country. He or she is the leader of the country. He or she is in charge of most of the country's business. He or she is usually a very wealthy man. He or she has a lot of money. He or she also has a lot of power. Presidents are different from other people because they can make all the decisions in the country. They are the most important people in our country. The president can make all the important business decisions. He or she can make all the important decisions about schools, jobs, and money. The president can make all the important decisions about the army and the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. There are 5, 000 inhabitants in a square city with 100 buildings. A company is now trying to locate a location for a new office. The only two possible locations are the city center and the perimeter of the city.
    
    The capital of France is located inside the city, and the city is rectangular in shape. The perimeter of the city is 2000 m and the city center has a population density of 0.3 people per square meter. The perimeter of the city is 2000 meters, and the city center has a population density of 0.3 people
    ===============================
    Prompt: The future of AI is
    Generated text:  not only about complex algorithms, but also about simplification, particularly in how humans interact with AI. This simplification helps individuals, organizations, and governments to better understand and interact with AI, which can lead to better decision-making and innovation.
    One example of how simplification can help with AI is in natural language processing. Natural language processing involves using AI to understand and interpret human language, and it can be particularly helpful in fields such as healthcare, education, and social media. By simplifying the language that people use in these fields, AI can help to improve the accuracy and reliability of the data that is analyzed.
    Another example of how simpl


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your job or experience here]. I enjoy [insert a short description of your hobbies or interests here]. What do you like to do in your free time? I enjoy [insert a short description of your hobbies or interests here]. What's your favorite hobby or activity? I love [insert a short description of your favorite hobby or activity here]. What's your favorite book or movie? I love [insert a short
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic hub, known for its rich history, art, and cuisine. It is a popular tourist destination and a major economic and political center in Europe. The city is also home to the French Riviera, a popular tourist destination for its beaches and luxury resorts. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of people, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision, to create more complex and sophisticated systems.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. More personalized and context-aware AI: AI will become more personalized and context-aware, allowing systems to learn from individual users and adapt to their needs.
    
    4. Increased use of AI in
    


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
    Generated text:  [Name]. I'm a [职业] who is passionate about [职业] and has been [number of years in the job]. I've always been [description of what you'd call you], and I enjoy [description of what you'd call your interests and hobbies]. I'm always [positive adjective], and I'm always looking for new ways to grow as a [职业]. I'm [occupation level] in my job, which is [percentage of the time I spend on this job]. What's your occupation? What brings you to this position? [What you say about yourself] How do you see your career evolving? [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city and the nation’s political, cultural, and economic center. It is known for its stunning architecture, diverse cuisine, and rich history. The city is also famous for its annual "Les Misérables" opera festival and for its iconic Eiffel Tower. It is home to the Louvre Museum, the Notre-Dame Cathedral, and many other landmarks. Paris has a long and rich history dating back to the 13th century and is an important part of European culture and history. It is one of the most visited cities in the world and is a major destination for tourists. The city is also home to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a number of factors, including advances in computing power, the development of more advanced machine learning algorithms, and the integration of AI into a wider range of industries. Here are some possible future trends in AI:
    
    1. Increased focus on ethics and accountability: As more AI systems become integrated into our daily lives, there will be increased pressure to ensure that they are developed and used responsibly and ethically. This will likely lead to greater focus on developing AI systems that are transparent, accountable, and meet ethical standards.
    
    2. Greater use of AI for healthcare: AI has the potential to revolutionize healthcare by improving diagnosis, treatment


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

     and

     I

    'm

     a

     [

    job

     title

    ]

     with

     [

    previous

     job

     title

    ]

     experience

    .

     I

     have

     a

     passion

     for

     [

    occupation

    ],

     and

     I

     enjoy

     [

    job

     responsibilities

     or

     skills

    ].

     I

     have

     always

     been

     drawn

     to

     the

     outdoors

     and

     have

     spent

     a

     lot

     of

     time

     outside

    .

     I

     also

     enjoy

     [

    other

     hobbies

     or

     interests

    ].

     I

     am

     always

     looking

     for

     new

     experiences

     to

     add

     to

     my

     life

    ,

     and

     I

    'm

     looking

     forward

     to

     exploring

     and

     learning

     more

     about

     the

     world

    .

     How

     can

     you

     describe

     your

     interests

     and

     hobbies

    ?

     As

     an

     outdoor

     enthusiast

     and

     avid

     reader

    ,

     I

     enjoy

     spending

     time

     outdoors

    ,

     whether

     it

    's

     hiking

    ,

     kay

    aking

    ,

     or

     simply

     wandering

     through

     nature

    .

     I

     also

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     French

     capital

     city

    ,

     Paris

    ,

     is

     known

     for

     its

     rich

     history

    ,

     vibrant

     culture

    ,

     and

     beautiful

     architecture

    .

     The

     city

     is

     located

     on

     the

     western

     bank

     of

     the

     River

     Se

    ine

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

     a

     center

     for

     the

     French

     film

     industry

    ,

     and

     the

     city

     has

     a

     strong

     cultural

     and

     artistic

     scene

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     is

     often

     referred

     to

     as

     the

     "

    city

     of

     love

    "

     due

     to

     its

     beautiful

     beaches

     and

     romantic

     ambiance

    .

     Paris

     is

     home

     to

     many

     famous

     landmarks

     and

     attractions

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

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     advances

     in

     computing

     power

    ,

     improvements

     in

     data

     and

     machine

     learning

     techniques

    ,

     and

     emerging

     technologies

     such

     as

     quantum

     computing

     and

     bi

    otechnology

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     collaboration

     between

     humans

     and

     machines

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     it

     is

     likely

     to

     become

     more

     integrated

     with

     human

     AI

    ,

     allowing

     machines

     to

     learn

     and

     adapt

     to

     human

     needs

     and

     preferences

    .
    


    2

    .

     Greater

     emphasis

     on

     ethical

     and

     legal

     considerations

    :

     As

     AI

     becomes

     more

     ubiquitous

    ,

     there

     is

     a

     growing

     demand

     for

     transparency

     and

     accountability

     from

     AI

     systems

    .

     Governments

     and

     organizations

     may

     become

     more

     rigorous

     in

     their

     regulation

     and

     oversight

     of

     AI

     development

     and

     deployment

    



```python
llm.shutdown()
```
