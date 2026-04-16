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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-16 06:46:08] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.34it/s]


    2026-04-16 06:46:13,014 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 06:46:13] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:02<00:03, 12.27it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:02<00:03, 12.27it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:02<00:03, 12.27it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.27it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.27it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.27it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.27it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.27it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:03<00:03, 12.27it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 19.53it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 19.53it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 19.53it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 19.53it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 19.53it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 19.53it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 19.53it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 19.53it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 26.22it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 26.22it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 26.22it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 26.22it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 26.22it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 26.22it/s]

    Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 26.22it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:03<00:00, 26.22it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 32.62it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 32.62it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 32.62it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 32.62it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 32.62it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 32.62it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 32.62it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 32.62it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 38.81it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 38.81it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 38.81it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 38.81it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 38.81it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 38.81it/s]

    Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 38.81it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 38.81it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 38.81it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 47.11it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 47.11it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 47.11it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 47.11it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 14.76it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 14.76it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 14.76it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 15.66it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 15.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.75 GB):   7%|▋         | 4/58 [00:00<00:03, 15.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.75 GB):  10%|█         | 6/58 [00:00<00:03, 16.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.75 GB):  10%|█         | 6/58 [00:00<00:03, 16.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  10%|█         | 6/58 [00:00<00:03, 16.90it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  10%|█         | 6/58 [00:00<00:03, 16.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.78it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:02, 20.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.50it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.50it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.62it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.62it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.62it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:01<00:03, 12.95it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:01<00:03, 12.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.69 GB):  33%|███▎      | 19/58 [00:01<00:03, 12.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.69 GB):  36%|███▌      | 21/58 [00:01<00:02, 12.97it/s]Capturing num tokens (num_tokens=960 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:01<00:02, 12.97it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:01<00:02, 12.97it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:01<00:02, 12.97it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:01<00:02, 15.45it/s]Capturing num tokens (num_tokens=768 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:01<00:02, 15.45it/s]Capturing num tokens (num_tokens=704 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:01<00:02, 15.45it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:01<00:02, 15.45it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:01<00:02, 15.45it/s]

    Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 19.25it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:01<00:01, 19.25it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 19.25it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 19.25it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:01, 19.25it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.75it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.75it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.75it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.75it/s]Capturing num tokens (num_tokens=288 avail_mem=76.67 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.75it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.75it/s]

    Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 28.25it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 28.25it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 28.25it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 28.25it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 28.25it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:02<00:00, 28.25it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:02<00:00, 31.91it/s]Capturing num tokens (num_tokens=160 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:02<00:00, 31.91it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:02<00:00, 31.91it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:02<00:00, 31.91it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:02<00:00, 31.91it/s]

    Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  72%|███████▏  | 42/58 [00:02<00:00, 31.91it/s] Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:02<00:00, 34.72it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:02<00:00, 34.72it/s]Capturing num tokens (num_tokens=64 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:02<00:00, 34.72it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:02<00:00, 34.72it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:02<00:00, 34.72it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.18it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.18it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.18it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.18it/s]

    Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.18it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  88%|████████▊ | 51/58 [00:02<00:00, 35.18it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:02<00:00, 37.05it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:02<00:00, 37.05it/s] Capturing num tokens (num_tokens=4 avail_mem=76.60 GB):  97%|█████████▋| 56/58 [00:02<00:00, 37.05it/s]Capturing num tokens (num_tokens=4 avail_mem=76.60 GB): 100%|██████████| 58/58 [00:02<00:00, 23.18it/s]


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
    Generated text:  Mark, and I have been a great fan of The Godfather movies. But, I was a little confused about the setting for the The Godfather movie. When did the showings take place? The Godfather, based on the true story of the Italian-American mafia, was first filmed in New York City in 1969. The movie adaptation of the novel is set in a different time and place, with the showings taking place in the late 1950s in a small town called Claremont, California. The film's production began in 1972 and was completed in 197
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to print more paper or print fewer. The president knows that the amount of paper they use depends on the number of products they manufacture. If the president uses too much paper, they will have to print more. If the president uses too little paper, they will run out of paper. The president has a policy of printing 20% of their products at a time, and they have 100 products to manufacture.
    
    a) When the president uses 100 products, how many different ways can they decide how much paper to print?
    
    b) What is the minimum amount of paper they can print if
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. For the upcoming three months, the hotel prices in Paris are given by the function:
    
    $$
    f(x)=\begin{cases}
    5 & \text{if } x\leq 5 \\
    2x+3 & \text{if } x>5
    \end{cases}
    $$
    
    Calculate the total cost of staying in the hotel for three months, considering the changes in hotel prices from January to July. The prices are as follows:
    - January: Hotel price = 5 euros
    - February: Hotel price = 3 euros
    - March: Hotel price = 4 euros
    - April: Hotel
    ===============================
    Prompt: The future of AI is
    Generated text:  not just about how well it works, but how it is used by people. Can you predict what kind of AI will be used by humans in the future?
    
    Yes, I can predict that the future of AI will be used in a wide range of applications, from improving healthcare, to enhancing education, to improving transportation. AI will be used to analyze data, identify patterns and trends, and make predictions. It will also be used to automate tasks, freeing up human workers to focus on more complex and creative activities. Additionally, AI will be used to assist in decision-making, from making recommendations for customer service to optimizing supply chain logistics. Finally


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a major tourist destination. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a major tourist destination and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This includes issues such as bias, transparency, and accountability. AI developers will need to be more mindful of the potential impact of their technology on society.
    
    2. Integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more efficient and effective decision-making processes.
    
    3
    


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
    Generated text:  [Name] and I'm a [Job Title] at [Company]. I'm passionate about [Your Passion], a field or area of interest, and I'm always looking to learn more about it. My dedication to [Your Profession] is evident in my constant involvement in [Your Activity], and I'm always eager to share my knowledge and insights with others. What's your background and what led you to choose this particular role or field of interest? And most importantly, what's something you're looking forward to trying this year? Welcome, [Name], I'm excited to meet you and learn more about your passion and interests! Let
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital and largest city of France, and the second most populous city in the European Union after Rome. The city is known for its iconic Eiffel Tower, beautiful landmarks such as the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe, and its rich history dating back to the Roman Empire. It is also known as the "City of Light" and is the birthplace of several French national symbols such as the flag, the coat of arms of France, and the French national anthem "La Marseillaise". It is a world-renowned city that has a diverse population of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by significant advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in artificial intelligence:
    
    1. Increased use of AI in healthcare: AI will continue to play a crucial role in healthcare, as it can help doctors to diagnose diseases and develop personalized treatment plans. AI algorithms will be used to analyze large amounts of medical data to identify patterns and predict outcomes.
    
    2. Emergence of AI in finance: AI will be used in the finance industry to improve risk management, fraud detection, and portfolio optimization. AI-powered algorithms will be able to analyze financial data in real-time and


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

     Jane

    .

     I

    'm

     an

     experienced

     marketing

     professional

     with

     over

     

    1

    0

     years

     of

     experience

     in

     the

     industry

    .

     I

     have

     a

     passion

     for

     helping

     businesses

     succeed

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     improve

     my

     skills

     and

     knowledge

    .

     I

    'm

     confident

     in

     my

     ability

     to

     create

     and

     execute

     successful

     marketing

     campaigns

     and

     I

    'm

     committed

     to

     always

     striving

     for

     excellence

    .

     I

    'm

     looking

     forward

     to

     the

     opportunity

     to

     work

     with

     you

     and

     help

     you

     achieve

     your

     marketing

     goals

    .

     What

     do

     you

     do

     for

     a

     living

    ?

     I

    'm

     a

     marketing

     professional

     with

     over

     

    1

    0

     years

     of

     experience

     in

     the

     industry

    .

     I

     specialize

     in

     creating

     and

     executing

     successful

     marketing

     campaigns

     and

     helping

     businesses

     succeed

    .

     I

    'm

     committed

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     of

     love

    .

     Paris

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     romantic

     op

    ium

     shops

    ,

     and

     lively

     nightlife

    .

     The

     city

     is

     also

     home

     to

     the

     Lou

    vre

     Museum

    ,

     the

     E

    iff

    el

     Tower

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

     a

     city

     of

     contrasts

     and

     a

     hub

     of

     culture

     and

     entertainment

    .

     With

     its

     beautiful

     architecture

    ,

     rich

     history

    ,

     and

     diverse

     population

    ,

     it

     is

     a

     true

     Paris

    ian

     destination

    .

     Let

     me

     know

     if

     you

     need

     more

     details

     or

     have

     any

     other

     questions

    !

     **

    Paris

    ,

     the

     City

     of

     Love

    **

     is

     renowned

     for

     its

     iconic

     **

    E

    iff

    el

     Tower

    **,

     romantic

     **

    op

    ium

     shops

    **,

     and

     **

    l

    ively

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     advance

     rapidly

    ,

     driven

     by

     technological

     innovations

    ,

     policy

     changes

    ,

     and

     societal

     shifts

    .

     Some

     possible

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     complex

     and

     autonomous

    ,

     there

     is

     a

     growing

     recognition

     that

     it

     will

     need

     to

     adhere

     to

     ethical

     and

     social

     norms

    .

     This

     includes

     things

     like

     privacy

    ,

     fairness

    ,

     and

     transparency

    .
    


    2

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

     healthcare

     to

     improve

     patient

     outcomes

     and

     reduce

     costs

    .

     As

     AI

     becomes

     more

     advanced

    ,

     it

     could

     play

     a

     larger

     role

     in

     healthcare

    ,

     particularly

     in

     areas

     like

     diagnostics

    ,

     personalized

     medicine

    ,

     and

     patient

     care

     coordination

    .
    


    3

    .

     Greater

     integration

     with

    



```python
llm.shutdown()
```
