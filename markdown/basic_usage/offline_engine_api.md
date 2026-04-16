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
    [2026-04-16 22:45:41] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.90it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.89it/s]


    2026-04-16 22:45:46,268 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 22:45:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]

    Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:10,  4.86it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:10,  4.86it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:10,  4.86it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:10,  4.86it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:10,  4.86it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:03<00:10,  4.86it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:03<00:10,  4.86it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:03<00:10,  4.86it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:03<00:10,  4.86it/s]Compiling num tokens (num_tokens=1792):  16%|█▌        | 9/58 [00:03<00:10,  4.86it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.89it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.89it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.89it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.89it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.89it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.89it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.89it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.89it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:03<00:03, 11.89it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=320):  45%|████▍     | 26/58 [00:03<00:01, 19.08it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 28.35it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 28.35it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 28.35it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 28.35it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 28.35it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 28.35it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 28.35it/s]

    Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 28.35it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 34.86it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 34.86it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 34.86it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 34.86it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 34.86it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 34.86it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 34.86it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 34.86it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 39.03it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 39.03it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 39.03it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 39.03it/s]

    Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 39.03it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 39.03it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 39.03it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 39.03it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 44.31it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 44.31it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 44.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=134.95 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=134.95 GB):   2%|▏         | 1/58 [00:00<00:12,  4.74it/s]Capturing num tokens (num_tokens=7680 avail_mem=134.92 GB):   2%|▏         | 1/58 [00:00<00:12,  4.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=134.91 GB):   2%|▏         | 1/58 [00:00<00:12,  4.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=134.91 GB):   5%|▌         | 3/58 [00:00<00:06,  9.14it/s]Capturing num tokens (num_tokens=6656 avail_mem=134.90 GB):   5%|▌         | 3/58 [00:00<00:06,  9.14it/s]Capturing num tokens (num_tokens=6144 avail_mem=134.89 GB):   5%|▌         | 3/58 [00:00<00:06,  9.14it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=134.89 GB):   9%|▊         | 5/58 [00:00<00:04, 12.65it/s]Capturing num tokens (num_tokens=5632 avail_mem=134.90 GB):   9%|▊         | 5/58 [00:00<00:04, 12.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=134.90 GB):   9%|▊         | 5/58 [00:00<00:04, 12.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=134.87 GB):   9%|▊         | 5/58 [00:00<00:04, 12.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=134.87 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=134.87 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.68it/s]Capturing num tokens (num_tokens=3840 avail_mem=134.86 GB):  14%|█▍        | 8/58 [00:00<00:02, 16.68it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=134.86 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.43it/s]Capturing num tokens (num_tokens=3584 avail_mem=134.85 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=134.85 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=134.85 GB):  21%|██        | 12/58 [00:00<00:02, 17.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=134.82 GB):  21%|██        | 12/58 [00:00<00:02, 17.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=134.82 GB):  21%|██        | 12/58 [00:00<00:02, 17.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=134.34 GB):  21%|██        | 12/58 [00:00<00:02, 17.13it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=134.34 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.50it/s]Capturing num tokens (num_tokens=2304 avail_mem=134.33 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.50it/s]Capturing num tokens (num_tokens=2048 avail_mem=134.22 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.50it/s]Capturing num tokens (num_tokens=1792 avail_mem=134.16 GB):  26%|██▌       | 15/58 [00:01<00:02, 19.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=134.15 GB):  26%|██▌       | 15/58 [00:01<00:02, 19.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=134.15 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.23it/s]Capturing num tokens (num_tokens=1280 avail_mem=134.14 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.23it/s]Capturing num tokens (num_tokens=1024 avail_mem=134.12 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.23it/s]Capturing num tokens (num_tokens=960 avail_mem=134.13 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.23it/s] Capturing num tokens (num_tokens=896 avail_mem=134.13 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.23it/s]

    Capturing num tokens (num_tokens=896 avail_mem=134.13 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=832 avail_mem=133.05 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=768 avail_mem=133.04 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=704 avail_mem=132.88 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=640 avail_mem=122.95 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.14it/s]Capturing num tokens (num_tokens=640 avail_mem=122.95 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.94it/s]Capturing num tokens (num_tokens=576 avail_mem=118.76 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.94it/s]Capturing num tokens (num_tokens=512 avail_mem=118.38 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.94it/s]Capturing num tokens (num_tokens=480 avail_mem=118.40 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.94it/s]

    Capturing num tokens (num_tokens=448 avail_mem=118.39 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.94it/s]Capturing num tokens (num_tokens=448 avail_mem=118.39 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=416 avail_mem=118.39 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=384 avail_mem=118.38 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=352 avail_mem=118.37 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=320 avail_mem=118.37 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=320 avail_mem=118.37 GB):  60%|██████    | 35/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=288 avail_mem=118.36 GB):  60%|██████    | 35/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=256 avail_mem=118.36 GB):  60%|██████    | 35/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=240 avail_mem=118.36 GB):  60%|██████    | 35/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=224 avail_mem=118.36 GB):  60%|██████    | 35/58 [00:01<00:00, 34.03it/s]

    Capturing num tokens (num_tokens=208 avail_mem=118.35 GB):  60%|██████    | 35/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=208 avail_mem=118.35 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=192 avail_mem=118.35 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=176 avail_mem=118.35 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=160 avail_mem=118.35 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=144 avail_mem=118.34 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=128 avail_mem=118.34 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.01it/s]Capturing num tokens (num_tokens=128 avail_mem=118.34 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=112 avail_mem=118.34 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=96 avail_mem=118.33 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.82it/s] Capturing num tokens (num_tokens=80 avail_mem=118.33 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.82it/s]

    Capturing num tokens (num_tokens=64 avail_mem=118.33 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=48 avail_mem=118.32 GB):  78%|███████▊  | 45/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=48 avail_mem=118.32 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=32 avail_mem=118.32 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=28 avail_mem=118.31 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=24 avail_mem=118.31 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=20 avail_mem=118.31 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=16 avail_mem=118.31 GB):  86%|████████▌ | 50/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=16 avail_mem=118.31 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=12 avail_mem=118.30 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.94it/s]Capturing num tokens (num_tokens=8 avail_mem=118.30 GB):  95%|█████████▍| 55/58 [00:01<00:00, 40.94it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=118.30 GB):  95%|█████████▍| 55/58 [00:02<00:00, 40.94it/s]Capturing num tokens (num_tokens=4 avail_mem=118.30 GB): 100%|██████████| 58/58 [00:02<00:00, 28.67it/s]


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
    Generated text:  Lisa.
    We'll look at the text below.
    John left on a plane. The next day, he traveled to another city. The next day, he bought a ticket on a different plane. What city did he travel to?
    John traveled to a different city. The next day, he bought a ticket on a different plane. Based on the information provided, John traveled to a different city and then bought a ticket on a different plane the following day. Therefore, the city he traveled to is the one he traveled to before making the purchase. However, without additional context, we cannot determine the specific city he traveled to before making the purchase
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He is in charge of all the other people. He has a lot of power. For example, he can veto any bill he wants. He can also sign any bill he wants. In addition, the president is the leader of the country. He has the power to make decisions about the country's policy. In addition, he can tell the public what to do. He can issue a statement to the public. Also, he can take any action which he wants to. All of these things show that the president is very powerful. They can do everything he wants. Which of the following statements is
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The place of death of Napoleon Bonaparte was the Place de la Concorde, on the Left Bank of the Seine in the city center. There, a monument to his grave was erected. The monument was erected by Louis Alexandre Martin, a man of the distinguished career of that time.
    He was born in 1808 and died in 1870. He was a brilliant politician. In 1848 he was the candidate of the extreme right wing for the presidency of the Assembly. In 1848 he was the President of the Committee of Public Safety. In 1
    ===============================
    Prompt: The future of AI is
    Generated text:  coming sooner than you think, and we already see signs of this in a variety of industries. But while we can have a fairly good idea of what AI is, there's still a lot to learn about the technology and its impact on society. Here are 15 interesting facts about AI that you may not have known about.
    1. The first AI was created by John McCarthy in 1956.
    2. It's harder to build a small computer than it is to build an entire planet.
    3. The first AI that was created using the same programming language was created by Alan Turing.
    4. The first AI that was


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the seat of the French government and the largest city in the European Union. Paris is known for its rich history, art, and culture, and is a popular tourist destination. The city is also home to many famous museums, including the Musée d'Orsay and the Musée Rodin. Paris is a vibrant and dynamic city with a rich cultural heritage that continues to inspire and captivate visitors from around the world. The city is also known for its fashion industry, with many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation: As AI continues to advance, we can expect to see more automation in various industries, including manufacturing, transportation, and healthcare. This will lead to increased efficiency, reduced costs, and improved productivity.
    
    2. Personalization: AI will enable more personalized experiences for users, with algorithms that can learn from user behavior and preferences to provide tailored recommendations and interactions.
    
    3. Ethical and responsible AI: As AI becomes more integrated into our daily lives,
    


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
    Generated text:  [name] and I'm a/an [job title or category] with over [number] of years of experience in [job title or field]. I'm excited to start a new journey in [job title or field] and am always eager to learn new things and take on new challenges. Thank you for taking the time to meet me, [name]. [Name] is a/an [job title or category] with over [number] of years of experience in [job title or field]. I'm excited to start a new journey in [job title or field] and am always eager to learn new things and take on new challenges
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is the most populous city and the third-largest city in the European Union. It is located on the Seine river and is known for its architecture, museums, and world-renowned landmarks. Paris is also an important cultural center and host to many museums, theaters, and opera houses. The city is home to a diverse population of 20 million people, making it one of the most populous cities in the world. Paris is a historic city with a rich history dating back to ancient times, and it is a UNESCO World Heritage site. The city is also a popular tourist destination, with millions of visitors each year. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bound to be exciting and transformative, with a number of trends expected to shape its trajectory. Some potential trends include:
    
    1. Increased use of AI in healthcare: AI will play a significant role in the healthcare industry, helping to diagnose and treat diseases more accurately and efficiently. AI-powered tools will be able to analyze large amounts of medical data quickly and provide personalized treatment plans.
    
    2. Improved efficiency in manufacturing: AI will help manufacturers optimize their production processes, reducing waste and improving quality control. AI-powered robots will be able to perform tasks more efficiently, such as assembly line operations or precision manufacturing.
    
    3. Increased use of AI in customer service:


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

     am

     a

     [

    role

    ]

     with

     [

    number

    ]

     of

     years

     experience

     in

     [

    field

     or

     hobby

    ].


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

    ],

     and

     I

     have

     been

     working

     hard

     to

     [

    what

     you

     do

    ].

     I

    'm

     a

     [

    reason

     for

     success

    ]

     with

     [

    how

     many

     years

     of

     experience

    ]

     and

     [

    field

     or

     hobby

    ].

     In

     my

     spare

     time

    ,

     I

     enjoy

     [

    what

     you

     do

    ],

     which

     I

     believe

     makes

     me

     [

    interest

     or

     passion

    ].

     What

    's

     your

     story

    ?

     As

     a

     [

    job

     title

    ]

     at

     [

    company

     name

    ],

     I

    've

     been

     working

     hard

     to

     [

    what

     you

     do

    ]

     for

     [

    number

    ]

     years

    .

     I

     believe

     my

     success

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     statement

     is

    :

     Paris

    ,

     the

     largest

     city

     in

     France

    ,

     is

     the

     capital

     of

     the

     country

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     wide

     range

     of

     exciting

     and

     unpredictable

     developments

    ,

     as

     well

     as

     an

     increasing

     focus

     on

     ethical

     and

     social

     implications

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

     autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     becoming

     increasingly

     common

     in

     our

     daily

     lives

    ,

     and

     the

     rise

     of

     AI

     in

     autonomous

     vehicles

     could

     revolution

    ize

     transportation

     and

     transportation

     systems

    .

     AI

     can

     help

     to

     make

     autonomous

     vehicles

     safer

    ,

     more

     efficient

    ,

     and

     reliable

    .
    


    2

    .

     Improved

     healthcare

    :

     AI

     can

     be

     used

     to

     help

     doctors

     and

     researchers

     to

     diagnose

     diseases

     more

     accurately

     and

     quickly

    ,

     and

     to

     develop

     new

     treatments

     and

     drugs

    .

     AI

     can

     also

     help

     to

     personalize

     treatment

     plans

     based

     on

     individual

     patient

     data

    ,

     improving

     patient

     outcomes

    



```python
llm.shutdown()
```
