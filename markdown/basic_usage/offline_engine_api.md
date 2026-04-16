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
    [2026-04-16 05:34:15] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.96it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.96it/s]


    2026-04-16 05:34:19,343 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 05:34:19] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:39,  2.81s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:30,  1.79it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:30,  1.79it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.50it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.73it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.73it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.73it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.73it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.73it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.73it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.73it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.73it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.73it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.73it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.73it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.73it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.73it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.73it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.73it/s]

    Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:01, 24.73it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 31.54it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 31.54it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 31.54it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 31.54it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 31.54it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 31.54it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 31.54it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 31.54it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 38.05it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 38.05it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 38.05it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 38.05it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 38.05it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 38.05it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 38.05it/s]

    Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 38.05it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 44.47it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 44.47it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 44.47it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 44.47it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 44.47it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 44.47it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   2%|▏         | 1/58 [00:00<00:06,  8.19it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   2%|▏         | 1/58 [00:00<00:06,  8.19it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:09,  5.96it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:09,  5.96it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:09,  5.96it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:06,  8.81it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:06,  8.81it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:06,  8.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:06,  8.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.34it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.34it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.34it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.00it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.72 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.71 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.53it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.71 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.53it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.70 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.53it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 24.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 24.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.68 GB):  31%|███       | 18/58 [00:00<00:01, 24.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.18 GB):  31%|███       | 18/58 [00:01<00:01, 24.52it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  31%|███       | 18/58 [00:01<00:01, 24.52it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  31%|███       | 18/58 [00:01<00:01, 24.52it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.34it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.34it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.34it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.34it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.34it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.34it/s]

    Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.04it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.91it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.91it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.91it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.91it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.91it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.91it/s]

    Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.91it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=208 avail_mem=76.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=192 avail_mem=76.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.83it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.83it/s]

    Capturing num tokens (num_tokens=48 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=48 avail_mem=75.97 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.72it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 32.43it/s]


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
    Generated text:  Jennifer and I am a regular contributor to my local community radio station. I have an MBA in finance and corporate strategy from the University of Southern California and have been on the road in South America to do business development with financial institutions and small businesses. In my spare time, I enjoy playing guitar and listening to music.
    The disc jockey, Jennifer, is a seasoned professional in the entertainment industry and has worked in a variety of roles such as radio, television, and online media. She has extensive experience in business development and financial strategy, which has allowed her to pivot her career and gain a deep understanding of the music industry.
    In addition to
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office. It is the most powerful office in the government and holds the power to appoint federal judges, and to be the head of the executive branch of the government. He/she also appoints ambassadors, judges, and other federal officials.
    The position is chosen by the President. The term of office is usually 4 years.
    Each state gets a number of representatives on the US Senate based on their population. Each senator gets a certain number of votes based on the size of the state, and the state with the most population gets the most votes. The number of votes is divided up based on the number of seats for each state.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The French name of Paris is Paris. Paris is the capital of France, but it is also the capital of another country, namely Luxembourg. Now, if we list the cities in the order of their capital's name, then the 6th city in the list would be Luxembourg. The French name of Luxembourg is Luxembourg. Luxembourg is the capital of France. Which of the following statements is incorrect?
    
    A: Paris is the capital of France.
    
    B: Luxembourg is the capital of France.
    
    C: Paris is the capital of Luxembourg.
    
    D: There are two capital cities of France, Paris and Luxembourg. To determine which statement is
    ===============================
    Prompt: The future of AI is
    Generated text:  fast and changing. Do you want to know what the future of AI will be like? Do you want to know what the future of AI will be like? Do you want to know what the future of AI will be like? Do you want to know what the future of AI will be like? Can you imagine a world where the lines between human and machine are blurred, where machines are as intelligent as humans? With the rapid development of AI technology, the future of AI is fascinating. But how to deal with the challenges of AI technology? What will the future of AI be like? Here's our answer. However, one thing is


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


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history dating back to the Middle Ages. It is located in the north of the country and is the largest city in France by population. The city is known for its beautiful architecture, vibrant culture, and annual festivals such as the Eiffel Tower and the Louvre Museum. Paris is also home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is a major economic and cultural center in France and is known for its role in the French Revolution and the French Revolution. It is also a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This will lead to the creation of more efficient and cost-effective solutions, as well as the creation of new jobs in areas such as manufacturing and logistics.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues
    


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
    Generated text:  [Name] and I am [Age]. I am an [occupation] who has been in the [field] for [number of years] years. I am passionate about [occupation] because I enjoy [reason for passion]. [Name] has a [number of years] year professional experience in the [occupation]. [Name] has a unique ability that has helped me achieve my goals. I am [description of ability]. I am always looking for new challenges and opportunities to learn and grow. I have a strong work ethic and always aim to make a positive impact in the world. I enjoy [occupation] because it gives me a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a sprawling metropolis with a rich cultural history dating back to Roman times. The city is located in the Île de France region and is the largest city in France, with a population of around 17.5 million. Paris is known for its unique architecture, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its food, music, and fashion. The city is a cultural and economic hub, and Paris is the world's third most visited city after New York City and Los Angeles. It is known for its flamboyant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very exciting and potentially transformative. Here are some possible trends that could shape the technology in the next decade:
    
    1. Increased Integration of AI into Everyday Life: AI is already being integrated into everyday life, from self-driving cars to virtual assistants like Siri and Alexa. We can expect more integration into our daily lives in the future, such as smart homes, robots in manufacturing, and more advanced healthcare technologies.
    
    2. Enhanced Personalization: As AI becomes more integrated into daily life, we can expect to see more personalized experiences. For example, AI-powered chatbots can provide personalized recommendations, while AI-powered virtual assistants can tailor their conversations to individual users


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

     am

     an

     AI

     language

     model

    .


    I

     am

     here

     to

     help

     you

     with

     any

     questions

     or

     concerns

     you

     may

     have

    .


    If

     you

     need

     anything

     specific

    ,

     don

    't

     hesitate

     to

     ask

    .

     I

    'm

     always

     here

     to

     assist

     you

    .


    Let

    's

     get

     started

    !

     How

     can

     I

     assist

     you

     today

    ?

     


    [

    Name

    ]

     AI

     Language

     Model

    .

     I

    'm

     a

     language

     model

     AI

     that

     can

     assist

     you

     with

     any

     question

     or

     concern

     you

     may

     have

    .

     I

    'm

     here

     to

     help

     you

     with

     any

     task

     or

     information

     you

     need

    .

     If

     you

     have

     any

     questions

     or

     concerns

    ,

     please

     feel

     free

     to

     ask

     me

    .

     I

    'm

     here

     to

     help

     you

    !

     [

    Name

    ]

     AI

     Language

     Model

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     and

     seat

     of

     government

     of

     France

     and

     the

     largest

     metropolitan

     area

     in

     the

     world

    .

     It

     is

     located

     on

     the

     Se

    ine

     River

     and

     the

     Î

    le

     de

     France

    .

     It

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     a

     thousand

     glass

    ".

     The

     city

     is

     a

     center

     of

     culture

    ,

     arts

    ,

     design

    ,

     and

     technology

     and

     is

     one

     of

     the

     world

    's

     most

     influential

     cities

    .

     Paris

     is

     home

     to

     many

     well

    -known

     museums

    ,

     theaters

    ,

     and

     art

     galleries

    ,

     as

     well

     as

     a

     vibrant

     nightlife

     and

     a

     long

     history

     dating

     back

     to

     Roman

     times

    .

     It

     is

     also

     a

     popular

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     annually

    .

     Its

     status

     as

     a

     global

     capital

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     constantly

     evolving

    .

     Here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     the

     field

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     concerns

     about

     AI

    's

     impact

     on

     society

     and

     the

     environment

     grow

    ,

     there

     will

     be

     increased

     focus

     on

     developing

     ethical

     guidelines

     and

     standards

     for

     AI

     systems

    .

     This

     could

     involve

     creating

     guidelines

     for

     AI

     developers

     to

     follow

     in

     order

     to

     ensure

     that

     their

     systems

     are

     safe

    ,

     transparent

    ,

     and

     accountable

    .
    


    2

    .

     Greater

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

     to

     improve

     the

     accuracy

     of

     diagnoses

     and

     treatment

     decisions

     in

     healthcare

    .

     As

     AI

     becomes

     more

     advanced

    ,

     it

     could

     be

     used

     to

     improve

     the

     accuracy

     of

     medical

     procedures

     and

     to

    



```python
llm.shutdown()
```
