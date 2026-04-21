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
    [2026-04-21 03:35:04] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.63it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.63it/s]


    2026-04-21 03:35:08,551 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-21 03:35:08] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:26,  2.06it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:26,  2.06it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:26,  2.06it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:14,  3.54it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:14,  3.54it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:14,  3.54it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:14,  3.54it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=1792):  16%|█▌        | 9/58 [00:03<00:07,  6.14it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:02, 16.78it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:02, 16.78it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:02, 16.78it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:02, 16.78it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:02, 16.78it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:02, 16.78it/s]

    Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:02, 16.78it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:02, 16.78it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:03<00:02, 16.78it/s]Compiling num tokens (num_tokens=640):  31%|███       | 18/58 [00:03<00:02, 16.78it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 27.80it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 27.80it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 27.80it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 27.80it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 27.80it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 27.80it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 27.80it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 30.13it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 30.13it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 30.13it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 30.13it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 30.13it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 30.13it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 30.13it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:03<00:00, 30.13it/s]Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:03<00:00, 30.13it/s]Compiling num tokens (num_tokens=176):  57%|█████▋    | 33/58 [00:03<00:00, 30.13it/s]Compiling num tokens (num_tokens=160):  57%|█████▋    | 33/58 [00:03<00:00, 30.13it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 42.68it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:03<00:00, 42.68it/s]

    Compiling num tokens (num_tokens=24):  74%|███████▍  | 43/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 53.78it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 53.78it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 53.78it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 53.78it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 53.78it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 53.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.09 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.09 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.08 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.08 GB):   7%|▋         | 4/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.09 GB):   7%|▋         | 4/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.08 GB):   7%|▋         | 4/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   7%|▋         | 4/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.64it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.64it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.64it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.64it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.64it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=960 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.65it/s] Capturing num tokens (num_tokens=896 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=768 avail_mem=76.02 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=704 avail_mem=76.02 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=704 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.45it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.78it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.78it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.78it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.78it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.78it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.78it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:00<00:00, 44.34it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:00<00:00, 44.34it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:00<00:00, 44.34it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.34it/s]

    Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=160 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.72it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=64 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.72it/s]

    Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.50it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.67it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.67it/s] Capturing num tokens (num_tokens=4 avail_mem=75.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.67it/s]Capturing num tokens (num_tokens=4 avail_mem=75.93 GB): 100%|██████████| 58/58 [00:01<00:00, 39.74it/s]


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
    Generated text:  J.D. Do you happen to know what the meaning of this sentence: "You know this is going to be a long time to understand"? 
    Your response should be concise and include the word "mathematics".
    In mathematics, the phrase "You know this is going to be a long time to understand" expresses the idea that something is difficult or takes a long time to grasp, often through logical or analytical reasoning. It implies that the subject matter requires careful consideration and effort to fully comprehend, which can be a significant barrier to understanding. In the context of learning, this could refer to a challenging subject like mathematics that requires deep understanding and
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to hold an election in person or by video conference. To help him make an informed decision, he needs to understand the expected number of votes a candidate will receive if he holds an election in person. The candidate plans to receive 20% of the votes from the public in person. If the average voter turnout in the U.S. is 40%, how many more votes does the candidate need to get from the public in person to have an 80% chance of winning the election?
    To determine how many more votes the candidate needs to get from the public in person to have an 80%
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the capital city of the country of France, and it is located in the north of France. The climate of Paris is considered to be Mediterranean, which is the type of climate that makes the climate of Paris warm and sunny at most of the time.
    
    The city is known for its fashion, art, food, and history. Paris is the capital of France and was the capital of the French Republic. Paris is also one of the most important cities in the world of architecture and design.
    
    The city has a long history, dating back to Roman times. It was the capital of the Romans, and it was an important center
    ===============================
    Prompt: The future of AI is
    Generated text:  now
    
    What is the future of AI? That’s a question that’s been asked many times since the dawn of the AI age, and the answers have been mixed at best. The field of AI has seen a dizzying development since the 1980s and, as we mark the year 2021, it’s clear that the world of AI is going to be forever changing. Here’s how the future of AI looks like.
    
    AI is now critical to many industries and businesses
    
    As technology continues to evolve, AI is starting to take center stage in industries that haven’t traditionally been keen on the technology. The


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic center that has played a significant role in French history and continues to be a major hub for international trade and diplomacy. The city is also known for its rich history, including the influence of the Roman Empire, the French Revolution, and the French Revolution. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and artificial intelligence: As automation and AI continue to advance, we are likely to see more and more jobs automated, which could lead to a shift in the way we work and live. This could also lead to the development of new technologies that are more efficient and effective at performing tasks.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be an increased need
    


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
    Generated text:  [Your Name], and I'm a [job title] with over [number of years] years of experience in [field]. I'm always looking to learn and grow, so I'm excited to share my knowledge and experience with you. What can you tell me about your background or skills? Remember, this is a fictional character, so please fill in any details that might make the introduction more engaging and personalized. [Your Name] is a [type of job] with over [number of years] years of experience in [field], and I'm always seeking new challenges and learning opportunities. [Your Name] is always looking to grow
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest city and the country’s cultural and political capital.
    France's capital city, Paris, is renowned for its rich history, beautiful architecture, and vibrant cultural scene. The city is home to the Eiffel Tower, the Louvre Museum, the Louvre Pyramid, and numerous historical landmarks. Paris is also known for its unique French cuisine and its role in hosting major international events such as the Olympic Games and the World Cup. The city has a diverse population and is home to many important organizations and institutions, including the French Academy of Fine Arts and the National Library of France. The city's rich culture and history have made
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve several key trends:
    
    1. Increased automation and specialization: As AI becomes more complex, it will become more efficient at performing specific tasks, which will lead to greater specialization and automation. This will enable machines to perform tasks that were previously done by humans, such as financial forecasting, medical diagnosis, and engineering.
    
    2. Improved ethics and accountability: As AI systems become more sophisticated, it will become increasingly difficult to ensure their ethical and accountability. This will require a more robust set of rules and regulations to govern the development and use of AI.
    
    3. Enhanced privacy and security: AI systems will become more sensitive to the personal information that


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

    Your

     Name

    ],

     and

     I

    'm

     a

     [

    career

    ]

     who

     is

     passionate

     about

     [

    career

     goal

    ].

     I

    ’m

     confident

     and

     organized

    ,

     and

     have

     always

     been

     a

     person

     who

     thr

    ives

     on

     challenges

     and

     has

     a

     natural

     ability

     to

     solve

     problems

    .

     I

     enjoy

     sharing

     my

     knowledge

     and

     helping

     others

    ,

     and

     I

    'm

     always

     eager

     to

     learn

     new

     things

    .

     I

    'm

     a

     [

    career

    ]

     who

     is

     passionate

     about

     [

    career

     goal

    ].

     I

    'm

     confident

     and

     organized

    ,

     and

     have

     always

     been

     a

     person

     who

     thr

    ives

     on

     challenges

     and

     has

     a

     natural

     ability

     to

     solve

     problems

    .

     I

     enjoy

     sharing

     my

     knowledge

     and

     helping

     others

    ,

     and

     I

    'm

     always

     eager

     to

     learn

     new

     things

    .

     I

    'm

     a

     [

    career

    
    
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

     most

     populous

     urban

     area

    .

     It

     is

     a

     cultural

     and

     economic

     center

    ,

     located

     on

     the

     Mediterranean

     coast

    .

     Its

     history

     dates

     back

     to

     the

     

    6

    th

     century

     AD

    ,

     when

     it

     was

     a

     site

     of

     ancient

     Roman

     and

     Frank

    ish

     kingdoms

    .

     Paris

     is

     noted

     for

     its

     iconic

     landmarks

    ,

     such

     as

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

     Dame

     Cathedral

    .

     The

     city

     is

     also

     famous

     for

     its

     art

    ,

     literature

    ,

     and

     fashion

    ,

     attracting

     visitors

     from

     around

     the

     world

    .

     Paris

     is

     known

     for

     its

     cuisine

    ,

     including

     its

     famous

     burgers

    ,

     fries

    ,

     and

     cro

    iss

    ants

    .

     It

     is

     a

     popular

     tourist

     destination

    ,

     with

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     fascinating

     and

     rapidly

     evolving

    ,

     with

     potential

     developments

     and

     challenges

     that

     will

     shape

     the

     field

    's

     trajectory

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

     **

    AI

     Automation

     and

     Self

    -

    Driving

     Cars

    **:

     AI

     is

     likely

     to

     become

     more

     integrated

     with

     everyday

     life

    ,

     potentially

     leading

     to

     self

    -driving

     cars

     that

     can

     operate

     autonom

    ously

    .

     These

     vehicles

     would

     be

     able

     to

     adapt

     to

     different

     conditions

     and

     environments

    ,

     reducing

     the

     need

     for

     human

     drivers

    .
    


    2

    .

     **

    Advanced

     Machine

     Learning

     and

     Deep

     Learning

    **:

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     learn

     and

     adapt

    ,

     leading

     to

     even

     more

     sophisticated

     models

    .

     This

     includes

     the

     development

     of

     new

     algorithms

     that

     are

     better

     at

     handling

     complex

     data

     and

     recognizing

     patterns

     that

    



```python
llm.shutdown()
```
