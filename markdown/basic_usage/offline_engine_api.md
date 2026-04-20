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
    [2026-04-20 10:02:09] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.08it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.07it/s]


    2026-04-20 10:02:14,748 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-20 10:02:14] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:44,  2.88s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:24,  2.20it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:24,  2.20it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:24,  2.20it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:24,  2.20it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:24,  2.20it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:03<00:24,  2.20it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:07,  6.42it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:07,  6.42it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:07,  6.42it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:07,  6.42it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:07,  6.42it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:07,  6.42it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:07,  6.42it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:07,  6.42it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:07,  6.42it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:07,  6.42it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.14it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.14it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.14it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.14it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.14it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.14it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.14it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.14it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.14it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 20.07it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 20.07it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 20.07it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 20.07it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 20.07it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 20.07it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 20.07it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 20.07it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 20.07it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 20.07it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 28.75it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 28.75it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 28.75it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 28.75it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 28.75it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 28.75it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 28.75it/s]

    Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 28.75it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 28.75it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:03<00:00, 28.75it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 37.73it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 46.86it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 46.86it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 46.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=122.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=122.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=122.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]Capturing num tokens (num_tokens=7168 avail_mem=122.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=122.37 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=122.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=122.38 GB):   9%|▊         | 5/58 [00:00<00:02, 20.71it/s]Capturing num tokens (num_tokens=5632 avail_mem=122.37 GB):   9%|▊         | 5/58 [00:00<00:02, 20.71it/s]Capturing num tokens (num_tokens=5120 avail_mem=122.37 GB):   9%|▊         | 5/58 [00:00<00:02, 20.71it/s]Capturing num tokens (num_tokens=4608 avail_mem=122.37 GB):   9%|▊         | 5/58 [00:00<00:02, 20.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=122.37 GB):   9%|▊         | 5/58 [00:00<00:02, 20.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=122.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=122.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.23it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=122.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=122.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=122.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=122.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=122.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=2560 avail_mem=122.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=2304 avail_mem=122.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=2048 avail_mem=122.34 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=2048 avail_mem=122.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=122.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.83it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=122.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=122.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.83it/s]Capturing num tokens (num_tokens=1024 avail_mem=122.31 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.83it/s]Capturing num tokens (num_tokens=1024 avail_mem=122.31 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.63it/s]Capturing num tokens (num_tokens=960 avail_mem=122.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.63it/s] Capturing num tokens (num_tokens=896 avail_mem=122.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.63it/s]Capturing num tokens (num_tokens=832 avail_mem=122.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.63it/s]

    Capturing num tokens (num_tokens=768 avail_mem=122.31 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.63it/s]Capturing num tokens (num_tokens=768 avail_mem=122.31 GB):  43%|████▎     | 25/58 [00:00<00:01, 28.38it/s]Capturing num tokens (num_tokens=704 avail_mem=122.35 GB):  43%|████▎     | 25/58 [00:00<00:01, 28.38it/s]Capturing num tokens (num_tokens=640 avail_mem=122.35 GB):  43%|████▎     | 25/58 [00:00<00:01, 28.38it/s]Capturing num tokens (num_tokens=576 avail_mem=122.35 GB):  43%|████▎     | 25/58 [00:00<00:01, 28.38it/s]Capturing num tokens (num_tokens=512 avail_mem=121.39 GB):  43%|████▎     | 25/58 [00:00<00:01, 28.38it/s]Capturing num tokens (num_tokens=480 avail_mem=121.03 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.38it/s]Capturing num tokens (num_tokens=480 avail_mem=121.03 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.82it/s]Capturing num tokens (num_tokens=448 avail_mem=121.03 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.82it/s]

    Capturing num tokens (num_tokens=416 avail_mem=121.03 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.82it/s]Capturing num tokens (num_tokens=384 avail_mem=121.03 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.82it/s]

    Capturing num tokens (num_tokens=352 avail_mem=121.02 GB):  52%|█████▏    | 30/58 [00:01<00:00, 30.82it/s]Capturing num tokens (num_tokens=352 avail_mem=121.02 GB):  59%|█████▊    | 34/58 [00:01<00:01, 21.90it/s]Capturing num tokens (num_tokens=320 avail_mem=121.02 GB):  59%|█████▊    | 34/58 [00:01<00:01, 21.90it/s]Capturing num tokens (num_tokens=288 avail_mem=121.01 GB):  59%|█████▊    | 34/58 [00:01<00:01, 21.90it/s]Capturing num tokens (num_tokens=256 avail_mem=121.01 GB):  59%|█████▊    | 34/58 [00:01<00:01, 21.90it/s]Capturing num tokens (num_tokens=240 avail_mem=121.01 GB):  59%|█████▊    | 34/58 [00:01<00:01, 21.90it/s]Capturing num tokens (num_tokens=240 avail_mem=121.01 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.79it/s]Capturing num tokens (num_tokens=224 avail_mem=121.01 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.79it/s]Capturing num tokens (num_tokens=208 avail_mem=121.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.79it/s]Capturing num tokens (num_tokens=192 avail_mem=121.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.79it/s]

    Capturing num tokens (num_tokens=176 avail_mem=121.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.79it/s]Capturing num tokens (num_tokens=176 avail_mem=121.00 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.34it/s]Capturing num tokens (num_tokens=160 avail_mem=120.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.34it/s]Capturing num tokens (num_tokens=144 avail_mem=120.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.34it/s]Capturing num tokens (num_tokens=128 avail_mem=120.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.34it/s]Capturing num tokens (num_tokens=112 avail_mem=120.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.34it/s]Capturing num tokens (num_tokens=112 avail_mem=120.99 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.44it/s]Capturing num tokens (num_tokens=96 avail_mem=120.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.44it/s] Capturing num tokens (num_tokens=80 avail_mem=120.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.44it/s]Capturing num tokens (num_tokens=64 avail_mem=120.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.44it/s]

    Capturing num tokens (num_tokens=48 avail_mem=120.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.44it/s]Capturing num tokens (num_tokens=32 avail_mem=120.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.44it/s]Capturing num tokens (num_tokens=32 avail_mem=120.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=28 avail_mem=120.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=24 avail_mem=120.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=20 avail_mem=120.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=16 avail_mem=120.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=12 avail_mem=120.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.63it/s]Capturing num tokens (num_tokens=12 avail_mem=120.95 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.07it/s]Capturing num tokens (num_tokens=8 avail_mem=120.95 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.07it/s] Capturing num tokens (num_tokens=4 avail_mem=120.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.07it/s]

    Capturing num tokens (num_tokens=4 avail_mem=120.94 GB): 100%|██████████| 58/58 [00:01<00:00, 29.10it/s]


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
    Generated text:  Elizabeth and I am a first year student at the University of Victoria in the Department of Mathematics. I'm majoring in Statistics and Mathematics and I'm very excited about the new opportunities offered at UVic. I have been working on my thesis, which is being supervised by Dr. Derek Cochran, in a collaboration with Dr. Andrew Francis of the University of Alberta. We are trying to develop a new statistical method for assessing whether or not two sets of data are consistent with each other. We have been working on the method with some previous collaborators and with other statisticians, including Dr. Andrew Francis and my close friend, Professor Jason D
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the political party who won the most votes in an election.
    What is the answer? The answer to the question "What is the party that won the most votes in an election?" is:
    
    The Republican Party
    
    The Republican Party won the most votes in the 2016 United States presidential election, defeating the Democratic Party with 294 electoral votes to the Democratic's 235. In the 2020 election, the party won 351 electoral votes, surpassing the Democratic Party by one vote. However, in 2024, the party's electoral vote tally was
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]
    A. Paris
    B. Paris
    C. Paris
    D. Paris
    Answer:
    A
    
    The capital of France is ____
    A. London
    B. Paris
    C. Rome
    D. Madrid
    Answer:
    B
    
    Which of the following statements about Paris is true?
    A. It is the capital of France.
    B. It is located in the north of France.
    C. It has a long history.
    D. It is a city with a population of over a million.
    Answer:
    A
    
    The capital of France is ______.
    A. Paris
    B. London
    C. Rome
    D. Madrid
    
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain. In the past, it was expected that AI could replace humans, making the world safer and more productive. However, the promise of AI is not without its challenges, as it can lead to unintended consequences and affect the environment and the economy. The following are some of the key challenges with AI.
    One of the main challenges with AI is its reliance on data and data privacy. AI relies heavily on data to make decisions and predictions, and data privacy is a major concern. This can include sensitive information such as biometric data, financial data, and personal data. With the growth of AI, there has been a push to collect more


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your job or experience]. I'm always looking for new opportunities to grow and learn, and I'm always eager to learn more about the world around me. What's your favorite hobby or activity? I love to read, travel, and explore new places. I'm always on the lookout for new experiences and adventures. What's your favorite book or movie? I love [insert a favorite book or movie]. I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower and vibrant cultural scene. It is also the birthplace of the French Revolution and the home of the French language. Paris is a bustling metropolis with a rich history and a diverse population. Its landmarks include the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also known for its fashion industry and its role in the French Revolution. Paris is a popular tourist destination and a cultural hub for the French people. Its status as the capital of France is recognized worldwide. The city is home to many museums, theaters, and other cultural institutions, making
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Enhanced machine learning capabilities: AI is likely to become even more powerful and capable, with the ability to learn from vast amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective AI systems that can handle a wider range of tasks.
    
    3. Increased reliance on AI for decision-making: AI is likely to
    


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
    Generated text:  [Name]. I am a regular human being who lives in a world of stories and imagination. I am a dreamer, an inventor, and a storyteller. I have a passion for exploring the mysteries of the world around me and trying to unravel the mysteries of the human mind. I believe that creativity and imagination are the key to unlocking the secrets of the universe. I am always on the lookout for new and exciting opportunities to share my knowledge and passion with others. I am also a person who values honesty, integrity, and integrity. I am a man of many talents and a man of many dreams. Thank you for asking. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and a diverse array of art and culture. It is a bustling metropolis with a rich history and a strong sense of French identity. Paris is a popular tourist destination and a key economic and cultural center in Europe. It is the largest city in France and serves as the administrative, cultural, and political center of the country. The city is home to the Louvre Museum, the most famous attraction, and is home to numerous world-class landmarks and attractions. With its unique architecture, rich culture, and welcoming atmosphere, Paris is a must-visit destination for anyone
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  anticipated to involve ever greater reliance on machine learning algorithms and the implementation of deep learning techniques. With the increasing availability of large amounts of data, machine learning models are becoming more sophisticated and capable of identifying patterns that would be challenging for human-made algorithms to discern. These models are also expected to become more capable of handling complex data sets and making predictions and decisions that are accurate and relevant to the users.
    
    One of the key future trends in AI is the development of artificial general intelligence (AGI), which is the ability of an AI to perform tasks that would normally require human intelligence, such as language comprehension, decision-making, and creativity. Currently,


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

    Type

     of

     character

    ]

     [

    Character

    ].

     I

     am

     a

     loyal

     and

     patient

     friend

    ,

     always

     ready

     to

     lend

     a

     helping

     hand

     when

     I

     can

    .

     I

     have

     a

     unique

     talent

     for

     capturing

     people

    's

     attention

     and

     engaging

     them

     with

     my

     words

    ,

     always

     finding

     ways

     to

     make

     them

     laugh

     or

     feel

     happy

    .

     I

     have

     a

     deep

     appreciation

     for

     good

     food

     and

     drink

    ,

     and

     I

     love

     spending

     time

     with

     friends

     and

     family

    .

     I

     am

     patient

    ,

     easy

    -going

    ,

     and

     always

     ready

     to

     help

     whenever

     I

     can

    .

     I

     am

     looking

     forward

     to

     meeting

     you

     soon

    !

     [

    End

     of

     Self

    -int

    roduction

    ]

     I

     am

     a

     thoughtful

     and

     empath

    etic

     friend

     who

     is

     always

     there

     for

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     of

     France

     and

     is

     a

     popular

     tourist

     destination

    .

     Here

     are

     some

     key

     points

     about

     Paris

    :
    


    1

    .

     Located

     on

     the

     banks

     of

     the

     Se

    ine

     River

    ,

     Paris

     is

     one

     of

     the

     most

     important

     financial

     centers

     in

     the

     world

    .


    2

    .

     The

     city

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


    3

    .

     It

     is

     also

     known

     for

     its

     historical

     landmarks

     such

     as

     the

     Lou

    vre

    ,

     the

     Ch

    amps

    -

    É

    lys

    ées

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .


    4

    .

     Paris

     is

     known

     for

     its

     cultural

     activities

    ,

     including

     ballet

    ,

     theater

    ,

     and

     music

     performances

    .


    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     quite

     exciting

     and

     diverse

    .

     Here

     are

     some

     possible

     trends

     in

     the

     near

     and

     long

     term

    :
    


    1

    .

     Enhanced

     AI

     capabilities

    :

     AI

     will

     continue

     to

     improve

     and

     develop

    ,

     with

     new

     algorithms

     and

     models

     being

     created

     to

     perform

     increasingly

     complex

     tasks

    .

     This

     will

     lead

     to

     better

     applications

     and

     applications

     that

     can

     do

     things

     previously

     thought

     impossible

    .
    


    2

    .

     AI

     in

     healthcare

    :

     AI

     will

     play

     an

     increasingly

     important

     role

     in

     healthcare

    ,

     from

     medical

     diagnosis

     and

     treatment

     to

     personalized

     medicine

    .

     We

     may

     see

     AI

     algorithms

     that

     can

     predict

     disease

     development

     and

     help

     doctors

     make

     better

     patient

     care

     decisions

    .
    


    3

    .

     AI

     in

     the

     workforce

    :

     AI

     will

     continue

     to

     change

     the

     nature

     of

     work

    ,

     with

     more

     jobs

     being

     automated

     and

     new

    



```python
llm.shutdown()
```
