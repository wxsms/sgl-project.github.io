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
    [2026-04-16 16:03:13] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.90it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.89it/s]


    2026-04-16 16:03:18,207 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 16:03:18] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<02:45,  2.91s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:07,  6.37it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:07,  6.37it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:07,  6.37it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:07,  6.37it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:07,  6.37it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:07,  6.37it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:07,  6.37it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:07,  6.37it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:07,  6.37it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:07,  6.37it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.03it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.03it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.03it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 20.74it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.23it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.23it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.23it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.23it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.23it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.23it/s]

    Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.23it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 29.23it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 29.23it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 36.94it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 36.94it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 36.94it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 36.94it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 36.94it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 36.94it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 36.94it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 36.94it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 36.94it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 36.94it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 46.10it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 46.10it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 46.10it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.73it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=117.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=117.08 GB):   2%|▏         | 1/58 [00:00<00:10,  5.60it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.05 GB):   2%|▏         | 1/58 [00:00<00:10,  5.60it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=117.05 GB):   2%|▏         | 1/58 [00:00<00:10,  5.60it/s]Capturing num tokens (num_tokens=7168 avail_mem=117.05 GB):   5%|▌         | 3/58 [00:00<00:04, 11.18it/s]Capturing num tokens (num_tokens=6656 avail_mem=117.05 GB):   5%|▌         | 3/58 [00:00<00:04, 11.18it/s]Capturing num tokens (num_tokens=6144 avail_mem=117.05 GB):   5%|▌         | 3/58 [00:00<00:04, 11.18it/s]Capturing num tokens (num_tokens=6144 avail_mem=117.05 GB):   9%|▊         | 5/58 [00:00<00:03, 13.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=117.04 GB):   9%|▊         | 5/58 [00:00<00:03, 13.98it/s]Capturing num tokens (num_tokens=5120 avail_mem=117.05 GB):   9%|▊         | 5/58 [00:00<00:03, 13.98it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=117.04 GB):   9%|▊         | 5/58 [00:00<00:03, 13.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=117.04 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=117.04 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=117.04 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=117.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=117.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=117.03 GB):  21%|██        | 12/58 [00:00<00:01, 23.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=117.03 GB):  21%|██        | 12/58 [00:00<00:01, 23.23it/s]Capturing num tokens (num_tokens=2816 avail_mem=117.03 GB):  21%|██        | 12/58 [00:00<00:01, 23.23it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=117.02 GB):  21%|██        | 12/58 [00:00<00:01, 23.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=117.02 GB):  21%|██        | 12/58 [00:00<00:01, 23.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=117.02 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=2048 avail_mem=117.02 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=117.01 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=117.01 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=117.00 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=117.00 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.98 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.73it/s]

    Capturing num tokens (num_tokens=960 avail_mem=117.00 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.73it/s] Capturing num tokens (num_tokens=896 avail_mem=117.00 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.73it/s]Capturing num tokens (num_tokens=896 avail_mem=117.00 GB):  40%|███▉      | 23/58 [00:00<00:01, 28.45it/s]Capturing num tokens (num_tokens=832 avail_mem=116.99 GB):  40%|███▉      | 23/58 [00:00<00:01, 28.45it/s]Capturing num tokens (num_tokens=768 avail_mem=116.99 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.45it/s]Capturing num tokens (num_tokens=704 avail_mem=116.99 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.45it/s]Capturing num tokens (num_tokens=640 avail_mem=116.98 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.45it/s]Capturing num tokens (num_tokens=640 avail_mem=116.98 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.77it/s]Capturing num tokens (num_tokens=576 avail_mem=116.98 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.77it/s]Capturing num tokens (num_tokens=512 avail_mem=116.97 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.77it/s]

    Capturing num tokens (num_tokens=480 avail_mem=116.97 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.77it/s]Capturing num tokens (num_tokens=448 avail_mem=116.97 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.77it/s]Capturing num tokens (num_tokens=448 avail_mem=116.97 GB):  53%|█████▎    | 31/58 [00:01<00:01, 26.62it/s]Capturing num tokens (num_tokens=416 avail_mem=116.96 GB):  53%|█████▎    | 31/58 [00:01<00:01, 26.62it/s]Capturing num tokens (num_tokens=384 avail_mem=116.87 GB):  53%|█████▎    | 31/58 [00:01<00:01, 26.62it/s]Capturing num tokens (num_tokens=352 avail_mem=116.46 GB):  53%|█████▎    | 31/58 [00:01<00:01, 26.62it/s]Capturing num tokens (num_tokens=320 avail_mem=116.37 GB):  53%|█████▎    | 31/58 [00:01<00:01, 26.62it/s]

    Capturing num tokens (num_tokens=320 avail_mem=116.37 GB):  60%|██████    | 35/58 [00:01<00:00, 29.69it/s]Capturing num tokens (num_tokens=288 avail_mem=116.36 GB):  60%|██████    | 35/58 [00:01<00:00, 29.69it/s]Capturing num tokens (num_tokens=256 avail_mem=116.36 GB):  60%|██████    | 35/58 [00:01<00:00, 29.69it/s]Capturing num tokens (num_tokens=240 avail_mem=116.36 GB):  60%|██████    | 35/58 [00:01<00:00, 29.69it/s]Capturing num tokens (num_tokens=224 avail_mem=116.36 GB):  60%|██████    | 35/58 [00:01<00:00, 29.69it/s]Capturing num tokens (num_tokens=224 avail_mem=116.36 GB):  67%|██████▋   | 39/58 [00:01<00:00, 30.22it/s]Capturing num tokens (num_tokens=208 avail_mem=116.35 GB):  67%|██████▋   | 39/58 [00:01<00:00, 30.22it/s]Capturing num tokens (num_tokens=192 avail_mem=116.35 GB):  67%|██████▋   | 39/58 [00:01<00:00, 30.22it/s]Capturing num tokens (num_tokens=176 avail_mem=116.35 GB):  67%|██████▋   | 39/58 [00:01<00:00, 30.22it/s]Capturing num tokens (num_tokens=160 avail_mem=116.34 GB):  67%|██████▋   | 39/58 [00:01<00:00, 30.22it/s]

    Capturing num tokens (num_tokens=144 avail_mem=116.34 GB):  67%|██████▋   | 39/58 [00:01<00:00, 30.22it/s]Capturing num tokens (num_tokens=144 avail_mem=116.34 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.69it/s]Capturing num tokens (num_tokens=128 avail_mem=116.34 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.69it/s]Capturing num tokens (num_tokens=112 avail_mem=116.33 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.69it/s]Capturing num tokens (num_tokens=96 avail_mem=116.33 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.69it/s] Capturing num tokens (num_tokens=80 avail_mem=116.33 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.69it/s]Capturing num tokens (num_tokens=80 avail_mem=116.33 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.90it/s]Capturing num tokens (num_tokens=64 avail_mem=116.32 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.90it/s]Capturing num tokens (num_tokens=48 avail_mem=116.32 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.90it/s]Capturing num tokens (num_tokens=32 avail_mem=116.31 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.90it/s]

    Capturing num tokens (num_tokens=28 avail_mem=116.31 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.90it/s]Capturing num tokens (num_tokens=28 avail_mem=116.31 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.40it/s]Capturing num tokens (num_tokens=24 avail_mem=116.31 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.40it/s]Capturing num tokens (num_tokens=20 avail_mem=116.30 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.40it/s]Capturing num tokens (num_tokens=16 avail_mem=116.30 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.40it/s]Capturing num tokens (num_tokens=12 avail_mem=116.30 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.40it/s]Capturing num tokens (num_tokens=12 avail_mem=116.30 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.12it/s]Capturing num tokens (num_tokens=8 avail_mem=116.29 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.12it/s] Capturing num tokens (num_tokens=4 avail_mem=116.28 GB):  97%|█████████▋| 56/58 [00:02<00:00, 35.12it/s]Capturing num tokens (num_tokens=4 avail_mem=116.28 GB): 100%|██████████| 58/58 [00:02<00:00, 28.52it/s]


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
    Generated text:  Lisa and I am the Assistant Director of the Community Service Corps. I have been working in the field of community service for over 12 years and have been the Director of the Community Service Corps for four years. I serve as a subject matter expert on the topic of community service. My role is to communicate with the community, provide information, and connect the community with the Corps’ goals and benefits.
    What are the benefits of being a part of the Community Service Corps?
    The benefits of being a part of the Community Service Corps can vary depending on the organization and the goals of the Corps. However, in general, the following are some
    ===============================
    Prompt: The president of the United States is
    Generated text:  married to a (n) ________.
    A. lady
    B. lady's
    C. lady of
    D. lady's
    Answer: C
    
    In the process of economic globalization, in which direction does the capital flow?
    A. From the developed countries to the developing countries
    B. From developed countries to developing countries
    C. From developing countries to developed countries
    D. From developing countries to developing countries
    Answer: B
    
    The era when the capital flow is from the developed countries to the developing countries is
    A. The 1970s
    B. The 1980s
    C.
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. London
    C. Shanghai
    D. Cairo
    Answer:
    A
    
    The core of the Four Cardinal Principles is ____
    A. Upholding the leadership of the Communist Party of China
    B. Upholding the socialist path
    C. Upholding people's democratic dictatorship
    D. Upholding Marxism-Leninism, Mao Zedong Thought
    Answer:
    A
    
    Which of the following is a representative work of the 19th-century American Romantic novel?
    A. "Wuthering Heights"
    B. "Ulysses"
    C. "Romeo and Juliet"
    D.
    ===============================
    Prompt: The future of AI is
    Generated text:  fully in our hands. With the increasing number of AI-related patents filed in the United States in the last few years, it is clear that AI is on the horizon of the next generation. The technology is developing at an incredible pace and the possibilities are truly amazing. A great example of this is the development of the latest AI systems and applications.
    One of the areas where AI is gaining the most attention is in the field of healthcare. There is no denying that AI has the potential to revolutionize healthcare in the next generation. It can be used to improve diagnosis, treatment, and patient outcomes, among other things.
    For example, AI can


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have [number] years of experience in [industry]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity], and I find it incredibly rewarding to be able to share my passion with others. What's your favorite book or movie? I love [book/movie], and I find it incredibly inspiring to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a cultural and economic center with a rich history dating back to the Roman Empire and the French Revolution. It is a popular tourist destination and a major hub for business and commerce in Europe. The city is also home to the French Riviera, a popular tourist destination for its beaches and luxury resorts. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the urban landscape
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives, from voice assistants like Siri and Alexa to self-driving cars. As AI technology continues to advance, we can expect to see even more integration into our daily routines.
    
    2. AI becoming more autonomous: As AI technology continues to improve, we can expect to see more autonomous vehicles on the roads. This could lead to a reduction in traffic congestion and accidents, but it could also lead to new challenges such as
    


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
    Generated text:  [Name] and I am a [Skill/Interest/Experience] who has been [Number of years of experience] years in this field. I bring a [Number of years of experience] of experience in [Skill/Interest/Experience], and I am [Age]. I am [Gender] and I am [Height/Weight], and I have a [Language/Attitude] personality. I am [Appearance/Background] and I have always been [Motivation]. I am excited to work with [Person] because [Reason for working with them]. I am [Available to work] at [Location]. If you would like
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "la Parisis the City of Light" or simply as "Paris".
    Paris is the largest city in France and the third-largest city in the world, home to 23 million people. Its status as the capital is due to its role as the political, economic, and cultural capital of France. The city has been the site of numerous important events in history, including the coronation of French monarchs and the founding of the French Republic. The city is also known for its beautiful architecture, including the Eiffel Tower and the Louvre Museum. Paris is a UNESCO World Heritage site and a major cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be a highly dynamic and rapidly evolving landscape, with many different trends and possibilities shaping the direction of the technology. Some of the most promising trends include:
    
    1. Increased integration with human emotions and behaviors: As AI systems become more capable of understanding and responding to human emotions, their capabilities may also expand to include more nuanced and emotional AI. This could mean that AI systems are able to understand and respond to human emotions and behaviors, rather than just being reactive to them.
    
    2. The development of more natural and intuitive AI: There is a growing interest in developing AI that is more intuitive and natural to humans, rather than just being mechanical


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

    ].

     I

     am

     a

     [

    background

     or

     occupation

    ]

     who

     has

     always

     been

     a

     curious

     and

     adventurous

     soul

    .

     I

     enjoy

     exploring

     new

     places

     and

     learning

     about

     different

     cultures

    .

     I

     am

     always

     eager

     to

     try

     new

     foods

    ,

     and

     I

     love

     sharing

     my

     experiences

     and

     opinions

     with

     others

    .

     I

     am

     a

     true

     believer

     in

     the

     power

     of

     creativity

     and

     have

     always

     been

     a

     fan

     of

     art

     and

     music

    .

     I

     believe

     that

     learning

     and

     growing

     in

     all

     aspects

     of

     life

     is

     important

    ,

     and

     I

     strive

     to

     live

     a

     fulfilling

     life

     that

     respects

     and

     encourages

     others

    .

     What

    's

     your

     name

     and

     what

     kind

     of

     job

     or

     career

     do

     you

     have

    ?

     I

    'm

     [

    Your

     Name

    ],

     a

     [

    background

     or

     occupation

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     promising

    ,

     with

     a

     wide

     range

     of

     potential

     developments

     and

     applications

    .

     Here

     are

     some

     possible

     trends

     in

     AI

    :
    


    1

    .

     Increased

     automation

    :

     As

     AI

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

     more

     automation

     in

     our

     daily

     lives

    .

     This

     could

     include

     more

     automated

     manufacturing

    ,

     autonomous

     vehicles

    ,

     and

     even

     more

     advanced

     AI

     that

     can

     help

     us

     with

     tasks

     such

     as

     healthcare

     and

     finance

    .
    


    2

    .

     Enhanced

     cognitive

     abilities

    :

     AI

     is

     getting

     even

     better

     at

     processing

     and

     analyzing

     information

    ,

     and

     we

     can

     expect

     to

     see

     even

     more

     advanced

     cognitive

     abilities

     in

     the

     future

    .

     This

     could

     include

     more

     sophisticated

     forms

     of

     speech

     recognition

    ,

     as

     well

     as

     more

     advanced

     machine

     learning

     algorithms

     that

     can

     analyze

     large

     amounts

    



```python
llm.shutdown()
```
