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
    [2026-04-17 09:42:19] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.18it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.17it/s]


    2026-04-17 09:42:23,189 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 09:42:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:40,  2.82s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:30,  1.78it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:30,  1.78it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:30,  1.78it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.47it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.61it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.61it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.61it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.61it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.61it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.61it/s]

    Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.61it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:02, 16.67it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:02, 16.67it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:02, 16.67it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:02, 16.67it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:02, 16.67it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:02, 16.67it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:03<00:02, 16.67it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 20.76it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 20.76it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 20.76it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 20.76it/s]

    Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 20.76it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 20.76it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 20.76it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 20.76it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 27.63it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 27.63it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 27.63it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 27.63it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 27.63it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 27.63it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 27.63it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 27.63it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s]

    Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.40it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.47it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.47it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.47it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.47it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.47it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.47it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.47it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.47it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.47it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.27 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.27 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.27 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.25 GB):   7%|▋         | 4/58 [00:00<00:03, 14.51it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.25 GB):   7%|▋         | 4/58 [00:00<00:03, 14.51it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.24 GB):   7%|▋         | 4/58 [00:00<00:03, 14.51it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.24 GB):  10%|█         | 6/58 [00:00<00:03, 14.71it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.75 GB):  10%|█         | 6/58 [00:00<00:03, 14.71it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.66 GB):  10%|█         | 6/58 [00:00<00:03, 14.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.59 GB):  10%|█         | 6/58 [00:00<00:03, 14.71it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.59 GB):  10%|█         | 6/58 [00:00<00:03, 14.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.59 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.58 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.57 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.52it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.52it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.52it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.52it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=71.56 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.52it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.34it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.34it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.34it/s] Capturing num tokens (num_tokens=896 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.34it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.34it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:01, 28.28it/s]Capturing num tokens (num_tokens=768 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:00<00:01, 28.28it/s]Capturing num tokens (num_tokens=704 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.28it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.28it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.28it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.18it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.18it/s]

    Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.18it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.18it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 31.18it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.47it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.47it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.47it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.47it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 33.47it/s]

    Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.09it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.09it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.09it/s]Capturing num tokens (num_tokens=224 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.09it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.09it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.09it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=160 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 35.22it/s]

    Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.87it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=64 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.87it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.92it/s]

    Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.55it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.55it/s] Capturing num tokens (num_tokens=4 avail_mem=75.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.55it/s]Capturing num tokens (num_tokens=4 avail_mem=75.93 GB): 100%|██████████| 58/58 [00:01<00:00, 31.68it/s]


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
    Generated text:  Dave. And, uh, I'm the best ice cream salesman in the world. That's right. I'm the best ice cream salesman in the world. How did you come to be the best ice cream salesman in the world? Can you tell me about your journey to success?
    I'm sorry Dave, but I can't assist with that.
    You know the drill - I'm not a superlative personality type like this, so I can't be your ice cream salesman. But I'd love to have a conversation about what really matters in life, rather than selling ice cream. Can you share your thoughts on what truly matters?
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a small village in need. He is kind enough to give $1000 as a gift to the village. How much would the president have received if he gives 10% of the money to the village board?
    To determine how much the president would have received if he gives 10% of $1000 to the village board, we can follow these steps:
    
    1. Calculate 10% of $1000.
    2. Subtract this amount from $1000 to find out how much the president would have received.
    
    First, let's calculate 10% of $1
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and the ancient city of Carthage is situated in which island of Tunisia? To determine the capital of France and the ancient city of Carthage that is located on the island of Tunisia, we can follow these steps:
    
    1. Identify the capital of France.
       - The capital of France is Paris.
    
    2. Identify the ancient city of Carthage.
       - The ancient city of Carthage is situated in the island of Tunisia.
    
    Therefore, the capital of France is Paris and the ancient city of Carthage that is situated on the island of Tunisia is Carthage.
    
    The final answer is \(\boxed
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it's not without challenges. How can we ensure that AI is used to benefit society and not to harm it?
    The future of AI is bright, but it's not without challenges. How can we ensure that AI is used to benefit society and not to harm it?
    
    AI can have a significant impact on society, providing new opportunities for innovation and improving quality of life. However, it also poses significant risks and challenges. Here are some key points to consider:
    
    1. Risk of Bias: AI systems can be biased if they are trained on biased data or if the algorithms are designed to prioritize certain outcomes over others. This can


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and passions. Let's chat about [topic of interest]. How can I assist you today? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and passions. Let's chat about [topic of interest]. How can I assist you today? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and passions. Let's chat about [topic of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination, known for its rich history, art, and cuisine. The city is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a vibrant and diverse city with a rich cultural heritage that continues to inspire and influence the world. The city is also known for its fashion industry, with Paris Fashion Week being one of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This integration will enable AI to perform more complex tasks and improve its ability to learn and adapt to new situations.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability.
    
    3. Increased use of
    


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
    Generated text:  Sarah. I'm a dedicated, friendly, and organized student who loves to learn new things and explore new places. I'm constantly on the go, from meeting new people to attending classes and trying out new activities. I'm friendly to all and enjoy helping others when I can. I'm always looking for new experiences and adventures to try, and I'm always excited to share my knowledge with others. I'm a strong believer in the power of learning and trying new things, and I'm always up for a challenge. I'm always eager to learn and grow, and I'm always looking for ways to improve myself. I love to travel
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is also the city's largest and most populous city. The city is located in the Île de France region of the northwestern region of the country, and is situated on the River Seine. It has a population of approximately 2.2 million people, making it the fourth-largest city in Europe by population. Paris is known for its rich history, art, fashion, and cuisine. It is also one of the world's most cosmopolitan cities, with a diverse mix of cultures and languages. Paris has been a center for literature, art, music, and the arts for centuries, and it continues to be an
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be very exciting and diverse. Here are some potential trends that are likely to shape the field in the coming years:
    
    1. Increased Use of Machine Learning: AI will continue to become more sophisticated and accurate, and will be used in a wider range of applications. Machine learning will allow AI to learn from data and improve its performance over time.
    
    2. Integration of AI into everyday life: AI will become more integrated into our everyday lives, from home assistants and robots to self-driving cars and virtual assistants. This will make life easier and more efficient, but also create new challenges for AI researchers.
    
    3. Personalization: AI will be


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

    'm

     a

     [

    type

     of

     character

    ]

     with

     [

    number

    ]

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

     [

    the

     main

     character

    's

     main

     occupation

     or

     field

     of

     work

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ].

     I

    'm

     a

     [

    type

     of

     character

    ]

     with

     [

    number

    ]

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

     [

    the

     main

     character

    's

     main

     occupation

     or

     field

     of

     work

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     [

    Your

     Name

    ],

     I

    'm

     a

     [

    type

     of

     character

    ]

     with

     [

    number

    ]

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

     [

    the

     main

     character

    's

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    France

    's

     capital

     city

    ,

     Paris

    ,

     is

     known

     for

     its

     beautiful

     architecture

    ,

     rich

     history

    ,

     and

     vibrant

     cultural

     scene

    .

     The

     city

     is

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

     and

     is

     home

     to

     numerous

     attractions

     such

     as

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

     famous

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     music

     scene

    ,

     and

     is

     a

     popular

     tourist

     destination

     for

     millions

     of

     visitors

     annually

    .

     The

     city

     has

     a

     diverse

     population

     with

     many

     different

     ethnic

    ities

     and

     cultures

    ,

     and

     is

     home

     to

     many

     historical

     and

     artistic

     landmarks

    .

     The

     French

     government

     and

     citizens

     work

     together

     to

     promote

     and

     preserve

     the

     city

    's

     unique

     identity

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     looking

     promising

     and

     exciting

    .

     Some

     possible

     trends

     that

     may

     occur

     in

     the

     near

     future

     include

    :
    


    1

    .

     Increased

     personal

    ization

    :

     AI

     will

     allow

     machines

     to

     learn

     and

     adapt

     to

     the

     user

    's

     behavior

     and

     preferences

    ,

     leading

     to

     more

     personalized

     and

     effective

     experiences

    .
    


    2

    .

     More

     intelligent

     virtual

     assistants

    :

     AI

     technology

     will

     continue

     to

     improve

     and

     become

     more

     capable

    ,

     leading

     to

     more

     sophisticated

     virtual

     assistants

     that

     can

     assist

     humans

     in

     various

     tasks

    .
    


    3

    .

     Improved

     safety

     and

     security

    :

     AI

     will

     be

     used

     to

     develop

     new

     technologies

     for

     enhancing

     safety

     and

     security

    ,

     such

     as

     smarter

     drones

    ,

     more

     secure

     cybersecurity

     systems

    ,

     and

     more

     efficient

     public

     transportation

     systems

    .
    


    4

    .

     Increased

     data

    -driven

     decision

    -making

    :

     AI

     will

     continue

    



```python
llm.shutdown()
```
