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
    [2026-04-22 19:56:33] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.48it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.48it/s]


    2026-04-22 19:56:37,838 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 19:56:37] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.25it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:03<00:23,  2.25it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.45it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.45it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.45it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.45it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.45it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.45it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.45it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.45it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.45it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.45it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.40it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.11it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.11it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.11it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 38.89it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 38.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.99 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.96 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.96 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.95 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.95 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.95 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.95 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.88 GB):   9%|▊         | 5/58 [00:00<00:02, 21.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.88 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.87 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.23it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.86 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.86 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.23it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=118.86 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.86 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.86 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.85 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.97it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.85 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.97it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.84 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.84 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.84 GB):  31%|███       | 18/58 [00:00<00:01, 33.21it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.84 GB):  31%|███       | 18/58 [00:00<00:01, 33.21it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.83 GB):  31%|███       | 18/58 [00:00<00:01, 33.21it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.81 GB):  31%|███       | 18/58 [00:00<00:01, 33.21it/s]

    Capturing num tokens (num_tokens=960 avail_mem=118.83 GB):  31%|███       | 18/58 [00:00<00:01, 33.21it/s] Capturing num tokens (num_tokens=896 avail_mem=118.83 GB):  31%|███       | 18/58 [00:00<00:01, 33.21it/s]Capturing num tokens (num_tokens=896 avail_mem=118.83 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.88it/s]Capturing num tokens (num_tokens=832 avail_mem=118.82 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.88it/s]Capturing num tokens (num_tokens=768 avail_mem=118.82 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.88it/s]Capturing num tokens (num_tokens=704 avail_mem=118.82 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.88it/s]Capturing num tokens (num_tokens=640 avail_mem=118.81 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.88it/s]Capturing num tokens (num_tokens=576 avail_mem=118.81 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.88it/s]Capturing num tokens (num_tokens=576 avail_mem=118.81 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.01it/s]Capturing num tokens (num_tokens=512 avail_mem=118.80 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.01it/s]Capturing num tokens (num_tokens=480 avail_mem=118.82 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.01it/s]

    Capturing num tokens (num_tokens=448 avail_mem=118.81 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.01it/s]Capturing num tokens (num_tokens=416 avail_mem=118.81 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.01it/s]Capturing num tokens (num_tokens=416 avail_mem=118.81 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.30it/s]Capturing num tokens (num_tokens=384 avail_mem=118.81 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.30it/s]Capturing num tokens (num_tokens=352 avail_mem=118.80 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.30it/s]Capturing num tokens (num_tokens=320 avail_mem=118.80 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=288 avail_mem=118.79 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=288 avail_mem=118.79 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=256 avail_mem=118.74 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=240 avail_mem=118.72 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]

    Capturing num tokens (num_tokens=224 avail_mem=118.72 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=208 avail_mem=118.71 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=192 avail_mem=118.71 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.07it/s]Capturing num tokens (num_tokens=192 avail_mem=118.71 GB):  71%|███████   | 41/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=176 avail_mem=118.71 GB):  71%|███████   | 41/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=160 avail_mem=118.71 GB):  71%|███████   | 41/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=144 avail_mem=118.70 GB):  71%|███████   | 41/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=128 avail_mem=118.70 GB):  71%|███████   | 41/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=112 avail_mem=118.70 GB):  71%|███████   | 41/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=112 avail_mem=118.70 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.58it/s]Capturing num tokens (num_tokens=96 avail_mem=118.69 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.58it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=118.69 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.58it/s]Capturing num tokens (num_tokens=64 avail_mem=118.69 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.58it/s]Capturing num tokens (num_tokens=48 avail_mem=118.69 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.58it/s]Capturing num tokens (num_tokens=32 avail_mem=118.68 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.58it/s]Capturing num tokens (num_tokens=32 avail_mem=118.68 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=28 avail_mem=118.68 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=24 avail_mem=118.67 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=20 avail_mem=118.67 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=16 avail_mem=118.67 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=12 avail_mem=118.66 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.18it/s]

    Capturing num tokens (num_tokens=12 avail_mem=118.66 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.14it/s]Capturing num tokens (num_tokens=8 avail_mem=118.66 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.14it/s] Capturing num tokens (num_tokens=4 avail_mem=118.66 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.14it/s]Capturing num tokens (num_tokens=4 avail_mem=118.66 GB): 100%|██████████| 58/58 [00:01<00:00, 36.98it/s]


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
    Generated text:  Jana and I am a freelance writer based in Zurich. I have a background in journalism and have written articles for various publications. I'm currently working on my first novel.
    Can you provide me with a general introduction to the novel you're writing about? Of course! My first novel is a historical romance set in the 1920s and 1930s. It follows a young woman named Emily who becomes entangled in a romantic affair with a wealthy man named John, a prominent figure in the world of politics and business. As the novel progresses, we see the conflicts and alliances that come with pursuing a love
    ===============================
    Prompt: The president of the United States is
    Generated text:  inaugurated on the second Monday after the third Monday in October.
    
    Does it follow that if the president of the United States is inaugurated on the second Monday after the third Monday in October, then it must be true that the president is inaugurated before November 4, 2020? 
    
    To determine if the given statement logically implies the conclusion, we need to analyze the timeline and the specific dates provided.
    
    1. The president is inaugurated on the second Monday after the third Monday in October.
    2. We need to determine if this inauguration occurs before November 4, 2020.
    
    To check the relationship between
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population of Paris is 2.1 million. The population of London is 8.5 million. The population of Tokyo is 11.9 million. The population of Moscow is 11.1 million. The population of New York City is 8.3 million. If we consider the areas within these cities as two-dimensional rectangles, what is the total area of the four largest cities multiplied by the area of Paris?
    To find the total area of the four largest cities multiplied by the area of Paris, we need to follow these steps:
    
    1. Calculate the area of the largest city (London).
    2
    ===============================
    Prompt: The future of AI is
    Generated text:  rosy, and it's not just about the capabilities of the technologies themselves but also about the way that they are being used and how they are being governed. It's a time of great innovation and change, where the boundaries between humans and machines are blurring and where the potential for great advancements in technology is limitless.
    The need for AI to play a role in shaping the future of society is clear. It can be used for a variety of purposes, from helping to improve healthcare and education, to creating more efficient and effective transportation systems, to enhancing the quality of life for people with disabilities.
    However, it's important to note that AI


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


    Generated text:  [Name] and I am a [occupation] who has been [number of years] in the industry. I am passionate about [reason for passion], and I am always looking for ways to [action or goal]. I am [age] years old, and I am [gender] and [race]. I am [occupation] and I am [gender] and [race]. I am [occupation] and I am [gender] and [race]. I am [occupation] and I am [gender] and [race]. I am [occupation] and I am [gender] and [race]. I am [occupation] and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris". It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a popular tourist destination, with millions of visitors each year. The city is known for its fashion, art, and food, and is a major hub for business and commerce in Europe. Paris is a cultural and political center of France and plays a significant role in the country's history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to improve, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare to transportation. This will likely lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more advanced healthcare applications, including
    


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
    Generated text:  [Name] and I'm a [Job Title] at [Company Name]. I'm here to provide [Purpose of the Job] or [Support the Team]. I thrive on learning and staying up-to-date with the latest technologies and trends in the industry. I'm always eager to apply my skills and knowledge to help solve problems and make a positive impact. I'm a team player and love to collaborate with others to achieve our goals. I'm confident and proactive, always ready to take on new challenges and challenges in my current role. I'm looking forward to [Next Steps] with [Company Name]. [Name] is a professional
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the fifth largest in the world. It is the seat of government and political power in France. The city is known for its rich history, diverse cultural scene, and iconic landmarks like Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. It is also one of the most expensive cities in the world, with a per capita income of over $80,000. Paris is a popular tourist destination, with millions of visitors annually, making it one of the world's largest cities. The city is home to several influential institutions and organizations, such as the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by an increase in machine learning and neural network algorithms, with the ability to learn and adapt to new data. This will likely lead to the development of more sophisticated algorithms that can perform tasks that were previously thought impossible, such as image and speech recognition, self-driving cars, and personalized medicine. The use of AI in healthcare and finance is likely to become more widespread, as AI algorithms can analyze large amounts of data and identify patterns that human analysts may not be able to detect. Additionally, AI is likely to become more integrated into everyday life, with applications such as voice assistants and virtual assistants becoming more common and prevalent. AI


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

     am

     a

     [

    role

    ]

    !

     I

     am

     passionate

     about

     [

    mention

     a

     hobby

     or

     activity

     that

     you

     enjoy

    ,

     such

     as

     hiking

    ,

     painting

    ,

     or

     dancing

    ].

     I

     am

     a

     [

    mention

     a

     profession

    ,

     such

     as

     a

     doctor

    ,

     lawyer

    ,

     or

     engineer

    ].

     I

     am

     [

    mention

     a

     personal

     characteristic

     or

     trait

    ,

     such

     as

     outgoing

    ,

     hard

    -working

    ,

     or

     patient

    ].

     I

     am

     [

    mention

     a

     unique

     skill

     or

     ability

    ,

     such

     as

     the

     ability

     to

     speak

     a

     foreign

     language

     or

     solve

     complex

     mathematical

     problems

    ].

     I

     am

     [

    mention

     a

     reason

     for

     your

     success

    ,

     such

     as

     having

     a

     strong

     work

     ethic

    ,

     being

     able

     to

     adapt

     to

     different

     challenges

    ,

     or

     being

     a

     natural

     at

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Please

     answer

     the

     following

     question

     about

     the

     statement

    :


    What

     is

     the

     population

     of

     Paris

    ?

     

    1

    .

    8

     million

     people

     live

     in

     Paris

    .

     
    


    2

    .

    3

     million

     people

     live

     in

     Paris

    ,

     making

     it

     the

     largest

     city

     in

     Europe

     and

     the

     second

     largest

     city

     in

     the

     world

     by

     population

    .

     
    


    Based

     on

     the

     given

     information

    ,

     what

     is

     the

     population

     density

     of

     Paris

    ?

     The

     population

     density

     of

     Paris

     is

     

    2

    ,

    3

    0

    0

     people

     per

     square

     kil

    ometer

    .

     
    


    Which

     city

     is

     the

     largest

     in

     Europe

     and

     the

     second

     largest

     in

     the

     world

     by

     population

    ?

     The

     city

     that

     is

     the

     largest

     in

     Europe

     and

     the

     second

     largest

     in

     the

     world

     by

     population

     is

     Paris

    .
    


    How

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     vast

     and

     unpredictable

    ,

     and

     there

     are

     many

     possible

     trends

     that

     could

     influence

     its

     development

    .

     Some

     of

     the

     most

     significant

     trends

     that

     could

     shape

     the

     future

     of

     AI

     include

    :
    


    1

    .

     Improved

     accuracy

     and

     efficiency

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     its

     accuracy

     and

     efficiency

     in

     solving

     complex

     problems

     will

     only

     improve

    .

     This

     means

     that

     AI

     systems

     will

     be

     able

     to

     process

     and

     analyze

     larger

     amounts

     of

     data

     more

     quickly

     and

     efficiently

    ,

     leading

     to

     better

     decision

    -making

     and

     predictions

    .
    


    2

    .

     Enhanced

     human

    -com

    puter

     interaction

    :

     With

     the

     advancement

     of

     AI

    ,

     we

     are

     likely

     to

     see

     an

     increased

     reliance

     on

     AI

    -powered

     tools

     and

     interfaces

     for

     human

    -com

    puter

     interaction

    .

     This

     could

     lead

     to

     the

     development

     of

     more

    



```python
llm.shutdown()
```
