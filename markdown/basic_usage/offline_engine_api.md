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
    [2026-04-22 15:27:45] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.17it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.15it/s]


    2026-04-22 15:27:49,814 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 15:27:49] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.79it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.79it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.79it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.79it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.79it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:02<00:06,  6.79it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:02<00:06,  6.79it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:02<00:06,  6.79it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.79it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.79it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.88it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.88it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.88it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.88it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.88it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.88it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.88it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.88it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.88it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.88it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 22.01it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 22.01it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 22.01it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 22.01it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 22.01it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 22.01it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 22.01it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 22.01it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 22.01it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 22.01it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.90it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.90it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.90it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.90it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.90it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.90it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.90it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.90it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.90it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.90it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.74it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.74it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.74it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.74it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.74it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.74it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.74it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.74it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.74it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.74it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.74it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 50.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:03, 18.12it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.12it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.12it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.12it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 20.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 20.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 20.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 20.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.54it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.54it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.54it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.54it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  21%|██        | 12/58 [00:00<00:01, 28.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  21%|██        | 12/58 [00:00<00:01, 28.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  21%|██        | 12/58 [00:00<00:01, 28.71it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  21%|██        | 12/58 [00:00<00:01, 28.71it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  21%|██        | 12/58 [00:00<00:01, 28.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  21%|██        | 12/58 [00:00<00:01, 28.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.89it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.89it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.89it/s]Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.89it/s] Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.70it/s]Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.70it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.70it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.70it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.70it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.70it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.00it/s]

    Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.56it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.56it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.56it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.56it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.56it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.56it/s]

    Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.39it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.39it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.39it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.39it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.39it/s]

    Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.39it/s] Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  81%|████████  | 47/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  81%|████████  | 47/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  81%|████████  | 47/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  81%|████████  | 47/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  81%|████████  | 47/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  81%|████████  | 47/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.97it/s]

    Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.97it/s] Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 38.20it/s]


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
    Generated text:  Lisa. My favorite subject is math. I want to learn more about people and their feelings. I like to have conversations and try to understand people's thoughts and feelings.
    This is a simple conversation between me and someone.
    Lisa: Hello, my name is Lisa. I want to learn more about people and their feelings. I like to have conversations and try to understand people's thoughts and feelings.
    Person: What subject are you in the middle of learning? Lisa: Math. Person: That's good. So you have a lot of questions about people and their feelings.
    Lisa: Yes, it's great. I'm always curious about people
    ===============================
    Prompt: The president of the United States is
    Generated text:  a noble title, and it is unique to him. An American person who has been elected as the president of the United States has the right to vote in the general election of the United States, and he/she can also be appointed to various government posts. A. Correct B. Incorrect
    Answer:
    
    A
    
    For a patient with an open pneumothorax, which of the following measures should NOT be taken?
    A. Administering oxygen
    B. Maintaining airway patency
    C. Preventing infection
    D. Performing thoracentesis
    E. Administering sedatives
    Answer:
    
    E
    
    The secondary key of
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is located in the center of the country. The city is a major cultural center with many famous landmarks and historical sites. Here are a few of the most famous landmarks in Paris:
    
    1. The Eiffel Tower - one of the most famous monuments in the world, the Eiffel Tower is a massive wrought iron tower that stands 324 meters tall. It was built as a military lookout tower during the French Revolution.
    
    2. The Louvre Museum - this is one of the most famous museums in the world, and it is home to many of the world's most famous art pieces, including the Mona Lisa
    ===============================
    Prompt: The future of AI is
    Generated text:  exciting and full of possibilities. However, these advancements should be accompanied by ethical considerations and concerns. Here are some key points to keep in mind:
    
    1. Bias: AI systems can perpetuate biases and discrimination if they are not trained on diverse data sets. Organizations should consider implementing diversity and inclusion training for all employees to help prevent these issues.
    
    2. Privacy: AI systems can potentially collect and analyze large amounts of personal data, which can be used to monitor and control individuals. Organizations should implement strong data privacy policies and practices to ensure that this data is protected.
    
    3. Transparency: AI systems should be transparent about how they make decisions and what


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and is home to many famous landmarks and attractions. It is also known for its rich history and cultural heritage, including the influence of French literature, art, and music. The city is also known for its cuisine, with many famous restaurants and food festivals. Overall, Paris is a vibrant and exciting city that is a must-visit for anyone interested in French culture and history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced natural language processing: AI will continue to improve its ability to understand and interpret human language, allowing machines to better communicate and interact with humans.
    
    3. Improved image and speech recognition: AI will continue to improve its ability to recognize and interpret images and speech, allowing machines to better understand and respond to human speech.
    
    4. Greater autonomy: AI will become more autonomous, allowing machines to make
    


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
    Generated text:  [Name], I’m [Age] years old. I’m from [Place], [City]. I’m a [Occupation] who loves [The thing you do best]. I’m [Your Character’s Summary]. As for my hobbies, I like [Other hobbies]. I enjoy spending time with my family and friends, and I have a keen sense of [Other hobbies]. I’m always looking for ways to [More hobbies]. I’m [Your Character’s Summary], and I’m here to be your guide through life. How can I help you today? [Name] (your name) has been searching for a life coach for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the world's largest city and the fifth-largest by population, home to over 10 million people.
    Paris is a cosmopolitan and vibrant city located on the Île de France, an island off the coast of the French mainland, and is known for its rich cultural heritage, iconic landmarks, and gastronomy. The city is home to many world-renowned institutions and attractions, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, as well as numerous museums, theaters, and parks. Paris is also known for its unique culture and food, including its famous dishes such as galette d'asc
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, with significant advancements in areas such as machine learning, robotics, natural language processing, and computer vision. Some potential future trends in AI include:
    
    1. Increased cognitive power: With the advent of artificial neural networks and deep learning, it is possible that AI systems will become even more capable of processing information and making decisions than current human beings.
    
    2. Autonomous systems: Autonomous vehicles, drones, and other self-driving vehicles are expected to become more common in the future, as AI technology improves and becomes more integrated into our daily lives.
    
    3. AI for healthcare: AI has the potential to revolutionize the way we diagnose and treat diseases


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

     ______

    .

     I

    'm

     a

    /an

     __

    ________

    __

    _.

     I

     come

     from

     __

    ________

    .

     I

    'm

     ___

     years

     old

    .

     I

     love

     ___

     and

     enjoy

     __

    _.

     I

    'm

     a

    /an

     __

    ________

    _

     (

    e

    .g

    .

     active

    ,

     shy

    ,

     outgoing

    ,

     intro

    verted

    )

     and

     I

     like

     to

     __

    ________

     (

    e

    .g

    .

     clean

    ,

     explore

    ,

     have

     fun

    ).

     I

    'm

     looking

     forward

     to

     a

    /an

     __

    ________

    _

     (

    e

    .g

    .

     fun

    ,

     adventure

    ,

     adventure

    ).

     I

    'm

     confident

     in

     my

     __

    ________

     (

    e

    .g

    .

     knowledge

    ,

     abilities

    ,

     appearance

    )

     and

     I

    'm

     always

     __

    ________

     (

    e

    .g

    .

     optimistic

    ,

     open

    -minded

    ,

     willing

     to

     learn

    ).

     My

     goal

     is

     to

     __

    ________

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     French

     department

     of

     Paris

    .

     It

     serves

     as

     the

     administrative

    ,

     cultural

    ,

     and

     economic

     center

     of

     the

     nation

     and

     is

     home

     to

     many

     of

     the

     world

    's

     major

     museums

     and

     art

     galleries

    .

     Paris

     is

     also

     known

     for

     its

     vibrant

     nightlife

     and

     unique

     shopping

     culture

    .

     In

     addition

    ,

     the

     city

     is

     home

     to

     several

     well

    -known

     cultural

     institutions

     such

     as

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     as

     well

     as

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     also

     known

     for

     its

     gastr

    onomic

     scene

    ,

     with

     famous

     French

     dishes

     such

     as

     the

     cre

    pes

     and

     la

     amb

    ass

    ade

    ur

    .

     Paris

     is

     considered

     one

     of

     the

     most

     cosm

    opolitan

     cities

     in

     the

     world

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     there

     are

     many

     possibilities

     for

     the

     development

     and

     improvement

     of

     this

     technology

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     focus

     on

     ethics

     and

     safety

    :

     AI

     systems

     that

     are

     designed

     to

     make

     decisions

     that

     could

     potentially

     harm

     individuals

     or

     society

     as

     a

     whole

     will

     be

     regulated

     and

     monitored

     more

     closely

    .

     This

     will

     require

     a

     deep

     understanding

     of

     ethical

     considerations

     and

     a

     greater

     emphasis

     on

     safety

     in

     the

     design

     of

     AI

    .
    


    2

    .

     Integration

     with

     natural

     language

     processing

    :

     As

     AI

     becomes

     more

     integrated

     with

     our

     everyday

     lives

    ,

     we

     may

     see

     more

     widespread

     adoption

     of

     natural

     language

     processing

    .

     This

     will

     allow

     AI

     to

     understand

     and

     respond

     to

     human

     language

    ,

     leading

     to

     more

     natural

     and

     effective

     interactions

    



```python
llm.shutdown()
```
