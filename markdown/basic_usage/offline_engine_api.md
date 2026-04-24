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
    [2026-04-24 12:00:07] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.53it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.52it/s]


    2026-04-24 12:00:11,583 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 12:00:11] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:40,  2.81s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.25it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.25it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:03<00:23,  2.25it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.57it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.42it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.42it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.42it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.42it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.42it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.42it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.42it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.42it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.42it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.42it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.32it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.32it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.32it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.32it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.32it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.32it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.32it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.32it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.32it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.32it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.97it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 29.97it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 38.66it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.03it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:04, 13.79it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:04, 13.79it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:04, 13.79it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:04, 13.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.56it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.56it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.56it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.56it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.56it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.47it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  31%|███       | 18/58 [00:00<00:01, 33.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 33.85it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 33.85it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  31%|███       | 18/58 [00:00<00:01, 33.85it/s]Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  31%|███       | 18/58 [00:00<00:01, 33.85it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  31%|███       | 18/58 [00:00<00:01, 33.85it/s]Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.65it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.46it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.46it/s]

    Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.46it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.46it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.46it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.46it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.87it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.87it/s]

    Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.32it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.32it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.32it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.32it/s]

    Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.32it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.32it/s]Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.09it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.03it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.03it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.03it/s]

    Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.03it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.03it/s] Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.03it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 36.53it/s]


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
    Generated text:  Simon. I'm an experienced language tutor and I'm working on a story. I would like you to help me with the story. I need you to go through the story and find all the characters, the dialogue, the dialogue between characters, the dialogue between the characters and dialogue between characters and the narrator. Once I have this information, I will help you in writing the story in a more natural and realistic way.
    
    Sure, I would love to help you with that. Please go ahead and share the story with me. Once I have it, I will go through it and help you in writing it in a more natural and realistic way
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term. There are 5 existing presidents who have served for 3 years, and the current president has served for 5 years. How many years of experience do the current president have compared to the existing presidents?
    
    To determine how many years of experience the current president has compared to the existing presidents, we need to follow these steps:
    
    1. Identify the number of years of experience the current president has.
    2. Identify the number of years of experience each of the existing presidents has.
    3. Add the years of experience of the current president to the years of experience of the existing presidents.
    
    From the problem, we know
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It was established in 1164 by the French king Charles I. The city was founded in 799 by the Romans, but it was abandoned. In 1267, the city was lost to the army of the Cathars. In 1358, the city was given to Philip IV, who made it his capital. In 1420, the city was given to Louis XIII, the last French king. In 1792, the French Revolution ended, and in 1795, the city was given to the king Louis XVI. In 
    ===============================
    Prompt: The future of AI is
    Generated text:  set to be shaped by the AI it develops. But what about the impact of AI on the education system? In a recent paper, researchers conducted a review on the impact of artificial intelligence on education. Their findings suggest that AI could help bridge the digital divide, improve student outcomes, and reduce the need for traditional education methods.
    The authors argue that AI has the potential to revolutionize education by automating tasks and providing personalized learning experiences. They also believe that AI can help address the digital divide by ensuring that all students have access to technology and educational resources. In addition, AI can help reduce the need for traditional teaching methods by automating administrative


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working here for [number of years] years. I am a [occupation] with a passion for [interest or hobby]. I am always looking for ways to [describe a new skill or initiative]. I am [age] years old and I am [gender]. I am [any other relevant information about your character]. I am [any other relevant information about your character]. I am [any other relevant information about your character]. I am [any other relevant information about your character]. I am [any other relevant information about your character]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a popular tourist destination, with a rich history and culture that attracts millions of visitors each year. Paris is a city of art, fashion, and cuisine, and is a major hub for business and commerce in Europe. The city is home to many famous landmarks and attractions, including the Palace of Versailles, the Champs-Élysées, and the Musée d'Orsay. Paris is a city that has been a center of politics, culture, and commerce for centuries,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see even more widespread use in healthcare.
    
    3. Increased use of AI in finance: AI is already being
    


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
    Generated text:  [Name]. I’m a young man with a passion for [occupation or hobby]. What brings you to the table today?
    
    As the character introduces themselves, they should include:
    - Their name (e.g., "My name is [Name].")
    - Their profession or hobby (e.g., "I'm a [occupation/hobby] with a passion for [interest or hobby].")
    - A brief history of their journey in the field (e.g., "My journey in the field started when [background].")
    - A statement of their current status (e.g., "I'm a [status] with a goal of [goal
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. The city is known for its rich history, stunning architecture, and vibrant culture. It's home to iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, as well as the many festivals and events that keep the city alive. The city has a diverse population of more than 2 million people, making it a vibrant and multicultural place. It's also home to some of the world's most famous museums, restaurants, and restaurants, making it a hub for entertainment and dining. Paris is a city that's not just for tourists, but for locals too, with its sophisticated culture, delicious cuisine
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid advancements in areas such as machine learning, deep learning, and natural language processing. Here are some possible trends that are currently being discussed and expected to continue in the coming years:
    
    1. Increased reliance on AI-powered automation: AI is being increasingly used to automate repetitive and mundane tasks, freeing up human resources to focus on more complex and creative work. This trend is expected to continue as AI becomes more sophisticated and capable of performing tasks that are currently performed by humans.
    
    2. The integration of AI with other technologies: AI is being integrated with other technologies, such as blockchain, to create new and innovative applications. This


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

    ]

     and

     I

     am

     a

     [

    occupation

    ]

     with

     a

     passion

     for

     [

    insert

     a

     hobby

     or

     activity

    ].

     I

     am

     [

    number

    ]

     of

     [

    years

    ]

     years

     old

    .

     I

     am

     a

     [

    any

     age

    ]

     person

     who

     loves

     [

    insert

     a

     hobby

     or

     activity

    ],

     which

     I

    've

     been

     doing

     for

     [

    number

     of

     years

    ]

     years

    .

     I

     am

     a

     [

    any

     age

    ]

     person

     who

     is

     always

     [

    insert

     a

     trait

     or

     quality

     that

     makes

     you

     unique

    ].

     I

     am

     a

     [

    any

     age

    ]

     person

     who

     is

     [

    insert

     a

     personal

     value

     or

     principle

    ].

     I

     am

     [

    any

     age

    ]

     person

     who

     is

     always

     [

    insert

     a

     characteristic

     that

     defines

     you

    ].

     I

     am

     a

     [

    any

     age

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

     and

     the

     European

     Capital

     of

     Culture

    .


    Paris

     is

     a

     cultural

     and

     artistic

     capital

     with

     a

     rich

     history

     dating

     back

     over

     

    2

    ,

    0

    0

    0

     years

    .

     The

     city

     is

     home

     to

     many

     world

    -ren

    owned

     museums

    ,

     theaters

    ,

     and

     galleries

    ,

     as

     well

     as

     a

     vibrant

     nightlife

     and

     food

     scene

    .

     It

     is

     also

     a

     bustling

     city

     with

     a

     population

     of

     over

     

    1

    .

     

    7

     million

     people

    .

     The

     city

     has

     been

     hosting

     many

     international

     events

     and

     festivals

     throughout

     the

     year

    ,

     including

     the

     famous

     Les

     Grand

    es

     Od

    alis

    ques

     exhibition

     and

     the

     World

     Cup

     football

     match

    .

     Paris

     has

     played

     an

     important

     role

     in

     French

     history

    ,

     with

     the

     French

     Revolution

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     undoubtedly

     one

     of

     rapid

     change

     and

     innovation

    .

     As

     we

     progress

     in

     technology

    ,

     the

     way

     we

     interact

     with

     machines

     and

     other

     intelligent

     systems

     is

     constantly

     evolving

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

     Autonomous

     systems

    :

     We

     may

     see

     more

     autonomous

     vehicles

    ,

     drones

    ,

     and

     other

     autonomous

     machines

     become

     more

     common

     in

     our

     daily

     lives

    .

     AI

     will

     likely

     improve

     the

     efficiency

     and

     safety

     of

     these

     systems

    .
    


    2

    .

     Personal

    ization

    :

     As

     we

     learn

     more

     about

     individual

     users

     and

     their

     behaviors

    ,

     AI

     will

     become

     more

     personalized

     and

     efficient

    .

     Machine

     learning

     algorithms

     will

     be

     able

     to

     analyze

     data

     and

     provide

     more

     accurate

     and

     relevant

     recommendations

    .
    


    3

    .

     Bi

    ometric

     security

    :

     Bi

    ometric

     systems

     will

     become

     more

    



```python
llm.shutdown()
```
