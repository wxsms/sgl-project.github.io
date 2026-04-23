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
    [2026-04-23 03:29:56] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.24it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.23it/s]


    2026-04-23 03:30:01,354 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 03:30:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.71it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.73it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.53it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.53it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.53it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.30it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.30it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.30it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.30it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.30it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.30it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.30it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.30it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.30it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.30it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.30it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.76it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.80it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.36 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.35 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.09it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=137.35 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.09it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.35 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.35 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=136.85 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=136.84 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=136.84 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=136.75 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=136.68 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=136.67 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.10it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=136.67 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=1280 avail_mem=136.67 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=1280 avail_mem=136.67 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=1024 avail_mem=136.65 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=960 avail_mem=136.66 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.40it/s] Capturing num tokens (num_tokens=896 avail_mem=136.66 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=832 avail_mem=136.65 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=768 avail_mem=136.65 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=768 avail_mem=136.65 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.10it/s]Capturing num tokens (num_tokens=704 avail_mem=136.65 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.10it/s]Capturing num tokens (num_tokens=640 avail_mem=136.64 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.10it/s]

    Capturing num tokens (num_tokens=576 avail_mem=136.64 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.10it/s]Capturing num tokens (num_tokens=512 avail_mem=136.63 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.10it/s]Capturing num tokens (num_tokens=480 avail_mem=136.65 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.10it/s]Capturing num tokens (num_tokens=480 avail_mem=136.65 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.57it/s]Capturing num tokens (num_tokens=448 avail_mem=136.65 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.57it/s]Capturing num tokens (num_tokens=416 avail_mem=136.64 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=384 avail_mem=136.64 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=352 avail_mem=136.64 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=320 avail_mem=136.63 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.57it/s]Capturing num tokens (num_tokens=320 avail_mem=136.63 GB):  60%|██████    | 35/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=288 avail_mem=136.63 GB):  60%|██████    | 35/58 [00:01<00:00, 39.11it/s]

    Capturing num tokens (num_tokens=256 avail_mem=136.63 GB):  60%|██████    | 35/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=240 avail_mem=136.62 GB):  60%|██████    | 35/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=224 avail_mem=136.62 GB):  60%|██████    | 35/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=208 avail_mem=136.62 GB):  60%|██████    | 35/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=208 avail_mem=136.62 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=192 avail_mem=136.62 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=176 avail_mem=136.61 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=160 avail_mem=136.61 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=144 avail_mem=136.61 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=128 avail_mem=136.60 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.41it/s]

    Capturing num tokens (num_tokens=128 avail_mem=136.60 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=112 avail_mem=136.60 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=96 avail_mem=136.60 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.15it/s] Capturing num tokens (num_tokens=80 avail_mem=136.59 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=64 avail_mem=136.59 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=48 avail_mem=136.59 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=48 avail_mem=136.59 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=32 avail_mem=136.58 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=28 avail_mem=136.58 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=24 avail_mem=136.58 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=20 avail_mem=136.57 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.45it/s]

    Capturing num tokens (num_tokens=16 avail_mem=136.57 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=16 avail_mem=136.57 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=12 avail_mem=136.57 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=8 avail_mem=136.56 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.16it/s] Capturing num tokens (num_tokens=4 avail_mem=136.56 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=4 avail_mem=136.56 GB): 100%|██████████| 58/58 [00:01<00:00, 35.60it/s]


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
    Generated text:  C. Hello, my name is C.
    
    Translate this into Spanish.
    
    Translation: Hello, my name is C. Hello, my name is C. 
    
    Spanish translation: Hola, mi nombre es C. Hola, mi nombre es C.
    You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can, with no instructions required. To achieve this, pay attention to the language used in the sentence and use your language understanding skills to create the equivalent translation. If the sentence is a statement, do not add or remove words. If the sentence contains a question,
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to increase the number of jobs in his country. His plan involves a job market survey. After the survey, the unemployment rate in the country is determined. Let's assume the unemployment rate is calculated using the formula $U = \frac{D}{D + I + G + X} = \frac{1}{2}$. In the sample data of the job market survey, there are 200 unemployed individuals, 50 workers, 30 companies, and 10,000 individuals in the country.
    
    1. **Calculate the total number of individuals in the country given the unemployment rate.**
    
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Nice
    B. Paris
    C. Lyon
    D. Marseille
    Answer:
    B
    
    A certain brand of ice cream is priced at 115 yuan, while another brand is priced at 105 yuan. When buying 100 ice cream bars of the first brand and 100 ice cream bars of the second brand, the total cost is 2300 yuan. What is the cost per ice cream bar of the first brand? 
    A. 1.05 yuan 
    B. 1.25 yuan 
    C. 1.20 yuan 
    D
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it can change the world in many ways. Whether it's transforming the way we live, work, and interact with each other, AI is going to change the landscape of every industry, every industry. With the right expertise and knowledge, we can develop and implement AI solutions that will help us make our lives easier, our work more efficient, and our industries better. But before we can do that, we need to be aware of the various types of AI, and understand what each one does.
    There are two main types of AI: Supervised learning and unsupervised learning. Supervised learning involves training a machine learning model


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a brief description of your personality or skills that you're passionate about]. I'm always looking for new challenges and opportunities to grow and learn. What are some of your favorite hobbies or interests? I enjoy [insert a hobby or interest that you're passionate about]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie? I love [insert a favorite book or movie that you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the annual Eiffel Tower Festival. It is also the seat of the French government and the largest city in France by population. Paris is a cultural and historical center with many famous landmarks, including the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major transportation hub, with the iconic Eiffel Tower serving as a landmark for millions of tourists each year. Paris is a vibrant and diverse city with a rich history and a thriving economy. Its status as the capital of France has made it a major economic and cultural center
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are already being used in a wide range of applications, from self-driving cars to personalized medicine to fraud detection. As these technologies continue to evolve, we can expect to see even more innovative applications emerge, such as autonomous robots, virtual assistants, and intelligent home systems. Additionally, AI will likely continue to be integrated into everyday life, with more people relying on AI-powered tools to perform tasks such as shopping, banking, and entertainment. Finally, AI will likely continue to be used to address some of the most pressing challenges of
    


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
    Generated text:  [Name]. I am a [Occupation] who has always been passionate about [Reason for Passion]. I believe that my unique combination of [Other Traits] makes me a valuable asset to any team or organization. I'm always eager to learn and grow, and I'm always up for challenges. And who knows, maybe one day I'll be able to turn my skills into something really cool! [Name] [Hire me here] or [Name] [Apply here]. What would you like to do next? [Name] [Hire me here] or [Name] [Apply here]. What would you like to do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and the fifth-largest city in the world. It is a major global economic, political, and cultural hub, with a rich history of art, literature, and music. The city is known for its iconic landmarks, such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. Paris is also home to numerous museums, theaters, and art galleries, as well as a diverse range of restaurants, shops, and food options. The city is known for its vibrant nightlife, and there are many clubs and bars in the French Quarter,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities and possibilities to be explored, but there are some potential trends to keep an eye on. Here are some of the most likely future trends in artificial intelligence:
    
    1. Increased Personalization: With AI, we can expect more personalized experiences for users. This could be through chatbots, virtual assistants, or even more sophisticated personalization algorithms. Personalization could also be used to offer personalized product recommendations, web browsing experiences, or even weather forecasts.
    
    2. Greater Robustness: AI is becoming more sophisticated and is being able to process and analyze large amounts of data more effectively. This means that AI models are becoming more robust


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

     Alex

    .

     I

    'm

     an

     aspiring

     author

    ,

     but

     I

    'm

     currently

     working

     as

     a

     copy

    writer

    .

     What

     brings

     you

     to

     the

     world

     of

     publishing

    ?

     I

    'm

     fascinated

     by

     the

     world

     of

     creativity

     and

     the

     way

     it

     can

     be

     used

     to

     tell

     stories

     and

     entertain

    .

     What

     do

     you

     love

     about

     the

     creative

     process

    ?

     I

     love

     being

     able

     to

     create

     and

     write

     my

     own

     stories

    ,

     and

     I

    'm

     excited

     to

     bring

     them

     to

     life

     on

     the

     pages

     of

     a

     book

    .

     What

     are

     your

     favorite

     genres

     or

     styles

     of

     writing

    ?

     I

     love

     to

     write

     science

     fiction

    ,

     but

     I

    'm

     also

     interested

     in

     horror

    ,

     mystery

    ,

     and

     romance

    .

     What

     inspires

     you

     in

     your

     work

    ?

     I

    'm

     inspired

     by

     the

     stories

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    .

     True

    


    B

    .

     False

    
    


    To

     determine

     whether

     the

     statement

     "

    Paris

     is

     the

     capital

     of

     France

    "

     is

     true

     or

     false

    ,

     I

     will

     analyze

     the

     information

     provided

    :
    


    1

    .

     Paris

     is

     the

     capital

     city

     of

     France

    .


    2

    .

     The

     capital

     city

     of

     France

     is

     Paris

    .
    


    Based

     on

     this

     information

    ,

     the

     statement

     is

     accurate

    .

     Paris

     is

     the

     capital

     of

     France

     and

     is

     the

     official

     and

     most

     populous

     city

     in

     the

     country

    .
    


    Therefore

    ,

     the

     correct

     answer

     is

    :
    


    A

    .

     True

    
    


    A

     French

     official

     statement

     that

     Paris

     is

     the

     capital

     of

     France

     is

     fact

    ually

     accurate

    .

     This

     is

     because

    :


    -

     It

     is

     the

     official

     capital

     city

     of

     France

    .


    -

     It

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     exciting

     and

     many

     potential

     trends

     can

     be

     identified

    .

     Here

     are

     some

     of

     the

     most

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     The

     rise

     of

     automation

     and

     AI

     in

     manufacturing

    :

     AI

     is

     set

     to

     transform

     the

     manufacturing

     industry

    ,

     with

     robots

     and

     automated

     processes

     becoming

     increasingly

     prevalent

    .

     This

     could

     lead

     to

     new

     job

     losses

    ,

     but

     it

     could

     also

     create

     new

     job

     opportunities

    ,

     including

     roles

     such

     as

     AI

     developers

     and

     robotics

     engineers

    .
    


    2

    .

     AI

     in

     healthcare

    :

     AI

     is

     already

     making

     strides

     in

     healthcare

    ,

     with

     medical

     diagnoses

     improving

    ,

     treatment

     plans

     being

     personalized

    ,

     and

     drug

     discovery

     accelerated

    .

     As

     AI

     becomes

     more

     advanced

     and

     widely

     available

    ,

     it

     could

     play

     a

     significant

     role

     in

     improving

     patient

     outcomes

    .
    


    3

    



```python
llm.shutdown()
```
