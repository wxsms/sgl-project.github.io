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
    [2026-04-22 06:40:05] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.17it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.16it/s]


    2026-04-22 06:40:10,026 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 06:40:10] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.70it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.70it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.70it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.70it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.70it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.70it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.70it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.70it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.70it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.70it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 20.71it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 20.71it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 20.71it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 20.71it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 20.71it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 20.71it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 20.71it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 20.71it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 20.71it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 20.71it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]

    Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 29.49it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 37.37it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 37.37it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 37.37it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 37.37it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 37.37it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 37.37it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 37.37it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 37.37it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 37.37it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:03<00:00, 37.37it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 46.41it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 46.41it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 46.41it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 46.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=136.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=136.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=136.72 GB):   3%|▎         | 2/58 [00:00<00:03, 18.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=136.71 GB):   3%|▎         | 2/58 [00:00<00:03, 18.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=136.71 GB):   3%|▎         | 2/58 [00:00<00:03, 18.26it/s]Capturing num tokens (num_tokens=6144 avail_mem=136.71 GB):   3%|▎         | 2/58 [00:00<00:03, 18.26it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=136.71 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=136.71 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=5120 avail_mem=136.71 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=4608 avail_mem=136.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.27it/s]Capturing num tokens (num_tokens=4608 avail_mem=136.70 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=136.70 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=136.70 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=136.70 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=136.69 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=136.69 GB):  21%|██        | 12/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=136.69 GB):  21%|██        | 12/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=136.69 GB):  21%|██        | 12/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=2560 avail_mem=136.68 GB):  21%|██        | 12/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=2304 avail_mem=136.68 GB):  21%|██        | 12/58 [00:00<00:01, 29.46it/s]Capturing num tokens (num_tokens=2304 avail_mem=136.68 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=136.68 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=136.67 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=136.67 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=136.67 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.72it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=136.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=136.65 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.73it/s]Capturing num tokens (num_tokens=960 avail_mem=136.66 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.73it/s] Capturing num tokens (num_tokens=896 avail_mem=136.66 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.73it/s]Capturing num tokens (num_tokens=832 avail_mem=136.65 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.73it/s]Capturing num tokens (num_tokens=768 avail_mem=136.65 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.73it/s]Capturing num tokens (num_tokens=704 avail_mem=136.65 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.73it/s]Capturing num tokens (num_tokens=704 avail_mem=136.65 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.63it/s]Capturing num tokens (num_tokens=640 avail_mem=136.64 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.63it/s]Capturing num tokens (num_tokens=576 avail_mem=136.64 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.63it/s]Capturing num tokens (num_tokens=512 avail_mem=136.63 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.63it/s]

    Capturing num tokens (num_tokens=480 avail_mem=136.65 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.63it/s]Capturing num tokens (num_tokens=448 avail_mem=136.65 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.63it/s]Capturing num tokens (num_tokens=448 avail_mem=136.65 GB):  53%|█████▎    | 31/58 [00:00<00:00, 38.93it/s]Capturing num tokens (num_tokens=416 avail_mem=136.64 GB):  53%|█████▎    | 31/58 [00:00<00:00, 38.93it/s]Capturing num tokens (num_tokens=384 avail_mem=136.64 GB):  53%|█████▎    | 31/58 [00:00<00:00, 38.93it/s]Capturing num tokens (num_tokens=352 avail_mem=136.64 GB):  53%|█████▎    | 31/58 [00:00<00:00, 38.93it/s]Capturing num tokens (num_tokens=320 avail_mem=136.63 GB):  53%|█████▎    | 31/58 [00:00<00:00, 38.93it/s]Capturing num tokens (num_tokens=288 avail_mem=136.63 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.93it/s]Capturing num tokens (num_tokens=288 avail_mem=136.63 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=256 avail_mem=136.63 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=240 avail_mem=136.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.69it/s]

    Capturing num tokens (num_tokens=224 avail_mem=136.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=208 avail_mem=136.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=192 avail_mem=136.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=192 avail_mem=136.62 GB):  71%|███████   | 41/58 [00:01<00:00, 40.40it/s]Capturing num tokens (num_tokens=176 avail_mem=136.61 GB):  71%|███████   | 41/58 [00:01<00:00, 40.40it/s]Capturing num tokens (num_tokens=160 avail_mem=136.61 GB):  71%|███████   | 41/58 [00:01<00:00, 40.40it/s]Capturing num tokens (num_tokens=144 avail_mem=136.61 GB):  71%|███████   | 41/58 [00:01<00:00, 40.40it/s]Capturing num tokens (num_tokens=128 avail_mem=136.60 GB):  71%|███████   | 41/58 [00:01<00:00, 40.40it/s]Capturing num tokens (num_tokens=112 avail_mem=136.60 GB):  71%|███████   | 41/58 [00:01<00:00, 40.40it/s]Capturing num tokens (num_tokens=112 avail_mem=136.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.77it/s]Capturing num tokens (num_tokens=96 avail_mem=136.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.77it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=136.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.77it/s]Capturing num tokens (num_tokens=64 avail_mem=136.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.77it/s]Capturing num tokens (num_tokens=48 avail_mem=136.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.77it/s]Capturing num tokens (num_tokens=32 avail_mem=136.58 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.77it/s]Capturing num tokens (num_tokens=32 avail_mem=136.58 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=28 avail_mem=136.58 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=24 avail_mem=136.58 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=20 avail_mem=136.57 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=16 avail_mem=135.52 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=12 avail_mem=135.51 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.78it/s]

    Capturing num tokens (num_tokens=12 avail_mem=135.51 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.23it/s]Capturing num tokens (num_tokens=8 avail_mem=135.45 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.23it/s] Capturing num tokens (num_tokens=4 avail_mem=128.32 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.23it/s]Capturing num tokens (num_tokens=4 avail_mem=128.32 GB): 100%|██████████| 58/58 [00:01<00:00, 37.06it/s]


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
    Generated text:  Daria and I'm a student at the University of Miami. I'd like to talk about the future of STEM education, or STEM for short. As you can imagine, STEM education is centered around the foundation of Science, Technology, Engineering, and Math, or STEM for short. Science is all about the study of the world and how things work. Technology is all about how to make things. Engineering is about designing and building things. Finally, math is all about numbers and logic.
    STEM education is the foundation of all these other subjects and is an important step for many students to take. STEM education is also known as STEM education because
    ===============================
    Prompt: The president of the United States is
    Generated text:  a candidate for how many terms? 1. **Understanding the Problem:**
       - The president of the United States serves a fixed term of 4 years.
       - We need to determine how many terms the president can serve.
    
    2. **Analyzing the Information:**
       - The president's term is fixed and lasts for 4 years.
       - No other specific information is provided about other terms or periods.
    
    3. **Conclusion:**
       - Since the president serves a fixed term of 4 years and no other specific term periods are mentioned, we cannot determine the number of additional terms or periods beyond the 4-year
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The French people use a french flag. In fact, Paris is a part of France, which is an administrative division of the country. The capital of France, in fact, is the city of Paris.
    
    Paris is a very famous city, and many famous films and TV shows are filmed there. It is also the world’s most populous city. Paris is a very historic city with a very rich culture.
    
    Why is Paris so famous? The city has a beautiful landscape, but it is also very fashionable. Paris is famous for the famous Eiffel Tower. The Eiffel Tower is a very tall tower that rises in Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  not here today, but in the coming decades. By 2050, it is forecasted that AI is expected to play a crucial role in all areas of business, from healthcare to transportation and security. The tech sector is projected to generate over $4.3 trillion in revenue by 2030. While AI is still in the early stages of development, it is expected to become more widely used over time.
    As AI technology continues to advance, there are certain trends that will be most influential in shaping its future. Some of these trends include:
    1. AI will become more ubiquitous: The widespread adoption of AI will


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for business, finance, and tourism, making it a popular destination for tourists and locals alike. The city is known for its fashion industry, art scene, and food culture, and is a major hub for international trade and diplomacy. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and the trend is expected to continue. AI-powered diagnostic tools, virtual assistants, and personalized treatment plans are expected to become more common in the future.
    
    2. Increased use of AI in transportation: AI is already being used in transportation to help improve safety, efficiency, and convenience. Autonomous vehicles, self-driving
    


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
    Generated text:  [Name], and I am a [type of profession]. I love [why I love this profession], and I have always been a [what does this mean?].
    
    I've always been fascinated by [what it is that fascinates me], and I've always dreamed of [what it is that I've always dreamed of]. I have a passion for [what it is that I'm passionate about], and I strive to be the best version of myself. I'm also a [what do you do best?].
    
    I believe that my [what do you believe in?] will always be my compass, and I will always strive
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic city renowned for its rich cultural heritage, diverse neighborhoods, and iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral. It is also the largest city in France, with a population of approximately 2.7 million people. Paris has a rich history dating back to the Roman Empire and continues to be a hub of creativity, arts, and commerce. It is known for its delicious cuisine, charming architecture, and annual cultural events such as the Eiffel Tower Parade and the Louvre Museum's "Tango" dance. As one of the world's largest cities, Paris has a large population and a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  looking very promising, with many exciting developments and applications to come. Some possible future trends in AI include:
    
    1. Increased AI efficiency: As AI technology advances, we can expect to see better use cases for AI, as well as increased efficiency and productivity in many industries.
    
    2. AI-based healthcare: AI could be used to create more accurate and personalized diagnoses, as well as to develop more effective treatments for diseases.
    
    3. AI-driven automation: AI could be used to automate many of the tasks that people do, reducing the workload of human workers and freeing up time for more important tasks.
    
    4. AI-powered personalization: AI could be


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

    'm

     a

     [

    Job

     Title

    ]

     at

     [

    Company

    ].

     I

    'm

     a

     fan

     of

     [

    Theme

     or

     Interest

    ]

     and

     have

     always

     been

     passionate

     about

     [

    What

     You

     Do

     Best

    ].

     I

    'm

     always

     eager

     to

     learn

     something

     new

     and

     share

     it

     with

     others

    .

     My

     aim

     is

     to

     always

     provide

     the

     best

     service

     and

     assistance

     to

     my

     clients

    .

     I

    'm

     always

     ready

     to

     help

     if

     there

    's

     anything

     I

     can

     do

     to

     assist

     you

     in

     your

     journey

     to

     success

    .

     My

     name

     is

     [

    Name

    ]

     and

     I

    'm

     a

     [

    Job

     Title

    ]

     at

     [

    Company

    ].

     I

    'm

     a

     fan

     of

     [

    Theme

     or

     Interest

    ]

     and

     have

     always

     been

     passionate

     about

     [

    What

     You

     Do

     Best

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     beautiful

     and

     historic

     city

     known

     for

     its

     rich

     history

    ,

     famous

     landmarks

    ,

     and

     vibrant

     culture

    .

     Paris

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

    ,

     among

     many

     other

     attractions

    .

     It

    's

     also

     a

     center

     for

     art

    ,

     fashion

    ,

     and

     science

    ,

     with

     numerous

     museums

    ,

     museums

    ,

     and

     cultural

     institutions

     offering

     visitors

     a

     glimpse

     into

     the

     city

    's

     rich

     history

     and

     cultural

     heritage

    .

     With

     its

     warm

     weather

    ,

     vibrant

     nightlife

    ,

     and

     romantic

     ambiance

    ,

     Paris

     is

     a

     must

    -

    visit

     destination

     for

     anyone

     interested

     in

     exploring

     the

     French

     capital

    .

     
    


    1

    .

     Paris

     is

     the

     capital

     of

     France

     and

     is

     known

     for

     its

     rich

     history

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     variety

     of

     trends

    ,

     including

    :
    


     

     

    1

    .

     Increasing

     reliance

     on

     machine

     learning

     and

     artificial

     intelligence

    :

     AI

     is

     becoming

     more

     and

     more

     prevalent

     in

     everyday

     life

    ,

     from

     self

    -driving

     cars

     to

     personalized

     health

     and

     wellness

     apps

    .


     

     

    2

    .

     Improved

     processing

     power

     and

     storage

    :

     The

     rise

     of

     SSD

    s

     and

     H

    SM

    s

     (

    High

     Speed

     Memory

    )

     in

     computing

     devices

     is

     expected

     to

     improve

     the

     processing

     power

     and

     storage

     capacity

     of

     AI

     models

    ,

     allowing

     them

     to

     process

     larger

     amounts

     of

     data

     and

     train

     them

     faster

    .


     

     

    3

    .

     Greater

     emphasis

     on

     ethical

     AI

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     there

     will

     be

     a

     greater

     emphasis

     on

     ensuring

     that

     it

     is

    



```python
llm.shutdown()
```
