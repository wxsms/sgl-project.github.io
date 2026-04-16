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
    [2026-04-16 16:59:40] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.60it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.60it/s]


    2026-04-16 16:59:45,015 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 16:59:45] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:03,  3.22s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:03,  3.22s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:03,  3.22s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:03,  3.22s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:09,  4.89it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:09,  4.89it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:09,  4.89it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:09,  4.89it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:09,  4.89it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:09,  4.89it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:09,  4.89it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:09,  4.89it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:09,  4.89it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:09,  4.89it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:03<00:09,  4.89it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:03, 11.97it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:03, 11.97it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 19.40it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 19.40it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 19.40it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 19.40it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 19.40it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 19.40it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 19.40it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 19.40it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 25.17it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 25.17it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 25.17it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 25.17it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 25.17it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 25.17it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 25.17it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 25.17it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:03<00:00, 25.17it/s]

    Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:03<00:00, 25.17it/s]Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:03<00:00, 25.17it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 35.88it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 35.88it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 35.88it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 35.88it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 35.88it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 35.88it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 35.88it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 35.88it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 35.88it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:03<00:00, 35.88it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:03<00:00, 35.88it/s]Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:03<00:00, 35.88it/s] Compiling num tokens (num_tokens=4):  79%|███████▉  | 46/58 [00:03<00:00, 35.88it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.91it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.63it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.39 GB):   3%|▎         | 2/58 [00:00<00:04, 12.97it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.39 GB):   3%|▎         | 2/58 [00:00<00:04, 12.97it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.39 GB):   3%|▎         | 2/58 [00:00<00:04, 12.97it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.39 GB):   7%|▋         | 4/58 [00:00<00:03, 15.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.39 GB):   7%|▋         | 4/58 [00:00<00:03, 15.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.38 GB):   7%|▋         | 4/58 [00:00<00:03, 15.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.38 GB):   7%|▋         | 4/58 [00:00<00:03, 15.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.38 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.38 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.72it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=74.38 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.37 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.37 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.37 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.36 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.36 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.36 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.44it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.46it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.46it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.46it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.46it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.46it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.34 GB):  31%|███       | 18/58 [00:00<00:01, 28.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.34 GB):  31%|███       | 18/58 [00:00<00:01, 28.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.34 GB):  31%|███       | 18/58 [00:00<00:01, 28.96it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.32 GB):  31%|███       | 18/58 [00:00<00:01, 28.96it/s]Capturing num tokens (num_tokens=960 avail_mem=74.33 GB):  31%|███       | 18/58 [00:00<00:01, 28.96it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=74.33 GB):  38%|███▊      | 22/58 [00:00<00:01, 25.22it/s]Capturing num tokens (num_tokens=896 avail_mem=74.33 GB):  38%|███▊      | 22/58 [00:00<00:01, 25.22it/s]Capturing num tokens (num_tokens=832 avail_mem=74.33 GB):  38%|███▊      | 22/58 [00:00<00:01, 25.22it/s]Capturing num tokens (num_tokens=768 avail_mem=74.32 GB):  38%|███▊      | 22/58 [00:00<00:01, 25.22it/s]Capturing num tokens (num_tokens=704 avail_mem=74.02 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.22it/s]Capturing num tokens (num_tokens=704 avail_mem=74.02 GB):  45%|████▍     | 26/58 [00:01<00:01, 26.92it/s]Capturing num tokens (num_tokens=640 avail_mem=74.29 GB):  45%|████▍     | 26/58 [00:01<00:01, 26.92it/s]Capturing num tokens (num_tokens=576 avail_mem=74.28 GB):  45%|████▍     | 26/58 [00:01<00:01, 26.92it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.27 GB):  45%|████▍     | 26/58 [00:01<00:01, 26.92it/s]Capturing num tokens (num_tokens=512 avail_mem=74.27 GB):  50%|█████     | 29/58 [00:01<00:01, 25.07it/s]Capturing num tokens (num_tokens=480 avail_mem=74.06 GB):  50%|█████     | 29/58 [00:01<00:01, 25.07it/s]Capturing num tokens (num_tokens=448 avail_mem=74.26 GB):  50%|█████     | 29/58 [00:01<00:01, 25.07it/s]Capturing num tokens (num_tokens=416 avail_mem=74.28 GB):  50%|█████     | 29/58 [00:01<00:01, 25.07it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.28 GB):  55%|█████▌    | 32/58 [00:01<00:01, 21.30it/s]Capturing num tokens (num_tokens=384 avail_mem=74.08 GB):  55%|█████▌    | 32/58 [00:01<00:01, 21.30it/s]Capturing num tokens (num_tokens=352 avail_mem=74.27 GB):  55%|█████▌    | 32/58 [00:01<00:01, 21.30it/s]Capturing num tokens (num_tokens=320 avail_mem=74.26 GB):  55%|█████▌    | 32/58 [00:01<00:01, 21.30it/s]Capturing num tokens (num_tokens=320 avail_mem=74.26 GB):  60%|██████    | 35/58 [00:01<00:01, 21.37it/s]Capturing num tokens (num_tokens=288 avail_mem=74.26 GB):  60%|██████    | 35/58 [00:01<00:01, 21.37it/s]Capturing num tokens (num_tokens=256 avail_mem=74.25 GB):  60%|██████    | 35/58 [00:01<00:01, 21.37it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.25 GB):  60%|██████    | 35/58 [00:01<00:01, 21.37it/s]Capturing num tokens (num_tokens=240 avail_mem=74.25 GB):  66%|██████▌   | 38/58 [00:01<00:00, 23.12it/s]Capturing num tokens (num_tokens=224 avail_mem=74.24 GB):  66%|██████▌   | 38/58 [00:01<00:00, 23.12it/s]Capturing num tokens (num_tokens=208 avail_mem=74.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 23.12it/s]Capturing num tokens (num_tokens=192 avail_mem=74.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 23.12it/s]Capturing num tokens (num_tokens=176 avail_mem=74.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 23.12it/s]Capturing num tokens (num_tokens=176 avail_mem=74.22 GB):  72%|███████▏  | 42/58 [00:01<00:00, 25.77it/s]Capturing num tokens (num_tokens=160 avail_mem=74.22 GB):  72%|███████▏  | 42/58 [00:01<00:00, 25.77it/s]Capturing num tokens (num_tokens=144 avail_mem=74.21 GB):  72%|███████▏  | 42/58 [00:01<00:00, 25.77it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.20 GB):  72%|███████▏  | 42/58 [00:01<00:00, 25.77it/s]Capturing num tokens (num_tokens=112 avail_mem=74.20 GB):  72%|███████▏  | 42/58 [00:01<00:00, 25.77it/s]Capturing num tokens (num_tokens=112 avail_mem=74.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=96 avail_mem=74.19 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.67it/s] Capturing num tokens (num_tokens=80 avail_mem=74.16 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=64 avail_mem=74.17 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=48 avail_mem=74.17 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=32 avail_mem=74.16 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=32 avail_mem=74.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.32it/s]Capturing num tokens (num_tokens=28 avail_mem=74.13 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.32it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.17 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.32it/s]Capturing num tokens (num_tokens=20 avail_mem=74.14 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.32it/s]Capturing num tokens (num_tokens=16 avail_mem=74.16 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.32it/s]Capturing num tokens (num_tokens=12 avail_mem=74.12 GB):  88%|████████▊ | 51/58 [00:02<00:00, 32.32it/s]Capturing num tokens (num_tokens=12 avail_mem=74.12 GB):  97%|█████████▋| 56/58 [00:02<00:00, 35.07it/s]Capturing num tokens (num_tokens=8 avail_mem=74.12 GB):  97%|█████████▋| 56/58 [00:02<00:00, 35.07it/s] Capturing num tokens (num_tokens=4 avail_mem=74.11 GB):  97%|█████████▋| 56/58 [00:02<00:00, 35.07it/s]Capturing num tokens (num_tokens=4 avail_mem=74.11 GB): 100%|██████████| 58/58 [00:02<00:00, 26.83it/s]


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
    Generated text:  Susan and I have just started working at a small company. I can use a few tools and see where I’m going, but I don’t know how to use them. What should I do?
    
    Choosing the right tools to use is an important step in setting up a successful career and can save you time and effort in the long run. Here are some steps to help you find the right tools for your needs:
    
    1. Identify your goals and objectives: Think about what you want to achieve in your career and how you will use the tools you have.
    
    2. Evaluate your current tools: Before you start looking for new tools, take a
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 10 inches tall. The vice president is 5 feet 7 inches tall. If 1 inch represents 3 feet, how tall is the president in terms of feet and inches?
    
    To determine the height of the president in terms of feet and inches, we need to convert the heights from feet and inches to just feet. We know that 1 inch is equivalent to 3 feet.
    
    First, let's convert the height of the president from feet and inches to just feet. The president's height is 5 feet 10 inches. To convert 10 inches to feet, we divide 10
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Berlin
    D. Moscow
    Answer:
    
    A
    
    The capital of France is:
    A. Paris
    B. London
    C. Berlin
    D. Moscow
    
    Answer:
    
    A
    
    What are the specific levels of the international regulatory structure?
    A. National level
    B. Regional level
    C. Industry level
    D. Corporate level
    E. Global level
    
    Answer:
    
    ABCD
    
    For the following sentences, identify the sentence with a specific question mark. Choose the correct answer from the options provided.
    1. Is this movie interesting?
    2. How are you?
    3. Does he
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the world's most influential minds. Join a thought-provoking discussion on the future of AI that will shape the future of the tech industry and beyond. The conversation will feature a panel of experts who will explore the future of AI, including the impact of AI on society, the ethical implications of AI, and the future of AI research. The conversation will be moderated by a distinguished AI researcher and will be followed by a Q&A session with the panelists.
    Panelists: 
    1. Professor X - AI researcher 
    2. Mr. Y - Tech entrepreneur 
    3. Dr. Z - AI ethicist 
    4


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working at [company name] for [number of years] years. I have always been passionate about [job title] and have always wanted to be a [job title] myself. I am always looking for new challenges and opportunities to grow and learn. I am a [job title] who is always looking for ways to improve my skills and knowledge. I am a [job title] who is always willing to learn and adapt to new situations. I am a [job title] who is always looking for ways to make a positive impact on
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is accurate and brief, capturing the essence of Paris' importance as the capital city of France. It provides a clear and concise overview of the capital's role in French politics, culture, and society. 
    
    To elaborate further, Paris is the largest city in France and the second-largest city in the European Union. It is home to the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and many other iconic landmarks. The city is also known for its rich history, including the French Revolution, the French Revolution, and the French Revolution. 
    
    Paris is a bustling metropolis with a diverse
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI technology continues to improve, we can expect to see more automation and AI-driven applications in various industries. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new opportunities for innovation and growth.
    
    2. Improved privacy and security: As AI technology becomes more advanced, we can expect to see more privacy and security concerns. This could lead to increased regulation and oversight of AI systems, as well as
    


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
    Generated text:  [Name] and I am a [field or profession]. I have been living in [location] for [number] years and I currently [occupation]. I have [number] years of experience in [occupation] and have been working in this field for [number] years. My primary [interest or hobby], [mention any relevant interests or hobbies], is [mention any relevant interests or hobbies]. I have always [mention any relevant personal qualities or values]. I am [mention any relevant achievements or accomplishments].
    Welcome to [Name], I am [Name] and I am a [field or profession]. I have been living in [location
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and is home to numerous museums, monuments, and cultural institutions. Paris has a rich history and is known for its art, literature, and cuisine. It is also a popular tourist destination and attracts millions of visitors each year. The city is home to many European Union cities and is a major hub for business and trade. Paris has a diverse population and is home to many minority communities. It is also known for its culinary traditions and has been recognized as a UNESCO World Heritage site.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are some potential trends that are likely to shape the technology and applications in the next few years:
    
    1. Increased automation: AI will become more efficient and accurate, and will be able to perform repetitive and mundane tasks more quickly and accurately than humans. This will lead to an increase in automation and automation-driven jobs, as well as a decrease in the need for human labor.
    
    2. Deep learning: Deep learning is a subset of machine learning that involves using neural networks to learn from large datasets. This will enable AI to perform tasks that were previously beyond the scope of human capability, such as image recognition, natural language processing


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

    'm

     [

    Age

    ]

     years

     old

    ,

     and

     I

     live

     in

     [

    Current

     Location

    ].

     I

    'm

     currently

     [

    Occup

    ation

    ]

     and

     have

     a

     [

    Professional

     Skill

    set

    ]

     background

    .

     I

    'm

     [

    H

    obbies

    ]

     that

     I

     enjoy

     spending

     time

     on

    ,

     which

     include

     [

    List

     of

     hobbies

    ].

     I

     also

     like

     [

    Activity

    ],

     which

     I

     do

     for

     fun

    .

     I

     have

     a

     [

    Friend

    ship

     Status

    ],

     and

     I

     love

     [

    Favorite

     Food

    ,

     Video

     Game

    ,

     Hobby

    ,

     etc

    .

    ].

     I

     believe

     in

     [

    Statement

     about

     the

     self

    ],

     and

     I

    'm

     always

     [

    Express

    ive

    ].

     I

    'm

     excited

     to

     meet

     you

    .

     What

    's

     your

     name

    ,

     and

     what

     kind

     of

     experience

     can

     you

     bring

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

     and

     European

     cultural

     and

     political

     center

    .

     It

     is

     the

     world

    's

     third

    -largest

     city

     by

     population

     and

     eighth

    -largest

     by

     area

    .
    


    To

     translate

     the

     provided

     factual

     statement

     into

     Spanish

    ,

     you

     can

     say

    :
    


    La

     capital

     de

     Franc

    ia

     es

     Par

    ís

    ,

     una

     ciudad

     histó

    rica

     y

     c

    ív

    ica

     europe

    a

     y

     cultural

     y

     político

     central

    .

     Es

     la

     ciudad

     número

     

    3

     más

     grande

     por

     población

     en

     la

     Un

    ión

     Europe

    a

     y

     número

     

    8

     más

     grande

     por

     superf

    icie

    .
    


    Here

    ,

     I

    've

     tried

     to

     accurately

     convey

     the

     meaning

     of

     each

     key

     term

     in

     the

     original

     statement

    ,

     including

    :
    


    -

     "

    Capital

     of

     France

    "

     -

     "

    la

     capital

     de

     Franc

    ia

    "


    -

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     unpredictable

    ,

     but

     here

     are

     some

     potential

     trends

     we

     can

     expect

     to

     see

     in

     the

     next

     decade

    :
    


    1

    .

     Integration

     of

     AI

     into

     everyday

     life

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     may

     see

     more

     and

     more

     applications

     of

     AI

     in

     our

     daily

     lives

    .

     This

     could

     include

     things

     like

     smart

     home

     devices

     that

     can

     recognize

     patterns

     in

     our

     behavior

     and

     recommend

     personalized

     recommendations

     to

     us

    ,

     or

     AI

    -powered

     virtual

     assistants

     that

     can

     understand

     our

     needs

     and

     make

     recommendations

     based

     on

     our

     preferences

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     With

     the

     rise

     of

     big

     data

     and

     artificial

     intelligence

    ,

     we

     may

     see

     more

     and

     more

     use

     of

     AI

     in

     healthcare

    .

     This

     could

     include

     the

     development

     of

     more

    



```python
llm.shutdown()
```
