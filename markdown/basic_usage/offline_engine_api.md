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
    [2026-04-21 21:33:35] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.02it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.01it/s]


    2026-04-21 21:33:42,143 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-21 21:33:42] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:43,  2.87s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:43,  2.87s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:43,  2.87s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:43,  2.87s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.75it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.75it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.75it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:30,  1.75it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:30,  1.75it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:30,  1.75it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:30,  1.75it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.38it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.38it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.38it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.38it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.38it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.38it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.38it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.38it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.38it/s]

    Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:03<00:03, 12.38it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:03<00:01, 20.29it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 27.88it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 27.88it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 27.88it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 27.88it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 27.88it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 27.88it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 27.88it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 27.88it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:03<00:00, 27.88it/s]

    Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:03<00:00, 27.88it/s]Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:03<00:00, 27.88it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 38.50it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 38.50it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 38.50it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 38.50it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 38.50it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 38.50it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 38.50it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 38.50it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 38.50it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:03<00:00, 38.50it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:03<00:00, 38.50it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 49.53it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 49.53it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 49.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.01 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.01 GB):   7%|▋         | 4/58 [00:00<00:03, 16.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.01 GB):   7%|▋         | 4/58 [00:00<00:03, 16.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.01 GB):   7%|▋         | 4/58 [00:00<00:03, 16.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.01 GB):  10%|█         | 6/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):  10%|█         | 6/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):  10%|█         | 6/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.00 GB):  10%|█         | 6/58 [00:00<00:02, 17.43it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:02, 21.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.99 GB):  21%|██        | 12/58 [00:00<00:02, 22.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  21%|██        | 12/58 [00:00<00:02, 22.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  21%|██        | 12/58 [00:00<00:02, 22.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.98 GB):  21%|██        | 12/58 [00:00<00:02, 22.65it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  21%|██        | 12/58 [00:00<00:02, 22.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.87it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.87it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.87it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.87it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.45it/s]Capturing num tokens (num_tokens=960 avail_mem=70.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.45it/s] Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.45it/s]

    Capturing num tokens (num_tokens=832 avail_mem=70.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.45it/s]Capturing num tokens (num_tokens=768 avail_mem=70.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.45it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.45it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.71it/s]Capturing num tokens (num_tokens=640 avail_mem=70.94 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.71it/s]Capturing num tokens (num_tokens=576 avail_mem=70.94 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.71it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.71it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.71it/s]Capturing num tokens (num_tokens=448 avail_mem=70.95 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.71it/s]Capturing num tokens (num_tokens=448 avail_mem=70.95 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=416 avail_mem=70.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=384 avail_mem=70.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.15it/s]

    Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=320 avail_mem=70.93 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.62it/s]Capturing num tokens (num_tokens=256 avail_mem=70.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.62it/s]Capturing num tokens (num_tokens=240 avail_mem=70.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.62it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.62it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.62it/s]Capturing num tokens (num_tokens=192 avail_mem=70.91 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.62it/s]Capturing num tokens (num_tokens=192 avail_mem=70.91 GB):  71%|███████   | 41/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  71%|███████   | 41/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  71%|███████   | 41/58 [00:01<00:00, 42.64it/s]

    Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  71%|███████   | 41/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=128 avail_mem=70.90 GB):  71%|███████   | 41/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=112 avail_mem=70.90 GB):  71%|███████   | 41/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=112 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.90it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.90it/s] Capturing num tokens (num_tokens=80 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.90it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.90it/s]Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.90it/s]Capturing num tokens (num_tokens=32 avail_mem=70.88 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.90it/s]Capturing num tokens (num_tokens=32 avail_mem=70.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.39it/s]

    Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=16 avail_mem=70.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.39it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.26it/s]Capturing num tokens (num_tokens=8 avail_mem=70.86 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.26it/s] Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.26it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 35.60it/s]


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
    Generated text:  Callum and I am 26 years old. I am a dancer, artist and writer. I enjoy writing and reading. My writing has been published in a variety of magazines and online. My biggest dream is to have a child and I have always felt a strong connection to nature and have a passion for the outdoors. 
    What is your favourite thing about being a writer? As a writer, I find that I'm able to express myself in a way that I have never felt before. It's like I can connect with characters, emotions, and experiences that I don't always have the chance to in other mediums. And most importantly
    ===============================
    Prompt: The president of the United States is
    Generated text:  inaugurated after what date?
    The president of the United States is inaugurated after January 20, 2021. The inauguration of the next president happens on January 20, 2021. The inauguration is a significant moment in American history, as it marks the beginning of the new president's presidency. The process for inducting the next president is a formal and diplomatic procedure that involves the president, the Vice President, the House of Representatives, and the Senate. The details of the inducting process can vary slightly depending on the specific circumstances and the previous president's position, but the general process
    ===============================
    Prompt: The capital of France is
    Generated text:  _________. A. Paris B. Nice C. Lyons D. Paris
    
    Answer:
    
    A
    
    Please select the correct Chinese translation for the following Japanese sentence.
    A. 「ああ、いいえ、それをやめる」
    B. 「ああ、いいえ、それをしない」
    C. 「ああ、いいえ、それを決める」
    D. 「ああ、いいえ、それを嫌な想い」
    Answer:
    
    B
    
    On July 4th, in the city of Paris, France, there was a ____.
    A. rainstorm
    B. snowstorm
    C. heatwave
    D
    ===============================
    Prompt: The future of AI is
    Generated text:  already here, and it's already changing how we work and live. But what does that mean for us? As an AI language model, I will try to provide you with a comprehensive overview of what the future of AI is, but note that the topic is complex and evolving rapidly, so it's essential to keep an open mind and stay informed about the latest developments in the field.
    
    What are some of the key trends and technologies that are shaping the future of AI?
    
    There are many different trends and technologies that are shaping the future of AI, and here are some of the most notable ones:
    
    1. Machine learning: Machine learning is the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its fashion industry, with many famous designers and boutiques. The city is a popular tourist destination and is home to many museums, theaters, and other cultural institutions. Paris is a major economic center and a major transportation hub, with many major airports and highways. It is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn and adapt in ways that are difficult for humans to do. This could lead to more efficient and effective decision-making, as well as more personalized and context-aware interactions with humans.
    
    2. Greater use of machine learning: Machine learning is becoming increasingly important in AI, as it allows AI systems to learn from data and improve their performance over time. This could lead to more accurate and effective predictions, as well as more personalized and adaptive interactions with humans.
    
    3
    


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
    Generated text:  [insert character's name here] and I'm a [insert profession, if applicable]. I'm a [insert your profession, if you are in a particular field of work]. I'm excited to be here and I'm looking forward to meeting all of you.
    
    [Include your name, profession, and how you first met your current employer or supervisor, and how your experience working with a diverse workforce has prepared you for this role.]
    
    Welcome to the team! I'm [insert name], and I'm excited to meet you all. I'm a [insert your profession], and I've had the privilege of working with a diverse workforce for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement summarizes the key information about the capital city in a concise format. It includes the name of the city (Paris) and its status as the capital of France. The statement is clear and to the point. It provides the essential facts about the city. For instance, the statement mentions the country, the capital city, and its significance in the overall context. If there are any further details or context needed, such as specific aspects of Paris' governance or its cultural significance, this information can be added without changing the core message of the statement. For example, if the statement was to include information about Paris' cuisine or
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a continued expansion and diversification of its applications and applications. Here are some possible trends that could occur:
    
    1. Increased sophistication of AI algorithms: AI algorithms will continue to get better and more sophisticated, with the goal of performing tasks that were previously thought to be impossible or impractical using traditional methods.
    
    2. Greater use of AI in healthcare: AI has the potential to revolutionize healthcare by enabling more precise diagnosis and treatment of diseases, personalized medicine, and improved patient outcomes. AI can also improve the efficiency and accuracy of medical research, drug development, and clinical trials.
    
    3. Greater use of AI in automation: AI will


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

     [

    Age

    ]

     years

     old

    ,

     and

     I

     currently

     live

     in

     [

    Your

     Location

    ].

     I

     enjoy

     [

    What

     you

     enjoy

     doing

    ].

     I

     am

     [

    Your

     Profession

     or

     Occupation

    ].

     
    


    Let

     me

     know

     if

     you

    'd

     like

     me

     to

     elaborate

     on

     any

     of

     these

     details

    ,

     or

     if

     there

    's

     anything

     else

     I

     can

     do

     for

     you

    .

     
    


    [

    Your

     Name

    ]

      


    [

    Your

     Position

    ]

      


    [

    Your

     Profession

    /O

    cc

    up

    ation

    ]

      


    Hello

    !

     My

     name

     is

     [

    Your

     Name

    ].

     I

     am

     [

    Age

    ]

     years

     old

    ,

     and

     I

     currently

     live

     in

     [

    Your

     Location

    ].

     I

     enjoy

     [

    What

     you

     enjoy

     doing

    ].

     I

     am

     [

    Your

     Profession

     or

     Occupation

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     


    (A

    )

     Yes

    


    (B

    )

     No

    


    (B

    )

     No

    


    The

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

     incorrect

    .

     While

     Paris

     is

     indeed

     the

     capital

     city

     of

     France

    ,

     it

     is

     not

     the

     only

     capital

     city

    .

     Other

     cities

     in

     France

     include

     Rome

     and

     Nice

    .

     The

     correct

     answer

     is

     (

    B

    )

     No

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

     and

     diverse

    ,

     with

     potential

     applications

     in

     numerous

     fields

     and

     industries

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     in

     the

     next

     few

     years

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

     and

     deep

     learning

    :

     As

     AI

     continues

     to

     improve

    ,

     we

     expect

     to

     see

     even

     more

     powerful

     algorithms

     that

     can

     perform

     complex

     tasks

     with

     high

     accuracy

     and

     speed

    .
    


    2

    .

     Increased

     integration

     with

     other

     technologies

    :

     AI

     is

     increasingly

     being

     integrated

     with

     other

     technologies

    ,

     such

     as

     voice

     recognition

    ,

     bi

    ometrics

    ,

     and

     sensor

     networks

    .

     This

     integration

     will

     enable

     AI

     to

     become

     more

     capable

     of

     understanding

     human

     behavior

     and

     emotions

    ,

     which

     could

     lead

     to

     new

     applications

     in

     areas

     such

     as

     healthcare

     and

     customer

     service

    .
    


    3

    .

     More

     diverse

     and

    



```python
llm.shutdown()
```
