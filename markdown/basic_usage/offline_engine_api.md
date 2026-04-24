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
    [2026-04-24 01:20:24] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.22it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.21it/s]


    2026-04-24 01:20:28,704 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 01:20:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.74s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]

    Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  4.96it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  4.96it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:09,  4.96it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:09,  4.96it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:09,  4.96it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:03<00:09,  4.96it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:03<00:09,  4.96it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:03<00:09,  4.96it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:03<00:09,  4.96it/s]Compiling num tokens (num_tokens=1792):  16%|█▌        | 9/58 [00:03<00:09,  4.96it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.10it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 19.35it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 19.35it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 19.35it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 19.35it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 19.35it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 19.35it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 19.35it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 19.35it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:03<00:01, 19.35it/s]Compiling num tokens (num_tokens=320):  45%|████▍     | 26/58 [00:03<00:01, 19.35it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 28.63it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 28.63it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 28.63it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 28.63it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 28.63it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 28.63it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 28.63it/s]

    Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 28.63it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:03<00:00, 28.63it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:03<00:00, 28.63it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 37.85it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 37.85it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 37.85it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 37.85it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 37.85it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 37.85it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 37.85it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:03<00:00, 37.85it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:03<00:00, 37.85it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:03<00:00, 37.85it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 46.58it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 46.58it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 46.58it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 46.58it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 46.58it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 46.58it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.39 GB):   3%|▎         | 2/58 [00:00<00:03, 16.66it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.38 GB):   3%|▎         | 2/58 [00:00<00:03, 16.66it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.38 GB):   3%|▎         | 2/58 [00:00<00:03, 16.66it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=116.38 GB):   3%|▎         | 2/58 [00:00<00:03, 16.66it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.62it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.62it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.37 GB):   9%|▊         | 5/58 [00:00<00:02, 19.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.52it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=116.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.52it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.36 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.52it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.36 GB):  21%|██        | 12/58 [00:00<00:01, 25.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.36 GB):  21%|██        | 12/58 [00:00<00:01, 25.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.36 GB):  21%|██        | 12/58 [00:00<00:01, 25.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.35 GB):  21%|██        | 12/58 [00:00<00:01, 25.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.35 GB):  21%|██        | 12/58 [00:00<00:01, 25.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.35 GB):  21%|██        | 12/58 [00:00<00:01, 25.70it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=116.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.31it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.31it/s]Capturing num tokens (num_tokens=960 avail_mem=116.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.31it/s] Capturing num tokens (num_tokens=960 avail_mem=116.33 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=896 avail_mem=116.30 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=832 avail_mem=116.29 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=768 avail_mem=116.29 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=704 avail_mem=116.29 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.80it/s]

    Capturing num tokens (num_tokens=640 avail_mem=116.28 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.80it/s]Capturing num tokens (num_tokens=640 avail_mem=116.28 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.34it/s]Capturing num tokens (num_tokens=576 avail_mem=116.28 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.34it/s]Capturing num tokens (num_tokens=512 avail_mem=116.27 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.34it/s]Capturing num tokens (num_tokens=480 avail_mem=116.29 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.34it/s]Capturing num tokens (num_tokens=448 avail_mem=116.28 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.34it/s]Capturing num tokens (num_tokens=416 avail_mem=116.28 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.34it/s]Capturing num tokens (num_tokens=416 avail_mem=116.28 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.21it/s]Capturing num tokens (num_tokens=384 avail_mem=116.28 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.21it/s]Capturing num tokens (num_tokens=352 avail_mem=116.27 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.21it/s]Capturing num tokens (num_tokens=320 avail_mem=116.27 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.21it/s]

    Capturing num tokens (num_tokens=288 avail_mem=116.27 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.21it/s]Capturing num tokens (num_tokens=256 avail_mem=116.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.21it/s]Capturing num tokens (num_tokens=256 avail_mem=116.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=240 avail_mem=116.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=224 avail_mem=116.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=208 avail_mem=116.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=192 avail_mem=116.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=176 avail_mem=116.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=176 avail_mem=116.25 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.88it/s]Capturing num tokens (num_tokens=160 avail_mem=116.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.88it/s]

    Capturing num tokens (num_tokens=144 avail_mem=116.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.88it/s]Capturing num tokens (num_tokens=128 avail_mem=116.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.88it/s]Capturing num tokens (num_tokens=112 avail_mem=116.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.88it/s]Capturing num tokens (num_tokens=112 avail_mem=116.24 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.81it/s]Capturing num tokens (num_tokens=96 avail_mem=116.23 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.81it/s] Capturing num tokens (num_tokens=80 avail_mem=116.23 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.81it/s]Capturing num tokens (num_tokens=64 avail_mem=116.22 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.81it/s]Capturing num tokens (num_tokens=48 avail_mem=116.22 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.81it/s]Capturing num tokens (num_tokens=32 avail_mem=116.22 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.81it/s]Capturing num tokens (num_tokens=32 avail_mem=116.22 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.90it/s]Capturing num tokens (num_tokens=28 avail_mem=116.21 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.90it/s]

    Capturing num tokens (num_tokens=24 avail_mem=116.21 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.90it/s]Capturing num tokens (num_tokens=20 avail_mem=116.21 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.90it/s]Capturing num tokens (num_tokens=16 avail_mem=116.21 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.90it/s]Capturing num tokens (num_tokens=12 avail_mem=116.20 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.90it/s]Capturing num tokens (num_tokens=12 avail_mem=116.20 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.17it/s]Capturing num tokens (num_tokens=8 avail_mem=116.20 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.17it/s] Capturing num tokens (num_tokens=4 avail_mem=116.20 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.17it/s]Capturing num tokens (num_tokens=4 avail_mem=116.20 GB): 100%|██████████| 58/58 [00:01<00:00, 35.77it/s]


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
    Generated text:  Jim and I would like to write a 2000 word essay on the topic "The Art of Negotiation". Can you provide me with some suggestions on how to write such an essay? Please provide specific tips and examples to help me better understand how to effectively write an essay on this topic.
    Certainly! Writing an essay on the art of negotiation can be quite challenging and requires careful consideration of the topic. Here are some tips and examples that can help you get started:
    1. Start with an Introduction: An introduction is where you will introduce the topic and grab the reader's attention. Here are some tips on how to write an
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking advice on how to handle a situation where one of their relatives is suspected of a criminal offense, and the president is worried about the possibility of the law enforcement agencies intervening too much and potentially compromising the president's personal privacy.
    As a solution, the president can consider the following:
    The president can work with the law enforcement agencies to establish clear guidelines on how the information will be handled and how it will be used. This can help prevent the misuse of the information, such as the possibility of it being used to identify political opponents or to commit crimes.
    The president can also work with the media to ensure that any information obtained during the investigation is
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Marseilles
    C. Rome
    D. London
    Answer:
    
    A
    
    In this question, the most appropriate word to fill in the blank is ____.
    A. The
    B. A
    C. An
    D. It
    
    For example:
    1. The bookstore is located in the corner of the mall.
    2. The cat is a pet.
    3. The man is a person.
    4. The boy is a student.
    5. The tree is a plant.
    6. The giraffe is a mammal.
    
    Answer: D
    
    In the question, "The capital of France is
    ===============================
    Prompt: The future of AI is
    Generated text:  inherently uncertain, but it’s an area of great promise. The growth in AI research and development is fueling demand for AI professionals and the development of new AI-based products and services. However, with the rapid pace of technological change, the future of AI is in doubt. The development of AI systems is not just about creating new technologies or improving existing ones, but also about developing new human skill sets. The rapid pace of development can create new job positions that are not yet being fully recognized or paid for. Furthermore, the future of AI is not just about creating new technologies or improving existing ones, but also about creating new human skill sets


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I enjoy [insert a short description of your hobbies or interests]. I'm [insert a short description of your personality or character]. I'm always looking for new challenges and opportunities to grow and learn. Thank you for taking the time to meet me. [Name] [Company name] [Phone number] [Email address] [LinkedIn profile link] [Social media handles] [Company website
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is also home to the Louvre Museum, the most famous museum in the world, and the Notre-Dame Cathedral, a stunning Gothic structure. Paris is a vibrant and diverse city with a rich cultural heritage and a lively nightlife. It is the capital of France and the largest city in the European Union. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a popular tourist destination and a major economic center, with a population of over 2 million
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability.
    
    2. Integration with other technologies: AI will continue to be integrated with other technologies such as IoT, blockchain, and quantum computing. This will create new opportunities for AI to be used in new and innovative ways.
    
    3. Development of new AI architectures: As AI becomes more complex, there will be a
    


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
    Generated text: ... what? Name? I'm a fictional character with no real name. However, I'm proud to be known as the kind-hearted, optimistic, and adventurous detective named Alex. I've been chasing the truth for over two decades, solving mysteries that have changed the lives of those who have fallen victim to fraud, deceit, and betrayal. 
    
    I'm always on the lookout for the next big case, and I'm always eager to help people find their way out of the dark side of life. Whether it's a lost wallet, a stolen watch, or a hidden treasure, I always have a solution. And I'm always ready to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is a city located in the south of France and is the capital of France. It is famous for its rich history, stunning architecture, and vibrant culture. The city is home to many famous landmarks such as Notre-Dame Cathedral, Eiffel Tower, and the Louvre Museum. Paris is also known for its fashion industry, wine, and cuisine. Its urban layout and natural beauty make it a popular tourist destination. The city has a rich history dating back to the Roman Empire and the French Revolution, and it continues to be a major city for artists, writers, and musicians. Paris is a cultural and economic center that
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but here are a few possible trends that are likely to shape the industry in the coming years:
    
    1. Increased use of AI for autonomous driving: With the widespread adoption of electric vehicles and advancements in self-driving technology, we may see a growing focus on AI that can assist in autonomous driving. This could lead to the development of self-driving cars that can operate in various weather conditions and have the ability to make decisions on their own, reducing the risk of accidents.
    
    2. AI for healthcare: AI is already being used in healthcare to improve patient outcomes, but there is likely to be further expansion in this field as AI becomes more capable


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

    'm

     a

     [

    character

     type

    ]

     who

     loves

     [

    occupation

    ]

     and

     [

    characters

    ]

     games

    .

     I

     like

     to

     [

    insert

     a

     personal

     trait

     or

     hobby

     that

     you

    'd

     enjoy

    ,

     like

     reading

    ,

     hiking

    ,

     or

     drawing

    ].

     I

    'm

     excited

     to

     share

     my

     thoughts

     and

     interests

     with

     you

    .

     How

     about

     you

    ?

     What

     makes

     you

     unique

     and

     why

     do

     you

     like

     to

     play

     games

    ?

     What

     brings

     you

     joy

    ?

     Let

    's

     connect

    !

     [

    Name

    ]

     [

    Contact

     Information

    ]

     [

    Date

    ]

     [

    Link

     to

     your

     profile

    ]


    Hey

     there

    !

     I

    'm

     [

    Your

     Name

    ],

     a

     [

    Character

     Type

    ]

     who

     really

     enjoys

     [

    occupation

    ].

     I

     love

     [

    characters

    ]

     games

     and

     I

     love

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     "

    City

     of

     Love

    ".

     It

     is

     a

     historic

     and

     culturally

     rich

     city

     with

     a

     world

    -ren

    owned

     art

     and

     music

     scene

    ,

     a

     rich

     culinary

     tradition

    ,

     and

     a

     vibrant

     nightlife

    .

     The

     city

     is

     home

     to

     several

     world

    -ren

    owned

     museums

    ,

     including

     the

     Lou

    vre

    ,

     the

     National

     Museum

     of

     Modern

     Art

    ,

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     Paris

     also

     boasts

     a

     number

     of

     international

     sporting

     venues

    ,

     including

     the

     E

    iff

    el Tower

    ,

     the

     Paris

     St

    .

     Ger

    main

    ,

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    .

     The

     city

     is

     also

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     which

     stands

     as

     a

     symbol

     of

     France

    's

     rich

     history

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     and

     there

     are

     many

     possible

     trends

     that

     could

     shape

     the

     industry

    's

     development

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

     Increasing

     reliance

     on

     AI

     for

     decision

    -making

    :

     As

     AI

     is

     becoming

     more

     and

     more

     advanced

    ,

     it

     is

     likely

     that

     decision

    -making

     will

     become

     increasingly

     reliant

     on

     AI

    .

     This

     could

     lead

     to

     more

     complex

     and

     sophisticated

     AI

     systems

     that

     are

     better

     able

     to

     handle

     a

     wide

     range

     of

     tasks

    .
    


    2

    .

     Integration

     of

     AI

     with

     other

     technologies

    :

     AI

     is

     already

     being

     integrated

     into

     other

     technologies

    ,

     such

     as

     smart

     homes

    ,

     self

    -driving

     cars

    ,

     and

     smart

     grids

    .

     It

     is

     possible

     that

     AI

     will

     continue

     to

     be

     integrated

     into

     more

     of

     these

     technologies

     in

     the

    



```python
llm.shutdown()
```
