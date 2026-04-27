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
    [2026-04-27 05:27:42] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.29it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.29it/s]


    2026-04-27 05:27:46,032 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-27 05:27:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:14,  4.46s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.35it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.77it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.73it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.73it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.73it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.73it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.73it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.73it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.73it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.73it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.73it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.73it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.73it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.66it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.66it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.66it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.66it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.66it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.66it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.66it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.75 GB):   3%|▎         | 2/58 [00:00<00:04, 13.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.91 GB):   3%|▎         | 2/58 [00:00<00:04, 13.10it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=69.91 GB):   3%|▎         | 2/58 [00:00<00:04, 13.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.91 GB):   7%|▋         | 4/58 [00:00<00:03, 15.31it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.91 GB):   7%|▋         | 4/58 [00:00<00:03, 15.31it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.90 GB):   7%|▋         | 4/58 [00:00<00:03, 15.31it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.90 GB):  10%|█         | 6/58 [00:00<00:03, 16.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.90 GB):  10%|█         | 6/58 [00:00<00:03, 16.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.89 GB):  10%|█         | 6/58 [00:00<00:03, 16.68it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=69.89 GB):  10%|█         | 6/58 [00:00<00:03, 16.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.89 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.89 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=3584 avail_mem=69.88 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=3328 avail_mem=69.88 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=3328 avail_mem=69.88 GB):  21%|██        | 12/58 [00:00<00:02, 21.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.88 GB):  21%|██        | 12/58 [00:00<00:02, 21.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.88 GB):  21%|██        | 12/58 [00:00<00:02, 21.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.87 GB):  21%|██        | 12/58 [00:00<00:02, 21.73it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=69.87 GB):  21%|██        | 12/58 [00:00<00:02, 21.73it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.01 GB):  21%|██        | 12/58 [00:00<00:02, 21.73it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.51it/s]Capturing num tokens (num_tokens=1280 avail_mem=69.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=68.99 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.51it/s]Capturing num tokens (num_tokens=960 avail_mem=69.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.51it/s] Capturing num tokens (num_tokens=960 avail_mem=69.00 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.54it/s]Capturing num tokens (num_tokens=896 avail_mem=69.00 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.54it/s]Capturing num tokens (num_tokens=832 avail_mem=68.99 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.54it/s]Capturing num tokens (num_tokens=768 avail_mem=68.99 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.54it/s]

    Capturing num tokens (num_tokens=704 avail_mem=68.99 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.54it/s]Capturing num tokens (num_tokens=640 avail_mem=68.98 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.54it/s]Capturing num tokens (num_tokens=576 avail_mem=68.98 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.54it/s]Capturing num tokens (num_tokens=576 avail_mem=68.98 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.47it/s]Capturing num tokens (num_tokens=512 avail_mem=68.97 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.47it/s]Capturing num tokens (num_tokens=480 avail_mem=68.98 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.47it/s]Capturing num tokens (num_tokens=448 avail_mem=68.98 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.47it/s]Capturing num tokens (num_tokens=416 avail_mem=68.98 GB):  48%|████▊     | 28/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=384 avail_mem=68.98 GB):  48%|████▊     | 28/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=352 avail_mem=68.97 GB):  48%|████▊     | 28/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=352 avail_mem=68.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=320 avail_mem=68.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=288 avail_mem=68.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.96it/s]

    Capturing num tokens (num_tokens=256 avail_mem=68.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=240 avail_mem=68.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=224 avail_mem=68.95 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=208 avail_mem=68.95 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=208 avail_mem=68.95 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.44it/s]Capturing num tokens (num_tokens=192 avail_mem=68.95 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.44it/s]Capturing num tokens (num_tokens=176 avail_mem=68.95 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.44it/s]Capturing num tokens (num_tokens=160 avail_mem=68.94 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.44it/s]Capturing num tokens (num_tokens=144 avail_mem=68.94 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.44it/s]Capturing num tokens (num_tokens=128 avail_mem=68.94 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.44it/s]Capturing num tokens (num_tokens=112 avail_mem=68.94 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.44it/s]Capturing num tokens (num_tokens=112 avail_mem=68.94 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.67it/s]Capturing num tokens (num_tokens=96 avail_mem=68.93 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.67it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=68.93 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.67it/s]Capturing num tokens (num_tokens=64 avail_mem=68.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.67it/s]Capturing num tokens (num_tokens=48 avail_mem=68.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.67it/s]Capturing num tokens (num_tokens=32 avail_mem=68.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.67it/s]Capturing num tokens (num_tokens=32 avail_mem=68.92 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.11it/s]Capturing num tokens (num_tokens=28 avail_mem=68.91 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.11it/s]Capturing num tokens (num_tokens=24 avail_mem=68.91 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.11it/s]Capturing num tokens (num_tokens=20 avail_mem=68.91 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.11it/s]Capturing num tokens (num_tokens=16 avail_mem=68.91 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.11it/s]Capturing num tokens (num_tokens=12 avail_mem=68.90 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.11it/s]Capturing num tokens (num_tokens=8 avail_mem=68.90 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.11it/s] Capturing num tokens (num_tokens=8 avail_mem=68.90 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.81it/s]Capturing num tokens (num_tokens=4 avail_mem=68.90 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.81it/s]

    Capturing num tokens (num_tokens=4 avail_mem=68.90 GB): 100%|██████████| 58/58 [00:01<00:00, 37.73it/s]


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
    Generated text:  Christopher and I am 28 years old. I am a full-time student in the spring semester of 2021, and I have been living in the United States for the past two years. I have a bachelor's degree in mathematics, a master's degree in computer science, and a minor in public health, and I have worked in the fields of data science and machine learning for a few years. I am deeply interested in computer science and the field of data science, and I have always loved to learn new things and challenge myself. I enjoy going on adventures and trying new things, and I love to travel. I
    ===============================
    Prompt: The president of the United States is
    Generated text:  a popular pastime, and many people in the country have worked hard to make the presidency a more appealing position. The president is a position of leadership in the United States government. The office of the president is responsible for the policies and direction of the government. The president is the head of state, the commander-in-chief of the armed forces, and the representative of the country to the world. All these roles require a certain level of executive skill and expertise. The president must also be able to handle the delicate balance of power in the country, and the ability to make decisions that affect the entire country.
    The United States has a long history
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, located in the French department of Paris, located in the Île-de-France region.
    The capital of France is located in the center of the country. It is in the city of Paris, a place in the department of Paris.
    The city of Paris is the capital of France and the national capital of France. It is the largest city in France and is the seat of the French government, the Senate and the Chamber of Deputies and the French presidential palace, the Louvre and the Centre Pompidou, and the headquarters of the French government and the French parliament, the Senate and the Chamber of Deputies. Paris is
    ===============================
    Prompt: The future of AI is
    Generated text:  still a mystery. In the coming years, we will see new technologies emerge and new ways of computing being invented. One of these new technologies is quantum computing. Quantum computing is a field of computing that uses quantum mechanics to solve problems and compute information. This technology is still in its infancy and is still being developed and refined.
    Quantum computing has the potential to revolutionize the way we use computers. It can be used for a variety of tasks, including encryption, simulating complex systems, and optimizing algorithms. By using quantum computing, we can solve problems that are too large for traditional computers to handle.
    One of the key advantages of quantum


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Monument. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. It is a popular tourist destination and a major economic center in Europe. The city is home to many famous museums, including the Louvre and the Musée d'Orsay. Paris is also known for its cuisine, including French cuisine and its famous cheese. It is a city that is constantly evolving and changing, with new
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI systems become more advanced, there will be a growing emphasis on ethical considerations, including issues such as bias, transparency, and accountability. This will lead to more rigorous testing and evaluation of AI systems, as well as greater transparency and accountability in their use.
    
    2. Greater integration with human intelligence
    


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
    Generated text:  [Your Name], and I'm a [specific role or profession] with a strong passion for [specific skill or hobby]. I'm currently [age] years old, and I live in [location].
    My hobbies include [list of hobbies, interests, or passions], and I'm always eager to learn new things. I enjoy [type of exercise or training], [type of food], and [type of entertainment], and I'm always looking for ways to stay healthy and enjoy life. As a [specific occupation or profession], I'm excited to help people and make a positive impact on the world around me. What's your name?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, and it is known for its iconic landmarks, such as the Eiffel Tower and Notre-Dame Cathedral, as well as its rich history and culture, including its romantic romance and annual Parisian Carnival. The city is also famous for its cafes, art galleries, and music venues, and it has a thriving food scene with its famous dishes like croissants and beignets. Paris is a must-visit destination for travelers from all over the world, and it remains one of the most diverse and vibrant cities in the world today. **Paris:** the city of love, romance, gastronomy, and culture. Known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid technological advancements and innovations, making it an ever-evolving field. Some potential trends that could be expected in the AI industry include:
    
    1. Increased Integration with Natural Language Processing: As AI becomes more integrated into everyday life, it's likely that we will see even more natural language processing capabilities being developed. This will enable machines to understand and respond to human speech and language in new and exciting ways.
    
    2. Enhanced Generative AI: As AI becomes more capable of generating content, it's possible that we'll see even more sophisticated AI that can generate content that looks, sounds, and feels like real human art,


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

     __

    ________

    .

     I

    ’m

     a

    /an

     __

    ________

    .

     I

     currently

     live

     in

     __

    ________

    .

     I

     enjoy

     __

    ________

    .

     If

     you

     could

     be

     anything

    ,

     what

     would

     it

     be

    ?

     I

     can

     learn

     lots

     of

     new

     things

     and

     make

     new

     friends

    .

     I

     have

     a

     lot

     of

     energy

     and

     enthusiasm

    ,

     and

     I

    'm

     always

     trying

     to

     make

     the

     world

     a

     better

     place

    .

     And

     I

    'm

     always

     eager

     to

     learn

     and

     grow

    .

     I

    'm

     __

    ________

    .

     Thank

     you

    !

     [

    Your

     name

    ]


    Hello

    ,

     my

     name

     is

     [

    Your

     name

    ].

     I

    ’m

     a

    /an

     [

    Your

     profession

    ].

     I

     currently

     live

     in

     [

    Your

     current

     address

    ].

     I

     enjoy

     [

    Your

     hobbies

     or

     interests

    ].

     If

     you

     could

     be

     anything

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     is

     fact

    ually

     correct

    .

     Paris

     is

     the

     largest

     city

     in

     France

     and

     serves

     as

     the

     capital

     of

     the

     country

    .

     It

     is

     also

     known

     as

     the

     City

     of

     Love

     due

     to

     its

     rich

     history

     and

     romantic

     ambiance

    .

     Paris

     has

     a

     rich

     cultural

     heritage

     and

     is

     a

     major

     tourist

     destination

     in

     Europe

    .

     It

     is

     home

     to

     many

     world

    -ren

    owned

     institutions

    ,

     including

     the

     Lou

    vre

     Museum

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Notre

     Dame

     Cathedral

    .

     The

     city

     is

     also

     home

     to

     numerous

     art

     galleries

    ,

     cafes

    ,

     and

     other

     cultural

     venues

    .

     Paris

     has

     a

     long

     history

     dating

     back

     to

     the

     Roman

     Empire

    ,

     making

     it

     a

     fascinating

     city

     to

     explore

    .

     Despite

     its

     large

     population

    
    
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

     factors

    ,

     including

     advancements

     in

     computing

     power

    ,

     advances

     in

     data

     analysis

    ,

     and

     changes

     in

     the

     way

     we

     interact

     with

     technology

    .

     Some

     potential

     trends

     include

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     everyday

     life

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     we

     may

     see

     more

     widespread

     adoption

     of

     AI

     in

     our

     daily

     lives

    ,

     from

     self

    -driving

     cars

     to

     smarter

     home

     appliances

    .
    


    2

    .

     Greater

     integration

     of

     AI

     into

     industries

    :

     AI

     is

     already

     being

     used

     in

     a

     wide

     range

     of

     industries

    ,

     from

     healthcare

     to

     finance

     to

     manufacturing

    .

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     may

     see

     more

     companies

     and

     governments

     investing

     in

     AI

     research

     and

     development

    ,

     and

     integrating

     AI

     into

    



```python
llm.shutdown()
```
