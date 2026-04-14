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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.13it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.12it/s]


    2026-04-14 07:47:04,156 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 07:47:04] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.09it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.09it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.09it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.09it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.09it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.09it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.09it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.09it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.23it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.23it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.23it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.23it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.23it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.23it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.23it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 23.63it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 23.63it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 28.71it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 28.71it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 28.71it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 28.71it/s]

    Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 28.71it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 28.71it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 28.71it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 22.19it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 22.19it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 22.19it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 22.19it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 22.19it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 22.19it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 25.48it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 25.48it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 25.48it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 25.48it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 25.48it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 25.48it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 25.48it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 31.09it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 31.09it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 31.09it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:04<00:00, 31.09it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:04<00:00, 31.09it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=118.06 GB):   3%|▎         | 2/58 [00:00<00:06,  8.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.95 GB):   3%|▎         | 2/58 [00:00<00:06,  8.31it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.95 GB):   3%|▎         | 2/58 [00:00<00:06,  8.31it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.95 GB):   7%|▋         | 4/58 [00:00<00:06,  8.35it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   7%|▋         | 4/58 [00:00<00:06,  8.35it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   7%|▋         | 4/58 [00:00<00:06,  8.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   7%|▋         | 4/58 [00:00<00:06,  8.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):  12%|█▏        | 7/58 [00:00<00:04, 12.64it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):  12%|█▏        | 7/58 [00:00<00:04, 12.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  12%|█▏        | 7/58 [00:00<00:04, 12.64it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  12%|█▏        | 7/58 [00:00<00:04, 12.64it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.99it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.99it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.99it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.65it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.65it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.65it/s] Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.75it/s]Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.75it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.75it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.75it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.75it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.75it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.66it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.66it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.66it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.66it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.66it/s]

    Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.66it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.06it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.06it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.06it/s]

    Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.06it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.06it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  71%|███████   | 41/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  71%|███████   | 41/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  71%|███████   | 41/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  71%|███████   | 41/58 [00:01<00:00, 37.95it/s]

    Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  71%|███████   | 41/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.45it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.45it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.45it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.45it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.45it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.45it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.49it/s]

    Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  86%|████████▌ | 50/58 [00:02<00:00, 31.49it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  95%|█████████▍| 55/58 [00:02<00:00, 35.02it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  95%|█████████▍| 55/58 [00:02<00:00, 35.02it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  95%|█████████▍| 55/58 [00:02<00:00, 35.02it/s] Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  95%|█████████▍| 55/58 [00:02<00:00, 35.02it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:02<00:00, 27.51it/s]


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
    Generated text:  Charlie and I'm a regular reader and frequent visitor to the internet. I'm very knowledgeable and experienced with all things related to physics, especially classical mechanics.
    
    I've been asked quite frequently to answer questions like these: "If I'm in an elevator, does it travel faster than the speed of light? How is it possible to tell whether it's moving slower than light?"
    
    I'll do my best to answer these questions:
    
    1. The speed of the elevator is constant. If you were to look down from the top of the elevator, you would observe that the elevator is moving at a constant speed because it's attached to a string,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking government official who is elected by the people to serve a specific term. What is a high-ranking government official? A president is a high-ranking government official. They hold the position of president and are responsible for leading the country and implementing the policies of the federal government. The term of office for a president is typically 4 years. The President of the United States is the head of state, the commander-in-chief of the armed forces, and the head of government of the United States. They also serve on various committees and boards that are important to their position. The president is elected through a process of voting for members of the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It has many landmarks including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a beautiful city with a rich history and culture. Visitors to Paris can explore the city and experience its culture and history in a unique way. The city is known for its art, music, and cuisine, and visitors can enjoy a variety of activities and attractions in Paris.
    In summary, Paris is a beautiful city with many landmarks, rich history, culture, art, music, cuisine, and activities to enjoy. Visitors can explore the city and experience its culture and history in a unique way.
    This summary is a
    ===============================
    Prompt: The future of AI is
    Generated text:  real, and there are no boundaries to it, but there are also issues with the complexity and potential dangers of AI, including biases in AI models, the lack of transparency of AI, and the potential misuse of AI. To combat these issues, the AI community needs to take steps to address the underlying issues with the technology. One such approach is to establish guidelines and regulations for AI development and use, which can help ensure that AI is developed and used in a responsible and ethical manner. Some examples of such guidelines and regulations include the General Data Protection Regulation (GDPR) in Europe and the European Union’s Artificial Intelligence, or AI, regulations


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French Quarter. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub and a major tourist destination, attracting millions of visitors each year. The city is known for its cuisine, fashion, and art, and is a popular destination for tourists and locals alike. Paris is a vibrant and dynamic city that continues to evolve and grow, with new developments
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more natural and intuitive interactions between humans and machines.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Increased use of AI for tasks that require human-like intelligence: AI is likely to become more integrated with human intelligence, allowing machines to perform tasks that require
    


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
    Generated text:  [Name]. I am a/an [Age], [Gender] person. I started my journey with a/an [Occupation], and I’ve always had a strong desire to [What you want to do]. I’m always on the lookout for opportunities and challenges, and I’m willing to put in the work to achieve my goals. What’s your story? What’s your purpose? What’s your passion? Tell me about yourself! #SelfIntro #Character
    
    ---
    
    Hey there! My name is [Name], and I’m a/an [Age] [Gender] girl who started my journey with a/an [Occupation] and has
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located on the banks of the River Seine. It is the largest city in the European Union and one of the world's most popular tourist destinations. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum, as well as its diverse culture, cuisine, and fashion industry. The city is home to many famous landmarks and attractions, and is a major hub for commerce, finance, and education in Europe. Paris also plays a significant role in French culture and politics, with its influence on its capital's politics, economy, and society. Paris is home to many important
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to involve a wide range of emerging trends, including:
    
    1. Automation of human tasks: As AI becomes more advanced, it is likely to replace humans in tasks that are repetitive or mundane, freeing up more time for human workers to focus on more complex and creative tasks.
    
    2. Personalized AI: As AI becomes more capable of understanding and learning from its environment, it is possible that AI will become more personalized, tailoring its responses and actions to individual users based on their preferences and behavior.
    
    3. Artificial intelligence in healthcare: AI is already being used to analyze medical images and data to assist with diagnosis and treatment. In the future


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

    Your

     role

    /

    occupation

    ]

     with

     a

     [

    Your

     profession

    ]

     degree

    .

     I

     have

     been

     studying

     for

     [

    Number

     of

     years

    ]

     years

     to

     pursue

     [

    Your

     goal

    ].

     Throughout

     my

     education

     and

     career

    ,

     I

    've

     been

     striving

     to

     [

    Why

     I

    'm

     pursuing

     this

     path

    ].

     I

     am

     always

     looking

     for

     ways

     to

     [

    How

     I

     try

     to

     improve

     myself

    ].

     I

    'm

     very

     enthusiastic

     and

     I

    'm

     always

     seeking

     to

     learn

     and

     grow

    .

     I

     hope

     to

     have

     a

     [

    Long

    -term

     goal

    ]

     at

     some

     point

     in

     the

     future

    .

     What

     would

     you

     like

     to

     learn

     more

     about

     about

     me

    ?

     [

    Name

    ]

     knows

     that

     learning

     about

     [

    Name

    ]

     is

     beneficial

     for

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     rich

     history

    ,

     iconic

     architecture

    ,

     and

     vibrant

     culture

    .

     It

     is

     also

     the

     birth

    place

     of

     many

     famous

     historical

     figures

    ,

     including

     Vol

    taire

     and

     Mozart

    ,

     and

     is

     a

     popular

     tourist

     destination

    .

     Paris

     has

     a

     unique

     blend

     of

     French

     and

     international

     influences

     and

     is

     renowned

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     art

    .

     It

     is

     a

     major

     cultural

     and

     economic

     center

    ,

     with

     many

     important

     museums

    ,

     theaters

    ,

     and

     landmarks

    .

     Paris

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Light

    "

     and

     a

     symbol

     of

     the

     European

     Union

    ,

     while

     also

     being

     considered

     one

     of

     the

     most

     dangerous

     cities

     in

     the

     world

    .

     It

     is

     a

     city

     that

     continues

     to

     evolve

     and

     change

     over

     time

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

     and

     potential

    ,

     and

     it

     is

     influenced

     by

     many

     factors

     including

     advances

     in

     computing

     power

    ,

     data

     availability

    ,

     and

     the

     growth

     of

     the

     AI

     research

     community

    .

     Here

     are

     some

     possible

     trends

     in

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     Efficiency

     and

     Automation

    :

     One

     of

     the

     most

     exciting

     future

     trends

     in

     AI

     is

     the

     increased

     efficiency

     and

     automation

     of

     AI

     systems

    .

     As

     AI

     technologies

     become

     more

     advanced

    ,

     they

     will

     become

     more

     effective

     at

     tasks

     that

     require

     routine

    ,

     repetitive

    ,

     and

     predictable

     work

    ,

     freeing

     up

     human

     workers

     to

     focus

     on

     more

     complex

     and

     innovative

     work

    .
    


    2

    .

     Personal

    ized

     AI

    :

     AI

     will

     continue

     to

     become

     more

     personalized

    ,

     taking

     into

     account

     various

     factors

     such

     as

     user

    



```python
llm.shutdown()
```
