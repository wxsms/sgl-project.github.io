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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.34it/s]


    2026-05-02 14:56:46,635 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 14:56:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:06<06:03,  6.38s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:06<06:03,  6.38s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:06<06:03,  6.38s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:06<06:03,  6.38s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:06<06:03,  6.38s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:51,  1.03it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:51,  1.03it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:06<00:51,  1.03it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:06<00:51,  1.03it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:06<00:51,  1.03it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:06<00:51,  1.03it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:06<00:51,  1.03it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:06<00:51,  1.03it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:14,  3.11it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:14,  3.11it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:06<00:14,  3.11it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:06<00:14,  3.11it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:06<00:14,  3.11it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:06<00:14,  3.11it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:06<00:14,  3.11it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:06<00:14,  3.11it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:06<00:14,  3.11it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:06<00:14,  3.11it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:06<00:05,  6.69it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:06<00:05,  6.69it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:06<00:05,  6.69it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:06<00:05,  6.69it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:06<00:05,  6.69it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:06<00:05,  6.69it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:06<00:05,  6.69it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:06<00:05,  6.69it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:06<00:05,  6.69it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:06<00:05,  6.69it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:06<00:02, 11.29it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:06<00:02, 11.29it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:06<00:02, 11.29it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:06<00:02, 11.29it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:06<00:02, 11.29it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:06<00:02, 11.29it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:06<00:02, 11.29it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:06<00:02, 11.29it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:06<00:02, 11.29it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:06<00:02, 11.29it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:06<00:02, 11.29it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:06<00:01, 17.70it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:06<00:01, 17.70it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:06<00:01, 17.70it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:06<00:01, 17.70it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:06<00:01, 17.70it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:06<00:01, 17.70it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:06<00:01, 17.70it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:06<00:01, 17.70it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:06<00:01, 17.70it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:07<00:01, 17.70it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:07<00:01, 17.70it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:07<00:00, 25.27it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:07<00:00, 25.27it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:07<00:00, 25.27it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:07<00:00, 25.27it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:07<00:00, 25.27it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:07<00:00, 25.27it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:07<00:00, 25.27it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:07<00:00, 25.27it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:07<00:00, 25.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  8.18it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.15it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.25it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.29 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.23 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.23 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.41it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=71.98 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=960 avail_mem=71.99 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.41it/s] Capturing num tokens (num_tokens=960 avail_mem=71.99 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.34it/s]Capturing num tokens (num_tokens=896 avail_mem=71.99 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.34it/s]Capturing num tokens (num_tokens=832 avail_mem=71.99 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.34it/s]Capturing num tokens (num_tokens=768 avail_mem=71.98 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.34it/s]Capturing num tokens (num_tokens=704 avail_mem=71.98 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.34it/s]Capturing num tokens (num_tokens=640 avail_mem=71.98 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.34it/s]Capturing num tokens (num_tokens=640 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=576 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=512 avail_mem=71.96 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=480 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.93it/s]

    Capturing num tokens (num_tokens=448 avail_mem=71.98 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=416 avail_mem=71.97 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=416 avail_mem=71.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.50it/s]Capturing num tokens (num_tokens=384 avail_mem=71.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.50it/s]Capturing num tokens (num_tokens=352 avail_mem=71.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.50it/s]Capturing num tokens (num_tokens=320 avail_mem=71.96 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.50it/s]Capturing num tokens (num_tokens=288 avail_mem=71.96 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.50it/s]Capturing num tokens (num_tokens=256 avail_mem=71.96 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.50it/s]Capturing num tokens (num_tokens=256 avail_mem=71.96 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.43it/s]Capturing num tokens (num_tokens=240 avail_mem=71.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.43it/s]Capturing num tokens (num_tokens=224 avail_mem=71.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.43it/s]Capturing num tokens (num_tokens=208 avail_mem=71.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.43it/s]

    Capturing num tokens (num_tokens=192 avail_mem=71.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.43it/s]Capturing num tokens (num_tokens=176 avail_mem=71.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.43it/s]Capturing num tokens (num_tokens=176 avail_mem=71.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=160 avail_mem=71.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=144 avail_mem=71.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=128 avail_mem=71.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=112 avail_mem=71.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.39it/s]Capturing num tokens (num_tokens=96 avail_mem=71.93 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.39it/s] Capturing num tokens (num_tokens=96 avail_mem=71.93 GB):  81%|████████  | 47/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=80 avail_mem=71.92 GB):  81%|████████  | 47/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=64 avail_mem=71.92 GB):  81%|████████  | 47/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=48 avail_mem=71.92 GB):  81%|████████  | 47/58 [00:01<00:00, 45.90it/s]

    Capturing num tokens (num_tokens=32 avail_mem=71.91 GB):  81%|████████  | 47/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=28 avail_mem=71.91 GB):  81%|████████  | 47/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=28 avail_mem=71.91 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.19it/s]Capturing num tokens (num_tokens=24 avail_mem=71.91 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.19it/s]Capturing num tokens (num_tokens=20 avail_mem=71.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.19it/s]Capturing num tokens (num_tokens=16 avail_mem=71.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.19it/s]Capturing num tokens (num_tokens=12 avail_mem=71.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.19it/s]Capturing num tokens (num_tokens=8 avail_mem=71.90 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.19it/s] Capturing num tokens (num_tokens=8 avail_mem=71.90 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.80it/s]Capturing num tokens (num_tokens=4 avail_mem=71.89 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.80it/s]Capturing num tokens (num_tokens=4 avail_mem=71.89 GB): 100%|██████████| 58/58 [00:01<00:00, 40.06it/s]


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
    Generated text:  J. For my upcoming test, I have to create a 3D model of an ancient Egyptian pyramid. I need to create an 8.5 inch tall tall by 10 inch wide by 3.5 inch depth model. Is it possible to create this model using the software Blender? Yes, it is possible to create an 8.5 inch tall, 10 inch wide, and 3.5 inch depth 3D model using Blender. However, Blender is a 2D modeling software, so you will need to convert the 3D model to 2D before you can create the model
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the Vice President. The Vice President is also represented by the Speaker of the House of Representatives. How many members of the House of Representatives are there if there are 130 members and one-third of them are members of the executive branch?
    
    To determine the number of members in the House of Representatives, we need to follow a systematic approach to find out the number of members in the executive branch and then use that to find the total number of members in the House of Representatives.
    
    1. **Calculate the number of members in the executive branch:**
       - We know that one-third of the 130 members of the
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Brussels
    C. Lyon
    D. London
    Answer:
    
    A
    
    There's a bit of a tradition, it's called a "Préambule" or "Préabat" in French language education. It's a document that sets the parameters of the entire educational process. Before class, some teacher students discuss the curriculum, assignments, and other issues that will be covered in class, along with the teacher's expectations for student participation and class behavior. What is this document called? 
    A. Public Notice
    B. Academic Calendar
    C. Teacher's Guide
    D. Pre-
    ===============================
    Prompt: The future of AI is
    Generated text:  already here, and it’s happening faster than we think. The technology has already made many advances in every field of human existence, and the future is only going to be bolder and more radical as the technological possibilities expand. It’s important to keep in mind that the future is not just for humans. There are animals, plants, and even non-human AI that can do amazing things, and they are all around us. As we explore the future of AI, it’s important to understand that there are always new technologies and ideas that are being developed, and it’s important to stay informed about what’s happening to keep up with the rapid


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your job or profession]. I enjoy [insert a brief description of your hobbies or interests]. I'm always looking for new experiences and learning new things. What's your favorite hobby or activity? I'm always up for a challenge and love to try new things. What's your favorite book or movie? I love to read and watch movies, and I'm always looking for new adventures. What's your favorite place to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third largest city in the world by population. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a cultural and economic hub of France and plays a significant role in the country's political and economic life. It is home to many famous landmarks and attractions, including the Louvre, the Champs-Élysées, and the Notre-Dame Cathedral. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence in the future, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for the development and use
    


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
    Generated text:  [Name], and I'm a [role or occupation] with [number] years of experience in the [industry] field. I have a [number] of years of experience in the [specific skill or area] field. I'm always ready to learn and continue my growth in this area, and I'm looking forward to meeting you at [date and location]. Let's chat and see what we can accomplish together! [Name's name]. [Name's job title] [Name's specialty] [Name's professional profile]. [Name's job title] [Name's specialty] [Name's professional profile].
    As a [Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La République" (the Republic).
    Paris is the largest city in France, located in the north-central region of the country. It is home to around 2.5 million people, making it one of the largest cities in the world, and is one of the oldest continuously inhabited cities in the world. The city has a rich cultural heritage and is known for its famous landmarks, including the Eiffel Tower, the Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its fashion, art, and cuisine. Paris is a major center for higher education, business, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see continued innovation and progress, with the technology becoming more sophisticated and adaptable. Here are some potential trends that could shape the future of AI:
    
    1. Increasingly advanced natural language processing: AI systems will continue to improve their ability to understand and process human language, allowing them to perform tasks that would be impossible for humans alone. This will require more advanced models and algorithms that can better capture and interpret complex social and emotional nuances.
    
    2. Deep learning and machine learning: As AI systems continue to grow in complexity and reach, they will increasingly rely on deep learning and machine learning to perform more sophisticated tasks. These methods will be used to


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

    ],

     and

     I

    'm

     here

     to

     meet

     you

     with

     my

     unique

     and

     versatile

     skills

    .

     My

     journey

     has

     been

     filled

     with

     learning

     and

     growth

    ,

     and

     I

    'm

     constantly

     striving

     to

     be

     a

     better

     version

     of

     myself

    .

     From

     a

     young

     age

    ,

     I

    've

     had

     the

     pleasure

     of

     experiencing

     many

     different

     cultures

     and

     cuis

    ines

    ,

     which

     has

     given

     me

     a

     unique

     perspective

     on

     the

     world

     and

     its

     diverse

     inhabitants

    .

     I

     believe

     in

     using

     my

     skills

     to

     help

     others

    ,

     and

     I

    'm

     always

     eager

     to

     learn

     and

     adapt

     to

     new

     challenges

    .

     I

    'm

     here

     to

     meet

     you

     and

     make

     a

     positive

     impact

    .

     Let

    's

     do

     this

     together

    !

     

    🌟

    👨

    ‍

    🍳

    👩

    ‍

    🍳

    
    


    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    France

    's

     capital

     city

     is

     Paris

    .

     This

     is

     because

     it

     is

     the

     largest

     and

     most

     populous

     city

     in

     France

    ,

     with

     a

     population

     of

     approximately

     

    6

    .

    8

     million

     people

    .

     It

     is

     home

     to

     the

     government

    ,

     government

     departments

    ,

     and

     many

     of

     the

     country

    's

     institutions

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     cultural

     attractions

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

     Dame

     Cathedral

    .

     As

     the

     capital

    ,

     Paris

     plays

     a

     vital

     role

     in

     France

    's

     governance

    ,

     politics

    ,

     and

     daily

     life

    .

     Its

     importance

     can

     be

     seen

     in

     its

     status

     as

     the

     French

     capital

    ,

     which

     has

     a

     history

     dating

     back

     to

     the

     

    8

    th

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     not

     just

     looking

     forward

     to

     future

     trends

    ,

     but

     also

     to

     what

     the

     present

     is

     about

     to

     become

    .

     In

     the

     coming

     years

    ,

     we

     can

     expect

     to

     see

     an

     increased

     focus

     on

     AI

    's

     impact

     on

     society

    ,

     including

     how

     it

     will

     shape

     our

     future

     work

    ,

     our

     daily

     lives

    ,

     and

     the

     way

     we

     live

    .

     Here

     are

     some

     of

     the

     most

     likely

     future

     trends

     in

     AI

    :
    


    1

    .

     AI

     will

     become

     more

     integrated

     with

     our

     everyday

     lives

    .

     With

     the

     rise

     of

     machine

     learning

     and

     deep

     learning

    ,

     AI

     is

     becoming

     more

     integrated

     into

     our

     daily

     lives

    .

     From

     voice

     assistants

     to

     self

    -driving

     cars

    ,

     we

     can

     expect

     AI

     to

     become

     even

     more

     ingr

    ained

     in

     our

     daily

     routines

    .
    


    2

    .

     AI

    



```python
llm.shutdown()
```
