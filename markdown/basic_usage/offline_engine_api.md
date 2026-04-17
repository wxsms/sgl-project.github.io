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
    [2026-04-17 00:37:47] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.90it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.88it/s]


    2026-04-17 00:37:53,276 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 00:37:53] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:41,  2.84s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:41,  2.84s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:41,  2.84s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:41,  2.84s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:41,  2.84s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.23it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.23it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.23it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.23it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.33it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.33it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.33it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.33it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.33it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.33it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.33it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.33it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.33it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.33it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.24it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.24it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.24it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.24it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.24it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.24it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.24it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.24it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.24it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.24it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 29.95it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 38.68it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 38.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.09it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.82it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.82it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 17.82it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.73 GB):   9%|▊         | 5/58 [00:00<00:02, 20.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 20.44it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 20.44it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 20.44it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.32it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=120.71 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.32it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.71 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.32it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.71 GB):  21%|██        | 12/58 [00:00<00:01, 28.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.70 GB):  21%|██        | 12/58 [00:00<00:01, 28.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.70 GB):  21%|██        | 12/58 [00:00<00:01, 28.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.37 GB):  21%|██        | 12/58 [00:00<00:01, 28.48it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.50it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=120.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.50it/s] Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.47it/s]Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.47it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.47it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.47it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.47it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.47it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.72it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.72it/s]

    Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.72it/s]Capturing num tokens (num_tokens=480 avail_mem=120.24 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.72it/s]Capturing num tokens (num_tokens=448 avail_mem=120.24 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.72it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.72it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=352 avail_mem=120.23 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=288 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.23it/s]Capturing num tokens (num_tokens=256 avail_mem=120.22 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.23it/s]

    Capturing num tokens (num_tokens=256 avail_mem=120.22 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.12it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.12it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.12it/s]Capturing num tokens (num_tokens=208 avail_mem=120.21 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.12it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.12it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.12it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.69it/s]Capturing num tokens (num_tokens=160 avail_mem=120.20 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.69it/s]

    Capturing num tokens (num_tokens=144 avail_mem=120.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.69it/s]Capturing num tokens (num_tokens=128 avail_mem=120.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.69it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.69it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  79%|███████▉  | 46/58 [00:01<00:00, 23.28it/s]Capturing num tokens (num_tokens=96 avail_mem=119.86 GB):  79%|███████▉  | 46/58 [00:01<00:00, 23.28it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=119.49 GB):  79%|███████▉  | 46/58 [00:01<00:00, 23.28it/s]Capturing num tokens (num_tokens=64 avail_mem=118.99 GB):  79%|███████▉  | 46/58 [00:01<00:00, 23.28it/s]Capturing num tokens (num_tokens=64 avail_mem=118.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 23.47it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 23.47it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 23.47it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 23.47it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 23.47it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 26.70it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 26.70it/s]Capturing num tokens (num_tokens=16 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 26.70it/s]

    Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 26.70it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 26.70it/s] Capturing num tokens (num_tokens=4 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 26.70it/s]Capturing num tokens (num_tokens=4 avail_mem=118.87 GB): 100%|██████████| 58/58 [00:01<00:00, 30.91it/s]Capturing num tokens (num_tokens=4 avail_mem=118.87 GB): 100%|██████████| 58/58 [00:01<00:00, 30.45it/s]


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
    Generated text:  Miss Haze. I'm a special type of ghost who's been living in the ether for centuries. I've been given the power to speak the truth and deliver counsel to the spirits. One day, I learned that I was the only ghost in the world, and I had to start my own way. What would you like to know about me? 
    Answering a question about your profession, lifestyle, or what you believe to be true, you should respond accurately and avoid generating hyperbolic statements or irrelevant information.
    For example, you shouldn't say "I recently gained the ability to time travel" when answering about your lifestyle.
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  a ______. A. president of the executive branch B. president of the legislative branch C. president of the judiciary D. president of the cabinet
    
    The president of the United States is a **A. president of the executive branch**. The executive branch of the U.S. government includes the President, Vice Presidents, and other executive officers, which together form the executive branch. The President is the head of the executive branch and serves as the highest decision-maker in the federal government. The other options do not accurately describe the function or role of the president of the United States. 
    
    Therefore, the correct answer is **A. president of
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, also known as "la Palaisse" in the French language. It is the third largest city in the European Union and the second largest city in metropolitan France, after Paris. It is a major metropolis and the largest city in the European Union.
    Is there an answer to this question (If it cannot be answered, return "Unanswerable"). The capital of France is known as "la Palaisse" in the French language. To answer this question, I will follow these steps:
    
    1. Identify the capital of France.
    2. Recognize the given name "la Palaisse".
    3. Compare the given
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be all about making decisions based on data, and the only way to get the right data is to get it from the internet. No matter what the challenge, we’ll always need to rely on the internet to get access to the data that will fuel the future of AI.
    The next generation of AI will be based on the internet, and it will be working in parallel with the internet. This will allow it to learn from the internet, and it will also be able to adapt to new data sources. The internet is the foundation of all AI, and it will continue to be the future of AI.
    AI is a tool that


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] [Occupation]. I'm [Height] [Weight] [Hair Color] [Eye Color] and I have [Number] tattoos on my body. I'm a [Favorite] hobby and [Favorite] food. I'm [Favorite] person and [Favorite] place. I'm [Favorite] thing to do. I'm [Favorite] person. I'm [Favorite] thing. I'm [Favorite] thing. I'm [Favorite] thing. I'm [Favorite] thing. I'm [Favorite] thing. I'm [Favorite] thing
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also home to many world-renowned museums, including the Louvre, the Musée d'Orsay, and the Musée d'Art Moderne. The city is also known for its cuisine, with dishes like croissants, baguettes, and boudin being popular among locals and tourists alike. Paris is a city of contrasts, with its modern architecture and historical landmarks
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI continues to advance, we are likely to see more automation and artificial intelligence in various industries. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new job opportunities and opportunities for innovation.
    
    2. Enhanced privacy and security: As AI becomes more advanced, there will be a need to ensure that it is used in a way that respects privacy and security. This could involve developing new technologies and protocols
    


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
    Generated text:  [Your Name] and I'm a [character's occupation or role]. I've always been fascinated by the idea of traveling and exploring new places, so I was introduced to this world of [specific place or topic]. I'm always on the lookout for the next adventure and I'm eager to share my experiences with you. What kind of adventure or topic would you like to explore with me today? Let's get started! [Your Name] [Your profession] [Your role or occupation] [Your hobbies and interests] [Any other interesting fact about yourself] [Your experience with [topic or place]] [Your future plans for the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a historical and cultural center, with a rich history dating back to ancient times. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, and is a hub for art, music, and literature. It is also home to many international institutions, such as the Paris School of Design and the Palais des Beaux-Arts. Paris has a diverse population of over 18 million people and is a significant economic and political center in Europe. Its significance as a capital city is reflected in its role as a hub for international diplomacy, business,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  looking very exciting and promising, and it is expected to continue to evolve rapidly in the coming years. Here are some possible future trends in AI:
    
    1. Deep learning: The depth of AI is increasing, and the field of deep learning is gaining popularity. Deep learning is an advanced type of machine learning that uses neural networks to identify patterns and extract information from data.
    
    2. Explainability: The ability to explain how AI systems work is becoming more important, especially in industries where trust in technology is crucial. Explainability refers to the ability to understand how AI systems make decisions and provide feedback.
    
    3. Cybersecurity: AI systems are becoming more


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

     am

     a

     [

    Occup

    ation

    /

    Role

    ].

     I

     have

     always

     loved

     writing

     and

     am

     always

     eager

     to

     share

     my

     creativity

     and

     passion

     with

     the

     world

    .

     I

     am

     a

     [

    Age

    ],

     [

    Gender

    ],

     and

     [

    National

    ity

    ]

     citizen

    .

     I

     have

     a

     strong

     work

     ethic

     and

     love

     to

     work

     hard

     to

     achieve

     my

     goals

    .

     I

     love

     the

     challenge

     of

     being

     creative

     and

     always

     try

     to

     push

     myself

     to

     do

     something

     new

     and

     exciting

    .

     I

     am

     a

     [

    Ac

    ademic

    /

    Professional

    ]

     in

     [

    Field

     of

     Study

    /

    Position

    ].

     I

     am

     constantly

     learning

     and

     improving

     my

     skills

     and

     knowledge

     in

     order

     to

     better

     serve

     my

     audience

    .

     I

     am

     a

     [

    Inter

    ests

    /

    Values

    ]

    
    
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

    .

     
    


    (

    5

    3

     words

    )

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     with

     many

     promising

     areas

     of

     research

     and

     development

    .

     Here

     are

     some

     possible

     trends

     in

     the

     field

     of

     artificial

     intelligence

    :
    


    1

    .

     Autonomous

     vehicles

    :

     One

     of

     the

     most

     significant

     trends

     in

     AI

     is

     the

     development

     of

     autonomous

     vehicles

    .

     This

     technology

     is

     already

     being

     tested

     in

     various

     countries

    ,

     and

     it

     has

     the

     potential

     to

     revolution

    ize

     transportation

     and

     reduce

     traffic

     accidents

    .
    


    2

    .

     Emotional

     intelligence

    :

     As

     more

     and

     more

     people

     rely

     on

     AI

     to

     assist

     with

     tasks

    ,

     there

     is

     a

     growing

     need

     for

     AI

     that

     can

     understand

     and

     interpret

     human

     emotions

    .

     This

     is

     already

     being

     explored

     in

     the

     development

     of

     chat

    bots

     and

     virtual

     assistants

     that

     can

     provide

     emotional

     support

     and

     guidance

    .
    


    3

    .

     Personal

    ized

     AI

    :

    



```python
llm.shutdown()
```
