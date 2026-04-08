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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.71it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.70it/s]


    2026-04-08 23:21:49,942 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 23:21:49] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:45,  2.90s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:45,  2.90s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:45,  2.90s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:45,  2.90s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:31,  1.74it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:31,  1.74it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:31,  1.74it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:31,  1.74it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:31,  1.74it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:31,  1.74it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:31,  1.74it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.36it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.41it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.41it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.41it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.41it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.41it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.41it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.41it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.41it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.28it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.28it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.28it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.28it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.28it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.28it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.28it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.28it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 23.79it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 23.79it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 23.79it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 23.79it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 23.79it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 23.79it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 23.79it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.21it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.21it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.21it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.21it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.21it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.21it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.21it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 37.65it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 46.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 14.61it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 14.61it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=131.63 GB):   3%|▎         | 2/58 [00:00<00:03, 14.61it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.63 GB):   7%|▋         | 4/58 [00:00<00:07,  6.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   7%|▋         | 4/58 [00:00<00:07,  6.79it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   7%|▋         | 4/58 [00:00<00:07,  6.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):  10%|█         | 6/58 [00:00<00:05,  9.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):  10%|█         | 6/58 [00:00<00:05,  9.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):  10%|█         | 6/58 [00:00<00:05,  9.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  10%|█         | 6/58 [00:00<00:05,  9.19it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  10%|█         | 6/58 [00:00<00:05,  9.19it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.57it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.73it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.73it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.73it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.73it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  26%|██▌       | 15/58 [00:01<00:01, 22.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  34%|███▍      | 20/58 [00:01<00:01, 28.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  34%|███▍      | 20/58 [00:01<00:01, 28.39it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  34%|███▍      | 20/58 [00:01<00:01, 28.39it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  34%|███▍      | 20/58 [00:01<00:01, 28.39it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  34%|███▍      | 20/58 [00:01<00:01, 28.39it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  34%|███▍      | 20/58 [00:01<00:01, 28.39it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.67it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.67it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.67it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.67it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.67it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.67it/s]

    Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.82it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.82it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.82it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.82it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.82it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.82it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 38.16it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 38.16it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 38.16it/s]

    Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 38.16it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  60%|██████    | 35/58 [00:01<00:00, 38.16it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  60%|██████    | 35/58 [00:01<00:00, 38.16it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.77it/s]

    Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.24it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.04it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.04it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.04it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.04it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.04it/s]

    Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.04it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.93it/s] Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.93it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 29.59it/s]


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
    Generated text:  Cynthia, I am a day laborer. I have no money. I have a few pieces of fabric and a sewing machine. I have an idea for a small home theater system. The goal is to have the system ready to install in the month of June. I am trying to make it work with the low interest that is available for a small business. How much money would it take to pay for the materials, the machine, and the parts for the system? Based on your description, it sounds like you have a small but ambitious project for your home theater system. You mention that you have an idea, materials, and parts,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. The president of the United States has been in office for ______ years. A. twenty years B. thirty years C. forty years D. sixty years D.
    
    Therefore, the answer is D. sixty years. As of the latest data available, the president of the United States has been in office for sixty years. While there have been some significant changes in the presidency over time, sixty years remains the most recent and accurate period for the current president. Other presidents have been in office for shorter periods, such as four years or five years, but sixty years has been the longest period for the current president.
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. Berlin C. Moscow D. Madrid
    Answer:
    
    A
    
    To ensure that the power system operates safely and reliably, which of the following tasks should be carried out first?
    A. Check the system's technical condition
    B. Conduct system maintenance and repair
    C. Implement risk assessment and risk management
    D. Conduct safety assessment and risk management
    Answer:
    
    C
    
    In the field of computer software, which of the following correctly describes the role of the Windows operating system?
    A. The Windows operating system controls the creation and management of files.
    B. The Windows operating system processes and executes all programs.
    C
    ===============================
    Prompt: The future of AI is
    Generated text:  about AI.
    Artificial intelligence is a complex topic with many applications ranging from healthcare and finance to education and transportation. AI is also a growing field of interest for companies as they try to create more efficient and smarter products and services. However, the topic of AI can also be a source of controversy and debate. One of the biggest concerns is the potential for AI to be used to create autonomous weapons and other dangerous technologies.
    In this blog post, we will explore some of the key challenges and opportunities of AI and how companies and individuals can work together to ensure that AI is used for the benefit of society and not for the harm of individuals.
    


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a few key points about you, such as your personality, interests, or skills]. How can I help you today? I'm always here to assist you with any questions or concerns you may have. What's your name, and what can you tell me about yourself? I'm [insert a few key points about you, such as your personality, interests, or skills]. How can I help you today? I'm always here to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination and is home to many world-renowned artists, writers, and musicians. The city is also known for its rich history, including the influence of French colonialism and the influence of the French Revolution. Paris is a vibrant and dynamic city that continues to be a major cultural and economic hub in France. 
    
    The statement is: Paris is a major cultural and economic center in France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to become more integrated into our daily lives, from manufacturing to healthcare. This could lead to increased automation and the creation of new jobs, but it could also lead to job displacement for some workers.
    
    2. AI ethics and privacy concerns: As AI becomes more advanced, it is likely to raise ethical and privacy concerns. This could lead to debates about the use of AI in areas such as surveillance,
    


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
    Generated text:  [Name] and I'm a [short, friendly, neutral name] who have been active in [specific, long-term, short-term, or ongoing] [field or area of study] for [number of years] years. I bring a wealth of [relevant, varied, or unique] skills to my work that I believe are essential for the success of my [industry or career] and I am eager to bring my [specific skill or experience] to any opportunity that may come my way. I am always looking to improve myself and to learn new things, and I am committed to staying up-to-date with the latest trends and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and the heart of the country. It is known for its rich cultural heritage, beautiful architecture, and vibrant nightlife. The city has a population of over 2 million people and is home to numerous museums, art galleries, and historical sites. Paris is also the seat of the government and is an important economic center in Europe. The city is known for its famous landmarks like the Eiffel Tower and the Louvre Museum, and its long-standing artistic and cultural legacy.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  filled with exciting possibilities and potential challenges. Here are some of the most common trends that are likely to shape AI in the next decade:
    
    1. Increased AI integration with traditional industries: AI is already being integrated into a wide range of industries, from manufacturing and healthcare to finance and retail. As these industries continue to adopt AI, we can expect to see more collaboration between AI and traditional industries, as well as the development of new AI applications that are tailored to specific industries.
    
    2. AI systems with more autonomy: As AI systems become more sophisticated, we can expect to see more autonomous AI systems that are capable of making decisions based on a wide


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

     am

     a

     [

    Age

    ]

     year

     old

    ,

     [

    Occup

    ation

    ].

     I

     am

     an

     [

    Ability

    /

    Interest

    ].

     I

     have

     always

     been

     an

     [

    Favorite

     Thing

    ].

     I

     am

     a

     [

    Favorite

     Color

    ].

     I

     am

     a

     [

    Favorite

     Quote

    ].

     I

     love

     [

    Favorite

     Food

    ].

     I

     am

     passionate

     about

     [

    Favorite

     Hobby

    ].

     I

     am

     a

     [

    Favorite

     Movie

    ].

     I

     enjoy

     [

    Favorite

     Music

    ].

     I

     am

     a

     [

    Favorite

     Book

    ].

     I

     am

     [

    Favorite

     Sport

    ].

     I

     love

     [

    Favorite

     Animal

    ].

     I

     am

     [

    Favorite

     Movie

    ].

     I

     am

     a

     [

    Favorite

     Movie

    ].

     I

     am

     a

     [

    Favorite

     Movie

    ].

     I

     am

     a

     [

    Favorite

     Movie

    ].

     I

     am

     a

     [

    Favorite

     Movie

    ].

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     capital

     of

     the

     country

    .

     It

     is

     home

     to

     the

     nation

    's

     cultural

     capital

    ,

     including

     the

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     also

     known

     for

     its

     vibrant

     arts

     scene

     and

     historic

     architecture

    .

     The

     city

     has

     a

     diverse

     population

     and

     a

     rich

     history

     dating

     back

     to

     the

     Middle

     Ages

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     cultural

     hub

     for

     Europe

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     the

     heart

     of

     the

     French

    -speaking

     world

    .

     Paris

     is

     known

     for

     its

     art

    ,

     culture

    ,

     and

     cuisine

    ,

     and

     is

     a

     major

     center

     of

     politics

     and

     politics

    .

     It

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

    ,

     and

     there

     are

     many

     potential

     trends

     that

     could

     shape

     the

     development

     of

     this

     technology

     in

     the

     coming

     years

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

     that

     could

     be

     expected

     to

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Automation

     and

     Automation

    :

     The

     automation

     of

     routine

     tasks

     and

     processes

     has

     the

     potential

     to

     greatly

     reduce

     the

     workload

     on

     humans

     and

     create

     new

     opportunities

     for

     AI

    -driven

     AI

    .

     This

     could

     lead

     to

     significant

     increases

     in

     productivity

     and

     efficiency

    ,

     and

     could

     also

     lead

     to

     the

     development

     of

     new

     types

     of

     AI

     algorithms

     that

     can

     perform

     tasks

     that

     were

     previously

     thought

     to

     be

     too

     complex

     for

     human

     AI

    .
    


    2

    .

     Eth

    ical

     Consider

    ations

    :

     As

     AI

     continues

     to

     develop

    



```python
llm.shutdown()
```
