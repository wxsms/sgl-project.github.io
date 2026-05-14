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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.64it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.64it/s]


    2026-05-14 06:37:05,763 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 06:37:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:06,  5.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:06,  5.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:06,  5.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:06,  5.39s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:06,  5.39s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:43,  1.21it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:14,  3.30it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:14,  3.30it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:14,  3.30it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:14,  3.30it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:14,  3.30it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:14,  3.30it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:14,  3.30it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:14,  3.30it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:06,  6.51it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:06,  6.51it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:06,  6.51it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:06,  6.51it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:06,  6.51it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:05<00:06,  6.51it/s]

    Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:05<00:06,  6.51it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:03,  9.85it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:03,  9.85it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:03,  9.85it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:03,  9.85it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:03,  9.85it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:03,  9.85it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:05<00:03,  9.85it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.90it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.90it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.90it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.90it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.90it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.90it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:06<00:02, 13.90it/s]

    Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:06<00:02, 13.90it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:01, 19.69it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:01, 19.69it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:01, 19.69it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:01, 19.69it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:06<00:01, 19.69it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:06<00:01, 19.69it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:06<00:01, 19.69it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:06<00:01, 19.69it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:06<00:01, 19.69it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 27.26it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 27.26it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 27.26it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 27.26it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:06<00:00, 27.26it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:06<00:00, 27.26it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:06<00:00, 27.26it/s]

    Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:06<00:00, 27.26it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:06<00:00, 27.26it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:06<00:00, 35.04it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:06<00:00, 35.04it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:06<00:00, 35.04it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:06<00:00, 35.04it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:06<00:00, 35.04it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:06<00:00, 35.04it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 16.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.27it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.27it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:02, 18.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:02, 18.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:02, 18.02it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:02, 18.02it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.71it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.71it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.75it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.93it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.93it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 26.35it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 26.35it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 26.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 26.35it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 26.35it/s] Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  38%|███▊      | 22/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.75it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.75it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.75it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.75it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.75it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.75it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.37it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.37it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.37it/s]

    Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.37it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.37it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.23it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.23it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.23it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.23it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.23it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.23it/s]

    Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.23it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.11it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.11it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.11it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.11it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.11it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.46it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.46it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.46it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.46it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.46it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 31.41it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 31.41it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 31.41it/s]

    Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 29.42it/s]


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
    Generated text:  Sun Myung- Ho. I am a student of Law. In the first part of my life, my father who was a lawyer was a fighter. He was a member of the resistance, and he was a guerrilla. He was a fighter for a good cause, which was to protect the people. He protected the people during the Korean War in Korea, and also during the Vietnam War. His life was a very long and difficult life, but he was always one who cared for the people and was always fighting for their interests.
    
    I was the second child. I was born in 1912. I was also the
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the Vice President. The Vice President is represented by the Speaker of the House of Representatives. The Speaker of the House of Representatives is also represented by the Representative from a particular district. If the representative from the district is from a state that is 50 miles away from the capital city, how many states are there in the United States? To determine how many states are in the United States, we need to analyze the given relationships and constraints step by step.
    
    1. **Identify the key positions:**
       - The President is the Vice President.
       - The Vice President is the Speaker of the House of Representatives.
      
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is located on the right bank of the Seine river, which is a major river of France. The Seine is one of the longest rivers in the world. The river Paris is the largest river in France.
    The Seine has a total length of 781.7 kilometers. In 2021, the Seine is approximately 17.57 kilometers longer than it was in 1970. The river has a width of 250 meters at its widest point.
    The Seine has been used as a waterway since the 15th century. The Se
    ===============================
    Prompt: The future of AI is
    Generated text:  bright
    
    Technology trends and the future of AI
    
      1. 16 August 2019
    
      2. 10 November 2019
    
      3. 21 May 2020
    
      4. 11 September 2020
    
      5. 15 September 2020
    
      6. 20 May 2021
    
      7. 23 June 2021
    
      8. 24 June 2021
    
      9. 1


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the arts and literature of the 19th century. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, making it a popular destination for tourists and locals alike. Paris is a city of contrasts, with its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Enhanced ethical considerations: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical implications. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Greater reliance on AI for decision-making: AI is likely to become more integrated with human decision-making processes, allowing machines to make more
    


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
    Generated text:  [First Name] and I'm excited to meet everyone. I'm a [occupation] with experience in [mention a specific job or role]. I enjoy [mention a hobby or interest that's currently or has been a passion of mine], and I'm always looking for ways to improve my skills. I'm a [mention your profession or ability] who is always ready to learn and grow. Thank you for having me! 😊 #Introducing #MyName #MyOccupation
    
    This self-introduction is neutral and straightforward, aiming to introduce the character confidently and build a connection. How can I improve my skills further based on the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Make sure to include an additional sentence that specifies Paris is also known as the "City of Light" and is known for its iconic Eiffel Tower. 
    Additionally, provide a comparative analysis of the importance of Paris in France's cultural and economic landscape by comparing its influence on Parisians, foreign visitors, and the country's overall economic output. 
    Lastly, provide a detailed description of Paris's location, including its geography and major landmarks. 
    Finally, provide a brief summary of the major industries and sectors that contribute to Paris's economy. 
    Lastly, provide a comparison of the educational system in Paris compared to other French cities
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly changing and unpredictable. Here are some potential trends in AI that may emerge:
    
    1. Deep learning will become even more sophisticated: Deep learning is the ability of machines to learn from large and complex data sets. As more data becomes available, it will become easier and more powerful to build models that can learn from even more data. This will enable AI systems to learn from complex and diverse data sets, which will have a profound impact on industries such as healthcare, finance, and transportation.
    
    2. AI will become more ubiquitous: As more and more devices and systems become connected to the internet, AI will become more pervasive. This will enable AI


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

     a

     [

    type

     of

     work

    /

    role

    ]

     with

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     [

    job

     or

     field

    ].

     My

     strength

     is

     [

    list

     of

     skills

     or

     qualities

    ],

     and

     my

     most

     important

     value

     is

     [

    value

     or

     trait

    ].

     I

    'm

     always

     eager

     to

     learn

     and

     improve

    ,

     and

     always

     strive

     to

     make

     a

     positive

     impact

    .

     I

    'm

     always

     up

     for

     a

     challenge

    ,

     and

     I

    'm

     passionate

     about

     [

    job

     or

     field

    ].

     I

     love

     to

     collaborate

     with

     others

    ,

     and

     I

     enjoy

     helping

     people

     achieve

     their

     goals

    .

     I

    'm

     a

     friendly

    ,

     approach

    able

     person

     with

     a

     strong

     sense

     of

     empathy

     and

     a

     willingness

     to

     listen

    .

     I

     value

     relationships

    ,

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     also

     known

     as

     "

    La

     Ré

    pub

    lique

    "

     and

     "

    La

     P

    ér

    ig

    ord

    ",

     is

     the

     most

     populous

     city

     in

     France

     and

     the

     fourth

    -largest

     city

     in

     Europe

    .

     It

     is

     located

     in

     the

     centre

     of

     the

     country

    ,

     between

     the

     Alps

     and

     the

     Py

    rene

    es

     mountains

    ,

     and

     serves

     as

     the

     cultural

     and

     political

     center

     of

     France

    .

     Paris

     is

     home

     to

     many

     landmarks

     and

     iconic

     buildings

    ,

     such

     as

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

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     also

     hosts

     numerous

     museums

    ,

     theaters

    ,

     and

     art

     institutions

    ,

     making

     it

     an

     important

     cultural

     center

     for

     France

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     is

     often

     referred

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     rapidly

     evolving

    ,

     with

     potential

     applications

     in

     many

     different

     areas

     of

     society

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

     AI

     will

     become

     more

     integrated

     into

     everyday

     life

    .

     AI

    -powered

     devices

     and

     systems

     will

     become

     more

     ubiquitous

    ,

     from

     smartphones

     and

     smart

     homes

     to

     self

    -driving

     cars

     and

     virtual

     assistants

    .
    


    2

    .

     AI

     will

     become

     more

     personal

     and

     tailored

     to

     individual

     needs

    .

     AI

     will

     be

     able

     to

     learn

     and

     adapt

     to

     human

     needs

     and

     preferences

    ,

     providing

     more

     personalized

     and

     context

    -aware

     experiences

    .
    


    3

    .

     AI

     will

     become

     more

     ethical

     and

     transparent

    .

     AI

     systems

     will

     be

     developed

     and

     deployed

     with

     a

     focus

     on

     ethical

     considerations

     and

     transparency

    .

     There

     will

     be

     greater

     regulations

     and

     standards

     to

     ensure

    



```python
llm.shutdown()
```
