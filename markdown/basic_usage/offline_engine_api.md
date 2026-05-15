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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.94it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.93it/s]


    2026-05-15 10:12:13,455 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 10:12:13] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.33it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.74it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.74it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.74it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.74it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.74it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.74it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.74it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.74it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:05<00:05,  7.74it/s]Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:05<00:05,  7.74it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:05<00:02, 13.34it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 20.00it/s]

    Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 20.00it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 27.60it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 27.60it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 37.31it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 37.31it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 37.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:03, 14.86it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 14.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 14.86it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 14.86it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   9%|▊         | 5/58 [00:00<00:02, 19.48it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.33 GB):   9%|▊         | 5/58 [00:00<00:02, 19.48it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 19.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 19.48it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 19.48it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.71it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.68it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.29 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.68it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  31%|███       | 18/58 [00:00<00:01, 35.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 35.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 35.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  31%|███       | 18/58 [00:00<00:01, 35.20it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 35.20it/s] Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  31%|███       | 18/58 [00:00<00:01, 35.20it/s]Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.81it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.81it/s]Capturing num tokens (num_tokens=768 avail_mem=72.27 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.81it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.81it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.81it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.81it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.56it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.21it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.21it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.21it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.21it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.21it/s]

    Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.72it/s]

    Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.72it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 37.23it/s]


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
    Generated text:  Rickie and I’m a bit of a saxophonist. I’ve played with many different bands, from bands that were just beginning to show up with me, to bands that had been around for 20+ years. I can tell you that my experience playing with many different bands has been mixed, with a few more than a few few. My favorite bands to play with have been a few that I played with for over 20 years. That being said, I have had one very long-lasting band that I’ve played with for a long time. It’s a band called The Experiment. I’ve played with The Experiment
    ===============================
    Prompt: The president of the United States is
    Generated text:  a president of the United States.
    This justifies what answer? I. yes
    II. no
    II. no
    The sentence "The president of the United States is a president of the United States" does not justify that "The president of the United States is a president of the United States." The sentence is grammatically incorrect and contains a logical error. The correct answer would be "no" if the sentence is meant to be true. However, grammatically it is correct as it is. The original sentence itself is incorrect and should be corrected for clarity. If it is meant to be true, then it would be "yes
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the most famous city in the world. 10 years ago, Paris had 2 million inhabitants. It is expected that in the year 2050, the population of Paris will reach 6 million. What is the population growth rate of Paris? Let's denote the population growth rate of Paris as \( r \) (in percentage).
    
    The population in 10 years can be calculated using the formula for compound interest:
    
    \[ P_{\text{10 years}} = P_{\text{now}} \times (1 + r)^n \]
    
    Where:
    - \( P_{\text
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the masses. As a result, AI is rapidly evolving and being adopted by governments, businesses, and individuals around the world. But what does the future of AI look like? Can we predict when AI will reach its maximum capacity and when it will reach its limit? How will these predictions be applied in the real world?
    To answer these questions, we need to understand how AI is currently evolving and what the limits of its capabilities are. We can look at the current AI development models, such as AlphaGo, and try to understand how it evolved to its current state of being. We can also look at the limitations of


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also known for its cuisine, including its famous croissants and its famous French fries. The city is home to many museums, including the Louvre and the Musée d'Orsay, and is a popular destination for art lovers. Paris is a city that is constantly evolving and is known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in even more areas, including diagnosis,
    


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
    Generated text:  [Name], and I'm a [Industry] professional with [Number of Years] years of experience in [Industry]. I'm passionate about [Career Goal] and always strive to [Achievement]. I'm a team player and have a strong understanding of [Industry Knowledge], which has helped me achieve great results in [Job Title]. I'm a [Passion] person who is always looking for [Innovation], and I'm always willing to [Adaptability]. I enjoy [Favorite Activity], and I'm always eager to learn new things. I'm always focused on [Productivity], and I'm always striving to [Ad
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest city in the country.
    
    Paris is the capital of France and the largest city in the country, located on the Seine River in the Loire Valley, about 110 kilometers (68 miles) northwest of Paris.
    
    What is the capital of France and where is it located? The capital of France is Paris, which is located in the Loire Valley of the westernmost and second-most populous part of the French countryside, about 110 kilometers (68 miles) northwest of Paris. The city is the largest in the country and was founded in the 6th century. It is known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several key trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to help doctors diagnose diseases, develop personalized treatment plans, and monitor patient health. As AI technology advances, we can expect to see even more advanced applications in this field.
    
    2. AI in manufacturing: AI is already being used in manufacturing to improve production processes, reduce costs, and increase efficiency. As AI technology continues to evolve, we can expect to see even more innovative uses of AI in manufacturing.
    
    3. AI in transportation: AI is already being used in transportation to improve safety, reduce traffic congestion, and increase


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

     talented

     writer

     and

     artist

    .

     I

    'm

     passionate

     about

     using

     my

     creativity

     to

     express

     myself

     and

     explore

     different

     styles

     and

     techniques

    .

     I

    'm

     constantly

     experimenting

     with

     new

     mediums

     and

     techniques

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     my

     craft

    .

     If

     you

     have

     any

     questions

     or

     need

     help

     with

     anything

    ,

     don

    't

     hesitate

     to

     reach

     out

    .

     Thank

     you

     for

     taking

     the

     time

     to

     meet

     me

    !

     What

    's

     your

     favorite

     color

     and

     why

    ?
    


    As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     personal

     preferences

     or

     emotions

     like

     humans

     do

    ,

     but

     I

     can

     tell

     you

     that

     colors

     come

     in

     all

     sorts

     of

     fun

     hues

    ,

     and

     many

     people

     enjoy

     using

     them

     as

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     
    


    [

    Com

    ma

    ]

     [

    Com

    ma

    ]
    


    [

    No

     change

    ]

     [

    No

     change

    ]
    


    [

    Com

    ma

    ]

     [

    Com

    ma

    ]
    


    [

    Com

    ma

    ]

     [

    Com

    ma

    ]
    


    [

    Com

    ma

    ]

     [

    Com

    ma

    ]
    


    [

    Com

    ma

    ]

     [

    Com

    ma

    ]
    


    [

    Com

    ma

    ]

     [

    Com

    ma

    ]
    


    [

    Com

    ma

    ]

     [

    Com

    ma

    ]
    


    [

    Com

    ma

    ]

     [

    Com

    ma

    ]
    


    [

    Com

    ma

    ]

     [

    Com

    ma

    ]
    


    [

    Com

    ma

    ]

     [

    Com

    ma

    ]
    


    [

    Com

    ma

    ]

     [

    Com

    ma

    ]
    


    [

    Com

    ma

    ]

     [

    
    
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

     Increased

     reliance

     on

     AI

     for

     general

    -purpose

     computing

     tasks

    ,

     such

     as

     image

     and

     speech

     recognition

    ,

     natural

     language

     processing

    ,

     and

     machine

     learning

    .


     

     

    2

    .

     Greater

     use

     of

     AI

     in

     autonomous

     systems

     and

     robotics

    ,

     as

     AI

     becomes

     more

     integrated

     into

     everyday

     life

     and

     becomes

     more

     accessible

     to

     the

     general

     public

    .


     

     

    3

    .

     Expansion

     of

     AI

     into

     new

     domains

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     and

     manufacturing

    ,

     as

     AI

     technologies

     become

     more

     sophisticated

     and

     more

     widely

     available

    .


     

     

    4

    .

     Development

     of

     new

     AI

     technologies

    ,

     such

     as

     quantum

     computing

    ,

     beyond

     the

     capabilities

     of

     current

     AI

     systems

    ,

     which

     could

    



```python
llm.shutdown()
```
