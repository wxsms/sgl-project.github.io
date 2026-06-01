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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.90it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.66it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.66it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.66it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:12,  3.92it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:12,  3.92it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:12,  3.92it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:12,  3.92it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:07,  6.18it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:07,  6.18it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:07,  6.18it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:07,  6.18it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  8.67it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  8.67it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  8.67it/s]

    Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:04,  8.67it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:03, 11.50it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:03, 11.50it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:03, 11.50it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:03, 11.50it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.30it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.30it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.30it/s]

    Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.30it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 17.01it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 17.01it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 17.01it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 17.01it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 17.01it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 21.05it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 21.05it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 21.05it/s]

    Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 21.05it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 21.05it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 24.95it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 24.95it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 24.95it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 24.95it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 24.95it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 28.33it/s]

    Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 28.33it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 33.90it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 33.90it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 33.90it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 33.90it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:06<00:00, 33.90it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:06<00:00, 35.52it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:06<00:00, 35.52it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:06<00:00, 35.52it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:06<00:00, 35.52it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:06<00:00, 35.52it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:06<00:00, 35.52it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:06<00:00, 37.85it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:06<00:00, 37.85it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:06<00:00, 37.85it/s]

    Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:06<00:00, 37.85it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:06<00:00, 37.85it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.09 GB):   2%|▏         | 1/58 [00:00<00:07,  7.36it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.06 GB):   2%|▏         | 1/58 [00:00<00:07,  7.36it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=56.06 GB):   3%|▎         | 2/58 [00:00<00:09,  6.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.02 GB):   3%|▎         | 2/58 [00:00<00:09,  6.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.02 GB):   5%|▌         | 3/58 [00:00<00:07,  6.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.99 GB):   5%|▌         | 3/58 [00:00<00:07,  6.88it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=55.99 GB):   5%|▌         | 3/58 [00:00<00:07,  6.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.99 GB):   9%|▊         | 5/58 [00:00<00:05,  9.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.80 GB):   9%|▊         | 5/58 [00:00<00:05,  9.25it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.80 GB):  10%|█         | 6/58 [00:00<00:06,  8.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.16 GB):  10%|█         | 6/58 [00:00<00:06,  8.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.16 GB):  10%|█         | 6/58 [00:00<00:06,  8.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.16 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.17 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.19 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.94it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.19 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.94it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.19 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.19 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.19 GB):  19%|█▉        | 11/58 [00:01<00:03, 15.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.19 GB):  19%|█▉        | 11/58 [00:01<00:03, 15.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.19 GB):  24%|██▍       | 14/58 [00:01<00:02, 18.36it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.19 GB):  24%|██▍       | 14/58 [00:01<00:02, 18.36it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.28 GB):  24%|██▍       | 14/58 [00:01<00:02, 18.36it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.28 GB):  24%|██▍       | 14/58 [00:01<00:02, 18.36it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=74.27 GB):  24%|██▍       | 14/58 [00:01<00:02, 18.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.27 GB):  31%|███       | 18/58 [00:01<00:01, 22.68it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.27 GB):  31%|███       | 18/58 [00:01<00:01, 22.68it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.26 GB):  31%|███       | 18/58 [00:01<00:01, 22.68it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.24 GB):  31%|███       | 18/58 [00:01<00:01, 22.68it/s]Capturing num tokens (num_tokens=960 avail_mem=74.23 GB):  31%|███       | 18/58 [00:01<00:01, 22.68it/s] Capturing num tokens (num_tokens=960 avail_mem=74.23 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.44it/s]Capturing num tokens (num_tokens=896 avail_mem=74.25 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.44it/s]Capturing num tokens (num_tokens=832 avail_mem=74.24 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.44it/s]Capturing num tokens (num_tokens=768 avail_mem=74.23 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.44it/s]

    Capturing num tokens (num_tokens=704 avail_mem=74.23 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.44it/s]Capturing num tokens (num_tokens=640 avail_mem=74.22 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.44it/s]Capturing num tokens (num_tokens=640 avail_mem=74.22 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.96it/s]Capturing num tokens (num_tokens=576 avail_mem=74.22 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.96it/s]Capturing num tokens (num_tokens=512 avail_mem=74.20 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.96it/s]Capturing num tokens (num_tokens=480 avail_mem=74.20 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.96it/s]Capturing num tokens (num_tokens=448 avail_mem=74.21 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.96it/s]Capturing num tokens (num_tokens=416 avail_mem=74.21 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.96it/s]Capturing num tokens (num_tokens=416 avail_mem=74.21 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=384 avail_mem=74.21 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=352 avail_mem=74.20 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.34it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.19 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=288 avail_mem=74.18 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=256 avail_mem=74.17 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.34it/s]Capturing num tokens (num_tokens=256 avail_mem=74.17 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=240 avail_mem=74.17 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=224 avail_mem=74.16 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=208 avail_mem=74.15 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=192 avail_mem=74.15 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=176 avail_mem=74.15 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=176 avail_mem=74.15 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.61it/s]Capturing num tokens (num_tokens=160 avail_mem=74.14 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.61it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.13 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.61it/s]Capturing num tokens (num_tokens=128 avail_mem=74.13 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.61it/s]Capturing num tokens (num_tokens=112 avail_mem=74.13 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.61it/s]Capturing num tokens (num_tokens=96 avail_mem=74.12 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.61it/s] Capturing num tokens (num_tokens=96 avail_mem=74.12 GB):  81%|████████  | 47/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=80 avail_mem=74.11 GB):  81%|████████  | 47/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=64 avail_mem=74.10 GB):  81%|████████  | 47/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=48 avail_mem=74.10 GB):  81%|████████  | 47/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=32 avail_mem=74.09 GB):  81%|████████  | 47/58 [00:02<00:00, 39.49it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.09 GB):  88%|████████▊ | 51/58 [00:02<00:00, 38.64it/s]Capturing num tokens (num_tokens=28 avail_mem=74.08 GB):  88%|████████▊ | 51/58 [00:02<00:00, 38.64it/s]Capturing num tokens (num_tokens=24 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:02<00:00, 38.64it/s]Capturing num tokens (num_tokens=20 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:02<00:00, 38.64it/s]Capturing num tokens (num_tokens=16 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:02<00:00, 38.64it/s]Capturing num tokens (num_tokens=16 avail_mem=74.06 GB):  95%|█████████▍| 55/58 [00:02<00:00, 38.74it/s]Capturing num tokens (num_tokens=12 avail_mem=74.06 GB):  95%|█████████▍| 55/58 [00:02<00:00, 38.74it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:02<00:00, 38.74it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  95%|█████████▍| 55/58 [00:02<00:00, 38.74it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:02<00:00, 26.22it/s]


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
    Generated text:  Jules. I'm a computer programmer. I work for a company in New York City.
    
    Jules says that a computer programmer needs to have 20 hours of experience before they can start working at a company. If Jules had 15 hours of experience when he started, how many more hours will he need to work to be eligible to start working at the company?
    
    Jules has 15 hours of experience. To find out how many more hours he needs, we subtract the hours he already has from the required 20 hours.
    
    \( 20 \text{ hours required} - 15 \text
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the government. Before he becomes the president, he usually goes to a college to get a __________. The answer is: college education. Before becoming the president, the president usually goes to a college to get a college education. This is a common requirement for election to office in the United States, where the president is elected by the people through a popular vote. In the United States, college education is generally required for people who want to run for president. This helps to ensure that the president is well-educated and well-prepared to make important decisions in the government. Other requirements for being elected to office, such
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. London
    C. Rome
    D. Moscow
    Answer:
    
    A
    
    Please identify the option that does not belong to the same category as the others in the following: 
    A. Yellow
    B. Green
    C. Blue
    D. Black
    Answer:
    
    D
    
    Which of the following is the correct expression for the proposition "The greatest number of students in the city is more than 100 students"?
    A. All the students in the city are more than 100 students.
    B. No student in the city is less than 100 students.
    C. More
    ===============================
    Prompt: The future of AI is
    Generated text:  here, but how can we make sure that the AI that is created or developed by the future of AI is up to par with its creators?
    The future of AI is here, but how can we make sure that the AI that is created or developed by the future of AI is up to par with its creators?
    Over the past few decades, the field of artificial intelligence has grown rapidly and has had a dramatic impact on various industries. However, there are also concerns about the potential ethical and safety implications of AI. In this post, we will explore the future of AI, its impact on society, and the ethical considerations that must be taken


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm passionate about [reason for job title] and I'm always looking for ways to [specific goal or activity]. I'm a [character trait or quality] and I'm always ready to [action or initiative]. I'm [character trait or quality] and I'm always [action or initiative]. I'm [character trait or quality] and I'm always [action or initiative]. I'm [character trait or quality] and I'm always [action or initiative]. I'm [character trait or quality] and I'm always [action or initiative].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" or simply "Paris". It is the largest city in France and the second-largest city in the European Union. Paris is a cultural, historical, and political center, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major transportation hub, with many major highways, airports, and train stations. Paris is home to many world-renowned museums, art galleries, and theaters, and is a popular tourist destination. The city is known for its cuisine, fashion, and music
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI is expected to become more integrated into various industries, leading to increased automation of tasks and processes. This could result in job displacement but also create new opportunities for workers.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be a growing concern about privacy and security. This will require advancements in encryption, data protection, and ethical AI practices.
    
    3. AI-driven healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more sophisticated, it is likely to be used in more advanced ways, such as
    


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
    Generated text:  ___________. I am a/an ___________, ___________ (if applicable). I have a/an ___________ degree, ___________ (if applicable). I currently live in ___________. I love to ___________ and I am always looking for ways to ___________ to help others. I am a/an ___________, ___________ (if applicable). I am passionate about ___________ and I believe that ___________ (if applicable). I am a/an ___________, ___________ (if applicable). I am always striving to ___________ and am always eager to ___________ what I have to offer
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    What is the answer? Paris is the capital city of:
    
    A) Japan
    B) Italy
    C) Switzerland
    D) France
    
    D) France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and there are several trends that we can expect to see in the coming years:
    
    1. Increased specialization: As AI becomes more advanced, we can expect to see an increase in the specialization of AI systems. For example, we may see more focused areas such as image recognition, natural language processing, and speech recognition that are specifically designed to perform certain tasks.
    
    2. Integration with other technologies: AI is becoming more integrated with other technologies, such as sensors, machines, and robots, and this trend is likely to continue. We can expect to see more widespread use of AI in our daily lives, such as self-driving cars,


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

    Age

    ]

     year

     old

    ,

     [

    Occup

    ation

    ]

     [

    H

    obby

    /

    Interest

    ].

     I

     am

     not

     a

     professional

    ,

     but

     I

     am

     a

     [

    Background

    ],

     and

     I

     have

     been

     working

     hard

     to

     [

    Achie

    vement

    ].

     I

     hope

     you

     find

     my

     introduction

     to

     be

     helpful

     and

     engaging

    .

     [

    Name

    ]

     [

    Age

    ]

     [

    Occup

    ation

    ]

     [

    Background

    ]

     [

    Achie

    vement

    ]

     [

    Other

     Interesting

     Facts

     or

     Personal

    ities

    ]

     [

    Name

    ]

     [

    Age

    ]

     [

    Occup

    ation

    ]

     [

    Background

    ]

     [

    Achie

    vement

    ]

     [

    Other

     Interesting

     Facts

     or

     Personal

    ities

    ]

     [

    Name

    ]

     [

    Age

    ]

     [

    Occup

    ation

    ]

     [

    Background

    ]

     [

    Achie

    vement

    ]

     [

    Other

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     historical

     landmarks

    ,

     vibrant

     music

     scene

    ,

     and

     world

    -class

     museums

     and

     art

     galleries

    .

     It

     is

     a

     bustling

     city

     with

     a

     rich

     cultural

     heritage

     and

     is

     a

     popular

     tourist

     destination

    .

     The

     city

     is

     home

     to

     many

     international

     organizations

     and

     influential

     figures

     in

     both

     France

     and

     the

     world

    .

     French

     cuisine

    ,

     known

     for

     its

     complex

     dishes

     and

     traditional

     methods

    ,

     is

     a

     hallmark

     of

     the

     city

    .

     The

     city

     is

     also

     home

     to

     many

     renowned

     universities

     and

     has

     a

     strong

     tradition

     of

     innovation

     and

     creativity

    .

     The

     city

     is

     a

     major

     player

     in

     the

     global

     economy

     and

     attracts

     many

     visitors

     from

     around

     the

     world

     every

     year

    .

     
    


    In

     summary

    ,

     Paris

     is

     a

     city

     with

     a

     rich

     history

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     and

     there

     are

     several

     potential

     trends

     that

     are

     likely

     to

     shape

     the

     technology

    's

     direction

    .

     Here

     are

     some

     of

     the

     most

     common

     trends

     in

     AI

     that

     are

     expected

     to

     shape

     the

     future

    :
    


    1

    .

     Increased

     emphasis

     on

     ethical

     considerations

    :

     As

     AI

     technology

     becomes

     more

     advanced

     and

     pervasive

    ,

     there

     is

     a

     growing

     emphasis

     on

     ethical

     considerations

    .

     This

     includes

     issues

     such

     as

     bias

    ,

     transparency

    ,

     and

     accountability

    .
    


    2

    .

     Greater

     integration

     with

     human

     decision

    -making

    :

     AI

     is

     increasingly

     being

     integrated

     into

     decision

    -making

     processes

    ,

     including

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    .

     This

     integration

     is

     expected

     to

     further

     enhance

     the

     accuracy

     and

     reliability

     of

     AI

     systems

    .
    


    3

    .

     Expansion

     of

     AI

    's

     applications

    



```python
llm.shutdown()
```
