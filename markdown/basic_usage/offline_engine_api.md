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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.98it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.98it/s]


    2026-05-13 06:55:46,747 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 06:55:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.10it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.10it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.10it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.10it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.10it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.10it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.10it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.10it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.10it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 12.85it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 12.85it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 12.85it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 12.85it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 12.85it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 12.85it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 12.85it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 12.85it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 17.65it/s]

    Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 22.12it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 26.65it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 26.65it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 26.65it/s]

    Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 26.65it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 26.65it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:06<00:00, 26.65it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:06<00:00, 26.65it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:06<00:00, 26.65it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:06<00:00, 26.65it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:06<00:00, 34.67it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:06<00:00, 34.67it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   2%|▏         | 1/58 [00:00<00:06,  8.53it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   2%|▏         | 1/58 [00:00<00:06,  8.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   2%|▏         | 1/58 [00:00<00:06,  8.53it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   5%|▌         | 3/58 [00:00<00:05, 10.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   5%|▌         | 3/58 [00:00<00:05, 10.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   5%|▌         | 3/58 [00:00<00:05, 10.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:04, 13.13it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:04, 13.13it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:04, 13.13it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.08 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.35 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.35 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.35 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.12 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.25it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.12 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.12 GB):  21%|██        | 12/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.33 GB):  21%|██        | 12/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.33 GB):  21%|██        | 12/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.33 GB):  24%|██▍       | 14/58 [00:00<00:02, 16.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.32 GB):  24%|██▍       | 14/58 [00:00<00:02, 16.79it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.22 GB):  24%|██▍       | 14/58 [00:00<00:02, 16.79it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.22 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.16 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.17 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.60it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.17 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.60it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.17 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.17 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.54it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.27 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.54it/s]Capturing num tokens (num_tokens=960 avail_mem=74.28 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.54it/s] Capturing num tokens (num_tokens=960 avail_mem=74.28 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.80it/s]Capturing num tokens (num_tokens=896 avail_mem=74.27 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.80it/s]Capturing num tokens (num_tokens=832 avail_mem=74.26 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.80it/s]Capturing num tokens (num_tokens=768 avail_mem=74.26 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.80it/s]Capturing num tokens (num_tokens=704 avail_mem=74.25 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.80it/s]

    Capturing num tokens (num_tokens=704 avail_mem=74.25 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.27it/s]Capturing num tokens (num_tokens=640 avail_mem=74.25 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.27it/s]Capturing num tokens (num_tokens=576 avail_mem=74.24 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.27it/s]Capturing num tokens (num_tokens=512 avail_mem=74.19 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.27it/s]Capturing num tokens (num_tokens=480 avail_mem=74.22 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.27it/s]Capturing num tokens (num_tokens=480 avail_mem=74.22 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.49it/s]Capturing num tokens (num_tokens=448 avail_mem=74.22 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.49it/s]Capturing num tokens (num_tokens=416 avail_mem=74.21 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.49it/s]Capturing num tokens (num_tokens=384 avail_mem=74.21 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.49it/s]Capturing num tokens (num_tokens=352 avail_mem=74.20 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.49it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.20 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.50it/s]Capturing num tokens (num_tokens=320 avail_mem=74.18 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.50it/s]Capturing num tokens (num_tokens=288 avail_mem=74.18 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.50it/s]Capturing num tokens (num_tokens=256 avail_mem=74.18 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.50it/s]Capturing num tokens (num_tokens=240 avail_mem=74.18 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.50it/s]Capturing num tokens (num_tokens=240 avail_mem=74.18 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=224 avail_mem=74.17 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=208 avail_mem=74.16 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=192 avail_mem=74.16 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=176 avail_mem=74.15 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.18it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.15 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=160 avail_mem=74.15 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=144 avail_mem=74.14 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=128 avail_mem=74.14 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=112 avail_mem=74.13 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=96 avail_mem=74.12 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.64it/s] Capturing num tokens (num_tokens=96 avail_mem=74.12 GB):  81%|████████  | 47/58 [00:01<00:00, 35.80it/s]Capturing num tokens (num_tokens=80 avail_mem=74.11 GB):  81%|████████  | 47/58 [00:01<00:00, 35.80it/s]Capturing num tokens (num_tokens=64 avail_mem=74.13 GB):  81%|████████  | 47/58 [00:02<00:00, 35.80it/s]Capturing num tokens (num_tokens=48 avail_mem=74.12 GB):  81%|████████  | 47/58 [00:02<00:00, 35.80it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.12 GB):  81%|████████  | 47/58 [00:02<00:00, 35.80it/s]Capturing num tokens (num_tokens=32 avail_mem=74.12 GB):  88%|████████▊ | 51/58 [00:02<00:00, 36.26it/s]Capturing num tokens (num_tokens=28 avail_mem=74.11 GB):  88%|████████▊ | 51/58 [00:02<00:00, 36.26it/s]Capturing num tokens (num_tokens=24 avail_mem=74.10 GB):  88%|████████▊ | 51/58 [00:02<00:00, 36.26it/s]Capturing num tokens (num_tokens=20 avail_mem=74.10 GB):  88%|████████▊ | 51/58 [00:02<00:00, 36.26it/s]Capturing num tokens (num_tokens=16 avail_mem=74.09 GB):  88%|████████▊ | 51/58 [00:02<00:00, 36.26it/s]Capturing num tokens (num_tokens=16 avail_mem=74.09 GB):  95%|█████████▍| 55/58 [00:02<00:00, 36.91it/s]Capturing num tokens (num_tokens=12 avail_mem=74.08 GB):  95%|█████████▍| 55/58 [00:02<00:00, 36.91it/s]Capturing num tokens (num_tokens=8 avail_mem=74.08 GB):  95%|█████████▍| 55/58 [00:02<00:00, 36.91it/s] Capturing num tokens (num_tokens=4 avail_mem=74.07 GB):  95%|█████████▍| 55/58 [00:02<00:00, 36.91it/s]

    Capturing num tokens (num_tokens=4 avail_mem=74.07 GB): 100%|██████████| 58/58 [00:02<00:00, 25.63it/s]


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
    Generated text:  Alex. I am currently studying at the University of California, Berkeley. I am an undergraduate student majoring in Statistics and computer science. I have been interested in statistics and computer science since I was a young child, and I have been working on various projects since I was a teenager.
    I have been working on projects with a variety of people, including scientists, software engineers, and data scientists. I have also been working on various other projects, such as building a system for monitoring and analyzing large amounts of data, and creating a tool for natural language processing.
    In my free time, I enjoy spending time with my family and friends, reading
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to use the budget to fund military spending or to fund social programs. The president has a budget of $10,000,000. The military budget is 15% of the budget. The president wants to maximize the use of the remaining budget for social programs. 
    
    If the social program budget increases by 20%, what is the new budget allocation for social programs? To determine the new budget allocation for social programs, we need to follow these steps:
    
    1. **Calculate the military budget:**
       The military budget is 15% of the total budget.
       \[
    
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A: Paris
    
    B: Lyon
    
    C: Marseille
    
    D: Nice
    
    To determine the capital of France, let's follow a step-by-step reasoning process:
    
    1. **Identify the key elements of the capital**:
       - The capital of France is typically a city where the government is headquartered and where important decisions are made.
    
    2. **Consider the options provided**:
       - Options are: Paris, Lyon, Marseille, and Nice.
    
    3. **Evaluate each option**:
       - **Paris**: While Paris is one of the most famous cities in France, it is not the capital. It is the largest city
    ===============================
    Prompt: The future of AI is
    Generated text:  complex. AI is the realisation of the future and the future of AI will be driven by the imagination of the human mind. AI is not a technology, but a concept. It is a reflection of the mindset of the mind of the human brain. If the human brain is a machine, then the machine has its own way of thinking, learning, and growing. AI is a reflection of the human brain, but it is a different way of thinking and learning. It is a way of thinking that is far more complex, dynamic, and adaptive than a machine could be.
    The world of AI is evolving at an ever-increasing pace


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] years in the field of [Occupation]. I'm passionate about [What I Love About My Profession]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [What I Do Best] person who is always ready to help others. I'm [What I Can Do For You]. I'm excited to meet you and learn more about you. [Name] [Age] [Occupation] [Skill] [What I Love About My Profession] [What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and music, and is a major tourist destination. Paris is a cultural and intellectual center of Europe, and is home to many famous museums, theaters, and art galleries. It is also a major financial center, with many international companies and institutions headquartered in the city. Paris is a city of contrasts, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread adoption in healthcare, with AI being used to diagnose diseases, predict patient outcomes, and optimize treatment plans.
    
    2. AI in manufacturing: AI is already being used in manufacturing to optimize production processes, reduce costs, and improve quality. As AI technology continues
    


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
    Generated text:  [Name], and I'm a/an [Occupation], [Career Type]. I'm passionate about [What interests me or excites me about the field], and I'm always learning and growing. I enjoy [What motivates me], and I strive to be a [Skill or Ability]. And my [What does I hope to achieve with my career?]. I'm looking forward to [What kind of day or week my career will bring me].
    This is a fantastic introduction for a fictional character! Here's a slightly refined version:
    
    "Hello, my name is [Name], and I'm a [Occupation], [Career Type
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and the seat of government, administration, and culture in the country. It is known for its towering architecture, vibrant culture, and long history. The city is home to the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and many other attractions. It is also a major financial hub and a world-renowned fashion and art center. Paris is the most visited city in the world, with over 70 million visitors annually. It has a rich cultural and historical heritage, making it an important center for education, research, and arts in France. The city's political and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be a rapidly evolving field. Here are some possible trends that may emerge in the coming years:
    
    1. Increased integration of AI into daily life: As AI becomes more widespread, we may see a growing integration of AI into our daily lives, such as in our smartphones, wearables, and autonomous vehicles.
    
    2. Greater focus on ethical and responsible AI: There will likely be greater emphasis on the ethical implications of AI, including issues related to bias, fairness, transparency, and accountability.
    
    3. Continued advancements in AI technology: AI is likely to continue to advance rapidly, with new technologies and techniques emerging regularly.
    
    4. Greater integration


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

    Gender

    ],

     [

    Role

    ]

     at

     [

    Company

    ].

     I

     have

     a

     [

    Role

    ]

     at

     [

    Company

    ],

     and

     I

    'm

     passionate

     about

     [

    Personal

     Interest

    ].

     If

     you

     need

     any

     information

     about

     [

    Company

    ]

     or

     [

    Role

    ],

     I

    'd

     be

     happy

     to

     provide

     it

     for

     you

    .

     How

     can

     I

     help

     you

     today

    ?

     [

    Name

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     historic

     landmarks

     and

     bustling

     shopping

     districts

    .

     The

     city

     is

     also

     a

     hub

     for

     the

     world

    ’s

     most

     prestigious

     art

     and

     cultural

     institutions

    ,

     including

     the

     Lou

    vre

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     It

    's

     one

     of

     the

     world

    's

     most

     famous

     cities

    ,

     and

     a

     popular

     tourist

     destination

    .

     Paris

     is

     also

     known

     for

     its

     rich

     history

     and

     culture

    ,

     with

     a

     diverse

     range

     of

     art

    ,

     music

    ,

     and

     cuisine

    .

     The

     French

     capital

     is

     a

     bustling

     center

     of

     politics

    ,

     business

    ,

     and

     culture

     in

     the

     world

    .

     Its

     architecture

    ,

     fashion

    ,

     and

     cuisine

     are

     all

     a

     testament

     to

     the

     city

    's

     unique

     identity

     as

     a

     major

     European

     capital

    .

     Based

     on

     the

     passage

     above

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

     and

     exciting

    ,

     and

     there

     are

     many

     potential

     developments

     that

     could

     shape

     the

     landscape

     of

     the

     technology

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

     Increased

     efficiency

     and

     productivity

    :

     AI

     can

     help

     automate

     repetitive

     tasks

    ,

     reduce

     the

     amount

     of

     time

     and

     effort

     required

     to

     complete

     tasks

    ,

     and

     enable

     faster

     decision

    -making

    .

     This

     could

     lead

     to

     increased

     productivity

     and

     better

     work

    -life

     balance

     for

     individuals

     and

     businesses

    .
    


    2

    .

     Improved

     healthcare

     and

     medicine

    :

     AI

     can

     help

     develop

     new

     treatments

     and

     drugs

    ,

     improve

     diagnostic

     tools

    ,

     and

     improve

     the

     delivery

     of

     care

    .

     This

     could

     lead

     to

     better

     outcomes

     and

     a

     more

     efficient

     healthcare

     system

    .
    


    3

    .

     Aug

    mented

     reality

     and

     virtual

     reality

    :

     AI

     can

     be

    



```python
llm.shutdown()
```
