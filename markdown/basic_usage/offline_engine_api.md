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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.74it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.73it/s]


    2026-05-12 23:04:17,935 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 23:04:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:33,  4.79s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:33,  4.79s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:33,  4.79s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:33,  4.79s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:50,  1.07it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:19,  2.60it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:19,  2.60it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:19,  2.60it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:19,  2.60it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:19,  2.60it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:05<00:19,  2.60it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:05<00:19,  2.60it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:05<00:19,  2.60it/s]Compiling num tokens (num_tokens=2304):  14%|█▍        | 8/58 [00:05<00:19,  2.60it/s]Compiling num tokens (num_tokens=2048):  14%|█▍        | 8/58 [00:05<00:19,  2.60it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:05,  7.26it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:05,  7.26it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:05,  7.26it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:05,  7.26it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:05,  7.26it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:05<00:05,  7.26it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:05<00:05,  7.26it/s]Compiling num tokens (num_tokens=832):  29%|██▉       | 17/58 [00:05<00:05,  7.26it/s]Compiling num tokens (num_tokens=768):  29%|██▉       | 17/58 [00:05<00:05,  7.26it/s]

    Compiling num tokens (num_tokens=704):  29%|██▉       | 17/58 [00:05<00:05,  7.26it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=320):  45%|████▍     | 26/58 [00:05<00:02, 13.03it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 19.94it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 19.94it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 19.94it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 19.94it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 19.94it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 19.94it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 19.94it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 19.94it/s]

    Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:01, 19.94it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 26.67it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 26.67it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 26.67it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 26.67it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 26.67it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 26.67it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 26.67it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:05<00:00, 26.67it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:05<00:00, 26.67it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 33.00it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 33.00it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.00it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.00it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.00it/s]

    Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.00it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.00it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.36 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.32 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.32 GB):   3%|▎         | 2/58 [00:00<00:03, 15.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.31 GB):   3%|▎         | 2/58 [00:00<00:03, 15.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.29 GB):   3%|▎         | 2/58 [00:00<00:03, 15.04it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.29 GB):   7%|▋         | 4/58 [00:00<00:03, 16.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.30 GB):   7%|▋         | 4/58 [00:00<00:03, 16.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.29 GB):   7%|▋         | 4/58 [00:00<00:03, 16.85it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.28 GB):   7%|▋         | 4/58 [00:00<00:03, 16.85it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.28 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.26 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.28 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.53it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.27 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.53it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.27 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.26 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.26 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.72it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.25 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.25 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.25 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.24 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.24 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.65it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.20 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.20 GB):  31%|███       | 18/58 [00:00<00:01, 30.46it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.20 GB):  31%|███       | 18/58 [00:00<00:01, 30.46it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.21 GB):  31%|███       | 18/58 [00:00<00:01, 30.46it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.19 GB):  31%|███       | 18/58 [00:00<00:01, 30.46it/s]Capturing num tokens (num_tokens=960 avail_mem=74.18 GB):  31%|███       | 18/58 [00:00<00:01, 30.46it/s] Capturing num tokens (num_tokens=960 avail_mem=74.18 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=896 avail_mem=74.17 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=832 avail_mem=74.17 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.27it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.16 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=704 avail_mem=74.16 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=640 avail_mem=74.15 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=640 avail_mem=74.15 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.16it/s]Capturing num tokens (num_tokens=576 avail_mem=74.15 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.16it/s]Capturing num tokens (num_tokens=512 avail_mem=74.13 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.16it/s]Capturing num tokens (num_tokens=480 avail_mem=74.16 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.16it/s]Capturing num tokens (num_tokens=448 avail_mem=74.15 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.16it/s]Capturing num tokens (num_tokens=416 avail_mem=74.15 GB):  47%|████▋     | 27/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=416 avail_mem=74.15 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.10it/s]Capturing num tokens (num_tokens=384 avail_mem=74.14 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.10it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.13 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.10it/s]Capturing num tokens (num_tokens=320 avail_mem=74.13 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.10it/s]Capturing num tokens (num_tokens=288 avail_mem=74.12 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.10it/s]Capturing num tokens (num_tokens=256 avail_mem=74.11 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.10it/s]Capturing num tokens (num_tokens=256 avail_mem=74.11 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.44it/s]Capturing num tokens (num_tokens=240 avail_mem=74.11 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.44it/s]Capturing num tokens (num_tokens=224 avail_mem=74.10 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.44it/s]Capturing num tokens (num_tokens=208 avail_mem=74.10 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.44it/s]Capturing num tokens (num_tokens=192 avail_mem=74.10 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.44it/s]Capturing num tokens (num_tokens=176 avail_mem=74.09 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.44it/s]Capturing num tokens (num_tokens=176 avail_mem=74.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=160 avail_mem=74.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.25it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.25it/s]Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.25it/s] Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  81%|████████  | 47/58 [00:01<00:00, 44.24it/s]Capturing num tokens (num_tokens=80 avail_mem=74.08 GB):  81%|████████  | 47/58 [00:01<00:00, 44.24it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  81%|████████  | 47/58 [00:01<00:00, 44.24it/s]Capturing num tokens (num_tokens=48 avail_mem=74.07 GB):  81%|████████  | 47/58 [00:01<00:00, 44.24it/s]Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  81%|████████  | 47/58 [00:01<00:00, 44.24it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  81%|████████  | 47/58 [00:01<00:00, 44.24it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.22it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 45.54it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 36.19it/s]


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
    Generated text:  Mary, I’m a 16 year old with ADHD. I know how to do things quickly and that I’m extremely good at math. I’m very good with science. I’m very good with writing. I’m also very good at coding.
    I’m currently learning how to code, but I think I’m good at math. My dad is a software engineer. My mom is a teacher. I know how to use an ATM in the US. I’m an expert at the kitchen, cooking, and doing laundry.
    My grandma is a nursing assistant. My brother is a doctor. My sister is a nurse. My mom has
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy on Monday, but on Friday, he has a meeting that lasts for 2 hours. If the duration of the president's meeting on Monday is represented by $m$ minutes, and the duration of the meeting on Friday is represented by $n$ minutes, find the value of $m$ if $m = 120$ minutes and $n = 300$ minutes.
    To determine the duration of the president's meeting on Friday, we start by noting that the meeting on Friday lasts for 2 hours. Since there are 60 minutes in an hour, the total duration of the meeting on Friday
    ===============================
    Prompt: The capital of France is
    Generated text:  ( ).
    A. Paris
    B. London
    C. Moscow
    D. New York
    Answer:
    
    A
    
    In the process of establishing a market economy, which of the following is most critical for ensuring market order and preventing market failures?
    A. Market supervision
    B. Market ethics
    C. Price regulation
    D. Price discovery
    Answer:
    
    A
    
    Which of the following statements about the advantages of market economies is incorrect?
    A. A market economy provides a competitive environment for businesses.
    B. The market economy can effectively allocate resources and reduce waste.
    C. The market economy is conducive to promoting social welfare.
    D. The market economy
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain and it is difficult to predict exactly how it will develop. However, one of the most significant advances in AI over the past 50 years has been the ability to create sophisticated models that can learn, understand and solve complex problems. Here are some of the most important AI developments that will shape the future of AI:
    
    1. Deep Learning: Deep learning is the technology that allows AI models to learn from large datasets and extract patterns and insights from them. It has revolutionized many fields, including natural language processing, computer vision, and speech recognition.
    
    2. Neural Networks: Neural networks are a type of machine learning model that simulate the


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


    Generated text:  [Name] and I am a [job title] at [company name]. I am a [job title] at [company name], and I have been with the company for [number of years] years. I have always been passionate about [job title] and I am always looking for ways to [job title] my skills and knowledge. I am a [job title] at [company name], and I have always been passionate about [job title] and I am always looking for ways to [job title] my skills and knowledge. I am a [job title] at [company name], and I have always been passionate
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third largest city in the world by population. It is also the seat of the French government and the country's cultural and political center. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and music, and is home to many world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination and a major economic hub in Europe. It is also a major center for science, technology, and innovation. The city is home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there will be a greater emphasis on developing ethical AI that is designed to minimize harm and maximize benefits.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, including smart homes, self-driving cars, and virtual assistants. As these technologies continue to evolve, we can expect to see even more integration of AI with other technologies, such as healthcare, finance,
    


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
    Generated text:  [Name]. I’m a [Level/Position] [Role]. I love [Job/Position], and I’m excited to join your team. My favorite thing about [Job/Position] is [Reason for Interest]. Looking forward to getting to know you! 🚀💪💼
    
    That's a great self-introduction! Could you please add some details about your role and the industry you're in? That would help the reader understand your experience better. 🚀💪💼
    
    Absolutely! Great! Here's a revised version:
    
    Hello, my name is [Name]. I’m a [Level/Position] [Role].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and the most populous city in the European Union. Paris is known for its beautiful architecture, rich culture, and historical landmarks. It is a major economic center and a global city. The city is also home to the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. As of 2021, Paris had an estimated population of over 2.2 million people. It is often considered one of the most important cities in the world due to its global importance in art, literature, music, and fashion. Additionally, the French government has taken some significant measures
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising and changing rapidly. Here are some possible trends that could shape the field in the coming years:
    
    1. AI will become more personalized: As AI systems become more advanced, they will be able to learn from individual users and provide more tailored and personalized experiences.
    
    2. AI will become more ethical: As AI systems are used for more complex tasks, such as decision-making in healthcare or finance, ethical considerations will become more important.
    
    3. AI will become more integrated with other technologies: As AI becomes more prevalent, it will be integrated with other technologies, such as machine learning, to create even more advanced and versatile systems.
    
    4.


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

     Sarah

    ,

     and

     I

     am

     an

     ambitious

    ,

     fearless

    ,

     and

     always

     ready

     to

     take

     on

     challenges

    .

     I

     come

     from

     a

     family

     of

     artists

    ,

     and

     I

     have

     always

     been

     fascinated

     by

     the

     world

     of

     creativity

    .

     I

     love

     to

     paint

     and

     photography

    ,

     and

     I

     have

     a

     passion

     for

     exploring

     the

     world

     and

     taking

     on

     new

     things

    .

     I

     am

     always

     looking

     for

     new

     opportunities

     to

     expand

     my

     hor

    izons

     and

     meet

     people

     who

     share

     my

     love

     of

     creativity

    .
    


    Sarah

     enjoys

     pursuing

     new

     hobbies

     and

     interests

     as

     well

    ,

     whether

     it

    's

     learning

     to

     cook

    ,

     play

     a

     musical

     instrument

    ,

     or

     pursue

     a

     new

     hobby

     such

     as

     writing

     or

     writing

     at

     work

    .

     I

     am

     a

     strong

     believer

     in

     the

     power

     of

     creativity

     and

    
    
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

    ."


    France

    's

     capital

     city

    ,

     Paris

    ,

     is

     known

     for

     its

     romantic

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

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     famous

     for

     its

     lively

     nightlife

    ,

     annual

     festivals

    ,

     and

     innovative

     cultural

     scene

    .

     Additionally

    ,

     Paris

     is

     home

     to

     several

     famous

     museums

     and

     art

     galleries

    ,

     including

     the

     Mus

    ée

     de

     l

    '

    Or

    anger

    ie

     and

     the

     Muse

    e

     d

    '

    Or

    say

    .

     Overall

    ,

     Paris

     is

     a

     fascinating

     city

     with

     a

     rich

     history

     and

     culture

     that

     continues

     to

     capt

    ivate

     visitors

     from

     around

     the

     world

    .

     Paris

     is

     the

     second

     most

     visited

     city

     in

     the

     world

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

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

     industry

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

    :
    


    1

    .

     Increased

     integration

     of

     AI

     with

     human

     intelligence

    :

     As

     AI

     continues

     to

     become

     more

     advanced

    ,

     it

     may

     become

     increasingly

     integrated

     with

     human

     intelligence

     in

     new

     ways

    .

     This

     could

     lead

     to

     more

     natural

     and

     intuitive

     interactions

     with

     machines

    ,

     as

     well

     as

     more

     human

    -like

     intelligence

     that

     is

     more

     personalized

     and

     adaptable

    .
    


    2

    .

     Advances

     in

     natural

     language

     processing

    :

     Natural

     language

     processing

     (

    N

    LP

    )

     is

     a

     key

     component

     of

     AI

    ,

     and

     it

     will

     likely

     continue

     to

     advance

     in

     the

     coming

     years

    .

     This

     will

     likely

     lead

     to

     more

     advanced

     models

     that

     can

     understand

     human

     language

     and

     generate

    



```python
llm.shutdown()
```
