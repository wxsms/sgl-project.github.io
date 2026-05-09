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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.22it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.21it/s]


    2026-05-09 09:56:32,208 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 09:56:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:04<00:07,  6.25it/s] 

    Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 11.94it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 11.94it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 11.94it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 11.94it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 11.94it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 11.94it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 11.94it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 11.94it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 17.71it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 17.71it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 17.71it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 17.71it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 17.71it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 17.71it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 17.71it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 17.71it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 17.71it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 25.22it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 25.22it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 25.22it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 25.22it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 25.22it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 25.22it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 25.22it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 25.22it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 25.22it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 33.31it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 33.31it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 33.31it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 33.31it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 33.31it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 33.31it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 33.31it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 33.31it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 33.31it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 33.31it/s]

    Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 42.89it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 42.89it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 42.89it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 42.89it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 42.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.24it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.14 GB):   3%|▎         | 2/58 [00:00<00:04, 13.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:04, 13.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:04, 13.37it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:04, 13.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   9%|▊         | 5/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.12 GB):   9%|▊         | 5/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.11 GB):  16%|█▌        | 9/58 [00:00<00:02, 24.43it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.11 GB):  16%|█▌        | 9/58 [00:00<00:02, 24.43it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.10 GB):  16%|█▌        | 9/58 [00:00<00:02, 24.43it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.10 GB):  16%|█▌        | 9/58 [00:00<00:02, 24.43it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.09 GB):  16%|█▌        | 9/58 [00:00<00:02, 24.43it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.09 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.09 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.09 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.09 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.08 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.40it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=72.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.40it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.40it/s]Capturing num tokens (num_tokens=960 avail_mem=72.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.40it/s] Capturing num tokens (num_tokens=960 avail_mem=72.07 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.25it/s]Capturing num tokens (num_tokens=896 avail_mem=72.07 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.25it/s]Capturing num tokens (num_tokens=832 avail_mem=72.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.25it/s]Capturing num tokens (num_tokens=768 avail_mem=72.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.25it/s]Capturing num tokens (num_tokens=704 avail_mem=72.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.25it/s]Capturing num tokens (num_tokens=640 avail_mem=72.05 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.25it/s]Capturing num tokens (num_tokens=640 avail_mem=72.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=576 avail_mem=72.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=512 avail_mem=72.04 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.75it/s]

    Capturing num tokens (num_tokens=480 avail_mem=72.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=448 avail_mem=72.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=416 avail_mem=72.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=416 avail_mem=72.05 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.23it/s]Capturing num tokens (num_tokens=384 avail_mem=72.05 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.23it/s]Capturing num tokens (num_tokens=352 avail_mem=72.04 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.23it/s]Capturing num tokens (num_tokens=320 avail_mem=72.03 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.23it/s]Capturing num tokens (num_tokens=288 avail_mem=72.03 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.23it/s]Capturing num tokens (num_tokens=256 avail_mem=72.03 GB):  55%|█████▌    | 32/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=256 avail_mem=72.03 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=240 avail_mem=72.03 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=224 avail_mem=72.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.72it/s]

    Capturing num tokens (num_tokens=208 avail_mem=72.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=192 avail_mem=72.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=176 avail_mem=72.01 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=176 avail_mem=72.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.37it/s]Capturing num tokens (num_tokens=160 avail_mem=72.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.37it/s]Capturing num tokens (num_tokens=144 avail_mem=72.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.37it/s]Capturing num tokens (num_tokens=128 avail_mem=72.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.37it/s]Capturing num tokens (num_tokens=112 avail_mem=72.00 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.37it/s]Capturing num tokens (num_tokens=96 avail_mem=72.00 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.37it/s] Capturing num tokens (num_tokens=96 avail_mem=72.00 GB):  81%|████████  | 47/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=80 avail_mem=72.00 GB):  81%|████████  | 47/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=64 avail_mem=71.99 GB):  81%|████████  | 47/58 [00:01<00:00, 44.93it/s]

    Capturing num tokens (num_tokens=48 avail_mem=71.99 GB):  81%|████████  | 47/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=32 avail_mem=71.99 GB):  81%|████████  | 47/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=28 avail_mem=71.98 GB):  81%|████████  | 47/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=28 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=24 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=20 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=16 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=12 avail_mem=71.97 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=8 avail_mem=71.97 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.25it/s] Capturing num tokens (num_tokens=8 avail_mem=71.97 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=4 avail_mem=71.96 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.61it/s]Capturing num tokens (num_tokens=4 avail_mem=71.96 GB): 100%|██████████| 58/58 [00:01<00:00, 38.78it/s]


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
    Generated text:  Matias and I'm from Buenos Aires, Argentina. I'm a human, but I like to turn my life into something else by writing.
    
    I don't like to go out for a long time, I'm very introverted. My favorite colors are red and yellow and I like to live in my apartment, the idea of having a sofa and some plants.
    
    I have a little brother named Armando. He's 3 years old. He's cute and brave and I like to play with him.
    
    I try to be a good person. I want to write books. I like to write about friends, family, love, relationships
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide what issue to present at the upcoming convention. He has a choice between two candidates, each of whom has a different perspective on the issue. Candidate A believes that the issue will be dealt with soon, and will likely lead to economic growth, while Candidate B believes that the issue will be dealt with late and will likely have negative economic consequences. The president decides to present the issue to a panel of voters who will be randomly selected from the entire country. What voting system should the president use to choose the panel? The president should use a random selection system to choose the panel of voters. Random selection ensures that every voter has an
    ===============================
    Prompt: The capital of France is
    Generated text:  located on the left bank of the river Seine in the south of the country. The Seine is the longest river in Europe, the second longest in the world, and has more than 1. 700 km of its course.
    The Seine is a tributary of the Rhine. The rivers the two are called the Oise and the Meuse. The Seine is the only river which passes through both the departments of the same name, it is the only river in France which passes through three different departments. Its length is over 400 km.
    The Seine has a long history. It
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, with the added power of IoT and big data all around us. AI can be used for the betterment of our communities by improving healthcare, manufacturing, and even education. So what are the hurdles to overcome? Here’s an overview of the top concerns and answers to those concerns.
    Why is the Internet of Things so important?
    The Internet of Things or IoT is a growing infrastructure that allows devices to exchange information. It can be used to collect data from devices on a network and store it in a central location. Once the data is collected, it can be used to improve the performance of devices. IoT can also be used for the


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your profession or role]. I enjoy [insert a short description of your hobbies or interests]. What's your favorite [insert a short description of your hobby or activity]? I'm always looking for new experiences and learning new things. What's your favorite [insert a short description of your hobby or activity]? I'm always eager to learn and grow as a person. What's your favorite [insert a short description of your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art, and cuisine. Paris is a vibrant and diverse city with a population of over 2.5 million people. It is a major tourist destination and a popular destination for business and leisure activities. The city is home to many famous landmarks and attractions, including the Louvre, the Eiffel Tower,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy.
    
    2. Integration with other technologies: AI is likely to become more integrated with other technologies, such as blockchain, quantum computing, and biotechnology. This will enable new applications and opportunities for AI to be developed.
    
    3. Development of more advanced models: As AI technology continues to advance,
    


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
    Generated text:  [Name], and I am a [Your profession or title] from [Location]. I am passionate about [Your passion or hobby], which has taken me on an incredible journey of discovery. I am always learning and always eager to learn more, whether through books, online courses, or just by asking questions. I am always open to new experiences, new ideas, and new perspectives. I am a [Your professional goal or motivation], and I am always working to achieve this by [Your approach to achieving your goal]. I am a [Your unique identifier or tagline], and I believe in [Your core belief or value]. If you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. Its history dates back to ancient times and has been a significant city in Europe for centuries. The city is known for its rich culture, art, and cuisine, and it is one of the most popular tourist destinations in the world. Paris has many famous landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. The city is also home to many influential organizations and institutions, including the French Academy of Sciences and the French Library. Paris is the capital of France and has a unique blend of old-world charm and modernity. It
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, and it is difficult to predict with certainty what will happen. However, there are several possible trends that could potentially shape the development of AI in the coming years:
    
    1. Increased collaboration between humans and machines: As AI becomes more advanced and complex, it is likely to require more humans to interact with it. This could lead to increased collaboration between humans and machines, with humans taking on tasks that require decision-making and problem-solving.
    
    2. AI-driven automation: As AI becomes more capable and widespread, it is possible that it could automate many of the tasks that humans currently perform. This could lead to a significant reduction in the need for


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

    age

    ]

     year

     old

     [

    gender

    ]

     [

    occupation

    ].

     I

    'm

     originally

     from

     [

    place

    ]

     and

     have

     lived

     here

     for

     [

    number

     of

     years

    ]

     years

    .

     I

     have

     a

     wide

    -ranging

     experience

     with

     [

    industry

     or

     field

    ]

     and

     am

     passionate

     about

     [

    current

     or

     related

     topic

     or

     project

    ].

     I

     enjoy

     [

    activities

     or

     hobbies

    ]

     and

     am

     always

     looking

     for

     opportunities

     to

     [

    achievement

     or

     growth

    ].

     I

     am

     always

     ready

     to

     learn

     and

     am

     eager

     to

     expand

     my

     knowledge

    .

     I

    'm

     [

    any

     other

     qualities

    ]

     and

     am

     always

     eager

     to

     learn

    .

     I

     believe

     in

     [

    f

    ounding

     principle

     or

     belief

    ].

     I

     am

     excited

     to

     [

    future

     aspirations

     or

     goals

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

     It

     is

     located

     on

     the

     left

     bank

     of

     the

     Se

    ine

     river

    ,

     the

     river

     that

     runs

     through

     the

     city

    .

     The

     city

     is

     known

     for

     its

     beautiful

     architecture

    ,

     rich

     history

    ,

     and

     vibrant

     culture

    .

     Paris

     is

     home

     to

     many

     famous

     landmarks

     and

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

     It

     is

     also

     an

     important

     center

     for

     politics

    ,

     arts

    ,

     and

     culture

     in

     the

     world

    .

     Paris

     is

     a

     city

     of

     contrasts

    ,

     with

     elegant

     pal

    aces

     and

     high

    -r

    ises

     on

     the

     right

     bank

     of

     the

     Se

    ine

    ,

     and

     charming

     neighborhoods

     on

     the

     left

     bank

    .

     Despite

     its

     fame

    ,

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     exciting

     and

     potentially

     transformative

    .

     Here

     are

     some

     potential

     trends

     that

     are

     currently

     being

     considered

     by

     experts

     and

     businesses

     alike

    :
    
    1

    .

     Increased

     AI

     accuracy

     and

     efficiency

    :

     As

     AI

     continues

     to

     improve

    ,

     we

     can

     expect

     to

     see

     more

     accurate

     and

     efficient

     applications

     of

     AI

    .

     This

     could

     include

     applications

     in

     healthcare

    ,

     finance

    ,

     manufacturing

    ,

     and

     more

    .
    


    2

    .

     AI

     integration

     with

     traditional

     industries

    :

     AI

     is

     already

     being

     used

     in

     many

     industries

    ,

     such

     as

     retail

    ,

     transportation

    ,

     and

     manufacturing

    .

     As

     AI

     technology

     continues

     to

     improve

    ,

     we

     can

     expect

     to

     see

     even

     more

     integration

     of

     AI

     with

     traditional

     industries

    .
    


    3

    .

     AI

    -driven

     automation

    :

     As

     AI

     technology

     improves

    ,

     we

     can

     expect

     to

     see

    



```python
llm.shutdown()
```
