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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.63it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.62it/s]


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    2026-05-08 21:46:36,818 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 21:46:36] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:46,  3.97s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:46,  3.97s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:46,  3.97s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:46,  3.97s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:46,  3.97s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03, 10.13it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03, 10.13it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03, 10.13it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03, 10.13it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03, 10.13it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03, 10.13it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03, 10.13it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03, 10.13it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03, 10.13it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03, 10.13it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.57it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.57it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.57it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.57it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.57it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.57it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.57it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.57it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.57it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.57it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 16.57it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 25.12it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 25.12it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 34.35it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 34.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.67it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.67it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.53 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.67it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.67it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  21%|██        | 12/58 [00:00<00:01, 27.61it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:01, 27.61it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:01, 27.61it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:01, 27.61it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:01, 27.61it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  21%|██        | 12/58 [00:00<00:01, 27.61it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.27it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.28 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.27it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.27it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.27it/s] Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.88it/s]Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.88it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.88it/s]Capturing num tokens (num_tokens=768 avail_mem=72.27 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.88it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.88it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.88it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.88it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.79it/s]Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.79it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.79it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.79it/s]Capturing num tokens (num_tokens=416 avail_mem=72.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.79it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.79it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.79it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.38it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.38it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.38it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.38it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.38it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.38it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  59%|█████▊    | 34/58 [00:01<00:00, 45.38it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.02it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.63it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.86it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.86it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.86it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.86it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.86it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.86it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.26it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.26it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.26it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.26it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 41.65it/s]


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
    Generated text:  Stephen Greenfield and I am a student of Software Engineering at the University of Birmingham. I am a fresh graduate from the University of the Witwatersrand in South Africa and I have been working in the field of Computer Science since my first year of university, where I was part of a team of five students who were designing a web application. I then moved to the University of Birmingham and completed my studies, before joining the University of Birmingham's Software Engineering team as a Software Developer. I have always had an interest in the field of Software Engineering and am eager to get into the field of Computer Science. My current interest is in learning about the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. The president of the United States is a man or a woman? - not enough information - a woman - a man - woman
    Answer the above question by writing a simple rationale for why the answer is correct, and by writing a simple rationale for why the answer is incorrect. Additionally, provide a reasoning for why there may be multiple correct answers.
    Answer: The rationale for the correct answer is as follows:
    - The President of the United States is a male official who serves as the head of state of the United States.
    - The "president" is a title given to the head of state.
    - The "United States
    ===============================
    Prompt: The capital of France is
    Generated text:  a place known as the seat of the government of France. The capital of France is the capital of France is called Paris. It is located on the right bank of the Seine river. The capital is on a place called the Île de France. It is the largest of the Paris Metro stations. Paris is a beautiful city with many tall buildings, beautiful gardens, and lovely parks. People from all over the world visit Paris for their vacations, the big parties, and the Paris Fashion Week. People visit Paris for its delicious cuisine and its wine. Paris is home to some of the most famous museums in the world including the Louvre
    ===============================
    Prompt: The future of AI is
    Generated text:  undoubtedly going to be in big data. With the increase in big data, companies are accessing a plethora of data and this data is being processed to a very high level of complexity. With this level of complexity, there is a tendency of massive errors.
    The need to process the data in such a complex way is not just limited to the processing of data. The whole process is governed by the algorithms used. The algorithms are used in order to provide the answers which the person seeking these answers is looking for.
    Here are some ways in which big data can be used to prevent and correct errors:
    1. Data cleaning: This is a very important


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic statement about your personality or skills]. I enjoy [insert a short, positive, enthusiastic statement about your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short, positive, enthusiastic statement about your favorite hobby or activity]. I'm always looking for new ways to challenge myself and expand my knowledge. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its cuisine, fashion, and art scene. Paris is a major tourist destination and a cultural hub, with many world-renowned museums, theaters, and art galleries. It is also home to many important political and economic institutions, including the French Parliament and the French National Library. The city is known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to become more integrated into our daily lives, from home automation to transportation. We may see more automation in manufacturing, healthcare, and other industries, as AI is increasingly used to perform tasks that were previously done by humans.
    
    2. Improved privacy and security: As AI becomes more advanced, it is likely to be used for more sensitive tasks, such as surveillance and data
    


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
    Generated text:  [Name], and I am a [Noun]. I am [Short, descriptive name]. I am an [occupation or profession]. I am passionate about [career goal or hobby]. I am always looking for [a challenge or opportunity]. I believe in [core value or principle]. What are your hobbies or interests?
    
    Always keep the tone of your response neutral and welcoming. Focus on your personality, experiences, and accomplishments, rather than your profession or background. Use [appropriate language and phrasing] to connect with your audience. Use [appropriate context and examples]. Good luck! Don't forget to include your contact information. [End with
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris, often abbreviated as "Paris," is the capital of France and the largest city in the country. It serves as the political, cultural, and economic center of France and is also home to the United Nations Headquarters. Paris is known for its historic landmarks, diverse neighborhoods, and annual celebrations, such as the Eiffel Tower and the annual Fête de la Feuille. Paris is an important global city and a popular tourist destination, hosting various cultural, artistic, and sporting events annually. The city is governed by the President of the Republic of France and is home to a diverse population of people from across the world.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are some of the most likely trends:
    
    1. Increased human oversight: As AI becomes more integrated into our daily lives, there will be more emphasis on human oversight and accountability. This could lead to stricter regulations and greater transparency in how AI is developed and used.
    
    2. AI that can learn and adapt: As AI becomes more integrated into our lives, we may see more AI that can learn from experience and adapt to new situations. This could lead to more efficient and effective use of resources and innovation.
    
    3. AI that can communicate: AI is already capable of communicating with humans, but more AI could develop the ability


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

     John

    .

     What

     brings

     you

     to

     this

     world

    ?


    I

    'm

     an

     avid

     reader

    ,

     writer

    ,

     and

     podcast

     listener

    .

     I

     love

     to

     explore

     new

     topics

     and

     ideas

    ,

     and

     I

     love

     to

     share

     what

     I

     learn

     with

     others

    .

     I

     also

     enjoy

     spending

     time

     in

     nature

     and

     exploring

     new

     places

    .

     What

     brings

     you

     to

     this

     world

    ?

     Write

     a

     short

    ,

     neutral

     self

    -int

    roduction

     for

     a

     fictional

     character

    .

     Hello

    ,

     my

     name

     is

     Jane

    .

     What

     brings

     you

     to

     this

     world

    ?


    I

    'm

     an

     artist

    ,

     writer

    ,

     and

     graphic

     designer

    .

     I

    'm

     constantly

     looking

     for

     new

     ideas

     and

     creative

     inspiration

    .

     I

     love

     to

     experiment

     with

     new

     techniques

     and

     mediums

    ,

     and

     I

     enjoy

     creating

     art

     that

    's

     both

     beautiful

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     historic

     city

     with

     many

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     known

     for

     its

     rich

     cultural

     heritage

     and

     its

     role

     as

     a

     center

     of

     France

    's

     government

     and

     politics

    .

     It

     is

     also

     famous

     for

     its

     fashion

     industry

     and

     its

     food

     scene

    ,

     particularly

     its

     famous

     cro

    iss

    ants

    .

     Paris

     is

     the

     most

     populous

     city

     in

     France

     and

     has

     a

     population

     of

     over

     

    2

    .

    5

     million

     people

    .

     It

     is

     the

     capital

     of

     France

     and

     one

     of

     the

     largest

     and

     most

     important

     cities

     in

     Europe

    .

     The

     city

     is

     home

     to

     many

     world

    -ren

    owned

     museums

    ,

     such

     as

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

     Mus

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     rapid

     advancements

     in

     areas

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     computer

     vision

    ,

     and

     robotics

    .

     These

     technologies

     are

     expected

     to

     become

     increasingly

     integrated

     into

     our

     daily

     lives

    ,

     enabling

     new

     applications

     and

     services

     that

     were

     previously

     impossible

    .
    


    One

     potential

     future

     trend

     is

     the

     widespread

     adoption

     of

     AI

     in

     healthcare

    .

     As

     AI

     improves

     its

     ability

     to

     analyze

     and

     interpret

     complex

     medical

     data

    ,

     it

     may

     become

     more

     accurate

     and

     reliable

     in

     diagn

    osing

     and

     treating

     diseases

    .

     AI

    -powered

     medical

     devices

    ,

     such

     as

     wearable

     sensors

     and

     smart

     implants

    ,

     may

     also

     be

     developed

     to

     monitor

     patients

    '

     health

     more

     closely

     and

     provide

     real

    -time

     feedback

     to

     doctors

    .
    


    AI

     may

     also

     have

     a

     significant

     impact

     on

     the

    



```python
llm.shutdown()
```
