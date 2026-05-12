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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.08it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.07it/s]


    2026-05-12 05:35:32,056 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 05:35:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  4.04it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.48it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.48it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.48it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.48it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.48it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.48it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.48it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:02, 12.37it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:02, 12.37it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:02, 12.37it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:02, 12.37it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:02, 12.37it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:02, 12.37it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:04<00:02, 12.37it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:04<00:02, 12.37it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:04<00:02, 12.37it/s]Compiling num tokens (num_tokens=352):  43%|████▎     | 25/58 [00:04<00:02, 12.37it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:04<00:01, 19.92it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:04<00:01, 19.92it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:04<00:01, 19.92it/s]

    Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:04<00:01, 19.92it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:04<00:01, 19.92it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:04<00:01, 19.92it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 24.02it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 34.07it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 34.07it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 34.07it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 34.07it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 34.07it/s]

    Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 34.07it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 34.07it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 34.07it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.16it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.09it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.09it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.09it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 19.78it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 19.78it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 19.78it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 19.78it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.65it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.65it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.65it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 27.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 27.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 27.59it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 27.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 27.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 27.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.91it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.91it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.91it/s] Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.63it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.63it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.63it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.63it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.63it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.63it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.75it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.75it/s]

    Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.75it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.75it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.75it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.75it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.11it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.11it/s]

    Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.18it/s]Capturing num tokens (num_tokens=240 avail_mem=76.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.18it/s]Capturing num tokens (num_tokens=224 avail_mem=76.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.18it/s]Capturing num tokens (num_tokens=208 avail_mem=76.64 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.18it/s]Capturing num tokens (num_tokens=192 avail_mem=76.64 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.18it/s]Capturing num tokens (num_tokens=192 avail_mem=76.64 GB):  71%|███████   | 41/58 [00:01<00:00, 35.35it/s]Capturing num tokens (num_tokens=176 avail_mem=76.64 GB):  71%|███████   | 41/58 [00:01<00:00, 35.35it/s]Capturing num tokens (num_tokens=160 avail_mem=76.62 GB):  71%|███████   | 41/58 [00:01<00:00, 35.35it/s]Capturing num tokens (num_tokens=144 avail_mem=76.14 GB):  71%|███████   | 41/58 [00:01<00:00, 35.35it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 35.35it/s]

    Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 35.35it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.98it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.98it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.98it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.98it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.98it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 36.98it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.80it/s]

    Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.29it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.29it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 35.63it/s]


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
    Generated text:  Mark. I am a software engineer and designer based in Chicago, IL. I have been working in software development for over 10 years. I have a master's in Electrical Engineering, and a B.S. in Computer Science. I have experience with design and development across the software development stack, including Java, C++, Python, and JavaScript. I am a self-taught software developer with a passion for learning and problem solving. I enjoy working with a variety of software development tools and technologies, as well as collaborating with other developers and teams to solve problems and achieve our goals. I have a strong sense of humor and enjoy spending
    ===============================
    Prompt: The president of the United States is
    Generated text:  an elected position, and it is not necessarily related to the current president. If $x$, $y$, and $z$ represent the positions of the president, vice president, and assistant vice president, respectively, then what is the value of $x+y+z$ if the president is currently Senatorial? To determine the value of \(x + y + z\) when the president is currently Senatorial, we need to know the current positions of the vice president and the assistant vice president. However, the problem does not provide specific values for these positions. Instead, it is implied that these positions are arbitrary and not necessarily related to the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Rome
    D. Madrid
    
    The correct answer is A. Paris. Paris, the capital of France, is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The other options are not capitals of France: London is the capital of England, and Rome is the capital of Italy. Madrid is the capital of Spain.
    ===============================
    Prompt: The future of AI is
    Generated text:  shaping the future of healthcare
    
    The world’s most advanced AI technologies are being developed and deployed in healthcare, to improve patient outcomes and to increase accessibility.
    
    AI has become an increasingly important tool in the healthcare industry, in terms of both predicting patient conditions and in helping doctors diagnose and treat patients more accurately. In fact, the use of AI in healthcare has increased by 75% over the past five years, and this trend is expected to continue.
    
    AI has the potential to transform the healthcare industry in several ways, including:
    
      1. More accurate and reliable diagnostics: AI algorithms are being used to analyze medical images, such as X


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive description of your personality or skills]. I enjoy [insert a short, positive description of your hobbies or interests]. I'm always looking for new opportunities to grow and learn, and I'm always eager to share my knowledge and experience with others. What's your favorite hobby or activity? I love [insert a short, positive description of your favorite hobby or activity]. I also enjoy [insert a short, positive description
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub and a major tourist destination. The city is known for its cuisine, fashion, and art, and is a popular tourist destination for its beautiful architecture and vibrant nightlife. Paris is a city of contrasts, with its modern and traditional architecture, and its diverse population. It is a city of innovation
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased automation: AI will continue to automate routine tasks, freeing up human workers to focus on more complex and creative work. This could lead to a shift in the job market, with many jobs being replaced by AI.
    
    2. Improved natural language processing: AI will continue to improve its ability to understand and respond to human language, leading to more sophisticated and nuanced interactions with humans.
    
    3. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be increased concerns about privacy and security. AI will need to be designed with privacy and security
    


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
    Generated text:  ____ and I'm an exceptional freelancer, _____.
    "Hi there, my name is [Name] and I'm an exceptional freelancer. With over 10 years of experience in the [industry], I specialize in [the specific type of work or services I provide], and I'm always here to help anyone need a hand. Whether you need a project completed, have questions for clarification, or just want to share your ideas, I'm your go-to expert." (Tell the reader of the character to write their name at the top of the introduction in all caps and include an industry section if applicable.) The introduction should be tailored to their
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    The statement is: Paris is the capital city of France. 
    
    Here's a simpler, more concise version:
    Paris is the largest city in France.
    This statement captures the essence of Paris being the capital city, including its geographical and cultural importance in France. 
    
    For a more detailed version, here's another option:
    
    The city of Paris, located on the left bank of the Seine river, is the largest and most populous city in France, with a population of over 2. 2 million people (as of 2023). It is the cultural and political center of France and has a rich history dating back
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting, and it will continue to evolve rapidly as technology advances. Here are some possible future trends in AI:
    
    1. Improved machine learning algorithms: As machine learning algorithms become more sophisticated, they will be able to learn from more data, improve their performance, and adapt to new situations.
    
    2. Increased transparency and explainability: AI systems will become more transparent and explainable, allowing users to understand how the system arrived at its decisions.
    
    3. Enhanced safety and privacy: AI systems will become more secure and ethical, with better safety measures in place to protect users' personal data and privacy.
    
    4. Increased human-computer interaction: AI systems


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

    name

    ].

     I

    'm

     a

     computer

     scientist

     with

     a

     passion

     for

     building

     and

     improving

     the

     world

     through

     technology

    .

     I

     have

     been

     working

     with

     programming

     languages

    ,

     algorithms

    ,

     and

     systems

     for

     over

     

    1

    0

     years

     now

    ,

     and

     I

     have

     a

     deep

     understanding

     of

     how

     to

     design

     and

     implement

     software

     that

     is

     efficient

     and

     scalable

    .

     I

    'm

     also

     proficient

     in

     Linux

    ,

     Node

    .js

    ,

     and

     Python

    .

     My

     goal

     is

     to

     help

     people

     use

     technology

     more

     effectively

     and

     make

     the

     world

     a

     better

     place

    .

     How

     can

     I

     help

     you

     today

    ?

     [

    Your

     name

    ]

     [

    age

    ]

     [

    occupation

    ]

     [

    h

    obbies

    ]

     [

    education

    ]

     [

    languages

    ]

     [

    interest

    s

    ]

     [work

     experience

    ]

     [

    potential

     areas

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     and

     its

     largest

     economy

    .


    Paris

    ,

     located

     on

     the

     Se

    ine

     River

    ,

     is

     the

     capital

     of

     France

     and

     the

     largest

     city

     in

     the

     country

    .

     It

     is

     the

     seat

     of

     the

     French

     government

    ,

     the

     head

     of

     state

    ,

     and

     the

     country

    's

     most

     populous

     city

    ,

     with

     an

     estimated

     population

     of

     

    2

    .

     

    6

     million

     as

     of

     

    2

    0

    2

    0

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     famous

     landmarks

    ,

     art

    ,

     fashion

    ,

     cuisine

    ,

     and

     its

     status

     as

     a

     cultural

     and

     financial

     center

     in

     the

     world

    .

     It

     also

     has

     a

     strong

     role

     in

     its

     European

     Union

     membership

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     home

     to

     many

     world

    -ren

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

     continued

     growth

    ,

     but

     it

     will

     also

     face

     significant

     challenges

     and

     challenges

    .

     Here

     are

     some

     possible

     trends

     to

     look

     out

     for

    :
    


    1

    .

     More

     autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     likely

     to

     become

     more

     common

     and

     widespread

     in

     the

     years

     ahead

     as

     they

     become

     safer

     and

     more

     efficient

    .

     AI

     will

     continue

     to

     improve

     as

     new

     technologies

     like

     machine

     learning

    ,

     natural

     language

     processing

    ,

     and

     computer

     vision

     come

     into

     use

    .
    


    2

    .

     Increased

     precision

     in

     AI

    :

     AI

     will

     continue

     to

     improve

     as

     new

     algorithms

     and

     models

     are

     developed

     to

     increase

     the

     accuracy

     and

     precision

     of

     predictions

     and

     decisions

    .
    


    3

    .

     Increased

     AI

     for

     healthcare

    :

     AI

     will

     be

     used

     for

     healthcare

     to

     improve

     diagnosis

     and

     treatment

     outcomes

    ,

     reduce

    



```python
llm.shutdown()
```
