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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.16it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.15it/s]


    2026-05-07 14:31:49,714 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 14:31:49] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.50it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.06it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.15it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.23it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.23it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.23it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.23it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.23it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.23it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.23it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.23it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.23it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.23it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 23.23it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 32.27it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 32.27it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 32.27it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 32.27it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 32.27it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 32.27it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 32.27it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.27it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.82 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.82 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.74 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.74 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.73 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.72 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.72 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.48it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.72 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.72 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.71 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.71 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.48it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=73.71 GB):  21%|██        | 12/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.71 GB):  21%|██        | 12/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.71 GB):  21%|██        | 12/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.70 GB):  21%|██        | 12/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.70 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.70 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.70 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.69 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.69 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.69 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.71it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=73.69 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.67 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.97it/s]Capturing num tokens (num_tokens=960 avail_mem=73.68 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.97it/s] Capturing num tokens (num_tokens=896 avail_mem=73.68 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.97it/s]Capturing num tokens (num_tokens=832 avail_mem=73.68 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.97it/s]Capturing num tokens (num_tokens=832 avail_mem=73.68 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.96it/s]Capturing num tokens (num_tokens=768 avail_mem=73.67 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.96it/s]Capturing num tokens (num_tokens=704 avail_mem=73.67 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.96it/s]Capturing num tokens (num_tokens=640 avail_mem=73.67 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.96it/s]Capturing num tokens (num_tokens=576 avail_mem=73.67 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.96it/s]

    Capturing num tokens (num_tokens=576 avail_mem=73.67 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.13it/s]Capturing num tokens (num_tokens=512 avail_mem=73.65 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.13it/s]Capturing num tokens (num_tokens=480 avail_mem=73.67 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.13it/s]Capturing num tokens (num_tokens=448 avail_mem=73.66 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.13it/s]Capturing num tokens (num_tokens=416 avail_mem=73.66 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.13it/s]Capturing num tokens (num_tokens=384 avail_mem=73.66 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.13it/s]Capturing num tokens (num_tokens=384 avail_mem=73.66 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=352 avail_mem=73.65 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=320 avail_mem=73.65 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=288 avail_mem=73.65 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=256 avail_mem=73.65 GB):  57%|█████▋    | 33/58 [00:01<00:00, 37.41it/s]

    Capturing num tokens (num_tokens=256 avail_mem=73.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=240 avail_mem=73.64 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=224 avail_mem=73.64 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=208 avail_mem=73.63 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=192 avail_mem=73.63 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=176 avail_mem=73.63 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=176 avail_mem=73.63 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.07it/s]Capturing num tokens (num_tokens=160 avail_mem=73.63 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.07it/s]Capturing num tokens (num_tokens=144 avail_mem=73.62 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.07it/s]

    Capturing num tokens (num_tokens=128 avail_mem=73.62 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.07it/s]Capturing num tokens (num_tokens=112 avail_mem=73.62 GB):  72%|███████▏  | 42/58 [00:01<00:00, 36.07it/s]Capturing num tokens (num_tokens=112 avail_mem=73.62 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.14it/s]Capturing num tokens (num_tokens=96 avail_mem=73.62 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.14it/s] Capturing num tokens (num_tokens=80 avail_mem=73.61 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.14it/s]Capturing num tokens (num_tokens=64 avail_mem=73.61 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.14it/s]

    Capturing num tokens (num_tokens=48 avail_mem=73.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 28.14it/s]Capturing num tokens (num_tokens=48 avail_mem=73.60 GB):  86%|████████▌ | 50/58 [00:01<00:00, 27.24it/s]Capturing num tokens (num_tokens=32 avail_mem=73.60 GB):  86%|████████▌ | 50/58 [00:01<00:00, 27.24it/s]Capturing num tokens (num_tokens=28 avail_mem=73.60 GB):  86%|████████▌ | 50/58 [00:01<00:00, 27.24it/s]Capturing num tokens (num_tokens=24 avail_mem=73.59 GB):  86%|████████▌ | 50/58 [00:01<00:00, 27.24it/s]Capturing num tokens (num_tokens=20 avail_mem=73.59 GB):  86%|████████▌ | 50/58 [00:01<00:00, 27.24it/s]Capturing num tokens (num_tokens=20 avail_mem=73.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 28.56it/s]Capturing num tokens (num_tokens=16 avail_mem=73.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 28.56it/s]Capturing num tokens (num_tokens=12 avail_mem=73.58 GB):  93%|█████████▎| 54/58 [00:01<00:00, 28.56it/s]

    Capturing num tokens (num_tokens=8 avail_mem=73.58 GB):  93%|█████████▎| 54/58 [00:01<00:00, 28.56it/s] Capturing num tokens (num_tokens=4 avail_mem=73.57 GB):  93%|█████████▎| 54/58 [00:01<00:00, 28.56it/s]Capturing num tokens (num_tokens=4 avail_mem=73.57 GB): 100%|██████████| 58/58 [00:01<00:00, 31.05it/s]Capturing num tokens (num_tokens=4 avail_mem=73.57 GB): 100%|██████████| 58/58 [00:01<00:00, 30.73it/s]


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
    Generated text:  Xiao Hong. I have two siblings. My older sister is named Xiao Hong. My younger sister is named Xiao Min. What is Xiao Hong's surname?
    A. Hong
    B. Min
    Answer: A
    
    What is the total number of digits in the product of 243 x 80?
    A. 12
    B. 13
    C. 14
    D. 15
    Answer: C
    
    The analytical methods for solving quadratic equations include:
    A. One-solution method
    B. Two-solution method
    C. Three-solution method
    D. Four-solution method
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to reduce the amount of waste generated by the country. In a city, there is a new park that has been created. The park has a total of 120 trees. The president decides to plant 10 more trees in the park every month. After how many months will the park have 140 trees? To determine how many months it will take for the park to have 140 trees, we start by noting the initial number of trees in the park and the number of trees added each month.
    
    The initial number of trees in the park is 120. Each month, the president plants
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population is approximately 2.2 million. It was the capital of France from 1804 to 1870, and again from 1871 to 1940. In 1940, when France had become an independent state again, Paris was the capital of France. After the war, the city became the capital of the United Kingdom. Today, Paris is France’s cultural and political capital, and home to many of the world’s major museums, museums, and buildings, including the Louvre, the Eiffel Tower, Notre-Dame Cathedral, and many
    ===============================
    Prompt: The future of AI is
    Generated text:  not what it seems to be, but rather the sum of many of its parts. The possibilities are vast. The first is that the 21st century is likely to be a world of pervasive AI, where AI will be part of almost everything, and in all its forms. On the other hand, the future of AI will not be defined by its ability to improve, but rather by its ability to make the world a better place.
    The impact of AI on the world is immense and far-reaching, and it is hard to overemphasize the potential of AI to change the world in unexpected ways. The benefits of AI are


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and the French Revolution. The city is home to many famous museums, including the Musée d'Orsay and the Musée Rodin, and is a popular tourist destination. Paris is also known for its cuisine, including its famous French dishes such as croissants, boudin, and escargot. The city is also home to many international organizations and events, including the World Cup and the Euro
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to become more prevalent in various industries, including manufacturing, healthcare, transportation, and finance. This will lead to increased automation and artificial intelligence, which will automate repetitive tasks and increase efficiency.
    
    2. Improved privacy and security: As AI becomes more advanced, there will be a need to ensure that it is used ethically and responsibly. This will involve developing new technologies and protocols to protect user data and
    


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
    Generated text:  __________ and I'm a/an __________. I'm a/an __________, __________. I'm a/an __________, __________. I'm a/an __________, __________. I'm a/an __________, __________. I'm a/an __________, __________. I'm a/an __________, __________. I'm a/an __________, __________. I'm a/an __________, __________. I'm a/an __________, __________. I'm a/an __________, __________. I'm a/an __________, __________. I'm a/an __________, __
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as Notre Dame Cathedral, the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. It's also the birthplace of French Revolution leader Robespierre. The city has a rich cultural heritage and is home to a diverse range of music, art, and cuisine. Paris is known for its fashion and wine industries, and is considered one of the most beautiful and economically vibrant cities in the world. Its status as the world’s most populous city has led to its nickname of "The City of Light" and its status as one of the most popular tourist destinations in the world
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating and changing rapidly, with potential implications on various sectors and aspects of our lives. Here are some possible future trends in AI:
    
    1. Artificial General Intelligence (AGI): AGI refers to the ability of machines to perform complex cognitive functions such as learning, reasoning, and decision-making. While AGI is still a theoretical concept, many experts believe that it is within our reach. AI research institutions and companies are investing heavily in research to develop AGI, and there are already some promising advancements in the field.
    
    2. Smart Cities: With the growing need for urban infrastructure, smart cities are becoming more common. AI is being used to


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

     lingu

    ist

     by

     training

    ,

     with

     a

     vast

     knowledge

     of

     all

     languages

     spoken

     on

     Earth

    .

     I

     have

     studied

     and

     taught

     in

     countries

     such

     as

     France

    ,

     Spain

    ,

     Russia

    ,

     and

     South

     Africa

    ,

     and

     have

     traveled

     extensively

     throughout

     the

     world

    .

     I

     can

     communicate

     in

     over

     

    2

    0

     languages

     and

     have

     a

     natural

     ability

     to

     learn

     new

     languages

     quickly

    .

     My

     approach

     to

     education

     is

     student

    -centered

    ,

     and

     I

     believe

     that

     every

     student

     has

     the

     potential

     to

     become

     an

     exceptional

     thinker

     and

     problem

     solver

    .

     Whether

     it

    's

     helping

     someone

     learn

     a

     new

     language

    ,

     teaching

     a

     student

     a

     new

     skill

    ,

     or

     guiding

     them

     through

     a

     complex

     problem

    ,

     I

     am

     always

     ready

     to

     assist

     them

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     is

     the

     seat

     of

     the

     French

     government

    ,

     the

     main

     economic

     hub

    ,

     and

     the

     largest

     city

     in

     the

     country

    .

     Paris

     is

     also

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     rich

     cultural

     and

     artistic

     heritage

    .

     It

     is

     home

     to

     several

     world

    -f

    amous

     landmarks

     such

     as

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    .

     Paris

     is

     also

     known

     for

     its

     extensive

     food

     and

     wine

     culture

    ,

     with

     its

     iconic

     restaurants

     and

     bars

     serving

     local

     and

     international

     cuisine

    .

     Additionally

    ,

     Paris

     is

     the

     birth

    place

     of

     the

     French

     Revolution

    ,

     a

     period

     of

     radical

     social

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     shaped

     by

     a

     wide

     range

     of

     factors

    ,

     including

     technological

     advances

    ,

     changing

     business

     models

    ,

     and

     shifting

     societal

     values

    .

     Here

     are

     some

     possible

     trends

     that

     may

     influence

     the

     future

     of

     AI

    :
    


    1

    .

     AI

     will

     continue

     to

     become

     more

     widespread

    ,

     with

     more

     businesses

     and

     organizations

     using

     AI

     in

     their

     operations

    .
    


    2

    .

     AI

     will

     continue

     to

     evolve

     and

     improve

    ,

     with

     new

     algorithms

     and

     techniques

     emerging

     to

     solve

     increasingly

     complex

     problems

    .
    


    3

    .

     AI

     will

     continue

     to

     be

     integrated

     into

     new

     and

     existing

     technologies

    ,

     such

     as

     autonomous

     vehicles

    ,

     smart

     home

     appliances

    ,

     and

     virtual

     assistants

    .
    


    4

    .

     AI

     will

     continue

     to

     be

     used

     for

     a

     range

     of

     applications

    ,

     from

     healthcare

     and

     finance

     to

     manufacturing

     and

    



```python
llm.shutdown()
```
