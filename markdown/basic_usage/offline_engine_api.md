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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.91it/s]


    2026-05-11 10:18:38,212 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 10:18:38] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.46it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.46it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.84it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.84it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.84it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.84it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.84it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.84it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.84it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.84it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 14.84it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 21.24it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 21.24it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 21.24it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 21.24it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 21.24it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 21.24it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 21.24it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 21.24it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 21.24it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 21.24it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 29.48it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 29.48it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 29.48it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 29.48it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 29.48it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 29.48it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 39.50it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 39.50it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 39.50it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.72it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 28.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 28.51it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 28.51it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 28.51it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 28.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 28.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=75.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=75.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.82it/s]Capturing num tokens (num_tokens=1024 avail_mem=75.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.82it/s]

    Capturing num tokens (num_tokens=960 avail_mem=75.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.82it/s] Capturing num tokens (num_tokens=960 avail_mem=75.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=896 avail_mem=75.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=832 avail_mem=75.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=768 avail_mem=75.05 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=704 avail_mem=75.05 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=640 avail_mem=75.05 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.60it/s]Capturing num tokens (num_tokens=640 avail_mem=75.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=576 avail_mem=75.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=512 avail_mem=75.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=480 avail_mem=75.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.58it/s]

    Capturing num tokens (num_tokens=448 avail_mem=75.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=416 avail_mem=75.04 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=416 avail_mem=75.04 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=384 avail_mem=75.04 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=352 avail_mem=75.04 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=320 avail_mem=75.03 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=288 avail_mem=75.03 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=256 avail_mem=75.03 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=256 avail_mem=75.03 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.39it/s]Capturing num tokens (num_tokens=240 avail_mem=75.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.39it/s]Capturing num tokens (num_tokens=224 avail_mem=74.69 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.39it/s]

    Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.39it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.39it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.39it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.26it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.26it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.26it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.26it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.26it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.26it/s] Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 42.26it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 42.26it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 42.26it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 42.26it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 42.26it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  81%|████████  | 47/58 [00:01<00:00, 42.26it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.90it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.90it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.90it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.90it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.90it/s]

    Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 39.90it/s] Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  98%|█████████▊| 57/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 37.41it/s]


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
    Generated text:  Andrew, and I am a software developer. I have been working with C++ for a while now, and I am looking for a challenge that would challenge me further. Can you suggest a project for me to work on?
    
    Sure, I can help you with that! Can you please provide me with more information about the project you would like to work on? What language are you familiar with, and what kind of project are you looking for? Also, please let me know if there are any specific requirements or constraints you have in mind for the project. Once I have the necessary information, I will be able to provide you with some suggestions
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to build in Asia. He has decided that the number of bases he should build is between 300 and 500. He knows that each base costs $100 million to build and that each military base in Asia will have 1000 personnel. The president wants to know if building any bases in Asia will be profitable. He can estimate the average cost per personnel per base in Asia as $100,000. How many military bases should the president build in Asia to maximize profit? To determine how many military bases the president should build in Asia to
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the most populous city in Europe. It is also the most visited city in the world.
    Yes, Paris is a beautiful city and the capital of France. It is located in the western region of France, on the Loire River in the center of the country. Paris is the fifth-largest city in the world by population and is one of the largest cities in the world by area. It is also one of the most economically vibrant cities in the world. Paris is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Palais des Fallingas.
    ===============================
    Prompt: The future of AI is
    Generated text:  likely to be decentralized with the rise of blockchain. Blockchain technology aims to ensure the privacy of data and enable secure and transparent transactions. As the use of blockchain in AI becomes more widespread, it is expected to transform the way we interact with technology and create new opportunities for innovation. The use of blockchain in AI is also expected to have a significant impact on the economic landscape, as it can reduce costs and increase efficiency.
    However, the adoption of blockchain in AI requires significant investment and understanding of its technical aspects. To make the most of the potential benefits of blockchain in AI, it is essential to have a deep understanding of the technology and its implications


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


    Generated text:  [Name] and I'm a [occupation] who has been [number of years] in the industry. I'm passionate about [reason for interest] and I'm always looking for ways to [action or goal]. I'm [age] years old and I'm currently [occupation] in [industry]. I'm [character's personality] and I'm always [character's attitude]. I'm [character's profession] and I'm [character's profession]. I'm [character's profession] and I'm [character's profession]. I'm [character's profession] and I'm [character's profession]. I'm [character's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French language. Paris is also home to many famous French artists, writers, and musicians. The city is a major center for the arts, music, and fashion, and is known for its vibrant nightlife and cultural events. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives, from voice assistants like Siri and Alexa to self-driving cars. As AI technology continues to improve, we can expect to see even more integration into our daily routines.
    
    2. Greater emphasis on ethical and social implications: As AI becomes more integrated into our lives, there will be increased scrutiny of its impact on society. This will likely lead to greater emphasis on ethical and social implications, as well as greater regulation
    


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
    Generated text:  [Your Name]. I am a [Your Profession] with a passion for [Your Field of Interest]. I'm a [Your Nickname] who believes in the importance of [Your Profession], and I strive to make the world a better place by sharing my knowledge and experience. I'm always looking to learn and grow, and I'm always excited to help others achieve their goals. What brings you to this field? What do you love most about your work? I hope to have the opportunity to connect with you and discuss our potential for collaboration and growth. To begin, let's talk about your profession and what drives you to pursue it
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its beautiful architecture, iconic landmarks, and vibrant culture. France's capital city, Paris, is a historic city that is home to the Eiffel Tower, the Louvre Museum, and many other cultural and tourist attractions. The city is also known for its art nouveau style, and the Notre-Dame Cathedral, which is a UNESCO World Heritage site. Paris has a rich history and is considered one of the world's most beautiful cities, which makes it a popular destination for tourists and locals alike. The city is also home to many international organizations and events, making it a hub for commerce and diplomacy in the region.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but some potential trends that are currently being explored and studied include:
    
    1. Increased autonomy of AI: As AI continues to become more complex and self-aware, there is potential for increased autonomy in AI systems. This could involve tasks that require decision-making, such as medical diagnosis or autonomous vehicles, where the AI system should make the best possible decision based on available information.
    
    2. Integration with human intelligence: AI is already becoming more integrated with human intelligence, with models that can interpret human language, recognize faces, and understand natural language. It is expected that this trend will continue, with AI systems becoming more capable of understanding and processing human


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

     am

     [

    Your

     Profession

    /

    Job

     Title

    ].

     Throughout

     my

     career

    ,

     I

     have

     been

     working

     in

     [

    Your

     Industry

    /

    Field

    ],

     and

     I

     have

     always

     been

     passionate

     about

     [

    Your

     Personal

     Interest

     or

     Hobby

    ].

     I

     am

     a

     dedicated

     and

     hard

    working

     individual

     who

     is

     always

     looking

     for

     ways

     to

     improve

     and

     enhance

     myself

    .

     I

     am

     always

     eager

     to

     learn

     and

     explore

     new

     opportunities

    ,

     and

     I

     believe

     that

     knowledge

     is

     power

    .

     If

     you

     are

     looking

     for

     someone

     to

     work

     with

    ,

     I

     am

     the

     person

     you

     will

     find

    .

     Thanks

     for

     having

     me

    !

     How

     can

     I

     help

     you

     today

    ?


    Hi

     there

    !

     I

    'm

     [

    Your

     Name

    ],

     a

     [

    Your

     Profession

    /

    Job

     Title

    
    
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

     serves

     as

     the

     capital

     of

     the

     country

    .

     It

     is

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     vibrant

     culture

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     landmarks

    .

     The

     city

     is

     also

     known

     for

     its

     rich

     culinary

     traditions

     and

     vibrant

     nightlife

    .

     As

     one

     of

     the

     world

    's

     most

     famous

     cities

    ,

     Paris

     is

     a

     symbol

     of

     France

     and

     has

     a

     unique

     blend

     of

     old

    -world

     charm

     and

     modern

    ity

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

     and

     exciting

    ,

     and

     there

     are

     several

     potential

     trends

     that

     are

     currently

     being

     explored

     and

     studied

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     become

     increasingly

     important

     to

     consider

     the

     ethical

     implications

     of

     its

     use

    .

     This

     will

     likely

     lead

     to

     more

     research

     into

     ways

     to

     design

     AI

     systems

     that

     are

     more

     transparent

    ,

     accountable

    ,

     and

     responsible

     for

     their

     decisions

    .
    


    2

    .

     Deep

     learning

     and

     neural

     networks

    :

     These

     are

     the

     core

     technologies

     behind

     AI

    ,

     and

     they

     are

     likely

     to

     continue

     to

     evolve

     and

     improve

    .

     Deep

     learning

     will

     likely

     become

     more

     powerful

    ,

     with

     the

     potential

     to

     solve

     increasingly

     complex

     problems

     that

     were

     previously

     impossible

     to

     tackle

    .
    


    3

    .

     Improved

     privacy

     and

     security

    



```python
llm.shutdown()
```
