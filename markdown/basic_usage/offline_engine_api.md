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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.53it/s]


    2026-05-11 23:03:32,329 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 23:03:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.83it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.83it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.83it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.83it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.83it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.83it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.83it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:17,  2.83it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.70it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.70it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.70it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.70it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.70it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.70it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:06,  6.70it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:06,  6.70it/s] Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:06,  6.70it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 12.24it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 12.24it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 12.24it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 12.24it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 12.24it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 12.24it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 12.24it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 12.24it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 12.24it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 18.77it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 27.27it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 27.27it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 27.27it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 27.27it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 27.27it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 27.27it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 27.27it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 27.27it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 27.27it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 27.27it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 36.32it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 36.32it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 36.32it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 36.32it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 36.32it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 36.32it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 36.32it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 36.32it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 36.32it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 36.32it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 45.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.26 GB):   3%|▎         | 2/58 [00:00<00:04, 13.71it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.26 GB):   3%|▎         | 2/58 [00:00<00:04, 13.71it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:04, 13.71it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.25 GB):   7%|▋         | 4/58 [00:00<00:03, 15.47it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.24 GB):   7%|▋         | 4/58 [00:00<00:03, 15.47it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.20 GB):   7%|▋         | 4/58 [00:00<00:03, 15.47it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.20 GB):  10%|█         | 6/58 [00:00<00:03, 17.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.23 GB):  10%|█         | 6/58 [00:00<00:03, 17.06it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=72.22 GB):  10%|█         | 6/58 [00:00<00:03, 17.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.21 GB):  10%|█         | 6/58 [00:00<00:03, 17.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.21 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.34it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.21 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.34it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.20 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.19 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.19 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.34it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.19 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.20 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.16 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.16 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.17 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.14 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.14 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.13 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.65it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.65it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=72.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=960 avail_mem=72.12 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.57it/s] Capturing num tokens (num_tokens=896 avail_mem=72.12 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=832 avail_mem=72.13 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=768 avail_mem=72.12 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=768 avail_mem=72.12 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=704 avail_mem=72.11 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=640 avail_mem=72.11 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=576 avail_mem=72.10 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=512 avail_mem=72.08 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.84it/s]

    Capturing num tokens (num_tokens=512 avail_mem=72.08 GB):  50%|█████     | 29/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=480 avail_mem=72.10 GB):  50%|█████     | 29/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=448 avail_mem=72.09 GB):  50%|█████     | 29/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=416 avail_mem=72.09 GB):  50%|█████     | 29/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=384 avail_mem=72.08 GB):  50%|█████     | 29/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=384 avail_mem=72.08 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.05it/s]Capturing num tokens (num_tokens=352 avail_mem=72.07 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.05it/s]Capturing num tokens (num_tokens=320 avail_mem=72.07 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.05it/s]Capturing num tokens (num_tokens=288 avail_mem=72.06 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.05it/s]Capturing num tokens (num_tokens=256 avail_mem=72.05 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.05it/s]

    Capturing num tokens (num_tokens=256 avail_mem=72.05 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=240 avail_mem=72.05 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=224 avail_mem=72.04 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=208 avail_mem=72.03 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=192 avail_mem=72.03 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=176 avail_mem=72.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.71it/s]Capturing num tokens (num_tokens=176 avail_mem=72.02 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.85it/s]Capturing num tokens (num_tokens=160 avail_mem=72.02 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.85it/s]Capturing num tokens (num_tokens=144 avail_mem=72.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.85it/s]Capturing num tokens (num_tokens=128 avail_mem=72.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.85it/s]Capturing num tokens (num_tokens=112 avail_mem=72.00 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.85it/s]

    Capturing num tokens (num_tokens=96 avail_mem=72.00 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.85it/s] Capturing num tokens (num_tokens=96 avail_mem=72.00 GB):  81%|████████  | 47/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=80 avail_mem=72.00 GB):  81%|████████  | 47/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=64 avail_mem=71.99 GB):  81%|████████  | 47/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=48 avail_mem=71.99 GB):  81%|████████  | 47/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=32 avail_mem=71.99 GB):  81%|████████  | 47/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=28 avail_mem=71.98 GB):  81%|████████  | 47/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=28 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.88it/s]Capturing num tokens (num_tokens=24 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.88it/s]Capturing num tokens (num_tokens=20 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.88it/s]Capturing num tokens (num_tokens=16 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.88it/s]

    Capturing num tokens (num_tokens=12 avail_mem=71.97 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.88it/s]Capturing num tokens (num_tokens=8 avail_mem=71.97 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.88it/s] Capturing num tokens (num_tokens=8 avail_mem=71.97 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=4 avail_mem=71.96 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=4 avail_mem=71.96 GB): 100%|██████████| 58/58 [00:01<00:00, 33.21it/s]


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
    Generated text:  Tom and I'm a software developer from the United States. I'm a very curious person who loves learning new things and trying out new things. I enjoy coding, building web applications, and always looking for new opportunities to learn new technologies. I love writing and sharing my ideas and experiences with others. I'm currently in the process of building my own website and I'm looking for a website builder to help me get started. What website builders do you recommend for beginners?
    When it comes to choosing a website builder for beginners, here are a few recommendations:
    
    1. Wix: This is a popular platform that offers a free trial for new
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking a new term. If this president is a woman, what is the probability that the next president will also be a woman? Express your answer as a common fraction.
    To determine the probability that the next president will be a woman if the current president is a woman, we need to follow these steps:
    
    1. Identify the total number of possible outcomes. Since there are 4 candidates for the next president, the total number of possible outcomes is 4.
    2. Identify the number of favorable outcomes. If the president is a woman, there is only 1 favorable outcome (the woman who will be the next president).
    3. Calculate
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    
    How would one implement this in Go?
    
    Certainly! In Go, we can define a struct to represent a capital city. Here's how you can implement this in Go:
    
    ```go
    package main
    
    import (
    	"fmt"
    )
    
    // CapitalCity represents a city
    type CapitalCity struct {
    	Name   string
    	Capital bool
    }
    
    func main() {
    	// Example data
    	cities := []CapitalCity{
    		{"Paris", true},
    		{"London", true},
    		{"Madrid", true},
    	}
    
    	// Print the cities
    	for _, city := range cities {
    		fmt.Printf("%s
    ===============================
    Prompt: The future of AI is
    Generated text:  mobile, and there are new areas where you can potentially invest: healthcare, education, and more. Here are some of the ways you can leverage mobile technology to help people in these areas.
    Mobile health systems will play a critical role in ensuring people receive the best possible care. They can help to detect and diagnose conditions, and can provide data on how people are using care, as well as link care providers to patients.
    Mobile health apps can be used to monitor patients in the hospital and provide real-time updates on their condition.
    Mobile education apps can be used to deliver educational resources to students in remote or underserved areas, as well as connect


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


    Generated text:  [Name] and I'm a [occupation] who has been working in the [industry] for [number] years. I'm passionate about [reason for interest] and have always been driven to achieve my goals. I'm always looking for new challenges and opportunities to learn and grow. I'm a [character trait] and I'm always ready to help others and make a positive impact. I'm excited to meet you and learn more about you. [Name] [Occupation] [Industry] [Number] [Reason for interest] [Character trait] [Character trait] [Character trait] [Character trait] [Character trait
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and vibrant cultural scene. It is also home to the Louvre Museum, the most famous museum in the world, and the Notre-Dame Cathedral. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub and a popular tourist destination. The city is known for its fashion industry, art scene, and cuisine, and is home to many famous landmarks and attractions. Paris is a city that has a unique blend of old-world charm and modernity, making it a popular destination for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more prevalent in many industries, including manufacturing, healthcare, and transportation. Automation will likely lead to increased efficiency and productivity, but it will also create new jobs and raise concerns about job displacement.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing pressure to address ethical and privacy concerns. This will likely lead to new regulations and standards for AI development and use.
    
    3. AI for
    


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
    Generated text:  [Name] and I'm a [age] year old [gender] girl/boy who [some specific skill or characteristic]. I'm passionate about [why I love my hobby, sport, or hobby]. If you would like to know more about me, just ask me how I can help you today. [Optional: mention any hobbies, interests, or goals for which I am passionate.] I'm always looking for new challenges and opportunities to learn and grow. How can I assist you today? Let's get started! What brings you here today? [Optional: ask about a specific event, job, or personal goal you're working
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the Île-de-France region in the south-central part of the country. It is the largest city in France and the 16th most populous city in the world. Paris is known for its art, fashion, and cuisine, and has hosted many famous events and cultural festivals over the years. The city is also an important economic hub and a major transportation hub, with its iconic Eiffel Tower serving as a symbol of the city's importance. 
    
    Paris has a rich and diverse history that dates back to the Roman Empire, and is known for its impressive architecture and medieval streets. The city is also home
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly diverse and dependent on a wide range of factors, including advancements in technology, policy decisions, and human behavior. Here are some potential trends that could emerge in the future of AI:
    
    1. Increased focus on ethical considerations: As AI is increasingly integrated into our daily lives, there is growing attention to the ethical implications of its use. This includes issues such as bias, privacy, and data privacy, as well as concerns about the potential impact on jobs and society as a whole.
    
    2. Integration of AI with other technologies: AI is becoming increasingly integrated into a wide range of other technologies, including robotics, healthcare, and finance. This integration


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

    ]

     and

     I

     am

     a

     skilled

     [

    occupation

     or

     hobby

    ].

     I

    've

     always

     been

     fascinated

     by

     the

     natural

     world

    ,

     and

     my

     passion

     for

     exploration

     led

     me

     to

     pursue

     my

     [

    major

     or

     career

     path

    ].

     I

    've

     always

     been

     an

     avid

     [

    interest

     or

     hobby

    ],

     and

     I

    'm

     always

     looking

     for

     the

     next

     big

     discovery

    .

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

     I

    'm

     always

     looking

     for

     ways

     to

     make

     the

     world

     a

     better

     place

    .

     And

     I

    'm

     here

     to

     share

     my

     knowledge

     and

     passion

     with

     you

    .

     What

    's

     your

     name

    ,

     and

     what

    's

     your

     occupation

     or

     hobby

    ?

     Can

     you

     tell

     me

     more

     about

     yourself

    ?

     I

    'm

     here

     to

     learn

     from

     you

    .

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     where

     the

     E

    iff

    el

     Tower

     stands

     as

     a

     symbol

     of

     the

     city

    's

     wealth

     and

     sophistication

    .


    Paris

    ,

     France

    's

     capital

    ,

     is

     renowned

     for

     its

     rich

     history

    ,

     diverse

     culture

    ,

     and

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

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Known

     for

     its

     art

    ,

     cuisine

    ,

     and

     fashion

    ,

     it

    's

     also

     home

     to

     the

     iconic

     Paris

     Metro

     and

     is

     a

     major

     transportation

     hub

     and

     cultural

     center

     in

     Europe

    .

     The

     city

    's

     global

     importance

     as

     the

     world

    's

     largest

     city

     and

     most

     visited

     tourist

     destination

     makes

     it

     a

     fascinating

     destination

     for

     travelers

     from

     all

     over

     the

     world

    .

     [

    4

    0

    ]

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     complex

     and

     multif

    ac

    eted

    ,

     driven

     by

     technological

     advancements

    ,

     changing

     societal

     needs

    ,

     and

     regulatory

     considerations

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     AI

     in

     the

     coming

     years

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

     more

     data

     and

     algorithms

     are

     analyzed

     to

     ensure

     fairness

     and

     transparency

     in

     AI

     decision

    -making

    ,

     there

     will

     be

     increased

     focus

     on

     developing

     ethical

     frameworks

     for

     AI

    .

     This

     could

     lead

     to

     more

     stringent

     regulations

    ,

     increased

     scrutiny

     of

     AI

     systems

    ,

     and

     a

     greater

     emphasis

     on

     accountability

     and

     transparency

    .
    


    2

    .

     Integration

     of

     AI

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     be

     integrated

     with

     other

     technologies

    ,

     such

     as

     sensors

    ,

     data

     analytics

    ,

     and

     machine

     learning

    .

     This

     integration

    



```python
llm.shutdown()
```
