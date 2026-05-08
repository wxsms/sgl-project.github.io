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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.39it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.38it/s]


    2026-05-08 04:39:07,805 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 04:39:07] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:04,  4.30s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.84it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.84it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.84it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.84it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.84it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.84it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.84it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.84it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.84it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.84it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 23.84it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 32.96it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 32.96it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 32.96it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 32.96it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 32.96it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 32.96it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 32.96it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 32.96it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 32.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.61 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.58 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.58 GB):   3%|▎         | 2/58 [00:00<00:02, 19.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.58 GB):   3%|▎         | 2/58 [00:00<00:02, 19.25it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.58 GB):   3%|▎         | 2/58 [00:00<00:02, 19.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.58 GB):   3%|▎         | 2/58 [00:00<00:02, 19.25it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.58 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.57 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.56 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.56 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.56 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.55 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.55 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.55 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.54 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=71.54 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.98it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.54 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.98it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.53 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.53 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.98it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.53 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.52 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.52 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.52 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.83it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.50 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.83it/s]Capturing num tokens (num_tokens=960 avail_mem=71.52 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.83it/s] Capturing num tokens (num_tokens=896 avail_mem=71.51 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.83it/s]

    Capturing num tokens (num_tokens=832 avail_mem=71.51 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.83it/s]Capturing num tokens (num_tokens=832 avail_mem=71.51 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=768 avail_mem=71.51 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=704 avail_mem=71.50 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=640 avail_mem=71.50 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=576 avail_mem=71.50 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=512 avail_mem=71.48 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=512 avail_mem=71.48 GB):  50%|█████     | 29/58 [00:00<00:00, 43.57it/s]Capturing num tokens (num_tokens=480 avail_mem=71.50 GB):  50%|█████     | 29/58 [00:00<00:00, 43.57it/s]Capturing num tokens (num_tokens=448 avail_mem=71.50 GB):  50%|█████     | 29/58 [00:00<00:00, 43.57it/s]Capturing num tokens (num_tokens=416 avail_mem=71.50 GB):  50%|█████     | 29/58 [00:00<00:00, 43.57it/s]Capturing num tokens (num_tokens=384 avail_mem=71.49 GB):  50%|█████     | 29/58 [00:00<00:00, 43.57it/s]

    Capturing num tokens (num_tokens=352 avail_mem=71.49 GB):  50%|█████     | 29/58 [00:00<00:00, 43.57it/s]Capturing num tokens (num_tokens=352 avail_mem=71.49 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.27it/s]Capturing num tokens (num_tokens=320 avail_mem=71.48 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.27it/s]Capturing num tokens (num_tokens=288 avail_mem=71.48 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.27it/s]Capturing num tokens (num_tokens=256 avail_mem=71.48 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.27it/s]Capturing num tokens (num_tokens=240 avail_mem=71.48 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.27it/s]Capturing num tokens (num_tokens=224 avail_mem=71.47 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.27it/s]Capturing num tokens (num_tokens=224 avail_mem=71.47 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.53it/s]Capturing num tokens (num_tokens=208 avail_mem=71.47 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.53it/s]Capturing num tokens (num_tokens=192 avail_mem=71.47 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.53it/s]Capturing num tokens (num_tokens=176 avail_mem=71.46 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.53it/s]Capturing num tokens (num_tokens=160 avail_mem=71.46 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.53it/s]

    Capturing num tokens (num_tokens=144 avail_mem=71.46 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.53it/s]Capturing num tokens (num_tokens=144 avail_mem=71.46 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.24it/s]Capturing num tokens (num_tokens=128 avail_mem=71.46 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.24it/s]Capturing num tokens (num_tokens=112 avail_mem=71.45 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.24it/s]Capturing num tokens (num_tokens=96 avail_mem=71.45 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.24it/s] Capturing num tokens (num_tokens=80 avail_mem=71.45 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.24it/s]Capturing num tokens (num_tokens=64 avail_mem=71.44 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.24it/s]Capturing num tokens (num_tokens=64 avail_mem=71.44 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.37it/s]Capturing num tokens (num_tokens=48 avail_mem=71.44 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.37it/s]Capturing num tokens (num_tokens=32 avail_mem=71.44 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.37it/s]Capturing num tokens (num_tokens=28 avail_mem=71.43 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.37it/s]Capturing num tokens (num_tokens=24 avail_mem=71.43 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.37it/s]

    Capturing num tokens (num_tokens=20 avail_mem=71.42 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.37it/s]Capturing num tokens (num_tokens=20 avail_mem=71.42 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.75it/s]Capturing num tokens (num_tokens=16 avail_mem=71.42 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.75it/s]Capturing num tokens (num_tokens=12 avail_mem=71.42 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.75it/s]Capturing num tokens (num_tokens=8 avail_mem=71.42 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.75it/s] Capturing num tokens (num_tokens=4 avail_mem=71.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.75it/s]Capturing num tokens (num_tokens=4 avail_mem=71.41 GB): 100%|██████████| 58/58 [00:01<00:00, 42.16it/s]


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
    Generated text:  Tessa, a hardworking professional, and I have been writing professionally since 2015. I work as a freelance writer and provide editing, proofreading, and copywriting services. My focus is on providing quality content that is well-written and concise, making it easy for businesses to communicate their messages effectively. As a business owner, I have a strong desire to support my customers, and I strive to provide them with the best possible service. My goal is to help my clients achieve their business objectives, and I am dedicated to making their projects successful. With over 30 years of experience in the business world, I am
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking government official who holds the office of the President of the United States. The office of the president of the United States is one of the three executive offices of the United States federal government. The office of the president of the United States has been held by the first four presidents, George Washington, John Adams, Thomas Jefferson and George Washington.
    Can we infer the following?
    The presidents of the United States are all of the same age. 
    OPT:
     (A). yes
     (B). it is not possible to tell
     (C). no
    The answer is (B). it is not possible to tell.
    We can infer
    ===============================
    Prompt: The capital of France is
    Generated text:  a city in the province of Lyon, which is in the north of France. The name of the city is Lyon, but it is called "Lyon" in French. The capital of France is in the southwest of France, near the Mediterranean Sea. It is the largest city in France and one of the most populous cities in Europe. It is a major transportation hub and a center of industry, culture, and commerce. Lyon is situated on the Jura Mountains, and the city has a mountainous climate. It is the administrative center of the Prefecture of Lyon, which is the second largest prefecture in France. The capital of
    ===============================
    Prompt: The future of AI is
    Generated text:  in motion. And for many applications, the complexity and sophistication of the AI algorithms will only increase. This is why it is crucial to ensure that the AI algorithms being used to serve your business are robust, scalable, and secure. This way, you can maintain your operational efficiency while also protecting your organization from potential risks. In this article, we’ll discuss the key issues you need to consider when building AI applications and how you can address them to ensure your organization is secure.
    AI algorithms are complex and need to be robust, scalable, and secure. Below are some of the key issues to consider when building AI applications:
    1. Robust


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    The statement is concise and accurately describes the capital city of France, providing only the key information needed to understand its significance. It does not include any additional details or context that would not be necessary to fully convey the information being conveyed. The statement is also grammatically correct and follows standard English syntax. 
    
    To further elaborate on the statement, it could be used to describe the cultural, historical, or political importance of Paris, or to provide information about its architecture, cuisine, or other aspects of the city. However, the statement itself is sufficient to convey the main point of the information being presented. 
    
    In conclusion, the statement
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from their environment and improve their performance over time.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as increased scrutiny of AI systems that are designed to benefit humans at the expense of other forms
    


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
    Generated text:  [Name] and I'm a [job title] at [company]. I'm an [occupation] in [industry] who has been [number of years] years in this role. I'm a [specific skill] that has honed over the years. I love [reason why I'm passionate about the role], and I’m [name of the next step in the career path] at [company] in [year]. [Name] is a well-rounded individual who is comfortable and confident in both their work and personal life. I'm always up for a challenge and eager to learn new things. I'm a great communicator,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and complex, but here are some possible trends to look out for:
    
    1. Increased integration: AI is becoming more integrated into everyday life, with more devices and applications learning from one another. This could lead to more complex systems that require human oversight.
    
    2. Increased diversity: AI is becoming more diverse, with more voices and perspectives being represented in the development of AI models. This could lead to more ethical and equitable AI systems.
    
    3. Autonomous systems: As AI becomes more advanced, it could be able to learn and adapt on its own, without human intervention. This could lead to more autonomous and self-driving vehicles, among other applications


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

     [

    insert

     any

     relevant

     professions

     or

     roles

     here

    ,

     such

     as

     "

    author

    ",

     "

    expl

    orer

    ",

     or

     "

    scient

    ist

    "].

     I

    'm

     a

     [

    insert

     any

     unique

     skill

     or

     ability

     here

    ,

     such

     as

     "

    creative

     writing

    ",

     "

    expl

    oration

    ",

     or

     "

    scientific

     research

    "].

     I

     enjoy

     [

    insert

     any

     interests

     or

     passions

     here

    ,

     such

     as

     "

    reading

    ",

     "

    travel

    ing

    ",

     "

    history

    ",

     or

     "

    science

    ".

    ].

     I

    'm

     constantly

     on

     the

     lookout

     for

     interesting

     topics

     to

     explore

    ,

     and

     I

    'm

     always

     eager

     to

     learn

     new

     things

    .

     Overall

    ,

     I

    'm

     a

     [

    insert

     any

     distinguishing

     traits

     or

     personality

     traits

     here

    ,

     such

     as

     "

    ext

    ro

    verted

    ",

    
    
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

    ,

     a

     bustling

     met

    ropolis

     with

     a

     rich

     cultural

     heritage

    .

     It

     is

     one

     of

     the

     world

    's

     most

     cosm

    opolitan

     cities

    ,

     home

     to

     a

     diverse

     range

     of

     art

    ,

     music

    ,

     and

     culinary

     traditions

    ,

     and

     is

     a

     global

     hub

     of

     politics

    ,

     business

    ,

     and

     fashion

    .

     The

     city

     is

     renowned

     for

     its

     historical

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    ,

     and

     for

     its

     role

     in

     shaping

     French

     culture

     and

     politics

    .

     Paris

     is

     also

     a

     major

     transportation

     hub

    ,

     with

     many

     important

     roads

    ,

     airports

    ,

     and

     rail

     networks

     connecting

     it

     to

     the

     rest

     of

     the

     country

     and

     the

     world

    .

     With

     its

     modern

     architecture

    ,

     vibrant

     culture

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     shaped

     by

     a

     variety

     of

     trends

    ,

     from

     rapid

     advancements

     in

     machine

     learning

     and

     natural

     language

     processing

     to

     new

     developments

     in

     robotics

     and

     autonomous

     systems

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

     focus

     on

     ethical

     and

     responsible

     AI

    :

     As

     the

     technology

     continues

     to

     advance

    ,

     we

     will

     see

     a

     greater

     emphasis

     on

     ethical

     and

     responsible

     AI

    .

     This

     includes

     concerns

     about

     privacy

    ,

     bias

    ,

     and

     transparency

    .

     We

     may

     see

     more

     regulations

     and

     standards

     being

     established

     to

     ensure

     that

     AI

     systems

     are

     designed

     and

     deployed

     in

     ways

     that

     meet

     ethical

     standards

    .
    


    2

    .

     Growth

     in

     AI

     integration

     with

     other

     technologies

    :

     AI

     is

     already

     being

     integrated

     into

     many

     different

     fields

    ,

     including

     healthcare

    ,

     finance

    ,

    



```python
llm.shutdown()
```
