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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.63it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.63it/s]


    2026-05-13 09:43:29,651 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 09:43:29] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:43,  4.97s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:43,  4.97s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  7.98it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  7.98it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.54it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.21it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.21it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.21it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.21it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.21it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.21it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.21it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.21it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.21it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.21it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 27.99it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 27.99it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 27.99it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 27.99it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 27.99it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 27.99it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 27.99it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 27.99it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 27.99it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 27.99it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 27.99it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 37.83it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 37.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.14it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.33 GB):   3%|▎         | 2/58 [00:00<00:03, 17.78it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.33 GB):   3%|▎         | 2/58 [00:00<00:03, 17.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.33 GB):   3%|▎         | 2/58 [00:00<00:03, 17.78it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.33 GB):   3%|▎         | 2/58 [00:00<00:03, 17.78it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.33 GB):   9%|▊         | 5/58 [00:00<00:02, 21.16it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.32 GB):   9%|▊         | 5/58 [00:00<00:02, 21.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.31 GB):   9%|▊         | 5/58 [00:00<00:02, 21.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.31 GB):   9%|▊         | 5/58 [00:00<00:02, 21.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.31 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.31 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.31 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.30 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.30 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.30 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.29 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.29 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.24 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.00 GB):  21%|██        | 12/58 [00:00<00:01, 29.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.00 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.00 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.59it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.00 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.99 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.59it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=57.99 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.97 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=960 avail_mem=57.99 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.54it/s] Capturing num tokens (num_tokens=896 avail_mem=57.98 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=832 avail_mem=57.98 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=768 avail_mem=57.98 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=704 avail_mem=57.98 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.54it/s]Capturing num tokens (num_tokens=704 avail_mem=57.98 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.04it/s]Capturing num tokens (num_tokens=640 avail_mem=57.97 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.04it/s]Capturing num tokens (num_tokens=576 avail_mem=57.97 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.04it/s]Capturing num tokens (num_tokens=512 avail_mem=57.96 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.04it/s]

    Capturing num tokens (num_tokens=480 avail_mem=57.97 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.04it/s]Capturing num tokens (num_tokens=448 avail_mem=57.97 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.04it/s]Capturing num tokens (num_tokens=448 avail_mem=57.97 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=416 avail_mem=57.97 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=384 avail_mem=57.97 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=352 avail_mem=57.96 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=320 avail_mem=57.95 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=288 avail_mem=57.95 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.58it/s]Capturing num tokens (num_tokens=288 avail_mem=57.95 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.70it/s]Capturing num tokens (num_tokens=256 avail_mem=57.95 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.70it/s]Capturing num tokens (num_tokens=240 avail_mem=57.95 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.70it/s]

    Capturing num tokens (num_tokens=224 avail_mem=57.94 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.70it/s]Capturing num tokens (num_tokens=208 avail_mem=57.94 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.70it/s]Capturing num tokens (num_tokens=192 avail_mem=57.94 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.70it/s]Capturing num tokens (num_tokens=192 avail_mem=57.94 GB):  71%|███████   | 41/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=176 avail_mem=57.93 GB):  71%|███████   | 41/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=160 avail_mem=57.93 GB):  71%|███████   | 41/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=144 avail_mem=57.93 GB):  71%|███████   | 41/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=128 avail_mem=57.93 GB):  71%|███████   | 41/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=112 avail_mem=57.92 GB):  71%|███████   | 41/58 [00:01<00:00, 39.75it/s]

    Capturing num tokens (num_tokens=112 avail_mem=57.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.81it/s]Capturing num tokens (num_tokens=96 avail_mem=57.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.81it/s] Capturing num tokens (num_tokens=80 avail_mem=57.92 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.81it/s]Capturing num tokens (num_tokens=64 avail_mem=57.91 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.81it/s]Capturing num tokens (num_tokens=48 avail_mem=57.91 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.81it/s]Capturing num tokens (num_tokens=32 avail_mem=57.91 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.81it/s]Capturing num tokens (num_tokens=32 avail_mem=57.91 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=28 avail_mem=57.90 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=24 avail_mem=57.90 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=20 avail_mem=57.90 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=16 avail_mem=57.90 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.21it/s]

    Capturing num tokens (num_tokens=12 avail_mem=57.89 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=12 avail_mem=57.89 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=8 avail_mem=57.89 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.24it/s] Capturing num tokens (num_tokens=4 avail_mem=57.88 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=4 avail_mem=57.88 GB): 100%|██████████| 58/58 [00:01<00:00, 37.73it/s]


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
    Generated text:  Fairewa and I am a digital marketer with a passion for online marketing and social media.
    In this role, I am responsible for creating and managing the overall digital marketing strategy for a digital agency, responsible for monitoring and analyzing the performance of the digital marketing campaigns, and identifying trends and areas for improvement.
    I enjoy learning new strategies, technologies, and methodologies to optimize and improve the digital marketing strategy for the agency.
    In my spare time, I enjoy reading books, playing board games, and spending time with my family and friends.
    What is the purpose of your digital marketing strategy? In your role as a digital marketer, what is your primary
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man, and it is known that he was born in the year 1981. In which year was he born?
    To determine the year the president of the United States was born, we need to check which year is a candidate year that fits all the given conditions: the president is a man, and he was born in the year 1981.
    
    Let's verify this step-by-step:
    
    1. **Check if 1981 is a candidate year:**
       - 1981 is the year that fits the condition that the president is a man.
       - 1981
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. London
    C. New York
    D. Sydney
    Answer: A
    
    The Chinese government has always maintained a firm stance against terrorism, actively protecting the lives and property safety of all people, and actively promoting global anti-terrorism cooperation. The figure below reflects some of the achievements in this regard. According to this, we can infer that ____.
    A. There are differences in views between governments at different levels
    B. The government is actively protecting the lives and property safety of people from all around the world
    C. There are differences in views within the government
    D. The government has always been actively
    ===============================
    Prompt: The future of AI is
    Generated text:  now, and it's not just the tech industry that needs to stay ahead. Small companies, startups, and individuals should be thinking of their AI opportunities now. These companies can take advantage of this new era of AI by leveraging their data, skills, and resources. By investing in AI, these companies can lead the charge in the future of technology and make a positive impact on society.
    In this blog post, we will discuss how small companies can harness the power of AI to drive innovation and growth. We will also explore the benefits of AI for startups and how they can use it to their advantage. Finally, we will provide some practical tips


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


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to many world-renowned museums, including the Musée d'Orsay and the Musée Rodin. Paris is a bustling city with a rich history and culture, and it is a popular tourist destination for visitors from around the world. The city is also known for its fashion industry, with many famous designers and boutiques located in the city center. Overall, Paris is a city that is a true reflection of French culture and history.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced machine learning capabilities: AI will continue to improve its ability to learn from data and make more accurate predictions and decisions.
    
    3. Increased use of AI in healthcare: AI will be used to improve the accuracy and efficiency of medical diagnosis and treatment, as well as to develop new treatments for diseases.
    
    4. Increased use of AI in finance: AI will be used to improve the accuracy and efficiency of
    


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
    Generated text:  [Name] and I am [Age]. I am a [job title or occupation] who is [description of your job]. I am passionate about [reason why you are passionate about your job]. I also enjoy [reason why you enjoy your job]. I believe that [reason why you believe this] is [reason for belief]. My [job title or occupation] is [position], [company name], or [specific company]. I am [percentage of age] years old. I am from [your hometown, city, or state]. I love [reason why you love your hometown]. I also like [reason why you like your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its iconic landmarks, rich history, and cultural attractions. The city is home to the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and many other notable structures and sites. As the seat of government, Paris is a bustling metropolis with a diverse population and a rich history dating back to the ancient Romans. The city is known for its cuisine, fashion, and art scene, and has become a popular tourist destination for centuries. Paris is home to many world-class museums, including the Louvre, Musée d'Orsay, and the Centre Pompidou. It is also a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve several key trends, including:
    
    1. Increased reliance on machine learning and deep learning: As the technology behind AI continues to improve, we can expect to see a greater reliance on machine learning and deep learning algorithms to model and solve complex problems.
    
    2. Increased focus on ethical considerations: With the increasing amount of data that AI systems are required to process, there will be greater emphasis on ethical considerations and data privacy.
    
    3. Advancements in natural language processing: AI will continue to improve its ability to process and understand natural language, leading to more sophisticated natural language processing tools and applications.
    
    4. Increased integration with human decision-making:


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

    insert

     character

    's

     name

    ],

     and

     I

    'm

     a

     [

    insert

     character

    's

     profession

    ,

     age

    ,

     or

     general

     identity

    ].

     I

    'm

     [

    insert

     character

    's

     age

     or

     any

     relevant

     information

    ].

     I

    've

     always

     been

     fascinated

     by

     the

     unknown

     and

     have

     always

     been

     eager

     to

     learn

     new

     things

    .

     I

     have

     a

     vast

     knowledge

     of

     [

    insert

     a

     few

     relevant

     facts

     about

     the

     character

    's

     field

     of

     expertise

     or

     interests

    ].

     I

    'm

     always

     willing

     to

     help

     those

     in

     need

     and

     I

    'm

     always

     looking

     for

     new

     challenges

     to

     overcome

    .

     If

     you

     have

     any

     questions

     or

     topics

     you

    'd

     like

     to

     discuss

    ,

     feel

     free

     to

     ask

     and

     I

    'll

     do

     my

     best

     to

     provide

     helpful

     and

     informative

     answers

    .

     Welcome

    ,

     [

    insert

    
    
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

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     rich

     culture

    ,

     and

     vibrant

     nightlife

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     is

     the

     birth

    place

     of

     many

     notable

     figures

     such

     as

     Napoleon

    ,

     Mar

    iano

     Ventura

    ,

     and

     Joan

     of

     Arc

    .

     The

     city

     is

     also

     known

     for

     its

     annual

     E

    iff

    el

     Tower

     Par

    c

     Fl

    ott

    if

    il

    é

     Festival

    ,

     which

     is

     one

     of

     the

     largest

     floating

     festivals

     in

     the

     world

    .

     
    


    Paris

     is

     also

     famous

     for

     its

     fashion

     industry

    ,

     with

     designers

     such

     as

     Louis

     V

    uit

    ton

     and

     Gu

    cci

    .

     The

     city

    's

     cuisine

    ,

     including

     its

     famous

     bou

    ill

    ab

    ais

    se

    ,

     is

     also

     renowned

     around

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     highly

     dynamic

     and

     unpredictable

    ,

     with

     potential

     breakthrough

    s

     and

     limitations

     that

     are

     difficult

     to

     predict

    .

     However

    ,

     here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     The

     integration

     of

     AI

     with

     human

     intelligence

     will

     likely

     increase

    .

     For

     example

    ,

     AI

    -powered

     systems

     may

     be

     used

     to

     assist

     with

     decision

    -making

    ,

     process

     data

    ,

     and

     perform

     tasks

     that

     require

     human

     empathy

    ,

     creativity

    ,

     and

     emotional

     intelligence

    .
    


    2

    .

     Enhanced

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     advanced

     and

     prevalent

    ,

     there

     is

     a

     risk

     of

     increased

     privacy

     and

     security

     concerns

    .

     This

     includes

     issues

     related

     to

     data

     privacy

    ,

     protection

     of

     sensitive

     information

    ,

    



```python
llm.shutdown()
```
