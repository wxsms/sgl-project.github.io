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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.12it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.11it/s]


    2026-04-13 23:49:53,809 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 23:49:53] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.35it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.16it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.45it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.45it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.45it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.45it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.45it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.45it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.45it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.45it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.51it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.51it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.51it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.51it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.51it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.51it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.51it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.51it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.04it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 34.90it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 34.90it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 34.90it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 34.90it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 34.90it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 34.90it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 34.90it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.82it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 18.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.27it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 18.27it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.43it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.43it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.43it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.43it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.35it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.35it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.35it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 27.41it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 27.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 27.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 27.41it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.63it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.63it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.63it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.63it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.63it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.63it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.01it/s] Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.01it/s]

    Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.02it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.02it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.02it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.02it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.02it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.02it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.68it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.68it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.68it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.68it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.68it/s]

    Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.68it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  60%|██████    | 35/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=240 avail_mem=120.25 GB):  60%|██████    | 35/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=224 avail_mem=120.25 GB):  60%|██████    | 35/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=208 avail_mem=119.02 GB):  60%|██████    | 35/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=208 avail_mem=119.02 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.23it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.23it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.23it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.23it/s]

    Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.23it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.23it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.59it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.59it/s]

    Capturing num tokens (num_tokens=64 avail_mem=118.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  86%|████████▌ | 50/58 [00:01<00:00, 21.08it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  86%|████████▌ | 50/58 [00:01<00:00, 21.08it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 21.08it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 21.08it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  86%|████████▌ | 50/58 [00:01<00:00, 21.08it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  86%|████████▌ | 50/58 [00:01<00:00, 21.08it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 25.34it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 25.34it/s]

    Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 25.34it/s] Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  95%|█████████▍| 55/58 [00:02<00:00, 25.34it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:02<00:00, 28.58it/s]


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
    Generated text:  Daniel. I'm a computer science major and I'm in the high school math group. What's your name? Hey Daniel! I'm your friend, Daniel. How are you doing today? I'm enjoying my high school math class a lot, and I've been trying to help others learn math as well. Do you have any favorite subjects that you like to help others with? Yes, I love teaching and helping others learn math. Sometimes I get so excited about helping someone that I forget to take a break and take a deep breath. I'm always here to help and to offer my support. Do you have any favorite books or
    ===============================
    Prompt: The president of the United States is
    Generated text:  a relatively high-ranking official with significant power, and the term of office is generally longer than that of a general. The president is responsible for national defense, foreign affairs, and the management of state and local governments. The president and vice president must be from the same party, which is usually the Democratic Party. They hold office by the popular vote or by election, with no upper limit on the term of office. The president and vice president can only be chosen by the electorate and not by the President, Vice President, or other members of the government.
    Does this next sentence follow, given the above text?
    The president has the ability to
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. New York
    D. Tokyo
    To determine the capital of France, we need to recall the capital cities of the countries in the European Union. The European Union consists of 28 member states, and each country is represented by one of its member states. The capitals of the member states are listed in descending order of their population sizes.
    
    The European Union's member states are:
    1. France
    2. Germany
    3. Italy
    4. Spain
    5. Greece
    6. Belgium
    7. Netherlands
    8. Luxembourg
    9. Ireland
    10. Austria
    
    ===============================
    Prompt: The future of AI is
    Generated text:  unpredictable
    
    Technology is advancing at a breakneck pace, and the future of AI looks uncertain. What are some of the most optimistic predictions, and what are the key factors that will determine its success?
    
    As technology continues to advance at a rapid pace, the future of AI looks uncertain. While many experts have predicted that AI will become ubiquitous and transform every aspect of society, it is now clear that the field is still in its infancy and the technology is still evolving. As the technology advances, it will be essential to be aware of the potential risks and challenges that it may present.
    
    However, there are also optimistic predictions about the future of AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, positive description of your personality or skills]. I enjoy [insert a short, positive description of your hobbies or interests]. I'm always looking for ways to [insert a short, positive description of your goals or aspirations]. I'm always eager to learn and grow, and I'm always willing to help others. I'm a [insert a short, positive description of your personality or skills]. I'm always looking for ways to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination and a major hub for international business and diplomacy. Its rich history and diverse culture make it a fascinating city to explore. The city is home to many famous French artists, writers, and musicians, and is known for its vibrant nightlife and delicious cuisine. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. Its status as the world
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can better understand and respond to human needs and preferences.
    
    2. Enhanced privacy and security: As AI systems become more complex and sophisticated, there will be a growing need for measures to protect privacy and security. This could include measures such as encryption, data anonymization, and regular
    


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
    Generated text:  [Your Name], and I am a [Your Profession]. I am passionate about [Your Specialization or Area of Expertise]. In my free time, I enjoy [Your Interests or hobbies]. I am always looking for new challenges and opportunities to grow and develop my skills. So, please tell me about yourself, and I will be happy to share some of the interesting parts of your background with you. So, what is your background and what do you like to do? [Your Name] [Your Profession] [Your Specialization or Area of Expertise] [Your Interests or hobbies] [Your Background and interests] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its historic monuments, fashion industry, and rich culinary traditions. 
    
    [Formula]
    
    City: Paris
    Institution: Capitale de la France
    City Location: Paris, Île-de-France, France
    Capital: 12,536,821 Population: 11.2 million
    Primary Industry: Fashion, Tourism
    Seasonal Agricultural Products: Veggies, Grains
    Major Physical Features: River Seine, Montmartre
    Major Artistic Figures: Claude Monet, Pablo Picasso
    Major Sports: Tennis, Football (soccer)
    Municipal Districts: 
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be marked by a combination of significant progress and challenges. Here are some potential trends that may shape the technology's development in the coming years:
    
    1. Increased development of ethical AI: The rapid advancement of AI is expected to lead to the development of AI that is more ethical and responsible. This could include AI that can be programmed to prioritize the well-being of the environment, human rights, and equality. In addition, AI that can be programmed to make ethical decisions in situations that are beyond human control will be essential.
    
    2. Increased focus on AI for healthcare and disease prevention: As AI is used in healthcare to assist doctors in diagn


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

    ].

     I

     am

     a

     [

    Your

     Profession

    ]

     who

     loves

     [

    Your

     Area

     of

     Expert

    ise

     or

     Passion

    ].

     I

     enjoy

     [

    Your

     Enjoy

    ment

     or

     Hobby

    ]

     and

     have

     always

     been

     fascinated

     by

     [

    Your

     Hobby

    /

    Interest

    ].

     I

     am

     a

     [

    Your

     Age

    /

    Generation

    ]

     with

     [

    Your

     Personality

     Traits

    ].

     I

     strive

     to

     be

     [

    Your

     Goal

     or

     Goal

     to

     Achie

    ve

    ].

     I

     am

     always

     looking

     for

     new

     challenges

     and

     adventures

    .

     What

    's

     your

     favorite

     hobby

     or

     interest

    ,

     and

     why

     do

     you

     enjoy

     it

    ?


    [

    Your

     Name

    ]


    (

    Answer

     the

     question

    :

     What

    's

     your

     favorite

     hobby

     or

     interest

    ,

     and

     why

     do

     you

     enjoy

     it

    ?)


    My

     favorite

     hobby

     is

     hiking

    .

     I

     have

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     central

     region

     of

     the

     country

     and

     served

     as

     the

     seat

     of

     government

     and

     the

     economic

     and

     cultural

     center

     of

     the

     country

     for

     centuries

    .

     It

     is

     home

     to

     numerous

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     also

     features

     a

     rich

     history

     of

     art

    ,

     literature

    ,

     and

     architecture

    ,

     and

     is

     known

     for

     its

     diverse

     cultural

     offerings

    ,

     including

     world

    -class

     museums

    ,

     theaters

    ,

     and

     cafes

    .

     Overall

    ,

     Paris

     is

     a

     vital

     and

     vibrant

     urban

     center

     in

     France

     that

     plays

     a

     crucial

     role

     in

     the

     country

    ’s

     political

    ,

     cultural

    ,

     and

     economic

     landscape

    .

     
    


    Key

     Facts

     about

     Paris

    :
    


    -

     Population

    :

     Over

     

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     key

     trends

    ,

     including

    :
    


     

     

    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     into

     other

     technologies

    ,

     such

     as

     the

     Internet

     of

     Things

     (

    Io

    T

    )

     and

     the

     Internet

     of

     Things

     (

    Io

    T

    ),

     the

     complexity

     and

     complexity

     of

     AI

     systems

     will

     likely

     increase

    .


     

     

    2

    .

     Increased

     reliance

     on

     data

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     they

     will

     require

     more

     and

     more

     data

     to

     train

     and

     learn

    ,

     which

     will

     make

     it

     increasingly

     difficult

     for

     humans

     to

     collect

     and

     analyze

     data

    .

     This

     could

     lead

     to

     an

     increased

     focus

     on

     using

     more

     data

     to

     improve

     AI

     performance

    ,

     as

     well

     as

     an

     increased

     reliance

     on

     data

     to

     train

    



```python
llm.shutdown()
```
