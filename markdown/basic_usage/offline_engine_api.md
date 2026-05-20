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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.96it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.95it/s]


    2026-05-20 22:32:43,127 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 22:32:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:45,  3.96s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:45,  3.96s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:45,  3.96s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:45,  3.96s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:45,  3.96s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.86it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.86it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.83it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.83it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.83it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.83it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.83it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.83it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.83it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.83it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.83it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.83it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.30it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.30it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.30it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.30it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.30it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.30it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.30it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.30it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.30it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.30it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.30it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.91it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.91it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.91it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.91it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.91it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.91it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.91it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.91it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.91it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.91it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.91it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=70.55 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.52 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.52 GB):   3%|▎         | 2/58 [00:00<00:03, 16.29it/s]Capturing num tokens (num_tokens=7168 avail_mem=70.51 GB):   3%|▎         | 2/58 [00:00<00:03, 16.29it/s]Capturing num tokens (num_tokens=6656 avail_mem=70.51 GB):   3%|▎         | 2/58 [00:00<00:03, 16.29it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=70.51 GB):   7%|▋         | 4/58 [00:00<00:03, 16.67it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.51 GB):   7%|▋         | 4/58 [00:00<00:03, 16.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=70.51 GB):   7%|▋         | 4/58 [00:00<00:03, 16.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.50 GB):   7%|▋         | 4/58 [00:00<00:03, 16.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.50 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.49 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.49 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.86it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=70.49 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.48 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.48 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.48 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.07it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.48 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.48 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.47 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.47 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.33it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.47 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.33it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.47 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.33it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=70.46 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.46 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.46 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.46 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.15it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.44 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.15it/s]Capturing num tokens (num_tokens=960 avail_mem=70.45 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.15it/s] Capturing num tokens (num_tokens=896 avail_mem=70.45 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.15it/s]Capturing num tokens (num_tokens=896 avail_mem=70.45 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.52it/s]Capturing num tokens (num_tokens=832 avail_mem=70.45 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.52it/s]Capturing num tokens (num_tokens=768 avail_mem=70.44 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.52it/s]

    Capturing num tokens (num_tokens=704 avail_mem=70.44 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.52it/s]Capturing num tokens (num_tokens=640 avail_mem=70.44 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.52it/s]Capturing num tokens (num_tokens=576 avail_mem=70.44 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.52it/s]Capturing num tokens (num_tokens=576 avail_mem=70.44 GB):  48%|████▊     | 28/58 [00:00<00:00, 32.71it/s]Capturing num tokens (num_tokens=512 avail_mem=70.42 GB):  48%|████▊     | 28/58 [00:00<00:00, 32.71it/s]Capturing num tokens (num_tokens=480 avail_mem=70.44 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.71it/s]Capturing num tokens (num_tokens=448 avail_mem=70.43 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.71it/s]Capturing num tokens (num_tokens=416 avail_mem=70.43 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.71it/s]Capturing num tokens (num_tokens=384 avail_mem=70.43 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.71it/s]

    Capturing num tokens (num_tokens=384 avail_mem=70.43 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=352 avail_mem=70.42 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=320 avail_mem=70.42 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=288 avail_mem=70.42 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=256 avail_mem=70.41 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=256 avail_mem=70.41 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=240 avail_mem=70.41 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=224 avail_mem=70.41 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=208 avail_mem=70.40 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=192 avail_mem=70.40 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=176 avail_mem=70.40 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.75it/s]

    Capturing num tokens (num_tokens=176 avail_mem=70.40 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=160 avail_mem=70.40 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=144 avail_mem=70.39 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=128 avail_mem=70.39 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=112 avail_mem=70.39 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.36it/s]Capturing num tokens (num_tokens=112 avail_mem=70.39 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.38it/s]Capturing num tokens (num_tokens=96 avail_mem=70.39 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.38it/s] Capturing num tokens (num_tokens=80 avail_mem=70.38 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.38it/s]Capturing num tokens (num_tokens=64 avail_mem=70.38 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.38it/s]

    Capturing num tokens (num_tokens=48 avail_mem=70.37 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.38it/s]Capturing num tokens (num_tokens=32 avail_mem=70.37 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.38it/s]Capturing num tokens (num_tokens=32 avail_mem=70.37 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=28 avail_mem=76.11 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=24 avail_mem=76.11 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=20 avail_mem=76.11 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=16 avail_mem=76.11 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.58it/s]

    Capturing num tokens (num_tokens=12 avail_mem=76.10 GB):  88%|████████▊ | 51/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=12 avail_mem=76.10 GB):  97%|█████████▋| 56/58 [00:01<00:00, 34.71it/s]Capturing num tokens (num_tokens=8 avail_mem=76.10 GB):  97%|█████████▋| 56/58 [00:01<00:00, 34.71it/s] Capturing num tokens (num_tokens=4 avail_mem=76.10 GB):  97%|█████████▋| 56/58 [00:01<00:00, 34.71it/s]Capturing num tokens (num_tokens=4 avail_mem=76.10 GB): 100%|██████████| 58/58 [00:01<00:00, 32.09it/s]


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
    Generated text:  Cui Chuang. I’m 34 years old, and my hobby is watching TV. I have several hobbies like reading, drawing, and playing sports. My wife and I have been happily married for 18 years. Our child is 14 years old. Now I want to ask my wife if she is interested in the video games. When should I ask her? 
    When should I ask her if she is interested in the video games? [Multiple Choice]
    A. In the first year of marriage
    B. In the second year of marriage
    C. In the third year of marriage
    D. In the
    ===============================
    Prompt: The president of the United States is
    Generated text:  25 years older than the president of Illinois, and the president of Illinois is half the age of the president of Texas. If the president of Texas is 50 years old, how old is the president of the United States?
    To determine the age of the president of the United States, we need to follow a step-by-step approach using the given information.
    
    1. Identify the age of the president of Texas:
       \[
       \text{Age of the president of Texas} = 50 \text{ years old}
       \]
    
    2. Determine the age of the president of Illinois:
       \[
       \
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, a city that has had a history of many different people and events. There are many people who have lived in Paris since 1170, when it was a royal capital. The history of Paris is quite long. The first people who lived in the area were people who came from England and Germany to escape religious persecution. After the 16th century, the people of Paris were Catholic.
    In 1250, a group of people came from Paris to stay in the area for 20 years. This group was called the First Church. They were sponsored by the King of France, Charles VII.
    ===============================
    Prompt: The future of AI is
    Generated text:  bright and it is evolving much faster than we can keep up with. It is anticipated that many of the AI technologies that we know today will have become obsolete in a few years. Here are some of the most anticipated technologies that will emerge in the coming decade. Let us see how AI will transform the way we live, work and play.
    
    1. Self-driving cars
    
    Self-driving cars are already here. The technology is on its way to become more advanced and more widely available. They will be available in the markets in the next few years. The journey to this will be a long one, however. It will take a while before you


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character, such as "a friendly, outgoing, and detail-oriented person" or "a creative, innovative, and passionate individual".] I enjoy [insert a short description of your hobbies or interests, such as "reading, cooking, or playing sports".] I'm always looking for new experiences and learning new things, and I'm always eager to share my knowledge and insights with others. What's your favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the seat of the French government and the country's cultural and political center. Paris is a major tourist destination and a popular tourist destination, attracting millions of visitors each year. It is also known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the development of modern France. Paris is a city of contrasts, with its elegant architecture, vibrant nightlife, and diverse cultural scene. It is a city that is steeped in history and culture,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced decision-making. This could lead to more personalized and context-aware AI that can better understand and respond to human emotions and behaviors.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be increased concerns about privacy and security. This could lead to more robust privacy protections and security measures to ensure that AI systems are not used for malicious purposes.
    
    3. Greater automation and efficiency: AI is likely to become more integrated with human intelligence, leading to
    


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
    Generated text:  [Name] and I'm a [job title] at [company name]. I'm currently [short duration since joining the company]. And I'm... [specific details about the character]. Excuse me, I was wondering if you could tell me more about your job title and company? Also, could you tell me a little bit about your background? I'm a [specific background or experiences] and I'm here to learn more about your company and get to know you better. How can I assist you today? And what's your daily routine like? I'm looking forward to hearing more about you. Hello, my name is [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city that serves as the heart and hub of the nation's culture, economy, and politics. Home to over a million residents, Paris is a vibrant, cosmopolitan metropolis with a rich history and diverse culture that has made it one of the world's most important cities for several centuries. Its iconic landmarks such as Notre Dame Cathedral, the Eiffel Tower, and the Louvre Museum, as well as its numerous museums, art galleries, and festivals, showcase its unique charm and cultural richness. As a global city, Paris is also known for its cuisine, fashion, and wine production, making it a world-renowned
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a number of trends and developments. Here are some possible future trends in AI:
    
    1. Increased use of AI in healthcare: AI can be used to improve the accuracy and efficiency of medical diagnosis and treatment, allowing for faster and more personalized care.
    
    2. Greater reliance on AI in automation: As AI becomes more capable of performing tasks that involve data analysis and decision-making, more of these tasks will be automated, freeing up human workers to focus on more complex and creative activities.
    
    3. Integration of AI with other technologies: AI will continue to integrate more with other technologies such as machine learning, natural language processing, and


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

    ...

     (

    insert

     your

     name

    )

     and

     I

    'm

     a

    /an

     (

    insert

     your

     occupation

     or

     profession

    )

     with

     a

    /an

     (

    insert

     your

     highest

     qualification

     or

     field

     of

     study

    )

     that

     has

     been

     in

     the

     field

     for

     (

    insert

     number

     of

     years

    )

     years

    .

     I

     believe

     that

     my

     experience

     in

     [

    insert

     your

     field

     or

     industry

    ]

     has

     helped

     me

     to

     develop

     my

     unique

     perspective

     and

     problem

    -solving

     skills

    ,

     which

     are

     essential

     for

     this

     position

    .

     I

     am

     confident

     that

     I

     would

     be

     a

     valuable

     addition

     to

     your

     team

     and

     am

     excited

     to

     bring

     my

     background

     and

     experience

     to

     your

     organization

    .

     If

     you

     have

     any

     questions

     or

     if

     you

     would

     like

     to

     learn

     more

     about

     me

    ,

     please

     don

    't

     hesitate

     to

     ask

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     city

     of

     France

     and

     serves

     as

     the

     political

    ,

     cultural

    ,

     and

     economic

     center

     of

     the

     country

    .

     It

     is

     known

     for

     its

     iconic

     architecture

    ,

     vibrant

     street

     life

    ,

     and

     rich

     history

    .

     Paris

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

     museums

    .

     The

     city

     is

     also

     known

     for

     its

     cuisine

    ,

     including

     French

     cuisine

    ,

     and

     its

     role

     in

     the

     French

    -speaking

     world

    .

     In

     addition

    ,

     Paris

     is

     a

     cultural

     hub

     and

     the

     birth

    place

     of

     several

     world

    -f

    amous

     artists

    ,

     including

     Pablo

     Picasso

    ,

     Co

    lette

    ,

     and

     Jean

    -P

    aul

     S

    art

    re

    .

     With

     its

     diverse

     neighborhoods

     and

     picturesque

     landscapes

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     currently

     focused

     on

     how

     to

     make

     it

     more

     efficient

    ,

     accurate

    ,

     and

     reliable

    ,

     as

     well

     as

     how

     to

     integrate

     it

     into

     a

     wide

     range

     of

     applications

    .

     Here

     are

     some

     possible

     future

     trends

    :
    


    1

    .

     Increased

     Personal

    ization

    :

     AI

     is

     already

     becoming

     more

     personalized

     than

     ever

     before

    ,

     and

     in

     the

     future

    ,

     it

     will

     become

     even

     more

     so

    .

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     be

     able

     to

     learn

     from

     the

     behavior

     and

     preferences

     of

     users

    ,

     providing

     more

     tailored

     and

     efficient

     results

    .
    


    2

    .

     AI

     in

     Healthcare

    :

     AI

     is

     already

     being

     used

     in

     the

     healthcare

     industry

     to

     diagnose

     and

     treat

     diseases

    ,

     and

     in

     the

     future

    ,

     it

     will

     be

     able

     to

     provide

     even

     more

     advanced

     and

     personalized

    



```python
llm.shutdown()
```
