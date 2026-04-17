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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-17 02:49:46] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.98it/s]


    2026-04-17 02:49:51,413 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 02:49:51] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 14.00it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 14.00it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 14.00it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 14.00it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:02<00:02, 14.00it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 14.00it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 14.00it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 14.00it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 14.00it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 14.00it/s]Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:03<00:02, 14.00it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 23.34it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 23.34it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 23.34it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 23.34it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 23.34it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 23.34it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 23.34it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 23.34it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 23.34it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 23.34it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:03<00:01, 23.34it/s]Compiling num tokens (num_tokens=192):  52%|█████▏    | 30/58 [00:03<00:01, 23.34it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 34.69it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:03<00:00, 34.69it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:03<00:00, 34.69it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 45.13it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 45.13it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 45.13it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 45.13it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 45.13it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 45.13it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 45.13it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 45.13it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.09 GB):   3%|▎         | 2/58 [00:00<00:02, 19.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.09 GB):   3%|▎         | 2/58 [00:00<00:02, 19.15it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.09 GB):   3%|▎         | 2/58 [00:00<00:02, 19.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.09 GB):   3%|▎         | 2/58 [00:00<00:02, 19.15it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.40it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.40it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.40it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.40it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.54it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.54it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.54it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.54it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.54it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.01it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.01it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.01it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.01it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.01it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.01it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.01it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.75it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.75it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.75it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.75it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.75it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.75it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.57it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.57it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.57it/s]Capturing num tokens (num_tokens=224 avail_mem=75.09 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.57it/s]Capturing num tokens (num_tokens=208 avail_mem=74.89 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.57it/s]Capturing num tokens (num_tokens=192 avail_mem=74.89 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.57it/s]Capturing num tokens (num_tokens=192 avail_mem=74.89 GB):  71%|███████   | 41/58 [00:01<00:00, 47.55it/s]Capturing num tokens (num_tokens=176 avail_mem=74.89 GB):  71%|███████   | 41/58 [00:01<00:00, 47.55it/s]Capturing num tokens (num_tokens=160 avail_mem=74.89 GB):  71%|███████   | 41/58 [00:01<00:00, 47.55it/s]Capturing num tokens (num_tokens=144 avail_mem=74.88 GB):  71%|███████   | 41/58 [00:01<00:00, 47.55it/s]Capturing num tokens (num_tokens=128 avail_mem=74.88 GB):  71%|███████   | 41/58 [00:01<00:00, 47.55it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.88 GB):  71%|███████   | 41/58 [00:01<00:00, 47.55it/s]Capturing num tokens (num_tokens=96 avail_mem=74.87 GB):  71%|███████   | 41/58 [00:01<00:00, 47.55it/s] Capturing num tokens (num_tokens=96 avail_mem=74.87 GB):  81%|████████  | 47/58 [00:01<00:00, 48.32it/s]Capturing num tokens (num_tokens=80 avail_mem=74.87 GB):  81%|████████  | 47/58 [00:01<00:00, 48.32it/s]Capturing num tokens (num_tokens=64 avail_mem=74.87 GB):  81%|████████  | 47/58 [00:01<00:00, 48.32it/s]Capturing num tokens (num_tokens=48 avail_mem=74.86 GB):  81%|████████  | 47/58 [00:01<00:00, 48.32it/s]Capturing num tokens (num_tokens=32 avail_mem=74.86 GB):  81%|████████  | 47/58 [00:01<00:00, 48.32it/s]Capturing num tokens (num_tokens=28 avail_mem=74.85 GB):  81%|████████  | 47/58 [00:01<00:00, 48.32it/s]Capturing num tokens (num_tokens=24 avail_mem=74.85 GB):  81%|████████  | 47/58 [00:01<00:00, 48.32it/s]Capturing num tokens (num_tokens=24 avail_mem=74.85 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.86it/s]Capturing num tokens (num_tokens=20 avail_mem=74.85 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.86it/s]Capturing num tokens (num_tokens=16 avail_mem=74.85 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.86it/s]Capturing num tokens (num_tokens=12 avail_mem=74.84 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.86it/s]

    Capturing num tokens (num_tokens=8 avail_mem=74.84 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.86it/s] Capturing num tokens (num_tokens=4 avail_mem=74.84 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.86it/s]Capturing num tokens (num_tokens=4 avail_mem=74.84 GB): 100%|██████████| 58/58 [00:01<00:00, 43.20it/s]


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
    Generated text:  Georgia and I am 23 years old. I have had a lot of experience in the military and in my community. My last name is Johnson. I have been working as a camp counselor for over 5 years now. I have been called to work at Camp Johnson to help kids with autism. One of the kids that I worked with was a boy named Nathan. He was a huge fan of the FC Tampa Bay Rays because of how much he enjoyed baseball and the fact that he was on a team. He really enjoyed spending time with me and my wife and it was a nice change from the normal work environment. However, after
    ===============================
    Prompt: The president of the United States is
    Generated text:  34 years older than the president of Brazil, and the president of Brazil is 25 years younger than the president of Russia. If the president of Russia is currently 20 years old, how old will the president of Russia be in 5 years?
    To determine the current age of the president of Russia, we start by noting the ages of the presidents of the United States, Brazil, and Russia given in the problem.
    
    1. The president of the United States is currently 34 years old.
    2. The president of Brazil is 25 years younger than the president of Russia. Therefore, the president of Brazil
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, located in the south of France. The city has a population of 2,169,000 (2020) and is the largest in France. The city is divided into 6 administrative regions: Paris I, Paris II, Paris III, Paris IV, Paris V, and Paris VI. The population of Paris was 1,066,000 in 2019. The district of Paris is composed of the six administrative regions and contains 64,392 inhabitants (2016). The population of the districts is made up of the populations of
    ===============================
    Prompt: The future of AI is
    Generated text:  in the air
    The future of AI is in the air and it will go far beyond the expectations of today’s tech workers. With the increasing number of jobs that are predicted to be automated, the need for more competent AI developers is becoming increasingly important. The technology is rapidly advancing, and there is no doubt that the future of AI will be exciting and interesting.
    In the coming years, AI will likely become a more integral part of our daily lives, from helping us with everyday tasks to assisting in decision-making processes. It will be a major driver of change in the way we work, communicate, and interact with each other.
    AI is


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic, or neutral description of yourself]. I enjoy [insert a short, positive, enthusiastic, or neutral description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What do you like to do in your free time? I enjoy [insert a short, positive, enthusiastic, or neutral description of your hobbies or interests]. I also enjoy [insert a short,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville-Marie" and "La Grande-Bretagne." It is the largest city in France and the third-largest city in the world by population. Paris is known for its rich history, art, and culture, and is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major center for business, finance, and tourism in France. Paris is a UNESCO World Heritage site and is a popular tourist destination for visitors from around the world. The city is also home to many important institutions such as the French Academy of Sciences,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs and preferences.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead to more robust and transparent AI
    


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
    Generated text:  [Name], and I'm a 28-year-old software engineer with a passion for open source software. I'm currently working as a Product Manager for [Company Name], where I lead the development of new software products and ensure that the company's products meet the needs of our customers. I'm always looking for ways to improve my skills and work on projects that challenge me. What's your favorite thing to do? I'm always up for a challenge, whether it's working on a difficult problem or trying something new in a field I'm passionate about. Thank you for asking! 😊 #SelfIntro
    
    As an AI language model
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower, historical landmarks such as Notre-Dame Cathedral, and rich cultural heritage. It is a major metropolis with a rich history and diverse population. Paris is often referred to as the "City of Light" and a melting pot of cultures. Its location near the Mediterranean Sea makes it a bustling port city, with attractions such as the Louvre Museum and the Opera Garnier. Paris is a modern city with a strong emphasis on sustainability and modernization. It is a highly livable city with a vibrant nightlife and a lively atmosphere. Paris is the second largest city in the European Union and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and highly dependent on technology, trends, and regulations. Some possible future trends in AI include:
    
    1. Personalization: AI systems will become more personalized to improve user experience and target specific user groups with tailored services.
    
    2. Autonomous systems: Autonomous vehicles, drones, and robots will become more prevalent, reducing human error and increasing efficiency.
    
    3. Artificial intelligence in healthcare: AI will be used to help diagnose and treat diseases, reduce costs, and improve patient outcomes.
    
    4. AI in finance: AI will be used to improve risk assessment, fraud detection, and investment strategies, and to provide more accurate predictions about future financial trends.
    
    5


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

     a

     [

    insert

     age

     range

    ]

     year

    -old [

    insert

     occupation

    ]

     who

     is

     passionate

     about

     [

    insert

     a

     unique

     hobby

     or

     interest

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

     and

     grow

     as

     a

     person

    .

     I

     believe

     in

     the

     power

     of

     self

    -im

    pro

    vement

     and

     believe

     that

     with

     hard

     work

     and

     dedication

    ,

     anything

     is

     possible

    .

     How

     would

     you

     describe

     your

     personality

    ?

     I

    'm

     outgoing

    ,

     independent

    ,

     and

     always

     ready

     to

     make

     new

     friends

     and

     try

     new

     things

    .

     What

     are

     your

     hobbies

     and

     interests

     outside

     of

     work

    ?

     My

     favorite

     hobbies

     include

     reading

    ,

     playing

     sports

    ,

     and

     trying

     new

     food

    .

    
    
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

     the

     third

     largest

     city

     in

     the

     world

     by

     population

    ,

     with

     a

     population

     of

     over

     

    2

    .

    7

     million

     inhabitants

    .

     The

     city

     is

     located

     in

     the

     Mar

    ne

     département

     of

     the

     Lo

    ire

    -

    Atl

    ant

    ique

     region

    .

     It

     is

     a

     city

     with

     a

     long

     and

     rich

     history

    ,

     known

     for

     its

     stunning

     architecture

    ,

     its

     historical

     sites

    ,

     and

     its

     vibrant

     culture

    .

     The

     city

     is

     also

     famous

     for

     its

     annual

     Paris

    ian

     festivals

    ,

     including

     the

     famous

     E

    iff

    el

     Tower

     party

     on

     Bast

    ille

     Day

    .

     Paris

     is

     a

     city

     of

     contrasts

     and

     beauty

    ,

     with

     its

     modern

     and

     ancient

     landmarks

    ,

     cultural

     offerings

    ,

     and

     vibrant

     life

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     variety

     of

     developments

     and

     trends

    ,

     including

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     analyze

     medical

     images

     and

     assist

     doctors

     in

     diagn

    osing

     diseases

    .

     It

     is

     likely

     that

     this

     trend

     will

     continue

    ,

     with

     more

     AI

    -based

     tools

     being

     developed

     to

     assist

     with

     medical

     research

     and

     treatment

    .
    


    2

    .

     Emer

    gence

     of

     AI

     in

     finance

    :

     AI

     is

     already

     being

     used

     in

     financial

     fraud

     detection

     and

     risk

     management

    ,

     and

     it

     is

     likely

     that

     this

     trend

     will

     continue

    ,

     with

     more

     AI

    -based

     tools

     being

     developed

     to

     improve

     the

     accuracy

     of

     financial

     predictions

     and

     investments

    .
    


    3

    .

     Integration

     of

     AI

     with

     IoT

    :

     AI

     is

     already

     being

     used

     to

     monitor

     and

     control

    



```python
llm.shutdown()
```
