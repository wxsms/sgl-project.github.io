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
    [2026-04-17 03:27:27] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.83it/s]


    2026-04-17 03:27:31,858 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 03:27:31] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.37it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.37it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.89it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.89it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.89it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.89it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.89it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:02<00:06,  6.89it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:02<00:06,  6.89it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:02<00:06,  6.89it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:02<00:06,  6.89it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:02<00:06,  6.89it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:02<00:06,  6.89it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:02<00:02, 14.94it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:02<00:02, 14.94it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:02, 14.94it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:03<00:02, 14.94it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:03<00:02, 14.94it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:03<00:02, 14.94it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:03<00:02, 14.94it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:03<00:02, 14.94it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 19.89it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 19.89it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 19.89it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 19.89it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 19.89it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 19.89it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 19.89it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 19.89it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 19.89it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 19.89it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:03<00:00, 28.88it/s] 

    Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:03<00:00, 28.88it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.97it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.97it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.97it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.97it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.97it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.97it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.97it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.97it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.97it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.97it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.97it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.65 GB):   3%|▎         | 2/58 [00:00<00:03, 15.22it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.65 GB):   3%|▎         | 2/58 [00:00<00:03, 15.22it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.64 GB):   3%|▎         | 2/58 [00:00<00:03, 15.22it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.64 GB):   7%|▋         | 4/58 [00:00<00:03, 15.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.65 GB):   7%|▋         | 4/58 [00:00<00:03, 15.99it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.64 GB):   7%|▋         | 4/58 [00:00<00:03, 15.99it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.64 GB):  10%|█         | 6/58 [00:00<00:03, 17.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.64 GB):  10%|█         | 6/58 [00:00<00:03, 17.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.64 GB):  10%|█         | 6/58 [00:00<00:03, 17.25it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=53.64 GB):  10%|█         | 6/58 [00:00<00:03, 17.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.63 GB):  10%|█         | 6/58 [00:00<00:03, 17.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.63 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.63 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.63 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.62 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.62 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.29it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.62 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.29it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.62 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.61 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.61 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.49it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=53.61 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.60 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.60 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.60 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.98it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.58 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.98it/s]Capturing num tokens (num_tokens=960 avail_mem=53.59 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.98it/s] Capturing num tokens (num_tokens=896 avail_mem=53.59 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.98it/s]Capturing num tokens (num_tokens=832 avail_mem=53.59 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.98it/s]Capturing num tokens (num_tokens=768 avail_mem=53.58 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.98it/s]Capturing num tokens (num_tokens=768 avail_mem=53.58 GB):  43%|████▎     | 25/58 [00:00<00:00, 40.15it/s]Capturing num tokens (num_tokens=704 avail_mem=53.58 GB):  43%|████▎     | 25/58 [00:00<00:00, 40.15it/s]Capturing num tokens (num_tokens=640 avail_mem=53.58 GB):  43%|████▎     | 25/58 [00:00<00:00, 40.15it/s]

    Capturing num tokens (num_tokens=576 avail_mem=53.58 GB):  43%|████▎     | 25/58 [00:00<00:00, 40.15it/s]Capturing num tokens (num_tokens=512 avail_mem=53.57 GB):  43%|████▎     | 25/58 [00:00<00:00, 40.15it/s]Capturing num tokens (num_tokens=480 avail_mem=53.58 GB):  43%|████▎     | 25/58 [00:00<00:00, 40.15it/s]Capturing num tokens (num_tokens=480 avail_mem=53.58 GB):  52%|█████▏    | 30/58 [00:00<00:00, 39.46it/s]Capturing num tokens (num_tokens=448 avail_mem=53.58 GB):  52%|█████▏    | 30/58 [00:00<00:00, 39.46it/s]Capturing num tokens (num_tokens=416 avail_mem=53.58 GB):  52%|█████▏    | 30/58 [00:00<00:00, 39.46it/s]

    Capturing num tokens (num_tokens=384 avail_mem=53.58 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.46it/s]

    Capturing num tokens (num_tokens=352 avail_mem=53.57 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=320 avail_mem=53.57 GB):  52%|█████▏    | 30/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=320 avail_mem=53.57 GB):  60%|██████    | 35/58 [00:01<00:01, 14.91it/s]Capturing num tokens (num_tokens=288 avail_mem=53.56 GB):  60%|██████    | 35/58 [00:01<00:01, 14.91it/s]

    Capturing num tokens (num_tokens=256 avail_mem=53.56 GB):  60%|██████    | 35/58 [00:01<00:01, 14.91it/s]Capturing num tokens (num_tokens=240 avail_mem=53.56 GB):  60%|██████    | 35/58 [00:01<00:01, 14.91it/s]Capturing num tokens (num_tokens=240 avail_mem=53.56 GB):  66%|██████▌   | 38/58 [00:01<00:01, 16.33it/s]Capturing num tokens (num_tokens=224 avail_mem=53.55 GB):  66%|██████▌   | 38/58 [00:01<00:01, 16.33it/s]Capturing num tokens (num_tokens=208 avail_mem=53.55 GB):  66%|██████▌   | 38/58 [00:01<00:01, 16.33it/s]Capturing num tokens (num_tokens=192 avail_mem=53.55 GB):  66%|██████▌   | 38/58 [00:01<00:01, 16.33it/s]Capturing num tokens (num_tokens=176 avail_mem=53.55 GB):  66%|██████▌   | 38/58 [00:01<00:01, 16.33it/s]Capturing num tokens (num_tokens=160 avail_mem=53.54 GB):  66%|██████▌   | 38/58 [00:01<00:01, 16.33it/s]Capturing num tokens (num_tokens=160 avail_mem=53.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 21.26it/s]Capturing num tokens (num_tokens=144 avail_mem=53.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 21.26it/s]Capturing num tokens (num_tokens=128 avail_mem=53.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 21.26it/s]

    Capturing num tokens (num_tokens=112 avail_mem=53.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 21.26it/s]Capturing num tokens (num_tokens=96 avail_mem=53.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 21.26it/s] Capturing num tokens (num_tokens=80 avail_mem=53.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 21.26it/s]Capturing num tokens (num_tokens=80 avail_mem=53.53 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.01it/s]Capturing num tokens (num_tokens=64 avail_mem=53.52 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.01it/s]Capturing num tokens (num_tokens=48 avail_mem=53.52 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.01it/s]Capturing num tokens (num_tokens=32 avail_mem=53.52 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.01it/s]Capturing num tokens (num_tokens=28 avail_mem=53.51 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.01it/s]Capturing num tokens (num_tokens=24 avail_mem=53.51 GB):  83%|████████▎ | 48/58 [00:02<00:00, 26.01it/s]Capturing num tokens (num_tokens=24 avail_mem=53.51 GB):  91%|█████████▏| 53/58 [00:02<00:00, 30.63it/s]Capturing num tokens (num_tokens=20 avail_mem=53.50 GB):  91%|█████████▏| 53/58 [00:02<00:00, 30.63it/s]Capturing num tokens (num_tokens=16 avail_mem=53.50 GB):  91%|█████████▏| 53/58 [00:02<00:00, 30.63it/s]

    Capturing num tokens (num_tokens=12 avail_mem=53.50 GB):  91%|█████████▏| 53/58 [00:02<00:00, 30.63it/s]Capturing num tokens (num_tokens=8 avail_mem=53.50 GB):  91%|█████████▏| 53/58 [00:02<00:00, 30.63it/s] Capturing num tokens (num_tokens=4 avail_mem=53.49 GB):  91%|█████████▏| 53/58 [00:02<00:00, 30.63it/s]Capturing num tokens (num_tokens=4 avail_mem=53.49 GB): 100%|██████████| 58/58 [00:02<00:00, 34.72it/s]Capturing num tokens (num_tokens=4 avail_mem=53.49 GB): 100%|██████████| 58/58 [00:02<00:00, 26.14it/s]


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
    Generated text:  Mac and I'm a high school student. I'm really good at reading and writing. But I'm not good at solving problems. I'm just a little bit confused about how to approach them. What should I do? Should I try to solve all the problems on my own, or should I ask my teacher for help? Also, what kind of questions should I ask my teacher to make sure I understand what I'm doing?
    
    It's frustrating because I think I'm really good at reading and writing but I'm not good at solving problems. I would really like to improve in math and be a better student. Can you help me
    ===============================
    Prompt: The president of the United States is
    Generated text:  a government official. That official is a very important leader in the country. He or she is the head of government and the commander-in-chief of the armed forces. The president is the leader of the United States, and in many countries around the world, the president is the head of government. The president is the head of the executive branch of the government. The executive branch is the branch of the government that deals with getting the day-to-day business of the government done, and making sure that the people are treated fairly. The president makes the laws that the other branches of the government have to follow, and he or she appoints other
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located in the south of France, close to the Mediterranean Sea. It is a unique city that has been rebuilt many times over the centuries. It has a rich history and is a major port city. The city has been occupied for thousands of years, and was a major city for most of the 5th and 6th centuries. It was also the capital of France during the 13th century and the 16th century. The present city has been a major port city since the 14th century, but it was rebuilt in the 18th century. The city is still undergoing
    ===============================
    Prompt: The future of AI is
    Generated text:  just starting
    
    Nvidia's Peter Wilmshurst, who is often referred to as the "father of AI", discusses how the field is evolving and what it means for the future.
    
    Peter Wilmshurst, the father of AI and one of the founders of the Neural Network Lab at Nvidia, talks about how the field of AI is evolving and what it means for the future.
    
    Wilmshurst believes that AI is currently at a crucial point in its evolution. He is optimistic that the development of AI will continue to advance in the coming years, and he is confident that it will be used in a wide range of applications across


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your character, such as "funny, witty, and always up for a good laugh."]. I enjoy [insert a short description of your character's interests or hobbies, such as "reading, playing sports, or cooking."]. I'm always looking for new experiences and learning new things, and I'm eager to explore new opportunities. What's your favorite hobby or activity? I love [insert a short description
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also a popular tourist destination and a major economic center. Paris is home to many famous French artists, writers, and musicians, and is a cultural hub for Europe. The city is also known for its rich history and diverse population, which has contributed to its status as a major global city. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a city that has played a significant role in shaping French culture and identity for centuries. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    2. Greater integration with human decision-making: AI is likely to become more integrated with human decision-making, allowing for more complex and nuanced decision-making. This could lead to more effective and efficient use of AI in various industries.
    
    3. Increased use of
    


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
    Generated text:  [Name] and I am a/an [Title] at [Company Name]. I come from a[country, city, or region]. I've always been passionate about [why you like what you do] and I'm very [insert any relevant personality trait or characteristic]. I'm excited to dive into [job title] and work with [specific company or organization], doing [specific task or project]. Please let me know if you have any questions about my background, experience, or goals. I'm looking forward to our first meeting! 🌟 #JobSeeker #Internship #CareerStart #ReadyToJoin
    [Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, often referred to as the "City of Light" and is the largest city in Europe. It is located on the Seine River and is home to the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and many other world-famous landmarks. Paris is known for its rich history, culture, and cuisine, and has been a major center for French politics, arts, and culture for over a millennium. Today, the city remains one of the most important and vibrant cities in Europe, attracting millions of visitors each year. 
    
    [Place France's capital city on a map with the capital city as the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and diverse, with many exciting developments on the horizon. Here are some of the most promising trends we can expect to see in the coming years:
    
    1. Better natural language processing: With the continuous growth of natural language processing, we can expect to see even more sophisticated AI-powered tools that can understand and generate human-like speech and text. This could lead to the creation of voice assistants and virtual assistants that can communicate more effectively with humans.
    
    2. Increased AI transparency: AI systems are becoming more complex and sophisticated, so we can expect to see greater transparency in how they are designed, trained, and used. This will make it easier for


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

    ].

     I

     am

     [

    insert

     character

    's

     age

    ].

     I

     am

     [

    insert

     character

    's

     profession

    ].

     I

     am

     a

     [

    insert

     character

    's

     favorite

     hobby

     or

     activity

    ].

     I

     enjoy

     [

    insert

     one

     or

     two

     things

     that

     make

     me

     happy

    ].

     I

     have

     a

     love

     for

     [

    insert

     one

     or

     two

     things

     that

     make

     me

     passionate

     about

    ].

     I

     love

     [

    insert

     one

     or

     two

     things

     that

     make

     me

     inspire

     others

    ].

     I

     am

     [

    insert

     character

    's

     personality

     trait

    ].

     I

     have

     a

     friendly

     and

     helpful

     personality

    ,

     always

     ready

     to

     help

     those

     in

     need

    .

     I

     have

     a

     deep

     and

     kind

     heart

     that

     makes

     me

     feel

     happy

     every

     day

    .

     I

     am

     [

    insert

     character

    's

     positive

     trait

    ].

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     

    2

    1

    st

     largest

     city

     in

     the

     world

    .

     It

     is

     the

     most

     populous

     city

     of

     France

     and

     has

     a

     population

     of

     around

     

    7

    .

    5

     million

     people

    .
    


    The

     name

     "

    Paris

    "

     comes

     from

     the

     ancient

     French

     word

     "

    par

    "

     meaning

     "

    city

    "

     and

     "

    sal

    "

     meaning

     "

    salt

    ,"

     which

     combined

     mean

     "

    the

     salt

     city

    ."
    


    Paris

     is

     a

     cosm

    opolitan

     city

     with

     a

     rich

     history

     dating

     back

     to

     the

     Middle

     Ages

    ,

     which

     has

     been

     influenced

     by

     various

     civilizations

     throughout

     its

     history

    .

     It

     is

     also

     known

     for

     its

     art

    ,

     architecture

    ,

     fashion

    ,

     food

    ,

     and

     wine

    .

     The

     city

     is

     also

     home

     to

     many

     notable

     landmarks

    ,

     such

     as

     the

     E

    iff

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     filled

     with

     incredible

     possibilities

     and

     challenges

    .

     Here

     are

     some

     possible

     trends

     we

     can

     expect

     to

     see

     in

     the

     next

     decade

    :
    


    1

    .

     Increased

     autonomy

     and

     self

    -aware

    ness

    :

     AI

     is

     getting

     better

     at

     understanding

     and

     making

     decisions

     without

     being

     explicitly

     programmed

    .

     This

     could

     lead

     to

     more

     autonomous

     machines

     that

     can

     make

     their

     own

     decisions

     and

     learn

     from

     their

     experiences

    .
    


    2

    . Integration

     with

     other

     technologies

    :

     AI

     is

     becoming

     more

     integrated

     with

     other

     technologies

     such

     as

     machine

     learning

    ,

     computer

     vision

    ,

     and

     natural

     language

     processing

    .

     We

     can

     expect

     to

     see

     more

     sophisticated

     applications

     of

     AI

     in

     fields

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    .
    


    3

    .

     Personal

    ization

     and

     adapt

    ability

    :

     AI

     is

     being

     used

     to

     personalize

     the

    



```python
llm.shutdown()
```
