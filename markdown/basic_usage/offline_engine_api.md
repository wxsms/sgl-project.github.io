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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.76it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.76it/s]


    2026-04-09 08:00:14,116 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 08:00:14] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:52,  3.02s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:52,  3.02s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<02:52,  3.02s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<02:52,  3.02s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:32,  1.67it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:32,  1.67it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:32,  1.67it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:32,  1.67it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:32,  1.67it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:32,  1.67it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:32,  1.67it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:09,  5.14it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.03it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.03it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 17.78it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 17.78it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 17.78it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 17.78it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 17.78it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 17.78it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 17.78it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 17.78it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:03<00:01, 17.78it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 25.23it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 25.23it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 25.23it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 25.23it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 25.23it/s]

    Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:00, 25.23it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:03<00:00, 25.23it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:03<00:00, 25.23it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 31.60it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 31.60it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 31.60it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 31.60it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 31.60it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 31.60it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 31.60it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:03<00:00, 31.60it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 37.94it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 37.94it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 37.94it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 37.94it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 37.94it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 37.94it/s]

    Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 37.94it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 37.94it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 37.94it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 37.94it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 37.94it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.97it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.92it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.92it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.92it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.92it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.92it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.98it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.98it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.98it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.98it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 31.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 31.99it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 31.99it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 31.99it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 31.99it/s] Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:01, 20.56it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:01, 20.56it/s]Capturing num tokens (num_tokens=832 avail_mem=76.69 GB):  38%|███▊      | 22/58 [00:01<00:01, 20.56it/s]

    Capturing num tokens (num_tokens=768 avail_mem=76.68 GB):  38%|███▊      | 22/58 [00:01<00:01, 20.56it/s]Capturing num tokens (num_tokens=768 avail_mem=76.68 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.02it/s]Capturing num tokens (num_tokens=704 avail_mem=76.68 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.02it/s]Capturing num tokens (num_tokens=640 avail_mem=76.68 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.02it/s]Capturing num tokens (num_tokens=576 avail_mem=76.67 GB):  43%|████▎     | 25/58 [00:01<00:01, 19.02it/s]Capturing num tokens (num_tokens=576 avail_mem=76.67 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.57it/s]Capturing num tokens (num_tokens=512 avail_mem=76.65 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.57it/s]Capturing num tokens (num_tokens=480 avail_mem=76.19 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.57it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.57it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.57it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.57it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  48%|████▊     | 28/58 [00:01<00:01, 20.57it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.90it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.90it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.90it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.90it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.90it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.90it/s]Capturing num tokens (num_tokens=208 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.90it/s]Capturing num tokens (num_tokens=208 avail_mem=76.00 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.10it/s]Capturing num tokens (num_tokens=192 avail_mem=76.00 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.10it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.10it/s]

    Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.10it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.10it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.10it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.10it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.07it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.07it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.07it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.07it/s]Capturing num tokens (num_tokens=48 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.07it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.07it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.49it/s]

    Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.49it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.49it/s] Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 31.95it/s]


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
    Generated text:  Jeffrey. I'm 18 years old, and I'm a student of an English school. At the beginning of the year, my teachers often ask me to make speeches, and I think I'm not good at it. So I wonder if there is any way to learn well. For example, I often go to the Internet to find information. But I can't find many useful books. I'm not sure if there is a way to learn well. My parents usually encourage me to learn English. But I'm not sure whether it's good for me. Sometimes I feel a bit scared when I speak in front of others.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking political officer appointed by the head of state, the president of the United States, and the state governments of the five states of the country. The president of the United States is the leader of the government of the United States and the chief executive of the executive branch of the federal government. The president serves a two-year term in office, unless removed from office by the president, the Senate, or the Supreme Court. The president is not elected by the people. The president is both the head of state and of the armed forces of the United States, and is the commander-in-chief of the US armed forces. He is sworn
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, a famous historical city. The city is located in the middle of the north-central Europe. It is situated on the left bank of the Seine River and is bordered by the Atlantic Ocean to the south, the English Channel to the east, the Mediterranean Sea to the west, and the English Channel to the north.
    
    Paris is a large city, with a population of around 2. 80 million people (2014). It is one of the largest cities in Europe, and the second largest city in the world, after Rome. It is divided into 17 districts, and each district has its own separate
    ===============================
    Prompt: The future of AI is
    Generated text:  clear: as technology advances, it will start to integrate into more and more areas of our lives. You'll be using AI in all sorts of ways, like in the virtual world of gaming, virtual assistants, virtual tutors, social media, virtual support groups, and more. This post will go over some of the things you should know about the future of AI and how you can use it in your own life.
    AI is all about making machines and computers do tasks that typically require human intelligence, like analyzing data, making decisions, and performing tasks that would otherwise require humans. This means that we can expect to see new and exciting ways to


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


    Generated text:  Paris, also known as "La Ville de Paris" and "La Ville de la Rose". It is the largest city in France and the second-largest city in the European Union. Paris is a cultural and historical center with many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major transportation hub, with many major highways and airports. Paris is known for its cuisine, fashion, and art, and is a popular tourist destination. It is also home to many important institutions of higher education, including the University of Paris and the Paris Observatory. The city is known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more robust AI systems that are designed to be transparent, accountable, and
    


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
    Generated text:  ___________ and I am a/an ___________.
    
    Hello, my name is __________________________ and I am a/an __________________________. I am here to ___________ to this person. If you have any questions or would like to discuss a particular topic, feel free to ___________. I am __________________________.
    
    When you're done, just let me know if you'd like to ___________. Let me know if you'd like to ___________. Let me know if you'd like to ___________. Let me know if you'd like to ___________. Let me know if you'd like to ___________. Let me know if
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is a major city in France located in the western part of the country. It serves as the capital of the country, and is also the political, cultural, and economic center of France. The city is well known for its historical sites, museums, art, and fashion. It is also a popular tourist destination, and is home to the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and other iconic landmarks. With a population of over 2 million people, Paris is a vibrant and diverse city that is a symbol of France. The city is also an important center for France's economy,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a dynamic and rapidly evolving field. Here are some possible future trends that are predicted to shape the future of AI:
    
    1. Improved Transparency and Explainability: As AI becomes more sophisticated and complex, it is becoming increasingly important for developers to provide clear and transparent explanations of their decision-making processes. This will allow users to trust the AI system and reduce the risk of bias or misinterpretation.
    
    2. Enhanced Personalization and Adaptability: AI will become even more personalized and adaptable, able to learn from user data and adjust its behavior accordingly. This will enable more effective communication and interactions between users and AI systems.
    
    3. Integration with Other Technologies


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

    ],

     and

     I

    'm

     a

     [

    role

     or

     profession

    ]

     with

     a

     passion

     for

     [

    role

     or

     profession

    ].

     I

    'm

     excited

     to

     meet

     you

     and

     learn

     more

     about

     you

    ,

     but

     first

     let

     me

     tell

     you

     a

     bit

     about

     me

    .

     I

    'm

     an

     [

    type

     of

     person

    ]

     who

     is

     always

     ready

     to

     learn

     and

     grow

    ,

     even

     when

     things

     are

     tough

    .

     I

     have

     a

     [

    specific

     interest

     or

     skill

    ]

     that

     I

    'm

     passionate

     about

    ,

     and

     I

    'm

     eager

     to

     share

     it

     with

     you

    .

     I

     thrive

     on

     having

     fun

     and

     making

     people

     smile

    ,

     and

     I

     believe

     that

    's

     what

     makes

     me

     special

    .

     I

    'm

     always

     up

     for

     a

     challenge

     and

     enjoy

     exploring

     new

     things

    .

     So

    ,

     if

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Paris

    ,

     often

     called

     the

     “

    City

     of

     Light

    ,”

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

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

     Christmas

     celebrations

    ,

     which

     include

     the

     "

    M

    ardi

     Gr

    as

    "

     and

     the

     "

    M

    ars

     Pom

    mes

     de

     Ter

    re

    "

     (

    or

     "

    P

    om

    mes

     de

     Mars

    ").

     Paris

     is

     the

     world

    's

     

    1

    5

    th

    -largest

     city

     and

     the

     

    1

    4

    th

    -most

    -pop

    ulous

     city

     in

     the

     world

    .

     Its

     nickname

     "

    La

     Ville

     Bl

    anche

    "

     means

     "

    White

     City

    "

     in

     French

    .

     In

     

    2

    0

    1

    5

    ,

     Paris

     had

     a

     population

     of

     approximately

     

    2

    .

    2

     million

     people

    .

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     continuous

     evolution

     and

     adaptation

    ,

     driven

     by

     new

     technologies

    ,

     scientific

     advancements

    ,

     and

     human

     creativity

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

     Enhanced

     decision

    -making

     capabilities

    :

     AI

     will

     continue

     to

     become

     more

     intelligent

     and

     able

     to

     make

     more

     accurate

     decisions

     based

     on

     data

    .

     This

     will

     require

     the

     development

     of

     new

     algorithms

    ,

     machine

     learning

    ,

     and

     deep

     learning

     techniques

     that

     can

     learn

     from

     complex

     patterns

     and

     patterns

     in

     data

    .
    


    2

    .

     Personal

    ization

     of

     AI

    :

     AI

     will

     become

     more

     personalized

     and

     tailored

     to

     individual

     users

    ,

     allowing

     them

     to

     receive

     more

     relevant

     and

     personalized

     recommendations

    .

     This

     will

     require

     the

     development

     of

     more

     advanced

     machine

     learning

     and

     natural

     language

     processing

     techniques

     that

     can

     understand

    



```python
llm.shutdown()
```
