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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.01it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.00it/s]


    2026-04-15 01:21:38,337 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-15 01:21:38] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.80it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.80it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.80it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.80it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.80it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:02<00:06,  6.80it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:02<00:06,  6.80it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:02<00:06,  6.80it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.80it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.80it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 22.07it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 22.07it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 22.07it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 22.07it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 22.07it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 22.07it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 22.07it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 22.07it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 22.07it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 22.07it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 31.02it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 31.02it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 31.02it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.95it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.95it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.95it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.95it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.95it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.95it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.95it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.95it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.95it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.95it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 50.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=133.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=133.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=133.21 GB):   3%|▎         | 2/58 [00:00<00:03, 18.43it/s]Capturing num tokens (num_tokens=7168 avail_mem=133.20 GB):   3%|▎         | 2/58 [00:00<00:03, 18.43it/s]Capturing num tokens (num_tokens=6656 avail_mem=133.20 GB):   3%|▎         | 2/58 [00:00<00:03, 18.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=133.20 GB):   3%|▎         | 2/58 [00:00<00:03, 18.43it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=133.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=133.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=133.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=133.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=133.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=133.20 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=133.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=133.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=133.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=133.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=133.18 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=133.18 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=133.18 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.90it/s]Capturing num tokens (num_tokens=2304 avail_mem=133.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.90it/s]Capturing num tokens (num_tokens=2048 avail_mem=133.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=133.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=133.17 GB):  31%|███       | 18/58 [00:00<00:01, 35.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=133.16 GB):  31%|███       | 18/58 [00:00<00:01, 35.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=133.16 GB):  31%|███       | 18/58 [00:00<00:01, 35.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=133.14 GB):  31%|███       | 18/58 [00:00<00:01, 35.19it/s]

    Capturing num tokens (num_tokens=960 avail_mem=133.15 GB):  31%|███       | 18/58 [00:00<00:01, 35.19it/s] Capturing num tokens (num_tokens=896 avail_mem=133.15 GB):  31%|███       | 18/58 [00:00<00:01, 35.19it/s]Capturing num tokens (num_tokens=896 avail_mem=133.15 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=832 avail_mem=133.15 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=768 avail_mem=133.14 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=704 avail_mem=133.14 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=640 avail_mem=133.14 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=576 avail_mem=133.14 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=576 avail_mem=133.14 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=512 avail_mem=133.12 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=480 avail_mem=133.14 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.71it/s]

    Capturing num tokens (num_tokens=448 avail_mem=133.14 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=416 avail_mem=133.14 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=384 avail_mem=133.13 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=384 avail_mem=133.13 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.05it/s]Capturing num tokens (num_tokens=352 avail_mem=133.13 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.05it/s]Capturing num tokens (num_tokens=320 avail_mem=133.12 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.05it/s]Capturing num tokens (num_tokens=288 avail_mem=133.12 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.05it/s]Capturing num tokens (num_tokens=256 avail_mem=133.12 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.05it/s]Capturing num tokens (num_tokens=240 avail_mem=133.12 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=240 avail_mem=133.12 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.76it/s]Capturing num tokens (num_tokens=224 avail_mem=133.11 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.76it/s]

    Capturing num tokens (num_tokens=208 avail_mem=133.11 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.76it/s]Capturing num tokens (num_tokens=192 avail_mem=133.11 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.76it/s]Capturing num tokens (num_tokens=176 avail_mem=133.11 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.76it/s]Capturing num tokens (num_tokens=160 avail_mem=133.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.76it/s]Capturing num tokens (num_tokens=160 avail_mem=133.10 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.46it/s]Capturing num tokens (num_tokens=144 avail_mem=133.10 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.46it/s]Capturing num tokens (num_tokens=128 avail_mem=133.10 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.46it/s]Capturing num tokens (num_tokens=112 avail_mem=133.09 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.46it/s]Capturing num tokens (num_tokens=96 avail_mem=133.09 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.46it/s] Capturing num tokens (num_tokens=80 avail_mem=133.09 GB):  74%|███████▍  | 43/58 [00:01<00:00, 42.46it/s]

    Capturing num tokens (num_tokens=80 avail_mem=133.09 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=64 avail_mem=133.08 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=48 avail_mem=133.08 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=32 avail_mem=133.07 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=28 avail_mem=133.07 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=24 avail_mem=133.07 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=24 avail_mem=133.07 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=20 avail_mem=133.06 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=16 avail_mem=133.06 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=12 avail_mem=133.06 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=8 avail_mem=132.97 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.09it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=132.33 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=4 avail_mem=132.33 GB): 100%|██████████| 58/58 [00:01<00:00, 43.42it/s]Capturing num tokens (num_tokens=4 avail_mem=132.33 GB): 100%|██████████| 58/58 [00:01<00:00, 38.73it/s]


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
    Generated text:  Ali and I'm a student at the University of Birmingham. I've been working as a research assistant for the last few months. I've been studying the effects of air pollution on the brain. Now, I want to explain to you some of the effects of air pollution on the brain. Unfortunately, pollution can have severe negative effects on the brain. For example, it can cause damage to the brain, which can lead to memory loss and problems with concentration. This is because the brain needs oxygen to function properly. If it doesn't get enough oxygen, it can suffer from brain damage. Air pollution can also cause problems with the function of
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she is the leader of the country. There are 5 vice presidents. The vice president is the leader of the cabinet, the group of leaders that the president does not serve in. There are usually 10 people in the cabinet. For the first president, the president had two vice presidents. After the first president, the president had 5 vice presidents. The vice president is chosen from the 10 people in the cabinet. 
    
    If the president of the United States had 10 people in the cabinet and the vice president was chosen from these 10 people, how many more people
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. If the sentence is written in English, how would it be written in Russian?
    The capital of France is Paris. Если бы вы написали этот оборот в английском языке, как бы вы его выразили в русском? Согласно правилам, оборот "The capital of France is Paris" в английском языке записывается как "The capital of France is Paris." Таким образом, русский перевод будет "Царство Франции – Париж." Используя свои знания английского языка, вы можете легко произносить
    ===============================
    Prompt: The future of AI is
    Generated text:  here.
    But while AI has made remarkable advances over the last few decades, it still struggles to make those significant leaps in accuracy that it has promised.
    A recent paper published in the IEEE Journal of Selected Topics in Quantum Information shows that, to the best of our knowledge, no algorithm for training of any kind has produced a state-of-the-art quantum computer (a machine that can perform the tasks of quantum computers). The authors examined the hardware and software for a large number of quantum computers, and used those results to predict the performance of a new state-of-the-art quantum computer: the “Tianhe-2” computer, which was the


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


    Generated text:  [Name] and I am a [job title] at [company name]. I am a [job title] at [company name] and I have been working here for [number of years] years. I have always been passionate about [job title] and have always wanted to be a [job title] myself. I am always looking for ways to [job title] and have always been inspired by [job title] and their work. I am a [job title] and I am always looking for ways to [job title] and have always been inspired by [job title] and their work. I am a [job
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a popular tourist destination and a major economic center. Paris is home to many famous French artists, writers, and musicians, and is a major hub for the French language and culture. The city is also known for its rich history, including the influence of the Roman Empire, French Revolution, and French Revolution. Paris is a vibrant and diverse city with a rich cultural and artistic heritage. It is a major transportation hub, with the Eiffel Tower serving as a symbol of Paris and a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and privacy. As a result, AI developers will need to be more mindful of the potential impact of their technology on society.
    
    2. Greater integration with human decision-making: AI is likely to become more integrated with human decision-making in the future. This will involve the use of AI to assist humans in making
    


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
    Generated text:  [Name]. I am a [Type] who enjoys [What interests you] and [What hobbies you enjoy]. I am always looking for new opportunities to learn and grow, and I am passionate about [What passion do you have]! [Name] is always looking to improve and learn, and I am thrilled to have the opportunity to share my knowledge and experience with you. I am a team player who always strive to make a positive impact in my community and strive to inspire others to do the same. I am excited to learn more about your life and to meet new people. [Name] is excited to meet you! [Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Explain why Paris is considered the "City of Love" and describe some of the iconic landmarks that make it a popular tourist destination. Paris is known as the "City of Love" due to its romantic atmosphere, which is evident in its architecture, festivals, and streets. Paris is a popular tourist destination due to its iconic landmarks such as Notre-Dame Cathedral, Louvre Museum, Eiffel Tower, and Marais district. These landmarks have captured the hearts of many tourists and make it a must-visit destination for anyone visiting France. 
    
    Some of the most famous landmarks in Paris include:
    
    1. Notre-Dame
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be one of rapid evolution and transformation, with many potential advancements that are yet to be realized. Here are some of the key trends that are shaping the future of AI:
    
    1. Increased Integration of AI with Traditional Industries: As AI becomes more advanced and ubiquitous, it is expected to integrate with various industries, including healthcare, finance, manufacturing, and transportation. This integration is likely to create new opportunities for AI-powered solutions that can improve efficiency, reduce costs, and increase productivity.
    
    2. Increased Adoption of AI for Healthcare: With the rising prevalence of chronic illnesses and a growing demand for personalized healthcare solutions, AI is likely to play a


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

    ]

     and

     I

     am

     a

     [

    Your

     Profession

    /

    Job

     Title

    ].

     I

     have

     always

     loved

     to

     learn

     and

     excel

     in

     my

     field

    ,

     and

     I

     strive

     to

     continuously

     improve

     my

     skills

    .

     I

     am

     a

     dedicated

     student

     with

     a

     love

     for

     reading

     and

     writing

    ,

     and

     I

     love

     to

     spend

     my

     free

     time

     exploring

     the

     world

     around

     me

    .

     I

     am

     confident

     in

     my

     ability

     to

     adapt

     and

     succeed

     in

     whatever

     challenges

     come

     my

     way

    ,

     and

     I

     am

     always

     ready

     to

     learn

     from

     the

     experiences

     of

     others

    .

     I

     am

     a

     positive

     and

     optimistic

     person

    ,

     and

     I

     strive

     to

     make

     a

     difference

     in

     the

     world

    .

     Thank

     you

     for

     considering

     me

     for

     your

     character

    .

     [

    Your

     Name

    ]

     [

    Your

     Profession

    
    
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

    .

     It

     is

     a

     bustling

     met

    ropolis

     with

     a

     rich

     history

     and

     culture

    ,

     famous

     for

     its

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

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     Paris

     is

     also

     known

     for

     its

     diverse

     cuisine

    ,

     fashion

    ,

     and

     arts

     scene

    .

     The

     city

     is

     home

     to

     over

     

    1

    0

     million

     residents

     and

     is

     considered

     one

     of

     the

     most

     important

     cities

     in

     the

     world

    .

     It

     is

     a

     hub

     for

     business

    ,

     politics

    ,

     and

     culture

    ,

     and

     is

     an

     important

     center

     of

     European

     diplomacy

    .

     Paris

     is

     considered

     the

     "

    City

     of

     love

    "

     and

     is

     home

     to

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

     many

     trends

     that

     are

     likely

     to

     shape

     the

     development

     of

     the

     technology

     and

     its

     applications

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

     in

     the

     future

    :
    


    1

    .

     Adv

    ancements

     in

     deep

     learning

    :

     Deep

     learning

     is

     a

     type

     of

     AI

     that

     uses

     artificial

     neural

     networks

     to

     perform

     tasks

     such

     as

     image

     and

     speech

     recognition

    ,

     natural

     language

     processing

    ,

     and

     decision

    -making

    .

     The

     use

     of

     deep

     learning

     will

     likely

     continue

     to

     grow

     as

     more

     data

     becomes

     available

     and

     as

     researchers

     develop

     new

     algorithms

     and

     models

     that

     can

     handle

     more

     complex

     tasks

    .
    


    2

    .

     Integration

     with

     other

     technologies

    :

     AI

     is

     likely

     to

     become

     more

     integrated

     with

     other

     technologies

     such

     as

     robotics

    ,

     autonomous

     vehicles

    ,

     and

     smart

     homes

    .

     These

    



```python
llm.shutdown()
```
