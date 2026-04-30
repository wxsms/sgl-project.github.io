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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.33it/s]


    2026-04-30 06:09:14,039 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 06:09:14] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:49,  5.08s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:49,  5.08s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:49,  5.08s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:49,  5.08s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:53,  1.01it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:53,  1.01it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:05<00:53,  1.01it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:05<00:53,  1.01it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:05<00:53,  1.01it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:20,  2.46it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:20,  2.46it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:20,  2.46it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:20,  2.46it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:20,  2.46it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:05<00:20,  2.46it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:05<00:20,  2.46it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:05<00:20,  2.46it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:07,  5.85it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:07,  5.85it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:07,  5.85it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:07,  5.85it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:07,  5.85it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:05<00:07,  5.85it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:05<00:07,  5.85it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:05<00:07,  5.85it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:05<00:07,  5.85it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03, 10.75it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03, 10.75it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03, 10.75it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03, 10.75it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03, 10.75it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03, 10.75it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03, 10.75it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03, 10.75it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:03, 10.75it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 16.51it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 16.51it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 16.51it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 16.51it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 16.51it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 16.51it/s]

    Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 16.51it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 16.51it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 16.51it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 23.35it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 23.35it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 23.35it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 23.35it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 23.35it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 23.35it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 23.35it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 23.35it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 23.35it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 30.90it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 30.90it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 30.90it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 30.90it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 30.90it/s]

    Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 30.90it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 30.90it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 30.90it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 37.16it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 37.16it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 37.16it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 37.16it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 37.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.69it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:03, 15.11it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:03, 15.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:03, 15.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:03, 15.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.25 GB):   9%|▊         | 5/58 [00:00<00:04, 12.76it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.24 GB):   9%|▊         | 5/58 [00:00<00:04, 12.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:04, 12.76it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:04, 12.76it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.23 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.41it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.23 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.23 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.22 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.41it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.22 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.41it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.22 GB):  21%|██        | 12/58 [00:00<00:02, 22.85it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.21 GB):  21%|██        | 12/58 [00:00<00:02, 22.85it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.21 GB):  21%|██        | 12/58 [00:00<00:02, 22.85it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=72.21 GB):  21%|██        | 12/58 [00:00<00:02, 22.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.21 GB):  21%|██        | 12/58 [00:00<00:02, 22.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.21 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.20 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.20 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.20 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.19 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.19 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.17 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.35it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.19 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.35it/s] Capturing num tokens (num_tokens=896 avail_mem=72.19 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=832 avail_mem=72.18 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=832 avail_mem=72.18 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.02it/s]Capturing num tokens (num_tokens=768 avail_mem=72.18 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.02it/s]Capturing num tokens (num_tokens=704 avail_mem=72.18 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.02it/s]Capturing num tokens (num_tokens=640 avail_mem=72.17 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.02it/s]

    Capturing num tokens (num_tokens=576 avail_mem=72.17 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.02it/s]Capturing num tokens (num_tokens=576 avail_mem=72.17 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.36it/s]Capturing num tokens (num_tokens=512 avail_mem=72.16 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.36it/s]Capturing num tokens (num_tokens=480 avail_mem=72.17 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.36it/s]Capturing num tokens (num_tokens=448 avail_mem=72.17 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.36it/s]Capturing num tokens (num_tokens=416 avail_mem=72.17 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.36it/s]Capturing num tokens (num_tokens=416 avail_mem=72.17 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.48it/s]Capturing num tokens (num_tokens=384 avail_mem=72.17 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.48it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.16 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.48it/s]Capturing num tokens (num_tokens=320 avail_mem=72.16 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.48it/s]Capturing num tokens (num_tokens=288 avail_mem=72.15 GB):  55%|█████▌    | 32/58 [00:01<00:00, 28.48it/s]Capturing num tokens (num_tokens=288 avail_mem=72.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.55it/s]Capturing num tokens (num_tokens=256 avail_mem=72.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.55it/s]Capturing num tokens (num_tokens=240 avail_mem=72.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.55it/s]Capturing num tokens (num_tokens=224 avail_mem=72.14 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.55it/s]Capturing num tokens (num_tokens=208 avail_mem=72.14 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.55it/s]

    Capturing num tokens (num_tokens=208 avail_mem=72.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.17it/s]Capturing num tokens (num_tokens=192 avail_mem=72.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.17it/s]Capturing num tokens (num_tokens=176 avail_mem=72.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.17it/s]Capturing num tokens (num_tokens=160 avail_mem=72.13 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.17it/s]Capturing num tokens (num_tokens=144 avail_mem=72.13 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.17it/s]Capturing num tokens (num_tokens=144 avail_mem=72.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=128 avail_mem=72.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=112 avail_mem=72.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=96 avail_mem=72.12 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.25it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=72.12 GB):  76%|███████▌  | 44/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=80 avail_mem=72.12 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=64 avail_mem=72.11 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=48 avail_mem=72.11 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=32 avail_mem=72.11 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=28 avail_mem=72.10 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.33it/s]Capturing num tokens (num_tokens=28 avail_mem=72.10 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=24 avail_mem=72.10 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=20 avail_mem=72.10 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=16 avail_mem=72.10 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.03it/s]Capturing num tokens (num_tokens=12 avail_mem=72.09 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.03it/s]

    Capturing num tokens (num_tokens=8 avail_mem=72.09 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.03it/s] Capturing num tokens (num_tokens=8 avail_mem=72.09 GB):  98%|█████████▊| 57/58 [00:01<00:00, 37.22it/s]Capturing num tokens (num_tokens=4 avail_mem=72.08 GB):  98%|█████████▊| 57/58 [00:01<00:00, 37.22it/s]Capturing num tokens (num_tokens=4 avail_mem=72.08 GB): 100%|██████████| 58/58 [00:02<00:00, 28.94it/s]


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
    Generated text:  Wanda and I am an AQA Standard Grade student. I have just started my first GCSE exam in England and I am in year 10. It's been a very long day and I'm getting a bit nervous. I am currently studying geometry and have a bit of homework today to do, and I am not sure what to expect in terms of questions and the grade I should get. I have learned a lot about lines, angles, and congruence but have not really had a chance to really think about the structure of the exam and how to tackle different types of questions. Can you help me with this?
    
    Certainly
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to build. He has two options. The first option would cost $100 billion and allow the US to expand its military. However, there are concerns about the potential harm that the military could cause. The second option would cost $200 billion, but the military would be allowed to expand at a lower rate, thus reducing the harm to the US. The president can choose one of these options and he is willing to spend $10 billion a year to avoid the potential harm. Which option should the president choose? The president should choose the second option. Despite the potential cost of $
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. There are 23 million people in France. If 30% of the population is female, how many people are there in Paris that are female? To determine the number of people in Paris that are female, we need to calculate 30% of the total population of Paris.
    
    First, we know the total population of Paris is 23 million people. We need to convert this population into a decimal for the calculation. Since 1 million equals 1,000,000, we have:
    
    \[ 23,000,000 \]
    
    Next, we calculate 
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but the future of healthcare is more complex and complex than ever before. The stakes are high, and the consequences of a misstep can be catastrophic. That is why it is essential to remain vigilant and proactive in protecting our data and systems, and to ensure that we are not exposed to a wide range of potential threats.
    
    Here are some key points to consider when protecting your data and systems in the age of AI:
    
    1. Secure your data: Make sure to keep all of your data in a secure location. This can include using a password, using a strong password for your email, and using a secure connection to your devices.
    
    


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] with [Number of Wheels] wheels. I'm [Favorite Food] and I love [Favorite Activity]. I'm [Favorite Book] and I enjoy [Favorite Music]. I'm [Favorite Movie] and I love [Favorite Sport]. I'm [Favorite Movie] and I love [Favorite Sport]. I'm [Favorite Movie] and I love [Favorite Sport]. I'm [Favorite Movie] and I love [Favorite Sport]. I'm [Favorite Movie] and I love [Favorite Sport]. I'm [Favorite Movie
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination, with millions of visitors annually. The city is known for its rich history, art, and cuisine, and is a major hub for international business and diplomacy. It is also home to the French Parliament and the French National Library. Paris is a vibrant and diverse city with a rich cultural heritage that continues to attract visitors from around the world. The city is also known for its use
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, but there is likely to be continued growth in its use in this area.
    
    4. Greater focus on AI in education: As AI becomes more advanced
    


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
    Generated text:  [Name] and I am a [Occupation]. I have always been passionate about [My Hobby/Interest/ Passion], and I have been working hard to develop my skills and expand my knowledge. I believe that my expertise in [My Expertise/Area of Expertise] will make me a valuable asset to any organization, and I am excited to join the team at [Company Name]. Let me know if you would like me to provide a more detailed introduction or have you already set up a meeting to discuss this opportunity? [Name]: Hello, my name is [Name] and I am a [Occupation]. I have always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower, and is a major cultural and economic center. It is the largest city in France and has a rich history dating back to ancient times. The city is home to many world-renowned museums, art galleries, and landmarks, including the Louvre and Notre-Dame Cathedral. Paris is also known for its vibrant nightlife, annual festivals, and food and wine scene. Its status as a major European city has made it a global hub of culture, art, and commerce.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be incredibly diverse and transformative, and there are a number of areas where we can expect to see significant growth. Here are some possible future trends in AI:
    
    1. Increased AI-powered automation: AI is becoming more and more integrated into our daily lives, with automation becoming more prevalent in sectors such as manufacturing, transportation, and customer service. By automating repetitive tasks, AI can increase efficiency and reduce the need for human workers.
    
    2. AI-powered healthcare: As AI becomes more advanced and capable, it may be able to provide more accurate diagnoses and treatment options for patients, leading to better health outcomes and a reduction in the need for


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

     your

     name

    ].

     I

     am

     an

     [

    insert

     your

     age

    ],

     [

    insert

     your

     profession

    ],

     and

     I

     am

     a

     [

    insert

     your

     notable

     achievement

     or

     accomplishment

    ].

     Here

    's

     a

     question

     for

     you

    :

     what

    's

     something

     you

    're

     passionate

     about

    ?

     Let

    's

     connect

    !

     [

    insert

     your

     name

    ]

     

    ✉

    ️

     

    ✉

    ️

    


    I

    'm

     excited

     to

     meet

     you

     and

     discuss

     our

     interests

    .

     Let

    's

     start

     with

     what

     you

    're

     passionate

     about

    !


    [

    insert

     your

     name

    ]

     

    ✉

    ️

     

    ✉

    ️

    


    I

    'm

     excited

     to

     meet

     you

     and

     discuss

     our

     interests

    .

     Let

    's

     start

     with

     what

     you

    're

     passionate

     about

    !

     [

    insert

     your

     name

    ]

     

    ✉

    ️

     

    ✉

    ️

    


    I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     iconic

     landmarks

     like

     Notre

    -D

    ame

     Cathedral

     and

     E

    iff

    el

     Tower

    ,

     as

     well

     as

     its

     rich

     cultural

     heritage

     and

     diverse

     population

    .

     Additionally

    ,

     Paris

     is

     a

     major

     hub

     for

     tourism

    ,

     with

     its

     many

     museums

    ,

     fashion

    ,

     and

     art

     galleries

    ,

     as

     well

     as

     its

     role

     in

     the

     French

     Republic

     and

     European

     politics

    .

     
    


    That

    's

     great

     information

     about

     Paris

    .

     Can

     you

     please

     provide

     more

     details

     on

     the

     cultural

     significance

     of

     the

     E

    iff

    el

     Tower

    ?

     For

     example

    ,

     how

     does

     it

     reflect

     Paris

    '

     history

     and

     society

    ?

     
    


    Sure

    ,

     I

    'd

     be

     happy

     to

     provide

     more

     details

     on

     the

     cultural

     significance

     of

     the

     E

    iff

    el

     Tower

     in

     Paris

    .

     The

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     combination

     of

     technical

     advances

    ,

     regulatory

     changes

    ,

     and

     societal

     changes

    .

     Here

     are

     some

     potential

     trends

     that

     could

     be

     expected

     in

     the

     AI

     landscape

     over

     the

     next

     decade

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

     and

     more

     AI

     systems

     are

     developed

    ,

     there

     is

     a

     growing

     awareness

     of

     the

     potential

     consequences

     of

     AI

     decision

    -making

    .

     As

     a

     result

    ,

     there

     is

     a

     greater

     emphasis

     on

     ethical

     AI

    ,

     which

     includes

     considerations

     of

     bias

    ,

     transparency

    ,

     accountability

    ,

     and

     fairness

    .
    


    2

    .

     More

     automation

    :

     One

     of

     the

     biggest

     potential

     trends

     in

     AI

     is

     the

     increasing

     automation

     of

     repetitive

     tasks

    .

     This

     could

     lead

     to

     the

     development

     of

     self

    -driving

     cars

    ,

     robotic

     processes

    ,

    



```python
llm.shutdown()
```
