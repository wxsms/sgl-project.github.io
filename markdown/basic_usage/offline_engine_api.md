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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.70it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.69it/s]


    2026-05-05 07:27:33,059 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 07:27:33] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.21it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.46it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.46it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.46it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.46it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.17it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.17it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.17it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.17it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.17it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.17it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.17it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.17it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.17it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.17it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.17it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.07it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.07it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.07it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.07it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.07it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.07it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.07it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.07it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 17.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 17.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 17.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 17.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.13 GB):   9%|▊         | 5/58 [00:00<00:02, 20.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.12 GB):   9%|▊         | 5/58 [00:00<00:02, 20.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.11 GB):   9%|▊         | 5/58 [00:00<00:02, 20.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.11 GB):   9%|▊         | 5/58 [00:00<00:02, 20.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.64it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.64it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.64it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.64it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 29.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 29.27it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 29.27it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.27it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 29.27it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.12it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.12it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.12it/s] Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=896 avail_mem=74.64 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=704 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.98it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.98it/s]Capturing num tokens (num_tokens=512 avail_mem=74.61 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.98it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.98it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.98it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.98it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.28it/s]Capturing num tokens (num_tokens=384 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.28it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.28it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.28it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.28it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.28it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.74it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.74it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.74it/s]

    Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.74it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=112 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.23it/s] Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 46.71it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  81%|████████  | 47/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.65it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.65it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.65it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.65it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.65it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.65it/s] Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.70it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.70it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 40.49it/s]


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
    Generated text:  Daniel and I am 18 years old. I am a high school student. I am getting a job in July. I want to know how I can become a professional basketball player. I would like to know some ways to practice and improve my skills. Can you give me some advice? Thank you very much. Daniel
    Sure, I'd be happy to help! Here are a few tips that may help you improve your basketball skills and become a professional player:
    
      1. Start with basics: Before you can focus on shooting or teamwork, it's important to learn the basics of basketball, such as passing, dribbling,
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide what lunch to have with his Cabinet. He has decided to have a sandwich and a glass of water. The sandwich has a price of $10 and the glass of water has a price of $2. If the president has $50, how many different combinations of sandwiches and glasses of water can he buy?
    
    To determine how many different combinations of sandwiches and glasses of water the president can buy, we need to consider all possible values for the number of sandwiches and the number of glasses of water that satisfy the budget of $50. Let's denote the number of sandwiches by \( x \) and the number of
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.____
    A. New York
    B. Paris
    C. Beijing
    D. London
    Answer:
    
    B
    
    The capital of France is ______.
    A. New York
    B. Paris
    C. Beijing
    D. London
    Answer:
    
    B
    
    Which of the following statements about the central government of France is incorrect?
    A. The central government is the highest administrative organ of France.
    B. The capital of France is Paris.
    C. The central government consists of the President and the Prime Minister.
    D. The executive power of the central government is exercised by the President.
    Answer:
    
    C
    
    Which of the following statements about
    ===============================
    Prompt: The future of AI is
    Generated text:  fascinating, and in particular, the field of AI in medicine is ripe for innovation. By developing AI-powered applications that can process, analyze, and interpret complex data sets, researchers and healthcare providers can make more informed decisions, improve patient outcomes, and help clinicians diagnose and treat diseases more accurately and efficiently.
    
    AI is making its way into the healthcare field, and the potential applications are vast. Here, we’ll take a closer look at how AI is being used in the field, and explore some of the promising developments that could transform the way we think about healthcare.
    
    ### How AI is Making its Way into Healthcare
    
    The field of AI is expanding


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your profession or role]. I enjoy [insert a brief description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a hobby or activity you enjoy]. I'm always looking for ways to improve myself and make the world a better place. What's your favorite book or movie? I love [insert a favorite book
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich cultural heritage and a vibrant nightlife. It is the largest city in France and the second-largest city in the European Union. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also a major center for art, music, and fashion. Paris is known for its romantic atmosphere, historical significance, and cultural diversity. It is a popular tourist destination and a major economic center in France. The city is also home to many international organizations and institutions. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in more complex and personalized ways, potentially leading to even more effective and efficient healthcare systems.
    
    3. Increased focus on ethical AI: As
    


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
    Generated text:  [insert name]. I'm here to help you if you need anything. How can I assist you today? Let's get started! Do you have a specific question or task you need help with? I'm here to listen and provide any information or assistance I can. I look forward to hearing from you! 🌟
    Your friendly neighborhood superhero! 🦁✨
    [Insert name] 🧡✨
    How can I assist you today? 🚀✨
    Your friend from the future! 🚀✨
    [Insert name] 🧡✨
    I'm here to help, so come on in!
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the largest metropolitan area in Europe. It has a population of over 10 million people and is home to many cultural and historic landmarks. The city is known for its romantic history and is a popular tourist destination. Paris is also home to many world-renowned art museums, including the Louvre and the Musée d'Orsay. It is a cultural hub with a rich history dating back to the ancient Roman Empire, and continues to be a major center of European culture, politics, and diplomacy. The city is also home to some of the world's top universities, including the University
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be a rapidly evolving landscape with many potential trends to watch out for. Here are some possible future trends in AI:
    
    1. Increased autonomy: As AI becomes more advanced and integrated into our daily lives, we may see increased autonomy in the forms of autonomous vehicles, robots, and other intelligent machines that can operate independently.
    
    2. Personalized and context-aware AI: AI will become more personalized and context-aware, which will enable machines to learn and adapt to the unique needs and behaviors of individual users.
    
    3. Cognitive computing: Cognitive computing will continue to advance, with machines becoming increasingly intelligent and capable of understanding and solving complex problems.
    
    4


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

     an

     [

    Age

    ]

     year

    -old

     [

    Occup

    ation

    ].

     I

    'm

     the

     founder

     of

     [

    Your

     Company

    /

    Project

    ],

     and

     I

    'm

     passionate

     about

     [

    Your

     passion

    ].

     Let

     me

     know

     if

     you

     would

     like

     more

     information

     on

     me

    .

     I

    'm

     always

     up

     for

     a

     challenge

    ,

     so

     please

     don

    't

     hesitate

     to

     ask

     for

     my

     help

    .

     Thank

     you

    !

     [

    Name

    ]

     [

    Age

    ]

     [

    Occup

    ation

    ]

     [

    Your

     Company

    /

    Project

    ]

     [

    Your

     passion

    ]


    As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     an

     age

    ,

     occupation

    ,

     or

     specific

     occupation

    .

     I

     am

     here

     to

     assist

     you

     with

     any

     questions

     or

     tasks

     you

     may

     have

    .

     Please

     let

     me

     know

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Its

     historical

     and

     cultural

     significance

     as

     a

     world

    -ren

    owned

     met

    ropolis

     and

     a

     major

     cultural

     hub

     has

     made

     it

     a

     major

     city

     of

     immense

     importance

     in

     France

    .

     The

     city

     is

     known

     for

     its

     picturesque

     architecture

    ,

     diverse

     museums

    ,

     and

     vibrant

     arts

     scene

    ,

     and

     is

     a

     popular

     tourist

     destination

    .

     Paris

     has

     an

     international

     reputation

     for

     being

     a

     melting

     pot

     of

     cultures

    ,

     with

     the

     city

    's

     diverse

     population

     contributing

     to

     its

     unique

     culinary

     and

     social

     scene

    .

     The

     French

     people

     have

     a

     rich

     cultural

     heritage

     and

     continue

     to

     create

     new

     artistic

     forms

     and

     traditions

    ,

     further

     cement

    ing

     Paris

    's

     status

     as

     a

     capital

     city

     of

     artistic

     excellence

    .

     As

     the

     largest

     and

     most

     populous

     city

     in

     Europe

    ,

     Paris

     is

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     complex

     and

     uncertain

    ,

     but

     there

     are

     several

     trends

     that

     are

     likely

     to

     shape

     the

     field

     in

     the

     years

     ahead

    .

     Here

     are

     some

     potential

     areas

     where

     we

     can

     expect

     to

     see

     significant

     changes

     in

     AI

    :
    


    1

    .

     Increased

     Integration

     of

     AI

     into

     Everyday

     Life

    :

     AI

     is

     already

     playing

     a

     significant

     role

     in

     our

     daily

     lives

    ,

     such

     as

     through

     virtual

     assistants

     like

     Siri

     and

     Alexa

    .

     In

     the

     future

    ,

     we

     can

     expect

     this

     trend

     to

     continue

    ,

     with

     AI

     becoming

     more

     integrated

     into

     our

     technology

     and

     daily

     routines

    .
    


    2

    .

     Improved

     Personal

    ized

     AI

    :

     As

     we

     learn

     more

     about

     how

     to

     process

     and

     analyze

     large

     amounts

     of

     data

    ,

     we

     can

     expect

     AI

     to

     become

     more

     personalized

     and

     adaptive

    .

     This

     could

    



```python
llm.shutdown()
```
