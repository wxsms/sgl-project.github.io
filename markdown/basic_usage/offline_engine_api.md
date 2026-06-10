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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  8.98it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.98it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.92it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.92it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.92it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.94it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.94it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.94it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.94it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.94it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.94it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.94it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.94it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.94it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.94it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.94it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 18.35it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.13 GB):   3%|▎         | 2/58 [00:00<00:03, 18.35it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.12 GB):   3%|▎         | 2/58 [00:00<00:03, 18.35it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.12 GB):   3%|▎         | 2/58 [00:00<00:03, 18.35it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=75.12 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.12 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.11 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.79it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.79it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.09 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.79it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.79it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 28.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 28.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 28.71it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 28.71it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 28.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 28.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.45it/s]Capturing num tokens (num_tokens=1792 avail_mem=75.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.45it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.45it/s]Capturing num tokens (num_tokens=1280 avail_mem=75.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.45it/s]Capturing num tokens (num_tokens=1024 avail_mem=75.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.45it/s]Capturing num tokens (num_tokens=960 avail_mem=75.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.45it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=75.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=896 avail_mem=75.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=832 avail_mem=75.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=704 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.33it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.81it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.81it/s]Capturing num tokens (num_tokens=512 avail_mem=74.61 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.81it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.81it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.81it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.81it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.14it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.29it/s]

    Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.81it/s] Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 42.65it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 42.65it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 42.65it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 42.65it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 42.65it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  81%|████████  | 47/58 [00:01<00:00, 42.65it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.96it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.48it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.48it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 37.94it/s]


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
    Generated text:  Daniel and I am a patient of the University of Maryland School of Dentistry. I’ve been here for 3 years now and I have been here since I was 11 years old. I’m 6'3" tall and I wear braces for my teeth. I wear braces 3 times a day because my front teeth are yellow and I have a lot of decay. My dentist really likes my smile and is very proud of me. I have received many awards for my braces and my work with my patients. It is very fulfilling to have a job that I can help people, and I feel like I am making a difference
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of which of the following?
    A. The Senate
    B. The House of Representatives
    C. The Supreme Court
    D. The Executive Branch
    Answer:
    
    D
    
    According to the Civil Code of the People's Republic of China, which of the following is an invalid civil act? 
    A. A contract signed by a person with limited civil capacity 
    B. A contract entered into by a person with full civil capacity under the age of 8 
    C. A contract entered into by a person with full civil capacity over 8, with the consent of the other party 
    D. A contract signed by a person with full
    ===============================
    Prompt: The capital of France is
    Generated text:  ( ).
    A. Paris
    B. Toulouse
    C. Nice
    D. Marseille
    Answer: A
    
    When a department manager is preparing to establish a learning management system, which of the following statements is incorrect?
    A. The department manager should clearly identify the goals of the learning management system, defining the scope and requirements.
    B. The department manager should establish clear guidelines for the use and maintenance of the system.
    C. The department manager needs to choose appropriate tools and platforms for the learning management system.
    D. The department manager should establish detailed plans for system implementation, including cost estimates and timeframes for system deployment and implementation.
    Answer
    ===============================
    Prompt: The future of AI is
    Generated text:  not in the future but in the present. Let us explore how it can be used in the context of the Black Box Problem and how it can improve our lives in the present. Our mission is to use the black box problem to solve the problems of the present.
    The Black Box Problem
    The Black Box Problem is a well-known term in the field of Artificial Intelligence. It was first proposed in 1964 by Claude Shannon and Edward Teller. They were working on the Large Hadron Collider at the University of California Los Alamos, and they wanted to know what was inside the machine. They wanted to know what was inside


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry]. I enjoy [reason for interest in the industry]. I'm always looking for ways to [reason for interest in the industry]. I'm a [reason for interest in the industry] and I'm always looking for ways to [reason for interest in the industry]. I'm a [reason for interest in the industry] and I'm always looking for ways to [reason for interest in the industry]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is factually correct and provides a clear and concise overview of the capital city's location and significance. However, it could be expanded to include additional information about Paris's cultural, historical, or political importance. For example:
    
    - Paris is the capital of France and the largest city in the European Union.
    - It is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.
    - Paris is home to many museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin.
    - The city is also known for its gastr
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can better understand and respond to human needs and preferences.
    
    2. Enhanced privacy and security: As AI systems become more integrated with human intelligence, there will be increased concerns about privacy and security. There will be a need for more robust privacy and security measures to protect the data and information that is generated and processed by AI systems.
    
    3. Greater reliance on AI for decision-making
    


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
    Generated text:  [Name] and I am [Age]. I am a [occupation] with a deep appreciation for [profession or hobby]. I am dedicated to [career goal] and I am always looking to improve myself. I am always looking for ways to make my life more meaningful and fulfilling. I am a [role] with a passion for [interest or hobby]. I am passionate about [purpose or reason]. I am confident in my abilities and I am always willing to learn and grow. I am someone who always tries to be a good friend, a good listener, and a good leader. I am a [character type] with [abilities
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light.
    
    That's a great statement! How can I help you further with Paris? 
    
    I'm interested in learning more about the cuisine in Paris. Can you recommend some must-try dishes and how they relate to the city's historical significance? Certainly! Paris is renowned for its gastronomy, and the city has a rich culinary tradition that reflects its history, architecture, and cultural diversity. Here are some must-try dishes that showcase the city's culinary heritage:
    
    1. The famous croissants: Paris is famous for its croissants, a staple of French bread. The cheese dough is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of technological and societal trends that will shape the way we interact with machines and interact with each other. Some possible trends include:
    
    1. Increased automation: AI is already revolutionizing industries such as manufacturing, healthcare, and finance, and it is likely to continue to do so in the future. Automation will bring new benefits such as increased efficiency and productivity, but it will also lead to job displacement for many people.
    
    2. Enhanced creativity: AI is already being used to generate art, music, and other creative outputs. In the future, we may see even more AI-generated art and music, and we may see even


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

     John

    ,

     and

     I

    'm

     a

     

    2

    8

    -year

    -old

     marketing

     manager

    .

     I

     have

     a

     deep

     love

     for

     the

     outdoors

     and

     love

     to

     travel

     and

     explore

     the

     great

     outdoors

    .

     I

    'm

     always

     eager

     to

     learn

     new

     things

     about

     the

     world

     and

     connect

     with

     others

     through

     outdoor

     adventures

    .

     My

     passion

     for

     creating

     a

     positive

     impact

     on

     the

     world

     and

     using

     technology

     to

     help

     others

     is

     a

     driving

     force

     in

     my

     career

    .

     I

    'm

     very

     active

    ,

     and

     I

     love

     to

     run

     and

     spend

     my

     free

     time

     hiking

     and

     cycling

    .

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     learning

     new

     things

    .

     I

     enjoy

     working

     with

     people

     and

     have

     a

     great

     team

     player

     attitude

    .

     I

    'm

     always

     looking

     for

     new

     ways

     to

     make

    
    
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

     the

     City

     of

     Many

     Faces

    ,

     and

     the

     City

     of

     Love

    .

     Located

     on

     the

     Se

    ine

     River

     and

     surrounded

     by

     the

     historic

     center

     of

     the

     city

    ,

     Paris

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     European

     Union

     by

     area

     and

     the

     third

     largest

     by

     population

    .

     It

     is

     a

     cosm

    opolitan

     city

     with

     a

     rich

     cultural

     heritage

     and

     a

     lively

     nightlife

    .

     Paris

     has

     a

     history

     dating

     back

     to

     the

     Roman

     Empire

    ,

     and

     it

     is

     a

     melting

     pot

     of

     cultures

    ,

     languages

    ,

     and

     traditions

    .

     The

     city

     is

     famous

     for

     its

     art

    ,

     music

    ,

     and

     literature

    ,

     as

     well

     as

     its

     famous

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Lou

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     potential

     opportunities

     and

     challenges

    .

     Here

     are

     some

     possible

     trends

     in

     the

     field

    :
    


    1

    .

     Increasing

    ly

     collaborative

     AI

    :

     AI

     is

     becoming

     more

     collaborative

     and

     open

    -source

    ,

     allowing

     developers

     to

     build

     on

     each

     other

    's

     work

     and

     improve

     upon

     it

    .

     This

     trend

     will

     lead

     to

     more

     efficient

     and

     effective

     AI

     systems

     that

     can

     work

     together

     to

     solve

     complex

     problems

    .
    


    2

    .

     Enhanced

     AI

    :

     AI

     will

     continue

     to

     get

     better

     at

     recognizing

     patterns

     and

     making

     decisions

     based

     on

     data

    .

     This

     will

     lead

     to

     more

     accurate

     and

     helpful

     AI

     systems

     that

     can

     handle

     a

     wide

     range

     of

     tasks

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     will

     be

     increasingly

     integrated

     into

     healthcare

     systems

     to

     improve

     patient

     care

     and

     reduce

     errors

    .

    



```python
llm.shutdown()
```
