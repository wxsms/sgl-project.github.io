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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.12it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.11it/s]


    2026-05-06 03:04:24,168 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 03:04:24] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.07it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.07it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.64it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.27it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.27it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.27it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.27it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.27it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.27it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.27it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.27it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.27it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.27it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.05it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 28.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.51it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.51 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.48 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.48 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.48 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.48 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.48 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.48 GB):   9%|▊         | 5/58 [00:00<00:02, 21.58it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.47 GB):   9%|▊         | 5/58 [00:00<00:02, 21.58it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.46 GB):   9%|▊         | 5/58 [00:00<00:02, 21.58it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.46 GB):   9%|▊         | 5/58 [00:00<00:02, 21.58it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.46 GB):   9%|▊         | 5/58 [00:00<00:02, 21.58it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.46 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.93it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.45 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.45 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.93it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.45 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.93it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=73.44 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.44 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.21it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.44 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.21it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.44 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.43 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.43 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.43 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.43 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.42 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.13it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=73.42 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.42 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.40 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.62it/s]Capturing num tokens (num_tokens=960 avail_mem=73.42 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.62it/s] Capturing num tokens (num_tokens=896 avail_mem=73.41 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.62it/s]Capturing num tokens (num_tokens=832 avail_mem=73.41 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.62it/s]Capturing num tokens (num_tokens=832 avail_mem=73.41 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.58it/s]Capturing num tokens (num_tokens=768 avail_mem=73.41 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.58it/s]Capturing num tokens (num_tokens=704 avail_mem=73.40 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.58it/s]Capturing num tokens (num_tokens=640 avail_mem=73.40 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.58it/s]

    Capturing num tokens (num_tokens=576 avail_mem=73.40 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.58it/s]Capturing num tokens (num_tokens=512 avail_mem=73.39 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.58it/s]Capturing num tokens (num_tokens=512 avail_mem=73.39 GB):  50%|█████     | 29/58 [00:00<00:00, 35.51it/s]Capturing num tokens (num_tokens=480 avail_mem=73.40 GB):  50%|█████     | 29/58 [00:00<00:00, 35.51it/s]Capturing num tokens (num_tokens=448 avail_mem=73.40 GB):  50%|█████     | 29/58 [00:00<00:00, 35.51it/s]Capturing num tokens (num_tokens=416 avail_mem=73.40 GB):  50%|█████     | 29/58 [00:00<00:00, 35.51it/s]Capturing num tokens (num_tokens=384 avail_mem=72.97 GB):  50%|█████     | 29/58 [00:01<00:00, 35.51it/s]Capturing num tokens (num_tokens=352 avail_mem=72.96 GB):  50%|█████     | 29/58 [00:01<00:00, 35.51it/s]Capturing num tokens (num_tokens=352 avail_mem=72.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=320 avail_mem=72.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=288 avail_mem=72.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.55it/s]

    Capturing num tokens (num_tokens=256 avail_mem=72.95 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=240 avail_mem=72.95 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=224 avail_mem=72.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 37.55it/s]Capturing num tokens (num_tokens=224 avail_mem=72.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=208 avail_mem=72.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=192 avail_mem=72.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=176 avail_mem=72.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=160 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.41it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.89it/s]

    Capturing num tokens (num_tokens=112 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=96 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.89it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=64 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=64 avail_mem=72.49 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.01it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.01it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.01it/s]Capturing num tokens (num_tokens=28 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.01it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.01it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.01it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.52it/s] Capturing num tokens (num_tokens=4 avail_mem=72.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=4 avail_mem=72.46 GB): 100%|██████████| 58/58 [00:01<00:00, 36.01it/s]


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
    Generated text:  Lucy. I have a lot of interesting stories to tell, but I can only write the first part of a story. Can you please let me know the other parts? 
    
    As an AI language model, I am here to assist you in writing your story. I will not write any of the other parts of the story for you. Please share the first part of the story that you would like me to write, and I will do my best to help you write the rest of the story. Let me know if you have any other questions or if there's anything else I can help you with. 
    
    Your story starts with:
    "The day
    ===============================
    Prompt: The president of the United States is
    Generated text:  now considering a policy to use a new wireless technology for government services. The government needs to create a new bill to get this technology. The president is currently evaluating the cost of developing the technology and the potential benefits of this technology.
    
    Based on the president's analysis, they have narrowed down their options to two options: Option A and Option B. Both options will result in an increase in government services and a decrease in the cost of developing the technology. However, Option B will result in more savings in the long run.
    
    The president wants to ensure that the policy they choose results in a balanced benefit-to-cost ratio. The benefit to cost ratio
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. Rome
    C. London
    D. Istanbul
    Answer:
    A
    
    Which of the following represents a well-designed interview question?
    A. Open-ended question
    B. Closed-ended question
    C. Question with partial answer
    D. Question with restricted answer
    Answer:
    A
    
    In the process of heat transfer, what is the primary basis for calculating the heat transfer coefficient h?
    A. The heat absorption rate of the medium
    B. The heat release rate of the medium
    C. The temperature difference of the medium
    D. The specific heat capacity of the medium
    Answer:
    C
    
    Which of the
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the developers, and as such it’s important for us as AI researchers to be ethical.
    In this talk, I will discuss my experiences working on the real-life AI project called SILENT. I’ll highlight some of the ethical dilemmas that arise when developing AI, as well as some of the steps that I took to address them.
    This talk will also include some of the main things that I have learned about how to build AI that is ethical, and that focuses on the need to be transparent and accountable as well as in the design of the models and the data that they are built with. This will be discussed


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast. I'm passionate about [What you like to do]. I'm always looking for new challenges and opportunities to learn and grow. I'm a [What you do for a living] who is always looking for ways to improve my skills and knowledge. I'm always eager to learn and grow, and I'm always willing to share my knowledge with others. I'm a [What you do for a living] who is always looking for ways to improve my skills and knowledge. I'm always eager to learn and grow
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of French literature, cinema, and music, and is a major economic and cultural center. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is also known for its diverse cuisine and vibrant nightlife. The French capital is a bustling metropolis with a rich history and culture. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is a UNESCO World Heritage site and a UNESCO City of Literature and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there is a growing emphasis on developing AI that is more ethical and responsible. This could mean developing AI that is designed to minimize harm to individuals and society as a whole.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, including healthcare, transportation, and manufacturing. As more of these technologies become integrated with AI, we can expect to see even more
    


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
    Generated text:  [Name] and I'm a [Title] [Degree] from [School]. I've always been passionate about [Your Passion], and I'm always eager to learn and grow. I'm an [Age], [Gender], and [Race/ethnicity], and I come from [Your Origin]. I'm a [Occupation] and I'm always looking for opportunities to inspire and challenge myself. I'm confident in my abilities and I'm ready to make a positive impact on the world. [Name] looks forward to meeting you and sharing my journey with you.
    I'm a [Title] [Degree] from [School],
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is also known as the "City of Light" for its long-standing cultural influence and technological advancements. The city is home to many world-famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre Dame Cathedral. It is also known for its rich and diverse history, with influences from various cultures and religions throughout its history. Paris has a vibrant and multicultural community, with many people of different nationalities living in the city. It is a major tourist destination and an important economic center for France. The city has a long history of being a center of culture and science, and continues to be an
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be transformative and unpredictable. Some possible trends in AI include:
    
    1. Deep Learning: With the increasing use of big data, AI will rely more on deep learning, which will enable computers to learn from data without being explicitly programmed.
    
    2. Explainability: AI systems will become more explainable, which will help in building trust and ensuring that AI is used safely.
    
    3. Ethical AI: There will be a growing focus on ethical AI, which will address issues such as bias, discrimination, and privacy.
    
    4. Personalized AI: AI will become more personalized, with systems that can learn from individual preferences and patterns to provide


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

     a

     [

    Age

    ]

     year

     old

     [

    Gender

    ]

     [

    Sex

    ].

     I

     was

     born

     in

     [

    Birth

    place

    ]

     and

     I

     currently

     reside

     in

     [

    Current

     Location

    ].

     I

     enjoy

     [

    Favorite

     Activity

    /

    Activity

    ,

     Hobby

    /

    Interest

    ]

     and

     I

     love

     [

    Reason

     Why

     I

     Love

     This

     Activity

    ].

     I

     am

     passionate

     about

     [

    Most

     Important

     Thing

    ],

     [

    Reason

     For

     Passion

    ],

     and

     I

    'm

     always

     eager

     to

     [

    Action

    /

    Act

    ].

     I

     have

     a

     [

    H

    obby

    /

    Interest

    ,

     Skill

    ,

     etc

    .]

     that

     I

     find

     incredibly

     fulfilling

    ,

     and

     I

    'm

     constantly

     learning

     new

     things

     that

     I

     believe

     will

     further

     enhance

     my

     abilities

    .

     I

     also

     enjoy

     [

    Favorite

     Book

    ,

     Film

    ,

     etc

    
    
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

    .

     The

     most

     notable

     historical

     sites

     and

     landmarks

     in

     Paris

     include

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     Notre

     Dame

     du

     Mont

    mart

    re

    ,

     and

     the

     Latin

     Quarter

    .

     Paris

     is

     also

     home

     to

     many

     notable

     museums

    ,

     including

     the

     Mus

    ée

     Rod

    in

    ,

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     the

     Mus

    ée

     de

     l

    '

    Or

    anger

    ie

    .

     The

     French

     Quarter

    ,

     in

     particular

    ,

     is

     famous

     for

     its

     historic

     French

    -style

     architecture

    ,

     lively

     nightlife

    ,

     and

     French

     cuisine

    .

     Paris

     is

     a

     cultural

     and

     tourist

     hub

     that

     has

     played

     a

     significant

     role

     in

     French

     history

     and

     continues

     to

     attract

     millions

     of

     visitors

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     uncertainties

    ,

     but

     some

     potential

     trends

     are

     becoming

     more

     likely

    .

     Here

     are

     a

     few

     examples

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

     companies

     and

     governments

     start

     to

     look

     into

     the

     ethical

     implications

     of

     AI

    ,

     there

     will

     be

     a

     push

     towards

     more

     ethical

     AI

    ,

     such

     as

     designing

     AI

     that

     considers

     the

     ethical

     implications

     of

     its

     actions

    .
    


    2

    .

     Automation

     of

     routine

     tasks

    :

     Automation

     of

     routine

     tasks

     will

     continue

     to

     increase

     as

     AI

     continues

     to

     become

     more

     advanced

    .

     This

     will

     involve

     designing

     AI

     that

     can

     perform

     tasks

     that

     are

     typically

     done

     by

     humans

    ,

     such

     as

     stock

     trading

    ,

     customer

     service

    ,

     and

     manufacturing

    .
    


    3

    .

     AI

     becoming

     more

     integrated

     into

     daily

     life

    :

     With

    



```python
llm.shutdown()
```
