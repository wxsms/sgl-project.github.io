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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.46it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.46it/s]


    2026-05-20 07:28:47,288 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 07:28:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 22.56it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 22.56it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 22.56it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 22.56it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 22.56it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 22.56it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 22.56it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 22.56it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 22.56it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 22.56it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 30.65it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 30.65it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 30.65it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.65it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.65it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.65it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.65it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.65it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.65it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.65it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.65it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 40.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   2%|▏         | 1/58 [00:00<00:08,  7.10it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   2%|▏         | 1/58 [00:00<00:08,  7.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   2%|▏         | 1/58 [00:00<00:08,  7.10it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   5%|▌         | 3/58 [00:00<00:04, 13.32it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   5%|▌         | 3/58 [00:00<00:04, 13.32it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   5%|▌         | 3/58 [00:00<00:04, 13.32it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   5%|▌         | 3/58 [00:00<00:04, 13.32it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):  10%|█         | 6/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  10%|█         | 6/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  10%|█         | 6/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  10%|█         | 6/58 [00:00<00:02, 19.02it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  10%|█         | 6/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.95it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.95it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.95it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.21it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  26%|██▌       | 15/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.00it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.00it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.97it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.97it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.97it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.97it/s]

    Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.97it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.97it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.36it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.36it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.36it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.36it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.36it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 41.36it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 43.22it/s]

    Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=208 avail_mem=76.65 GB):  60%|██████    | 35/58 [00:01<00:00, 43.22it/s]Capturing num tokens (num_tokens=208 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.84it/s]Capturing num tokens (num_tokens=192 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.84it/s]Capturing num tokens (num_tokens=176 avail_mem=76.64 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.84it/s]Capturing num tokens (num_tokens=160 avail_mem=76.64 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.84it/s]Capturing num tokens (num_tokens=144 avail_mem=76.64 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.84it/s]

    Capturing num tokens (num_tokens=144 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=128 avail_mem=76.55 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=112 avail_mem=76.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=96 avail_mem=76.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.95it/s] Capturing num tokens (num_tokens=80 avail_mem=76.13 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=64 avail_mem=76.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=64 avail_mem=76.04 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.07it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.07it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.07it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.07it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.07it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.07it/s]

    Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.07it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.07it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.07it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.07it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.07it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 35.95it/s]


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
    Generated text:  M. O. S. B. Nwabua, I am a Nigerian Speaker. I am an expert in law and human rights. I have been living in the country for 30 years and I have been very active in the profession. I am a member of the Nigerian Bar Association. My profession includes the study of international human rights law, the law of the situation in Nigeria, the law of conflict of laws and the law of jurisdiction. I have also been involved in the field of legal education. I am a law professor and a senior lecturer at the University of Nigeria, Nsukka. I am an active
    ===============================
    Prompt: The president of the United States is
    Generated text:  a powerful man, he can do anything he wants. What does he want? The president does not want to be president. He wants to be rich. He wants to be president to get rich.
    (1) What does the president want to do to get rich? (2) Why does the president want to be president? (3) If the president wants to be president, he should do what? (4) Why? (5) What is the president's character?
    (1) According to the text, the president wants to be president to get rich. (2) The president wants to be president to get rich because
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A: London
    
    B: Paris
    
    C: Madrid
    
    D: Rome To determine the capital of France, we need to recall a fundamental fact about the countries of Europe. The capital of France is Paris. 
    
    Let's break it down step by step:
    
    1. Identify the capital of France: The capital of France is Paris.
    2. Verify the answer: The capital of France is indeed Paris, as stated in the problem.
    
    Therefore, the correct answer is \boxed{B}.
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly approaching, and the most promising technology in the world is the AI-driven visualization system that can visualize complex systems in a way that makes sense to everyone, no matter their expertise. By visualizing complex systems in a way that makes sense to everyone, the AI-driven visualization system can help people make sense of the data and gain insights that can inform their decisions and actions. This system is designed to be user-friendly, intuitive, and accessible to anyone, no matter their expertise. The system is also designed to be scalable, making it possible to visualize large datasets in real-time, and to handle large amounts of data in a way that is both


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or experience here]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm always looking for new challenges and opportunities to grow
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a popular tourist destination, with many visitors coming to explore its rich history and culture. Paris is a major economic and financial center, with a diverse range of restaurants, shops, and entertainment venues. The city is also home to many notable museums and art galleries, including the Musée d'Orsay and the Musée Rodin. Paris is a vibrant and dynamic city that is known for its vibrant nightlife and cultural events. It is a city that is constantly evolving and changing, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems.
    
    2. Enhanced machine learning capabilities: AI is likely to become more capable of learning from large amounts of data and making more accurate predictions and decisions. This could lead to more efficient and effective use of AI in various industries.
    
    3. Greater reliance on AI for decision-making: AI is likely to become more integrated into decision-making processes, allowing machines to make more informed and accurate decisions without human
    


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
    Generated text:  [Your Name], and I'm a [Your Profession] with over [Your Industry Experience] years of experience in [Your Industry]. I'm an energetic, positive, and creative individual with a passion for [Your Interest/ passion]. I'm a problem solver with a strong understanding of [Your Area of Expertise] and always strive to find innovative solutions to complex problems. I have a knack for turning complex issues into clear, concise, and actionable plans. I'm also a team player with a strong work ethic and I enjoy working with diverse teams, as well as fostering a positive and collaborative culture in the workplace. I'm a reliable
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A brief explanation of Paris's significance:
    Paris is the capital of France, serving as its political, cultural, and economic center. It is known as the "City of Love" due to its romantic history and a number of iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of many French leaders and is a popular tourist destination. Paris is a cultural and artistic hub, hosting a wide range of museums, galleries, and theaters, including the Opéra Garnier, the Musée Rodin, and the Louvre. Paris is also known
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by several key trends, including:
    
    1. Increased automation: AI systems are expected to become more efficient and accurate in performing tasks that are routine or repetitive, allowing humans to focus on more creative and strategic tasks.
    
    2. Integration with human consciousness: AI is expected to become more closely integrated with human consciousness, allowing it to learn and adapt to the needs of humans and make decisions that are influenced by human emotions and preferences.
    
    3. Increased AI ethics: There is a growing concern that AI systems may become too powerful and potentially harmful, and there is a need for ethical guidelines and regulations to ensure that AI is used in a


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

    'm

     [

    Your

     Age

    ],

     a

     [

    Your

     Specialty

     or

     Occupation

    ]

     with

     [

    Your

     Historical

     Background

    ].

     I

    've

     always

     been

     [

    Your

     Unique

     Attribute

     or

     Strength

    ],

     and

     I

    'm

     here

     to

     learn

     from

     and

     be

     a

     part

     of

     your

     journey

    .

     
    


    I

    'm

     a

     [

    Your

     Special

    ization

    ],

     passionate

     about

     [

    Your

     Expert

    ise

    ].

     I

    've

     been

     [

    Your

     Experience

     with

     That

     Expert

    ise

    ]

     and

     I

    'm

     here

     to

     share

     my

     knowledge

     with

     you

    .

     Let

    's

     work

     together

     to

     uncover

     the

     mysteries

     of

     the

     world

     around

     us

    .
    


    Let

    's

     see

     where

     this

     adventure

     leads

     us

    !

     

    🌐

    📖

    🗺

    ️

    
    


    Always

     remember

    :

     You

    're

     not

     just

     any

     other

     human

    .

    
    
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

     
    


    **

    Question

    :**

     
    


    How

     many

     presidents

     have

     ever

     held

     the

     position

     of

     the

     President

     of

     France

    ?

     


    A

    .

     

    0

     


    B

    .

     

    1

     


    C

    .

     

    2

     


    D

    .

     

    4

     
    


    **

    Answer

    :**

     D

    
    


    **

    Explanation

    :**

     France

     has

     never

     had

     a

     president

     in

     its

     history

    .

     It

     was

     established

     as

     a

     republic

     in

     

    1

    8

    7

    5

    ,

     and

     the

     first

     president

     was

     not

     elected

     until

     

    1

    9

    1

    4

    .

     The

     current

     president

     is

     Emmanuel

     Macron

    .

     No

     other

     person

     has

     held

     the

     position

     of

     the

     President

     of

     France

    .
    


    The

     correct

     answer

     is

     D

    .

     

    4

    .

     However

    ,

     the

     provided

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     and

     it

     is

     expected

     to

     continue

     to

     grow

     in

     many

     different

     areas

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

     Increased

     Integration

     with

     IoT

    :

     AI

     is

     already

     integrated

     into

     many

     different

     devices

    ,

     but

     there

     is

     also

     potential

     for

     it

     to

     integrate

     even

     more

     widely

     into

     IoT

     (

    Internet

     of

     Things

    )

     devices

    .

     This

     would

     allow

     for

     more

     efficient

     and

     connected

     systems

     that

     could

     improve

     overall

     efficiency

     and

     productivity

    .
    


    2

    .

     Enhanced

     Predict

    ive

     Analytics

    :

     AI

     is

     being

     used

     for

     a

     wide

     range

     of

     applications

    ,

     including

     fraud

     detection

    ,

     healthcare

     diagnostics

    ,

     and

     predicting

     customer

     behavior

    .

     As

     AI

     becomes

     more

     advanced

    ,

     there

     is

     potential

     for

     it

     to

     provide

     even

     more

     accurate

     and

     predictive

     analytics

    



```python
llm.shutdown()
```
