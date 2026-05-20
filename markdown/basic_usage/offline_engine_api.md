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


    2026-05-20 09:07:13,268 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 09:07:13] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:31,  1.66it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:31,  1.66it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.93it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.99it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.99it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.99it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.99it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.99it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.99it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.99it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.99it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.99it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.99it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.55it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.55it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.55it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.55it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.55it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.55it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.55it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.55it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.55it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.55it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.55it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.78it/s]

    Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.78it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 29.62it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 29.62it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 29.62it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 29.62it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 29.62it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 29.62it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 29.62it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 29.62it/s]

    Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:04<00:00, 34.68it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:04<00:00, 34.68it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:04<00:00, 34.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.01 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.01 GB):   2%|▏         | 1/58 [00:00<00:06,  9.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.98 GB):   2%|▏         | 1/58 [00:00<00:06,  9.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.98 GB):   2%|▏         | 1/58 [00:00<00:06,  9.40it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=55.98 GB):   5%|▌         | 3/58 [00:00<00:04, 11.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.98 GB):   5%|▌         | 3/58 [00:00<00:04, 11.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.98 GB):   5%|▌         | 3/58 [00:00<00:04, 11.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.98 GB):   9%|▊         | 5/58 [00:00<00:03, 14.45it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.97 GB):   9%|▊         | 5/58 [00:00<00:03, 14.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.96 GB):   9%|▊         | 5/58 [00:00<00:03, 14.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.95 GB):   9%|▊         | 5/58 [00:00<00:03, 14.45it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=55.95 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.36it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.95 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.36it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.95 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.94 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.94 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.94 GB):  21%|██        | 12/58 [00:00<00:01, 23.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.94 GB):  21%|██        | 12/58 [00:00<00:01, 23.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.94 GB):  21%|██        | 12/58 [00:00<00:01, 23.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.93 GB):  21%|██        | 12/58 [00:00<00:01, 23.04it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=55.93 GB):  21%|██        | 12/58 [00:00<00:01, 23.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.93 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.92 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.56it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.92 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.56it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.92 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.56it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.92 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.56it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.90 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.56it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.90 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.21it/s]Capturing num tokens (num_tokens=960 avail_mem=55.91 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.21it/s] Capturing num tokens (num_tokens=896 avail_mem=55.91 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.21it/s]Capturing num tokens (num_tokens=832 avail_mem=55.90 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.21it/s]

    Capturing num tokens (num_tokens=768 avail_mem=55.90 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.21it/s]Capturing num tokens (num_tokens=704 avail_mem=55.90 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.21it/s]Capturing num tokens (num_tokens=704 avail_mem=55.90 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.06it/s]Capturing num tokens (num_tokens=640 avail_mem=55.90 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.06it/s]Capturing num tokens (num_tokens=576 avail_mem=55.90 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.06it/s]Capturing num tokens (num_tokens=512 avail_mem=55.88 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.06it/s]Capturing num tokens (num_tokens=480 avail_mem=55.90 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.06it/s]Capturing num tokens (num_tokens=448 avail_mem=55.89 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.06it/s]Capturing num tokens (num_tokens=416 avail_mem=55.89 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.06it/s]Capturing num tokens (num_tokens=416 avail_mem=55.89 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.33it/s]Capturing num tokens (num_tokens=384 avail_mem=55.89 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.33it/s]Capturing num tokens (num_tokens=352 avail_mem=55.88 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.33it/s]

    Capturing num tokens (num_tokens=320 avail_mem=55.88 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.33it/s]Capturing num tokens (num_tokens=288 avail_mem=55.88 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.33it/s]Capturing num tokens (num_tokens=256 avail_mem=55.87 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.33it/s]Capturing num tokens (num_tokens=256 avail_mem=55.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=240 avail_mem=55.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=224 avail_mem=55.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=208 avail_mem=55.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.60it/s]

    Capturing num tokens (num_tokens=192 avail_mem=55.83 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=176 avail_mem=55.83 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=176 avail_mem=55.83 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=160 avail_mem=55.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=144 avail_mem=55.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=128 avail_mem=55.81 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.00it/s]

    Capturing num tokens (num_tokens=112 avail_mem=55.81 GB):  72%|███████▏  | 42/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=112 avail_mem=55.81 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.11it/s]Capturing num tokens (num_tokens=96 avail_mem=55.79 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.11it/s] Capturing num tokens (num_tokens=80 avail_mem=55.79 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.11it/s]Capturing num tokens (num_tokens=64 avail_mem=55.78 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.11it/s]Capturing num tokens (num_tokens=48 avail_mem=55.77 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.11it/s]Capturing num tokens (num_tokens=48 avail_mem=55.77 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.72it/s]Capturing num tokens (num_tokens=32 avail_mem=55.77 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.72it/s]

    Capturing num tokens (num_tokens=28 avail_mem=55.76 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.72it/s]Capturing num tokens (num_tokens=24 avail_mem=55.75 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.72it/s]Capturing num tokens (num_tokens=20 avail_mem=55.75 GB):  86%|████████▌ | 50/58 [00:01<00:00, 28.72it/s]Capturing num tokens (num_tokens=20 avail_mem=55.75 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.50it/s]Capturing num tokens (num_tokens=16 avail_mem=55.74 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.50it/s]Capturing num tokens (num_tokens=12 avail_mem=55.73 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.50it/s]Capturing num tokens (num_tokens=8 avail_mem=55.73 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.50it/s] Capturing num tokens (num_tokens=4 avail_mem=55.72 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.50it/s]

    Capturing num tokens (num_tokens=4 avail_mem=55.72 GB): 100%|██████████| 58/58 [00:01<00:00, 31.30it/s]Capturing num tokens (num_tokens=4 avail_mem=55.72 GB): 100%|██████████| 58/58 [00:01<00:00, 29.45it/s]


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
    Generated text:  Bill. My favorite field of study is Economics. I am a research assistant in the Economics Department at the University of California, Los Angeles, and I have been working on a project to find out why people prefer to take a risk. To do that, we have to come up with a statistical model that takes into account the factors that affect people's choice of risk-taking behavior. We are going to start by collecting data on various variables that can affect people's risk-taking behavior such as the amount of money, the time of day, and the personal information of the people we will study. After collecting the data, we will use statistical techniques
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to introduce a bill that will prevent the government from spending money on a new stadium for the national basketball team. The president estimates that each year the value of the stadium will increase by 2% of its current value. The current value of the stadium is $100 million. How much will the stadium be worth in 10 years?
    To determine the value of the stadium in 10 years, we can use the formula for compound interest, which is given by:
    
    \[ V = P (1 + r)^n \]
    
    where:
    - \( V \) is the future value of the investment/loan, including
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The French parliament is in the city of Paris. The French government's office is also in Paris. The Parisian people work hard to make Paris beautiful, and to keep the streets clean, making it a nice place to visit. As a foreigner, you have to be very careful if you visit Paris. You can't smoke in the public places there. In Paris, you also have to take a lot of care to protect yourself from the sun. You can't have a burning sun on your skin. If you are a foreigner, you can't drink alcohol while you are in Paris either. The French people are very
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of many different companies. This article will look at the different types of companies that will dominate the future of AI and how they will impact the overall landscape of AI. AI will continue to drive innovation in healthcare, transportation, manufacturing, and many other sectors. In the coming years, we may see AI becoming more widespread and integrated into our daily lives, and potentially even changing the way we perceive the world around us. The following are some of the key areas that AI will be influencing:
    1. Healthcare – AI will be used to improve the accuracy and speed of medical diagnoses, predict disease outbreaks, and develop personalized treatment plans.


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm passionate about [reason for being at the company]. I'm always looking for ways to [what I enjoy doing at the company]. I'm excited to [what I hope to achieve at the company]. I'm looking forward to [what I hope to learn about the company]. I'm looking forward to [what I hope to do for the company]. I'm looking forward to [what I hope to do for the company]. I'm looking forward to [what I hope to do for the company]. I'm looking forward to [what I hope to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many of the world's most prestigious institutions. Paris is a popular tourist destination, with millions of visitors annually, and is a cultural and economic hub in Europe. It is also a major center for international diplomacy and has a rich history dating back to the Roman Empire. The city is known for its diverse cuisine, including French cuisine, and is home to many museums, theaters, and other cultural institutions. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems.
    
    2. Enhanced machine learning capabilities: Machine learning algorithms will continue to improve, allowing AI systems to learn from more complex data and make more accurate predictions and decisions.
    
    3. Greater emphasis on ethical considerations: As AI systems become more integrated with human intelligence, there will be increased focus on ethical considerations, such as privacy, bias, and accountability.
    
    4. Increased use of AI in healthcare:
    


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
    Generated text:  [Character's Name]. I am a [Occupation or Profession] who has been working in the [Industry/Field] for [Number of Years] years. I'm passionate about [Reason Why I Love the Profession]. My goal is to [Define My Goal]. I'm always looking to [Choose a Trait or Qualification You Value Most]. If you want to connect with someone like me, please feel free to reach out. Here's my contact information below:
    
    [Contact Information]
    
    ---
    
    [Your Name]
    
    [Your Position]
    
    [Your Contact Information]
    
    ---
    
    ---
    
    [Your Name]
    
    [Your Occupation/Profession]
    
    [Your Contact
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city that hosts the Eiffel Tower and is known for its rich history, art, and cultural scene. It is also the largest city in the country and a major economic hub. Paris has a rich history dating back to the ancient Romans, and today it is a global center of politics, culture, and fashion. The city is also home to some of the world's most famous landmarks, including the Louvre Museum, Notre Dame Cathedral, and the Arc de Triomphe.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a number of trends and developments. Here are some potential future trends in AI:
    
    1. Advancements in machine learning and deep learning: This trend is expected to continue as AI developers improve the models and algorithms used to train them. This will lead to greater accuracy and efficiency in AI systems, and could have a wide range of applications, from healthcare to finance.
    
    2. Integration of AI with other technologies: As AI becomes more integrated with other technologies, such as blockchain and virtual reality, we may see new applications emerge. For example, AI could be used to create more efficient supply chains, or to improve the quality


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

     your

     occupation

    ]

     with

     a

     passion

     for

     [

    insert

     the

     reason

     for

     your

     passion

    ].

     I

     enjoy

     reading

     and

     exploring

     the

     world

    ,

     and

     I

     love

     to

     travel

     to

     new

     places

     to

     experience

     new

     cultures

     and

     meet

     new

     people

    .

     I

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     learn

    .

     I

     am

     a

     [

    insert

     your

     age

    ]

     year

     old

     who

     is

     [

    insert

     your

     occupation

    ].

     I

     am

     always

     looking

     for

     ways

     to

     make

     the

     world

     a

     better

     place

     and

     to

     inspire

     others

     to

     do

     the

     same

    .

     What

    's

     your

     name

    ?

     What

    's

     your

     occupation

    ?

     What

     are

     your

     passions

    ?

     What

     are

     your

     goals

    ?

     I

    'm

     looking

     for

     someone

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     with

     the

     most

     populous

     population

     in

     the

     country

     and

     the

     seat

     of

     the

     government

    .

     It

     is

     the

     capital

     of

     France

     and

     the

     fourth

     most

     populous

     city

     in

     the

     world

    ,

     with

     an

     estimated

     population

     of

     over

     

    1

    3

     million

    .

     Paris

     is

     famous

     for

     its

     iconic

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     Notre

     Dame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    ,

     as

     well

     as

     its

     vibrant

     cultural

     scene

     and

     festive

     celebrations

    .

     It

     is

     also

     known

     for

     its

     rich

     history

    ,

     including

     the

     French

     Revolution

     and

     its

     influence

     on

     art

    ,

     literature

    ,

     and

     cuisine

    .

     The

     city

     is

     home

     to

     over

     

    3

    5

    0

     million

     visitors

     annually

     and

     is

     a

     major

     transportation

     hub

     for

     France

     and

     Europe

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     with

     many

     exciting

     developments

     on

     the

     horizon

    .

     Here

     are

     some

     potential

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

     Personal

    ized

     AI

    :

     AI

     will

     become

     even

     more

     personal

     and

     tailored

     to

     individual

     needs

    ,

     leading

     to

     more

     efficient

     and

     effective

     decision

    -making

     processes

    .

     This

     will

     be

     enabled

     by

     advanced

     machine

     learning

     algorithms

     that

     can

     analyze

     vast

     amounts

     of

     data

     to

     identify

     patterns

     and

     make

     predictions

     about

     future

     outcomes

    .
    


    2

    .

     Autonomous

     AI

    :

     Autonomous

     AI

     will

     be

     able

     to

     make

     decisions

     without

     human

     intervention

    ,

     leading

     to

     significant

     improvements

     in

     safety

     and

     efficiency

    .

     This

     technology

     will

     be

     enabled

     by

     the

     development

     of

     AI

     that

     can

     operate

     independently

     without

     human

     oversight

    .
    


    3

    .

     AI

     in

     healthcare

    :

    



```python
llm.shutdown()
```
