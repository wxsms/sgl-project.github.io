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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.78it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:06<06:32,  6.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:06<06:32,  6.89s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:06<06:32,  6.89s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:06<06:32,  6.89s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:06<06:32,  6.89s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:55,  1.05s/it]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:55,  1.05s/it]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:07<00:55,  1.05s/it]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:07<00:55,  1.05s/it]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:07<00:55,  1.05s/it]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:07<00:55,  1.05s/it]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:20,  2.32it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:20,  2.32it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:07<00:20,  2.32it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:07<00:20,  2.32it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:07<00:20,  2.32it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:07<00:20,  2.32it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:07<00:20,  2.32it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:07<00:20,  2.32it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:08,  4.90it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:08,  4.90it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:08,  4.90it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:08,  4.90it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:07<00:08,  4.90it/s]

    Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:07<00:08,  4.90it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:07<00:04,  7.22it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:07<00:04,  7.22it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:07<00:04,  7.22it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:07<00:04,  7.22it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:07<00:04,  7.22it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:07<00:04,  7.22it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:07<00:04,  7.22it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:07<00:02, 10.75it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:07<00:02, 10.75it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:07<00:02, 10.75it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:07<00:02, 10.75it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:07<00:02, 10.75it/s]

    Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:07<00:02, 10.75it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:07<00:02, 10.75it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:07<00:01, 14.96it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:07<00:01, 14.96it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:07<00:01, 14.96it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:07<00:01, 14.96it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:07<00:01, 14.96it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:07<00:01, 14.96it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:07<00:01, 18.76it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:07<00:01, 18.76it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:07<00:01, 18.76it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:07<00:01, 18.76it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:07<00:01, 18.76it/s]

    Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:07<00:01, 18.76it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:07<00:01, 18.76it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:07<00:00, 24.07it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:07<00:00, 24.07it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:07<00:00, 24.07it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:07<00:00, 24.07it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:07<00:00, 24.07it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:07<00:00, 24.07it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:07<00:00, 24.07it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:07<00:00, 24.07it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:07<00:00, 31.00it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:07<00:00, 31.00it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:07<00:00, 31.00it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:07<00:00, 31.00it/s]

    Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:07<00:00, 31.00it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:07<00:00, 31.00it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:07<00:00, 31.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 16.11it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:03, 16.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:03, 16.11it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.02 GB):   7%|▋         | 4/58 [00:00<00:03, 17.56it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   7%|▋         | 4/58 [00:00<00:03, 17.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.01 GB):   7%|▋         | 4/58 [00:00<00:03, 17.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.00 GB):   7%|▋         | 4/58 [00:00<00:03, 17.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.00 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.00 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.00 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.67it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.99 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.99 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.33it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.33it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.82it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.98 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.82it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.82it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.62it/s]Capturing num tokens (num_tokens=960 avail_mem=70.96 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.62it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.62it/s]Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  40%|███▉      | 23/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=832 avail_mem=70.95 GB):  40%|███▉      | 23/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=768 avail_mem=70.95 GB):  40%|███▉      | 23/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  40%|███▉      | 23/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=640 avail_mem=70.94 GB):  40%|███▉      | 23/58 [00:00<00:01, 29.40it/s]Capturing num tokens (num_tokens=640 avail_mem=70.94 GB):  47%|████▋     | 27/58 [00:00<00:00, 31.69it/s]Capturing num tokens (num_tokens=576 avail_mem=70.94 GB):  47%|████▋     | 27/58 [00:00<00:00, 31.69it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.69it/s]

    Capturing num tokens (num_tokens=480 avail_mem=70.94 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.69it/s]Capturing num tokens (num_tokens=448 avail_mem=70.94 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.69it/s]Capturing num tokens (num_tokens=448 avail_mem=70.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.70it/s]Capturing num tokens (num_tokens=416 avail_mem=70.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.70it/s]Capturing num tokens (num_tokens=384 avail_mem=70.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.70it/s]Capturing num tokens (num_tokens=352 avail_mem=70.93 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.70it/s]Capturing num tokens (num_tokens=320 avail_mem=70.93 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.70it/s]Capturing num tokens (num_tokens=320 avail_mem=70.93 GB):  60%|██████    | 35/58 [00:01<00:00, 30.47it/s]Capturing num tokens (num_tokens=288 avail_mem=70.92 GB):  60%|██████    | 35/58 [00:01<00:00, 30.47it/s]

    Capturing num tokens (num_tokens=256 avail_mem=70.92 GB):  60%|██████    | 35/58 [00:01<00:00, 30.47it/s]Capturing num tokens (num_tokens=240 avail_mem=70.92 GB):  60%|██████    | 35/58 [00:01<00:00, 30.47it/s]Capturing num tokens (num_tokens=224 avail_mem=70.91 GB):  60%|██████    | 35/58 [00:01<00:00, 30.47it/s]Capturing num tokens (num_tokens=224 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.76it/s]Capturing num tokens (num_tokens=208 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.76it/s]Capturing num tokens (num_tokens=192 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.76it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.76it/s]

    Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.10it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.10it/s]Capturing num tokens (num_tokens=144 avail_mem=70.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.10it/s]Capturing num tokens (num_tokens=128 avail_mem=70.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.10it/s]Capturing num tokens (num_tokens=112 avail_mem=70.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.10it/s]Capturing num tokens (num_tokens=112 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 30.76it/s]Capturing num tokens (num_tokens=96 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 30.76it/s] Capturing num tokens (num_tokens=80 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 30.76it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 30.76it/s]

    Capturing num tokens (num_tokens=48 avail_mem=70.88 GB):  79%|███████▉  | 46/58 [00:01<00:00, 30.76it/s]Capturing num tokens (num_tokens=48 avail_mem=70.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.98it/s]Capturing num tokens (num_tokens=32 avail_mem=70.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.98it/s]Capturing num tokens (num_tokens=28 avail_mem=70.87 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.98it/s]Capturing num tokens (num_tokens=24 avail_mem=70.87 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.98it/s]Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.98it/s]Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.29it/s]Capturing num tokens (num_tokens=16 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.29it/s]

    Capturing num tokens (num_tokens=12 avail_mem=70.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.29it/s]Capturing num tokens (num_tokens=8 avail_mem=70.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.29it/s] Capturing num tokens (num_tokens=8 avail_mem=70.86 GB):  98%|█████████▊| 57/58 [00:02<00:00, 29.05it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  98%|█████████▊| 57/58 [00:02<00:00, 29.05it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:02<00:00, 28.31it/s]


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
    Generated text:  Esteban and I have a lot of confidence in myself and my abilities. I have been a strong supporter of the LGBT community, and I have also been a huge supporter of the cause of the disabled community. My parents are both with me in the fight for the freedom and rights of everyone.
    
    My parents have been my biggest supporters and followers since my birth. I was born with a condition called the Congenital Heart Disease. I am a member of the Conservative Party and a member of the LGBT community. I have been in the military and I am also a former member of the Royal Military Institute. I am a member of the
    ===============================
    Prompt: The president of the United States is
    Generated text:  20 years older than the president of Brazil. The president of Brazil is 20 years younger than the president of the United States. How old is the president of the United States? Let's denote the age of the president of the United States as \( U \) and the president of Brazil as \( B \).
    
    From the information given, we have the following relationships:
    
    1. The president of the United States is 20 years older than the president of Brazil:
    \[ U = B + 20 \]
    
    2. The president of Brazil is 20 years younger than the president of the United States:
    \[
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Lille
    C. Strasbourg
    D. Lorraine
    答案：A
    解析：根据题意，巴黎是法国的首都，故选A。
    ===============================
    Prompt: The future of AI is
    Generated text:  predicting a quantum leap in machine learning, deep learning, and neural networks. The trend of AI-driven machine learning and deep learning is anticipated to grow at an annual rate of 17.8% and will be the backbone for the development of AI in the future. This growth will contribute to the advancement of AI technology and enable it to solve complex problems in fields such as finance, healthcare, and education. AI-driven machine learning and deep learning are expected to lead the development of AI in the future, and they will have a significant impact on the industry and society at large. Machine learning and deep learning algorithms can analyze large amounts of data


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your personality or skills]. I enjoy [insert a short description of your hobbies or interests]. What do you like to do in your free time? I enjoy [insert a short description of your hobbies or interests]. What's your favorite hobby? I love [insert a short description of your favorite hobby]. What's your favorite book or movie? I love [insert a short description of your favorite book or movie].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and the French Revolution. Paris is a popular tourist destination, with many famous landmarks and museums to explore. It is also home to the French Parliament and the French National Library. The city is known for its cuisine, including French cuisine, and is a popular destination for tourists and locals alike. Paris is a vibrant and dynamic city with a rich history and culture. The Eiffel Tower, Louvre
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced privacy and security: As AI becomes more prevalent in our daily lives, there will be an increased need for privacy and security measures to protect against data breaches and other forms of cyber attacks.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs.
    


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
    Generated text:  [name], and I'm a [job title] at [company name]. I specialize in [specific skill or expertise]. I've been working in this field for [number of years] years, and I'm constantly striving to [what you can do to improve]. I'm excited to bring my [specific area of expertise or experience] to your team. What can you tell me about yourself? [introduce yourself briefly, including a professional background, skills, and experience relevant to the position]. [introduce yourself in a neutral, neutral tone, with a neutral tone of voice, or the appropriate tone]. How can I help you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city and a major cultural, economic, and political center in Europe. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum, as well as its history, art, and cuisine. The city is also a center for scientific research, theater, and music. Paris is home to many different cultural and political movements, including the French Revolution, the French Resistance, and the French New Wave. Today, Paris remains one of the world's most important cities, hosting many of the world's most famous museums, landmarks, and events.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve several trends that are likely to shape how we use and develop AI technologies:
    
      1. Increased automation: As AI becomes more advanced, it is likely to be used in a wider range of tasks, from manufacturing to healthcare, and as a result, there will be an increased focus on automation in the use of AI. This will lead to the development of new tools and techniques for automating tasks and processes, which will make the work of humans more efficient and effective.
      2. Greater integration with human intelligence: As AI becomes more advanced, it is likely to be integrated more closely with human intelligence, which will


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

     ____

    _.

     I

    'm

     a

     young

     professional

     with

     a

     passion

     for

     technology

     and

     innovation

    .

     I

     love

     exploring

     new

     ideas

     and

     working

     on

     projects

     that

     push

     the

     boundaries

     of

     what

    's

     possible

    .

     I

    'm

     always

     looking

     for

     new

     opportunities

     and

     learning

     new

     skills

     to

     stay

     up

    -to

    -date

     with

     the

     latest

     trends

     in

     technology

    .

     I

     enjoy

     collaborating

     with

     other

     professionals

     and

     sharing

     my

     knowledge

     with

     others

    .

     And

     most

     of

     all

    ,

     I

    'm

     a

     person

     who

     values

     creativity

    ,

     hard

     work

    ,

     and

     the

     opportunity

     to

     make

     a

     meaningful

     impact

     on

     the

     world

     around

     us

    .

     Thank

     you

    .

     [

    Your

     name

    ]

     [

    Your

     profession

    ]

     [

    Your

     hobbies

     and

     interests

    ]

     [

    Your

     career

     goals

     and

     aspirations

    ]

     [

    Your

     personal

     qualities

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     cosm

    opolitan

     city

     known

     for

     its

     rich

     history

     and

     beautiful

     architecture

    .

     The

     city

     is

     also

     home

     to

     many

     museums

    ,

     theaters

    ,

     and

     culinary

     districts

     that

     attract

     visitors

     from

     all

     over

     the

     world

    .

     Paris

     is

     a

     major

     economic

     and

     cultural

     center

     and

     is

     a

     popular

     tourist

     destination

    .

     The

     city

     is

     also

     known

     for

     its

     world

    -ren

    owned

     landmarks

    ,

     including

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

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     a

     city

     of

     contrasts

     and

     beautiful

     vistas

    ,

     making

     it

     a

     popular

     destination

     for

     both

     locals

     and

     tourists

     alike

    .

     The

     city

     is

     home

     to

     the

     iconic

     E

    iff

    el

     Tower

     and

     the

     Arc

     de

     Tri

    omp

    he

    ,

     which

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     there

     are

     several

     possible

     trends

     that

     could

     shape

     its

     development

    .

     Here

     are

     some

     of

     the

     most

     likely

     ones

    :
    


    1

    .

     Increased

     Use

     of

     AI

     in

     Healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     improve

     patient

     outcomes

     and

     reduce

     medical

     errors

    .

     In

     the

     future

    ,

     we

     can

     expect

     AI

     to

     continue

     to

     be

     integrated

     into

     healthcare

     systems

    ,

     with

     more

     sophisticated

     algorithms

     and

     machine

     learning

     techniques

     being

     developed

     to

     improve

     patient

     care

    .
    


    2

    .

     Emer

    gence

     of

     AI

     in

     the

     Manufacturing

     Industry

    :

     AI

     is

     already

     being

     used

     to

     improve

     production

     efficiency

     and

     reduce

     errors

     in

     manufacturing

    .

     In

     the

     future

    ,

     we

     can

     expect

     AI

     to

     be

     even

     more

     integrated

     into

     the

     manufacturing

     industry

    ,

     with

     more

     advanced

     algorithms

     being

    



```python
llm.shutdown()
```
