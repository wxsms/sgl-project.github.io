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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.73it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:24,  2.08it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:24,  2.08it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:24,  2.08it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:24,  2.08it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:13,  3.68it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:13,  3.68it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:13,  3.68it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:13,  3.68it/s]

    Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:13,  3.68it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:07,  6.41it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:07,  6.41it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:07,  6.41it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:07,  6.41it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:07,  6.41it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:04<00:07,  6.41it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:04<00:07,  6.41it/s]Compiling num tokens (num_tokens=1280):  22%|██▏       | 13/58 [00:04<00:07,  6.41it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03, 12.53it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03, 12.53it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03, 12.53it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03, 12.53it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03, 12.53it/s]

    Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03, 12.53it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:03, 12.53it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 18.20it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 18.20it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 18.20it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 18.20it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 18.20it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:01, 18.20it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:01, 18.20it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 23.99it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 23.99it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 23.99it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 23.99it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 23.99it/s]

    Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 23.99it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 23.99it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 23.99it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 31.11it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 31.11it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 31.11it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 31.11it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 31.11it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 31.11it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 31.11it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 36.55it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 36.55it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 36.55it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 42.60it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 42.60it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 42.60it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 42.60it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 42.60it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 42.60it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 42.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.48it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.08 GB):   3%|▎         | 2/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.07 GB):   3%|▎         | 2/58 [00:00<00:02, 19.19it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.07 GB):   3%|▎         | 2/58 [00:00<00:02, 19.19it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.07 GB):   7%|▋         | 4/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.07 GB):   7%|▋         | 4/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.06 GB):   7%|▋         | 4/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.05 GB):   7%|▋         | 4/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.05 GB):  12%|█▏        | 7/58 [00:00<00:02, 23.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.05 GB):  12%|█▏        | 7/58 [00:00<00:02, 23.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.05 GB):  12%|█▏        | 7/58 [00:00<00:02, 23.39it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.05 GB):  12%|█▏        | 7/58 [00:00<00:02, 23.39it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.04 GB):  12%|█▏        | 7/58 [00:00<00:02, 23.39it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=59.04 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.04 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.03 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.03 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.03 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.03 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.03 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.02 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.02 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.02it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=59.02 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.01 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.01 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.00 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=960 avail_mem=59.01 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.57it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=59.01 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=896 avail_mem=59.01 GB):  40%|███▉      | 23/58 [00:00<00:01, 23.54it/s]Capturing num tokens (num_tokens=832 avail_mem=59.00 GB):  40%|███▉      | 23/58 [00:00<00:01, 23.54it/s]Capturing num tokens (num_tokens=768 avail_mem=59.00 GB):  40%|███▉      | 23/58 [00:00<00:01, 23.54it/s]Capturing num tokens (num_tokens=704 avail_mem=59.00 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.54it/s]

    Capturing num tokens (num_tokens=704 avail_mem=59.00 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.04it/s]Capturing num tokens (num_tokens=640 avail_mem=58.99 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.04it/s]Capturing num tokens (num_tokens=576 avail_mem=58.99 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.04it/s]Capturing num tokens (num_tokens=512 avail_mem=58.98 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.04it/s]Capturing num tokens (num_tokens=512 avail_mem=58.98 GB):  50%|█████     | 29/58 [00:01<00:01, 22.01it/s]Capturing num tokens (num_tokens=480 avail_mem=58.99 GB):  50%|█████     | 29/58 [00:01<00:01, 22.01it/s]Capturing num tokens (num_tokens=448 avail_mem=58.99 GB):  50%|█████     | 29/58 [00:01<00:01, 22.01it/s]

    Capturing num tokens (num_tokens=416 avail_mem=58.99 GB):  50%|█████     | 29/58 [00:01<00:01, 22.01it/s]Capturing num tokens (num_tokens=416 avail_mem=58.99 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.64it/s]Capturing num tokens (num_tokens=384 avail_mem=58.99 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.64it/s]Capturing num tokens (num_tokens=352 avail_mem=58.98 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.64it/s]Capturing num tokens (num_tokens=320 avail_mem=58.98 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.64it/s]Capturing num tokens (num_tokens=320 avail_mem=58.98 GB):  60%|██████    | 35/58 [00:01<00:01, 22.84it/s]Capturing num tokens (num_tokens=288 avail_mem=58.97 GB):  60%|██████    | 35/58 [00:01<00:01, 22.84it/s]

    Capturing num tokens (num_tokens=256 avail_mem=58.97 GB):  60%|██████    | 35/58 [00:01<00:01, 22.84it/s]Capturing num tokens (num_tokens=240 avail_mem=58.97 GB):  60%|██████    | 35/58 [00:01<00:01, 22.84it/s]Capturing num tokens (num_tokens=240 avail_mem=58.97 GB):  66%|██████▌   | 38/58 [00:01<00:00, 22.20it/s]Capturing num tokens (num_tokens=224 avail_mem=58.96 GB):  66%|██████▌   | 38/58 [00:01<00:00, 22.20it/s]Capturing num tokens (num_tokens=208 avail_mem=58.96 GB):  66%|██████▌   | 38/58 [00:01<00:00, 22.20it/s]Capturing num tokens (num_tokens=192 avail_mem=58.96 GB):  66%|██████▌   | 38/58 [00:01<00:00, 22.20it/s]

    Capturing num tokens (num_tokens=192 avail_mem=58.96 GB):  71%|███████   | 41/58 [00:01<00:00, 21.55it/s]Capturing num tokens (num_tokens=176 avail_mem=58.96 GB):  71%|███████   | 41/58 [00:01<00:00, 21.55it/s]Capturing num tokens (num_tokens=160 avail_mem=58.95 GB):  71%|███████   | 41/58 [00:01<00:00, 21.55it/s]Capturing num tokens (num_tokens=144 avail_mem=58.95 GB):  71%|███████   | 41/58 [00:01<00:00, 21.55it/s]Capturing num tokens (num_tokens=144 avail_mem=58.95 GB):  76%|███████▌  | 44/58 [00:01<00:00, 21.41it/s]Capturing num tokens (num_tokens=128 avail_mem=58.95 GB):  76%|███████▌  | 44/58 [00:01<00:00, 21.41it/s]Capturing num tokens (num_tokens=112 avail_mem=58.95 GB):  76%|███████▌  | 44/58 [00:01<00:00, 21.41it/s]

    Capturing num tokens (num_tokens=96 avail_mem=58.94 GB):  76%|███████▌  | 44/58 [00:01<00:00, 21.41it/s] Capturing num tokens (num_tokens=96 avail_mem=58.94 GB):  81%|████████  | 47/58 [00:02<00:00, 21.62it/s]Capturing num tokens (num_tokens=80 avail_mem=58.94 GB):  81%|████████  | 47/58 [00:02<00:00, 21.62it/s]Capturing num tokens (num_tokens=64 avail_mem=58.93 GB):  81%|████████  | 47/58 [00:02<00:00, 21.62it/s]Capturing num tokens (num_tokens=48 avail_mem=58.93 GB):  81%|████████  | 47/58 [00:02<00:00, 21.62it/s]Capturing num tokens (num_tokens=32 avail_mem=58.93 GB):  81%|████████  | 47/58 [00:02<00:00, 21.62it/s]Capturing num tokens (num_tokens=32 avail_mem=58.93 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.48it/s]Capturing num tokens (num_tokens=28 avail_mem=58.92 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.48it/s]Capturing num tokens (num_tokens=24 avail_mem=58.92 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.48it/s]Capturing num tokens (num_tokens=20 avail_mem=58.92 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.48it/s]

    Capturing num tokens (num_tokens=16 avail_mem=58.92 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.48it/s]Capturing num tokens (num_tokens=12 avail_mem=58.91 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.48it/s]Capturing num tokens (num_tokens=12 avail_mem=58.91 GB):  97%|█████████▋| 56/58 [00:02<00:00, 31.17it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:02<00:00, 31.17it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:02<00:00, 31.17it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:02<00:00, 23.88it/s]


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
    Generated text:  Sven Hulthén, a philosophy major and neuroscience major at the University of Minnesota. My research focuses on the relationship between language and the cerebral cortex. My work centers on understanding the way that the brain interprets and produces language, and how language and meaning are encoded and retrieved, and how language and memory are related. The work I do is interdisciplinary in nature; I use a combination of cognitive neuroscience and cognitive linguistics in my research, and I have been involved with a number of research projects focused on understanding the language processing system of the human brain. As a neuroscience major, I have worked closely with neuroscientists and cognitive
    ===============================
    Prompt: The president of the United States is
    Generated text:  a __________, representing the country in international organizations such as the United Nations, the G7, the G20, the OECD, the World Trade Organization, and the Organization for Security and Co-operation in Europe. (1 point) 
    A. Head of State 
    B. Head of Government 
    C. Head of Government 
    D. Head of the State 
    Answer:
    A
    
    A transformer's high-voltage side has 256 turns per volt, and its low-voltage side has 1024 turns per volt. The primary current of the transformer is 80 A. What is the secondary current?
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. [ ]
    A. Paris
    B. London
    C. New York
    D. Moscow
    Answer:
    
    A
    
    The capital of Canada is ________. [ ]
    A. Paris
    B. London
    C. New York
    D. Moscow
    Answer:
    
    A
    
    Which of the following pairs of countries has the capital in both Beijing and Beijing?
    A. France and Germany
    B. Japan and South Korea
    C. France and Japan
    D. Italy and Japan
    Answer:
    
    B
    
    Which of the following cities has a capital in both Beijing and Shanghai?
    A. Hong Kong
    B. Nanjing
    C. Sheny
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s revolutionizing the way we do things. From self-driving cars to intelligent speech recognition, AI is driving the future of technology. But how can you make sure that your AI project stays on track? In this article, we’ll explore the key aspects of creating a successful AI project and what to consider when selecting a tool for your project.
    First, it’s important to understand what AI is and how it can benefit your business. AI refers to the ability of machines to learn and process information from data, and it can be used to automate tasks, improve decision-making, and enhance user experience. By understanding the basics of


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive description of your personality or skills]. I enjoy [insert a short, positive description of your hobbies or interests]. I'm always looking for new opportunities to grow and learn, and I'm always eager to learn more about the world around me. What's your favorite hobby or activity? I love [insert a short, positive description of your favorite hobby or activity]. I'm always looking for new experiences and adventures,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also known for its vibrant culture, including its famous annual festivals such as the Eiffel Tower Parade and the Carnaval. The city is a popular tourist destination and a major economic center in France. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant culture. Its status as
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to more sophisticated and adaptive AI systems that can learn from human behavior and adapt to new situations.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as increased scrutiny of AI systems in the public eye.
    
    3. Increased use of AI in healthcare
    


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
    Generated text:  [Your Name], and I am a [insert your age] year old aspiring young professional. I am a [insert your profession] with a passion for [insert your area of interest or hobby]. I am always up for a challenge and always looking to learn more about my field. I am a [insert your personality trait or skill] and I am always looking for opportunities to grow and improve. I am a [insert your hobby or interest] and I am always looking for new ways to expand my knowledge. I am a [insert your profession or hobby] who is always looking to learn and improve. As you can see, I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located on the Seine River in the center of the country.
    Here's a concise factual statement about France's capital city:
    
    Paris is the capital and largest city of France. Its name means "City of Light," referring to its well-preserved Gothic architecture and vibrant cultural scene. The city is home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also known for its diverse food and wine cultures, as well as its fashion and film industry. The city is renowned for its atmospheric lighting and annual festivals like the Parisian Impressionism exhibition. (Note: While Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to evolve rapidly and include new technologies and applications. Some possible future trends include:
    
    1. Increased integration of AI into other areas of technology, such as healthcare, transportation, and manufacturing.
    
    2. AI will continue to be used for tasks that require human-like intelligence, such as language translation and problem-solving.
    
    3. AI will become more capable of learning and improving on its own, leading to greater efficiency and accuracy in various tasks.
    
    4. AI will become more integrated with the physical world, allowing for more seamless and interactive interactions between humans and machines.
    
    5. AI will continue to improve through the development of new algorithms, hardware


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

     fictional

     character

    's

     name

    ].

     I

     am

     a

     [

    insert

     fictional

     character

    's

     age

    ,

     gender

    ,

     and

     profession

     or

     occupation

    ].

     I

     have

     always

     been

     passionate

     about

     [

    insert

     a

     relevant

     hobby

     or

     interest

    ].

     I

     have

     [

    insert

     how

     I

     achieve

     my

     hobbies

     or

     interests

    ].

     I

     am

     always

     looking

     for

     ways

     to

     [

    insert

     how

     I

     achieve

     this

     goal

    ].

     I

    'm

     [

    insert

     how

     I

     got

     started

     in

     my

     field

     or

     profession

    ].

     I

     enjoy

     [

    insert

     how

     I

     spend

     my

     free

     time

    ].

     I

    'm

     [

    insert

     how

     I

     would

     describe

     my

     character

    ].

     I

     believe

     in

     [

    insert

     something

     that

     reflects

     my

     values

     or

     beliefs

    ].

     My

     passions

     have

     always

     been

     [

    insert

     a

     relevant

     word

     or

     phrase

    ],

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     heart

     of

     the

     French

     Riv

    iera

    ,

     known

     for

     its

     historic

     can

    als

    ,

     art

     nouveau

     architecture

    ,

     and

     vibrant

     cultural

     scene

    .

     It

     is

     also

     one

     of

     the

     most

     important

     cities

     in

     Europe

     and

     a

     major

     economic

    ,

     political

    ,

     and

     cultural

     center

    .

     France

    's

     capital

     is

     Paris

    ,

     also

     known

     as

     "

    La

     Petite

    -P

    atri

    e

    ,"

     and

     it

     is

     home

     to

     the

     most

     important

     cities

     in

     France

    .

     Its

     name

     "

    Paris

    "

     is

     derived

     from

     the

     Latin

     word

     "

    Paris

    ina

    ,"

     which

     means

     "

    red

     city

    ."

     The

     city

    's

     population

     is

     over

     

    2

    .

    5

     million

    ,

     and

     it

     is

     one

     of

     the

     largest

     cities

     in

     the

     world

    ,

     with

     its

     streets

     and

     landmarks

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

     and

     there

     are

     many

     exciting

     developments

     in

     the

     industry

     that

     are

     shaping

     its

     evolution

    .

     Here

     are

     some

     potential

     trends

     to

     watch

     out

     for

     in

     the

     near

     future

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

     and

     deep

     learning

    :

     AI

     is

     expected

     to

     continue

     to

     advance

     rapidly

     in

     the

     coming

     years

    ,

     with

     more

     sophisticated

     models

     being

     developed

    .

     This

     includes

     advancements

     in

     techniques

     like

     neural

     networks

     and

     gener

    ative

     advers

    arial

     networks

     (

    GAN

    s

    ),

     which

     have

     the

     potential

     to

     surpass

     human

     intelligence

     in

     certain

     domains

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     there

     is

     potential

     for

     it

     to

     be

     used

     in

     healthcare

     to

     improve

     patient

     outcomes

     and

     streamline

     medical

     practices

    .

     This

     could

    



```python
llm.shutdown()
```
