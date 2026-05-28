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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.37s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.37s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:09,  4.37s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:09,  4.37s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:06<00:04,  7.25it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:06<00:04,  7.25it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:06<00:04,  7.25it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:06<00:04,  7.25it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:06<00:04,  7.25it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:06<00:04,  7.25it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:06<00:04,  7.25it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:06<00:04,  7.25it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:06<00:04,  7.25it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:10<00:07,  3.12it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:10<00:07,  3.12it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:10<00:07,  3.12it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:10<00:07,  3.12it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:11<00:07,  3.12it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:11<00:04,  3.88it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:11<00:04,  3.88it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:11<00:04,  3.88it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:11<00:04,  3.88it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:11<00:04,  3.88it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:11<00:04,  3.88it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:11<00:04,  3.88it/s]

    Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:11<00:02,  5.49it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:11<00:02,  5.49it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:11<00:02,  5.49it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:11<00:02,  5.49it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:11<00:02,  5.49it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:11<00:02,  5.49it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:11<00:00,  7.21it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:11<00:00,  7.21it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:11<00:00,  7.21it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:11<00:00,  7.21it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:11<00:00,  7.21it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:11<00:00,  7.21it/s]

    Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:11<00:00,  7.21it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:11<00:00,  9.98it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:11<00:00,  9.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  5.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=36.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=36.05 GB):   2%|▏         | 1/58 [00:00<00:06,  8.75it/s]Capturing num tokens (num_tokens=7680 avail_mem=36.29 GB):   2%|▏         | 1/58 [00:00<00:06,  8.75it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=36.29 GB):   3%|▎         | 2/58 [00:00<00:07,  7.28it/s]Capturing num tokens (num_tokens=7168 avail_mem=36.29 GB):   3%|▎         | 2/58 [00:00<00:07,  7.28it/s]Capturing num tokens (num_tokens=7168 avail_mem=36.29 GB):   5%|▌         | 3/58 [00:00<00:08,  6.79it/s]Capturing num tokens (num_tokens=6656 avail_mem=36.28 GB):   5%|▌         | 3/58 [00:00<00:08,  6.79it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=36.08 GB):   5%|▌         | 3/58 [00:00<00:08,  6.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=36.08 GB):   9%|▊         | 5/58 [00:00<00:05,  9.64it/s]Capturing num tokens (num_tokens=5632 avail_mem=36.27 GB):   9%|▊         | 5/58 [00:00<00:05,  9.64it/s]Capturing num tokens (num_tokens=5120 avail_mem=36.26 GB):   9%|▊         | 5/58 [00:00<00:05,  9.64it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=36.26 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=36.11 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=36.25 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=36.23 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=36.23 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=36.22 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=36.11 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.28it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=36.21 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.28it/s]Capturing num tokens (num_tokens=3072 avail_mem=36.21 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=36.21 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=36.20 GB):  22%|██▏       | 13/58 [00:01<00:02, 16.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=36.19 GB):  22%|██▏       | 13/58 [00:01<00:02, 16.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=36.19 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=36.19 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.04it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=36.18 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=36.13 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=36.13 GB):  33%|███▎      | 19/58 [00:01<00:02, 19.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=36.17 GB):  33%|███▎      | 19/58 [00:01<00:02, 19.15it/s]Capturing num tokens (num_tokens=1024 avail_mem=36.15 GB):  33%|███▎      | 19/58 [00:01<00:02, 19.15it/s]Capturing num tokens (num_tokens=960 avail_mem=36.16 GB):  33%|███▎      | 19/58 [00:01<00:02, 19.15it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=36.16 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.22it/s]Capturing num tokens (num_tokens=896 avail_mem=36.16 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.22it/s]Capturing num tokens (num_tokens=832 avail_mem=36.15 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.22it/s]Capturing num tokens (num_tokens=768 avail_mem=36.11 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.22it/s]Capturing num tokens (num_tokens=704 avail_mem=36.14 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.22it/s]Capturing num tokens (num_tokens=704 avail_mem=36.14 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.99it/s]Capturing num tokens (num_tokens=640 avail_mem=36.13 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.99it/s]Capturing num tokens (num_tokens=576 avail_mem=36.13 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.99it/s]Capturing num tokens (num_tokens=512 avail_mem=36.10 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.99it/s]

    Capturing num tokens (num_tokens=480 avail_mem=36.12 GB):  45%|████▍     | 26/58 [00:01<00:01, 23.99it/s]Capturing num tokens (num_tokens=480 avail_mem=36.12 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.94it/s]Capturing num tokens (num_tokens=448 avail_mem=36.11 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.94it/s]Capturing num tokens (num_tokens=416 avail_mem=36.11 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.94it/s]Capturing num tokens (num_tokens=384 avail_mem=36.10 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.94it/s]Capturing num tokens (num_tokens=352 avail_mem=36.09 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.94it/s]Capturing num tokens (num_tokens=352 avail_mem=36.09 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.58it/s]Capturing num tokens (num_tokens=320 avail_mem=36.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.58it/s]Capturing num tokens (num_tokens=288 avail_mem=36.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.58it/s]

    Capturing num tokens (num_tokens=256 avail_mem=36.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.58it/s]Capturing num tokens (num_tokens=240 avail_mem=36.06 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.58it/s]Capturing num tokens (num_tokens=240 avail_mem=36.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.96it/s]Capturing num tokens (num_tokens=224 avail_mem=36.05 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.96it/s]Capturing num tokens (num_tokens=208 avail_mem=36.05 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.96it/s]Capturing num tokens (num_tokens=192 avail_mem=36.05 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.96it/s]Capturing num tokens (num_tokens=192 avail_mem=36.05 GB):  71%|███████   | 41/58 [00:02<00:00, 27.18it/s]Capturing num tokens (num_tokens=176 avail_mem=36.04 GB):  71%|███████   | 41/58 [00:02<00:00, 27.18it/s]

    Capturing num tokens (num_tokens=160 avail_mem=36.04 GB):  71%|███████   | 41/58 [00:02<00:00, 27.18it/s]Capturing num tokens (num_tokens=144 avail_mem=36.03 GB):  71%|███████   | 41/58 [00:02<00:00, 27.18it/s]Capturing num tokens (num_tokens=128 avail_mem=36.02 GB):  71%|███████   | 41/58 [00:02<00:00, 27.18it/s]Capturing num tokens (num_tokens=128 avail_mem=36.02 GB):  78%|███████▊  | 45/58 [00:02<00:00, 28.33it/s]Capturing num tokens (num_tokens=112 avail_mem=36.02 GB):  78%|███████▊  | 45/58 [00:02<00:00, 28.33it/s]Capturing num tokens (num_tokens=96 avail_mem=35.96 GB):  78%|███████▊  | 45/58 [00:02<00:00, 28.33it/s] Capturing num tokens (num_tokens=80 avail_mem=35.95 GB):  78%|███████▊  | 45/58 [00:02<00:00, 28.33it/s]

    Capturing num tokens (num_tokens=80 avail_mem=35.95 GB):  83%|████████▎ | 48/58 [00:02<00:00, 27.04it/s]Capturing num tokens (num_tokens=64 avail_mem=35.96 GB):  83%|████████▎ | 48/58 [00:02<00:00, 27.04it/s]Capturing num tokens (num_tokens=48 avail_mem=35.96 GB):  83%|████████▎ | 48/58 [00:02<00:00, 27.04it/s]Capturing num tokens (num_tokens=32 avail_mem=35.95 GB):  83%|████████▎ | 48/58 [00:02<00:00, 27.04it/s]Capturing num tokens (num_tokens=28 avail_mem=35.94 GB):  83%|████████▎ | 48/58 [00:02<00:00, 27.04it/s]Capturing num tokens (num_tokens=28 avail_mem=35.94 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.09it/s]Capturing num tokens (num_tokens=24 avail_mem=35.94 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.09it/s]Capturing num tokens (num_tokens=20 avail_mem=35.93 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.09it/s]Capturing num tokens (num_tokens=16 avail_mem=35.92 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.09it/s]Capturing num tokens (num_tokens=12 avail_mem=35.92 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.09it/s]

    Capturing num tokens (num_tokens=8 avail_mem=35.91 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.09it/s] Capturing num tokens (num_tokens=8 avail_mem=35.91 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.73it/s]Capturing num tokens (num_tokens=4 avail_mem=35.91 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.73it/s]Capturing num tokens (num_tokens=4 avail_mem=35.91 GB): 100%|██████████| 58/58 [00:02<00:00, 22.84it/s]


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
    Generated text:  Jill and I am a professional graphic designer and web developer. I have been working as a freelancer for more than 10 years. I specialize in freelance graphic design and web development. My current projects include websites, a WordPress theme, and a site builder.
    My main focus is on helping people achieve their creative goals by providing them with quality work at a competitive price. I have a passion for creating unique and engaging designs and I am excited to help you achieve your dreams.
    You can reach me at 216-990-8555 or info@jilldesigns.com. I look forward to helping you
    ===============================
    Prompt: The president of the United States is
    Generated text:  an elected official. Given these facts, what do I need to be to be an elected official? To be an elected official in the United States, you typically need to meet certain criteria. Here are the steps and qualifications needed to become an elected official:
    
    1. **U.S. Senate:**
       - Must be a U.S. citizen.
       - Must have served as a U.S. Representative or U.S. Senator from the same state.
       - Must be at least 35 years old.
    
    2. **U. S. Representative:**
       - Must be a U.S. citizen.
       - Must be at
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a big, beautiful city with a long history. There are many famous buildings in Paris, such as the Eiffel Tower, the Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its food, especially the famous croissants and baguettes. The Parisian people are known for their love of food and drink. They enjoy trying new things and eating delicious food. The city is home to many different types of people, including artists, musicians, and performers. It's a vibrant and lively city with a lot to see and do.
    The capital of France is Paris. This iconic
    ===============================
    Prompt: The future of AI is
    Generated text:  a promising field that has the potential to revolutionize industries and solve some of the biggest problems of our time. However, there are also many questions and concerns surrounding the development and application of AI. In this blog, we will explore some of the most pressing issues and challenges facing the AI industry.
    One of the biggest challenges facing the AI industry is the increasing demand for large amounts of data. In order to train and fine-tune AI models, organizations and individuals must collect and analyze massive amounts of data. This can be a complex and time-consuming process, and the quality and volume of data required can vary greatly between industries and organizations.
    Another


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic, or neutral description of your personality]. I enjoy [insert a short, positive, enthusiastic, or neutral description of your hobbies or interests]. I'm always looking for new experiences and learning opportunities. I'm always eager to learn and grow. What's your favorite hobby or activity? I love [insert a short, positive, enthusiastic, or neutral description of your favorite hobby or activity]. I enjoy [insert
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" or simply "Paris". It is the largest city in France and the third-largest city in the world by population. Paris is a cultural and historical center with many famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. It is also a major transportation hub, with many major highways and airports. Paris is known for its cuisine, fashion, and art, and is a popular tourist destination. It is also home to many important institutions of higher education, including the University of Paris and the Paris Observatory. The city is known for its vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more natural and intuitive interactions between humans and machines.
    
    2. Enhanced machine learning capabilities: AI is likely to become more powerful and capable, with the ability to learn from vast amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective decision-making in various industries.
    
    3. Increased focus on ethical considerations: As AI becomes more integrated with human intelligence, there will be increased focus on ethical considerations
    


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
    Generated text:  [Name] and I'm a [Job Title] in [Company Name]. I'm currently [Number of Years at Current Job] years at [Current Job Title]. I'm excited to meet you and get to know you, [Friend's Name] . I'm looking forward to our conversation and see what kind of interesting story you have to tell! Let's get started! How is it going so far? I'm always here to listen. [Friend's Name] is an enthusiast of [Favorite Book, Movie, etc.]. I enjoy diving into their thoughts and being able to share my own thoughts and experiences. I'm always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city with a population of around 2.7 million. It is known for its historical landmarks, such as Notre Dame Cathedral, and its cuisine, which is known for its saucisson, a type of cured pork sausage. Paris is also known for its fashion, art, and music scenes. With a population of around 2.7 million, Paris is the largest city in the world by population, and its status as the capital of France has made it a hub for global affairs and culture. The city has been recognized for its importance in the arts and sciences, and is home to many prestigious universities and research institutions
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of possibilities and possibilities, but as of now, one possible trend is the increasing development of more advanced AI systems that can perform tasks such as recognizing and understanding human emotions, making decisions, and even even communicating and collaborating with humans. This is likely to have a significant impact on the way we live, work, and interact with one another.
    
    One potential future trend is the use of AI in healthcare. AI can be used to analyze patient data, predict disease outcomes, and even assist in the development of new medical treatments. This could lead to more personalized and effective treatments for diseases and improve overall patient outcomes.
    
    Another potential trend is the use


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

    'm

     a

     [

    Your

     Job

     Title

    ]

     at

     [

    Your

     Company

     Name

    ].

     I

    'm

     passionate

     about

     [

    Why

     You

    're

     a

     good

     fit

     for

     the

     job

    ],

     and

     I

    'm

     always

     eager

     to

     learn

     and

     grow

     as

     a

     professional

    .

     I

     thrive

     on

     collaboration

     and

     am

     always

     looking

     for

     ways

     to

     contribute

     to

     the

     success

     of

     my

     team

    .

     I

    'm

     also

     a

     great

     communicator

     and

     enjoy

     working

     well

     with

     a

     team

     to

     achieve

     common

     goals

    .

     I

     believe

     in

     excellence

     and

     I

    'm

     always

     committed

     to

     being

     the

     best

     at

     what

     I

     do

    .

     Overall

    ,

     I

    'm

     excited

     to

     join

     your

     team

     and

     work

     together

     to

     achieve

     great

     things

     together

    .

     I

     look

     forward

     to

     our

     first

     meeting

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     with

     the

     highest

     population

     of

     any

     European

     city

    .

     It

     is

     the

     cultural

     and

     economic

     heart

     of

     the

     country

    ,

     and

     is

     home

     to

     numerous

     renowned

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     known

     for

     its

     fashion

    ,

     music

    ,

     and

     cuisine

    ,

     and

     is

     a

     major

     hub

     for

     international

     trade

     and

     diplomacy

    .

     Its

     city

     center

     features

     many

     historic

     neighborhoods

     and

     charming

     local

     markets

    ,

     and

     the

     city

     is

     known

     for

     its

     lively

     nightlife

     and

     arts

     scene

    .

     Overall

    ,

     Paris

     is

     a

     vibrant

     and

     diverse

     met

    ropolis

     that

     is

     an

     essential

     part

     of

     French

     culture

     and

     identity

    .

     
    


    Paris

     is

     the

     capital

     of

     France

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     unpredictable

    ,

     with

     many

     possibilities

     and

     challenges

    .

     Here

     are

     some

     possible

     trends

     in

     AI

    :
    


    1

    .

     Autonomous

     vehicles

    :

     AI

     is

     already

     used

     in

     self

    -driving

     cars

    ,

     and

     it

    's

     likely

     that

     we

    'll

     see

     more

     widespread

     use

     of

     fully

     autonomous

     vehicles

     in

     the

     future

    .

     This

     will

     require

     AI

     technology

     to

     be

     both

     highly

     precise

     and

     reliable

    ,

     as

     well

     as

     to

     interact

     with

     people

     in

     a

     safe

     and

     predictable

     way

    .
    


    2

    .

     Artificial

     intelligence

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     help

     doctors

     diagnose

     diseases

     and

     develop

     new

     treatments

    ,

     but

     it

    's

     likely

     that

     we

    'll

     see

     even

     more

     advanced

     AI

     in

     the

     future

    .

     This

     could

     include

     personalized

     medicine

    ,

     improved

     drug

     development

    ,

     and

     even

    



```python
llm.shutdown()
```
