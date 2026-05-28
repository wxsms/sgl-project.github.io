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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.01it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:04,  5.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:04,  5.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:04,  5.35s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:04,  5.35s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:04,  5.35s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  6.97it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  6.97it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  6.97it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  6.97it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  6.97it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  6.97it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  6.97it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  6.97it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 10.90it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 10.90it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 10.90it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 10.90it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 10.90it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 10.90it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 10.90it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 10.90it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:01, 15.66it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:01, 15.66it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:01, 15.66it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:01, 15.66it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:01, 15.66it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 15.66it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 15.66it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:06<00:01, 15.66it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:06<00:00, 21.34it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:06<00:00, 21.34it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:06<00:00, 21.34it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:06<00:00, 21.34it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:06<00:00, 21.34it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:06<00:00, 21.34it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:06<00:00, 21.34it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:06<00:00, 21.34it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:06<00:00, 27.38it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:06<00:00, 27.38it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:06<00:00, 27.38it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:06<00:00, 27.38it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:06<00:00, 27.38it/s]

    Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:06<00:00, 27.38it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:06<00:00, 27.38it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:06<00:00, 27.38it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:06<00:00, 27.38it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 35.10it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 35.10it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 35.10it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 35.10it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 15.01it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 15.01it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 15.01it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   7%|▋         | 4/58 [00:00<00:03, 14.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   7%|▋         | 4/58 [00:00<00:03, 14.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   7%|▋         | 4/58 [00:00<00:03, 14.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):  10%|█         | 6/58 [00:00<00:03, 14.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):  10%|█         | 6/58 [00:00<00:03, 14.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  10%|█         | 6/58 [00:00<00:03, 14.40it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  10%|█         | 6/58 [00:00<00:03, 14.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:02, 20.88it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.80it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.80it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.80it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.80it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 24.37it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 24.37it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 24.37it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  31%|███       | 18/58 [00:00<00:01, 24.37it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 24.37it/s] Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.61it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.61it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.61it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.61it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.61it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.18it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.18it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.18it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.18it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.18it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.84it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.84it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.84it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.84it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.84it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.17it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.17it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.17it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.17it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 29.17it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.23it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.23it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.23it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.23it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.23it/s]

    Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.76it/s]Capturing num tokens (num_tokens=160 avail_mem=74.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.76it/s]Capturing num tokens (num_tokens=144 avail_mem=74.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.76it/s]Capturing num tokens (num_tokens=128 avail_mem=74.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.76it/s]Capturing num tokens (num_tokens=128 avail_mem=74.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 24.28it/s]Capturing num tokens (num_tokens=112 avail_mem=74.25 GB):  78%|███████▊  | 45/58 [00:01<00:00, 24.28it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.18 GB):  78%|███████▊  | 45/58 [00:01<00:00, 24.28it/s] Capturing num tokens (num_tokens=80 avail_mem=74.18 GB):  78%|███████▊  | 45/58 [00:01<00:00, 24.28it/s]Capturing num tokens (num_tokens=80 avail_mem=74.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 24.62it/s]Capturing num tokens (num_tokens=64 avail_mem=74.17 GB):  83%|████████▎ | 48/58 [00:01<00:00, 24.62it/s]Capturing num tokens (num_tokens=48 avail_mem=74.17 GB):  83%|████████▎ | 48/58 [00:01<00:00, 24.62it/s]Capturing num tokens (num_tokens=32 avail_mem=74.17 GB):  83%|████████▎ | 48/58 [00:02<00:00, 24.62it/s]Capturing num tokens (num_tokens=32 avail_mem=74.17 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.06it/s]Capturing num tokens (num_tokens=28 avail_mem=74.16 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.06it/s]

    Capturing num tokens (num_tokens=24 avail_mem=73.75 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.06it/s]Capturing num tokens (num_tokens=20 avail_mem=73.62 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.06it/s]Capturing num tokens (num_tokens=16 avail_mem=73.59 GB):  88%|████████▊ | 51/58 [00:02<00:00, 25.06it/s]Capturing num tokens (num_tokens=16 avail_mem=73.59 GB):  95%|█████████▍| 55/58 [00:02<00:00, 27.34it/s]Capturing num tokens (num_tokens=12 avail_mem=73.58 GB):  95%|█████████▍| 55/58 [00:02<00:00, 27.34it/s]Capturing num tokens (num_tokens=8 avail_mem=73.58 GB):  95%|█████████▍| 55/58 [00:02<00:00, 27.34it/s] Capturing num tokens (num_tokens=4 avail_mem=73.58 GB):  95%|█████████▍| 55/58 [00:02<00:00, 27.34it/s]

    Capturing num tokens (num_tokens=4 avail_mem=73.58 GB): 100%|██████████| 58/58 [00:02<00:00, 27.40it/s]Capturing num tokens (num_tokens=4 avail_mem=73.58 GB): 100%|██████████| 58/58 [00:02<00:00, 25.12it/s]


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
    Generated text:  Alicia, a medical student, and I'm in second year of Medicine. I have been in this class since the beginning. I'm currently taking the American College of Physicians annual exam. I'm not sure what to prepare for. It's an objective exam, and I don't have a lot of time. I know I'll have to know about the following topics:
    
      1. Introduction to medical genetics
      2. Medical genetics
      3. Understanding genetic diseases and hereditary syndromes
      4. Ethical considerations in genetic testing and counseling
      5. Case study
    
    I'm just wondering
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to secure votes from the European Parliament to authorize the sale of the aircraft carrier USS Enterprise to the United States. In the United States, the presidential election is held annually in June. In the European Parliament, the European Parliament elections are held annually in July. Both the United States and the European Parliament take place on the same day.
    
    Does it follow that does the European Parliament hold its elections every other year?
    Choose your answer from: A). yes. B). it is not possible to tell. C). no.
    A). yes.
    The European Parliament does hold its elections every other year. The election cycle for the European Parliament typically lasts
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    What is the answer? The capital of France is:
    
    Paris
    
    Paris is the capital city of France. It's like the biggest and most important place where important decisions are made in France. Think of it like a giant house where the King and Queen live and where lots of people work and play.
    Paris is very special because it has beautiful buildings, lots of yummy food, and many fun places to visit. It's like a big playground for grown-ups and kids who want to go on fun adventures! So, if you ever want to know what's special about Paris, just say "Paris" and we'll tell you all
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of those who study it. The next generation of AI researchers is poised to make significant strides and drive the future of AI. With continued investment and support from governments, industries, and academic institutions, there is a bright future ahead for AI. For those who follow the progress of AI, it is clear that the field is set for continued growth and innovation. With the right approaches and investments, it is possible to unlock the full potential of AI and make it a game-changing tool for advancing human knowledge and advancing society.
    AI is the field of computer science that deals with the development of intelligent machines that can perform tasks that typically require


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I have a [job title] at [company name], and I enjoy [what I do best]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [what I do best], and I enjoy [what I do best]. I'm always looking for new ways to challenge myself and expand my knowledge. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third largest city in the world by population. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also known for its annual festivals and cultural events. Paris is a popular tourist destination and a major economic center in France. The city is home to many international organizations and has a strong economy that is heavily reliant on tourism and finance. The city is also known for its cuisine, with many famous French dishes such as croissants,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we can expect to see more automation and artificial intelligence in various industries, from manufacturing to healthcare to transportation. This could lead to increased efficiency, reduced costs, and improved quality of life for many people.
    
    2. Personalized AI: AI will become more personalized, with machines learning from data and adapting to individual needs and preferences. This could lead to more effective and efficient healthcare, education, and
    


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
    Generated text:  [Name]. I am a [X] with a passion for [Y], a [Z] artist who brings [X] and [Y] together in unique and captivating ways.
    
    What draws you to [X] and [Y]?
    
    I am always drawn to [X] for the way it inspires and excites people, and to [Y] for the beauty and diversity it brings. Together, we can create something truly amazing. 
    
    What are some things you enjoy doing at home?
    
    As a [X] artist, I enjoy creating a wide range of projects, from [Z] to [X] to [Y]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Explanation: Paris is the largest city in France and serves as the political, economic, cultural, and administrative center of the country. It is known for its iconic landmarks, such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also home to numerous museums, including the Musée d'Orsay, the Musée d'Orsay, and the Musée d'Orsay, which showcases an extensive collection of European art and modern art. Paris is also known for its bustling street life, vibrant nightlife, and diverse culinary scene. Overall, Paris is a vibrant and diverse
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be rapidly evolving, with several trends set to shape its direction in the coming years. Here are some potential trends that could shape the future of AI:
    
    1. Increased focus on ethics and safety: As more AI is used in everyday life, there will be an increased focus on ensuring that it is ethical and safe. This will require AI developers to integrate ethical considerations into the design and development of AI systems.
    
    2. More automation in various fields: As AI becomes more advanced, it will be integrated into more and more aspects of our daily lives, from healthcare to transportation to customer service. This will require AI to become more efficient,


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

    ].

     I

    'm

     a

     [

    Age

    ]

     year

     old

     female

     with

     [

    Gender

    ]

     skin

    ,

     [

    Occup

    ation

    ]

     in

     the

     field

     of

     [

    Field

     of

     Work

    ],

     and

     a love

     for

     [

    Favorite

     Activity

    ].

     I

     love

     [

    Job

     Title

    ]

     and

     I

     have

     a

     [

    Number

    ]

     of

     years

     of

     experience

     in

     [

    Field

     of

     Work

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    Achie

    ve

     Object

    ives

    ].

     What

    's

     your

     profession

     or

     occupation

    ,

     [

    Name

    ]?

     And

     what

    's

     your

     favorite

     activity

    ,

     [

    Name

    ]?

     
    


    Please

     include

     a

     brief

     summary

     of

     your

     experiences

    ,

     including

     your

     educational

     background

     and

     any

     relevant

     training

     you

    've

     received

    .

     
    


    Also

    ,

     if

     you

     have

     any

     particular

     interests

     or

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     rich

     history

    ,

     famous

     landmarks

    ,

     and

     diverse

     culture

    .

     The

     city

     has

     a

     population

     of

     over

     

    2

     million

     people

    ,

     and

     it

     is

     the

     most

     populous

     urban

     area

     in

     Europe

    .

     It

     has

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

     and

     has

     been

     a

     major

     hub

     for

     trade

     and

     commerce

     throughout

     history

    .

     Paris

     is

     also

     famous

     for

     its

     art

    ,

     architecture

    ,

     and

     cuisine

    .

     The

     city

     is

     home

     to

     many

     notable

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

     world

    -ren

    owned

     tourist

     destination

     and

     has

     been

     recognized

     as

     one

     of

     the

     world

    's

     most

     liv

    able

     cities

    .

     Its

     status

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     one

     of

     rapid

     growth

     and

     transformation

    .

     As

     technology

     continues

     to

     evolve

    ,

     we

     can

     expect

     to

     see

     a

     wide

     range

     of

     innovations

     and

     advancements

     in

     artificial

     intelligence

     that

     will

     have

     a

     profound

     impact

     on

     every

     aspect

     of

     society

    .

     Here

     are

     some

     of

     the

     potential

     trends

     that

     are

     likely

     to

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     One

     of

     the

     most

     significant

     areas

     of

     AI

     development

     is

     in

     healthcare

    .

     AI

     has

     the

     potential

     to

     improve

     diagnostic

     accuracy

    ,

     personalize

     treatment

     plans

    ,

     and

     help

     doctors

     make

     better

    -in

    formed

     decisions

    .

     As

     more

     people

     become

     eligible

     for

     access

     to

     high

    -quality

     healthcare

    ,

     AI

    -powered

     systems

     will

     likely

     become

     more

     common

    .
    


    2

    .

     Enhanced

    



```python
llm.shutdown()
```
