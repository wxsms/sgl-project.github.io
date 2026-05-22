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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.63it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.44it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.44it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.44it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.44it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.44it/s]

    Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.44it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.44it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 11.61it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 11.61it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 17.66it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 17.66it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 17.66it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 17.66it/s]

    Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 17.66it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 17.66it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 21.47it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 27.21it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 27.21it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 27.21it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 27.21it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 27.21it/s]

    Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 27.21it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 27.21it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 27.21it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 27.21it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 35.88it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 35.88it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 35.88it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 35.88it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.28it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.28it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.05 GB):   3%|▎         | 2/58 [00:00<00:03, 18.28it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.05 GB):   9%|▊         | 5/58 [00:00<00:02, 21.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.04 GB):   9%|▊         | 5/58 [00:00<00:02, 21.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.19 GB):   9%|▊         | 5/58 [00:00<00:02, 21.09it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=71.71 GB):   9%|▊         | 5/58 [00:00<00:02, 21.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.71 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.72 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.73 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.12it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.73 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.94 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.59it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.93 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.59it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.93 GB):  21%|██        | 12/58 [00:00<00:03, 13.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.91 GB):  21%|██        | 12/58 [00:00<00:03, 13.52it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=71.90 GB):  21%|██        | 12/58 [00:00<00:03, 13.52it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.90 GB):  24%|██▍       | 14/58 [00:00<00:03, 14.16it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.89 GB):  24%|██▍       | 14/58 [00:00<00:03, 14.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.76 GB):  24%|██▍       | 14/58 [00:01<00:03, 14.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.75 GB):  24%|██▍       | 14/58 [00:01<00:03, 14.16it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=71.75 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.86 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.15it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.86 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.85 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.85 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.83 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.18it/s]Capturing num tokens (num_tokens=960 avail_mem=71.76 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.18it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=71.76 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.18it/s]Capturing num tokens (num_tokens=896 avail_mem=71.76 GB):  40%|███▉      | 23/58 [00:01<00:01, 20.59it/s]Capturing num tokens (num_tokens=832 avail_mem=71.76 GB):  40%|███▉      | 23/58 [00:01<00:01, 20.59it/s]Capturing num tokens (num_tokens=768 avail_mem=71.80 GB):  40%|███▉      | 23/58 [00:01<00:01, 20.59it/s]Capturing num tokens (num_tokens=704 avail_mem=71.80 GB):  40%|███▉      | 23/58 [00:01<00:01, 20.59it/s]Capturing num tokens (num_tokens=704 avail_mem=71.80 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.56it/s]Capturing num tokens (num_tokens=640 avail_mem=71.75 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.56it/s]Capturing num tokens (num_tokens=576 avail_mem=71.73 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.56it/s]

    Capturing num tokens (num_tokens=512 avail_mem=71.76 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.56it/s]Capturing num tokens (num_tokens=480 avail_mem=71.75 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.56it/s]Capturing num tokens (num_tokens=480 avail_mem=71.75 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.05it/s]Capturing num tokens (num_tokens=448 avail_mem=71.74 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.05it/s]Capturing num tokens (num_tokens=416 avail_mem=71.75 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.05it/s]Capturing num tokens (num_tokens=384 avail_mem=71.74 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.05it/s]Capturing num tokens (num_tokens=352 avail_mem=71.73 GB):  52%|█████▏    | 30/58 [00:01<00:01, 25.05it/s]Capturing num tokens (num_tokens=352 avail_mem=71.73 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.22it/s]Capturing num tokens (num_tokens=320 avail_mem=71.70 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.22it/s]

    Capturing num tokens (num_tokens=288 avail_mem=71.73 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.22it/s]Capturing num tokens (num_tokens=256 avail_mem=71.72 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.22it/s]Capturing num tokens (num_tokens=240 avail_mem=71.71 GB):  59%|█████▊    | 34/58 [00:01<00:00, 27.22it/s]Capturing num tokens (num_tokens=240 avail_mem=71.71 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.31it/s]Capturing num tokens (num_tokens=224 avail_mem=71.70 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.31it/s]Capturing num tokens (num_tokens=208 avail_mem=71.69 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.31it/s]Capturing num tokens (num_tokens=192 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.31it/s]Capturing num tokens (num_tokens=176 avail_mem=71.65 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.31it/s]

    Capturing num tokens (num_tokens=176 avail_mem=71.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.57it/s]Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.57it/s]Capturing num tokens (num_tokens=144 avail_mem=71.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.57it/s]Capturing num tokens (num_tokens=128 avail_mem=71.64 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.57it/s]Capturing num tokens (num_tokens=112 avail_mem=71.64 GB):  72%|███████▏  | 42/58 [00:02<00:00, 30.57it/s]Capturing num tokens (num_tokens=112 avail_mem=71.64 GB):  79%|███████▉  | 46/58 [00:02<00:00, 31.55it/s]Capturing num tokens (num_tokens=96 avail_mem=71.62 GB):  79%|███████▉  | 46/58 [00:02<00:00, 31.55it/s] Capturing num tokens (num_tokens=80 avail_mem=71.61 GB):  79%|███████▉  | 46/58 [00:02<00:00, 31.55it/s]Capturing num tokens (num_tokens=64 avail_mem=71.60 GB):  79%|███████▉  | 46/58 [00:02<00:00, 31.55it/s]

    Capturing num tokens (num_tokens=48 avail_mem=71.59 GB):  79%|███████▉  | 46/58 [00:02<00:00, 31.55it/s]Capturing num tokens (num_tokens=48 avail_mem=71.59 GB):  86%|████████▌ | 50/58 [00:02<00:00, 31.40it/s]Capturing num tokens (num_tokens=32 avail_mem=71.59 GB):  86%|████████▌ | 50/58 [00:02<00:00, 31.40it/s]Capturing num tokens (num_tokens=28 avail_mem=71.57 GB):  86%|████████▌ | 50/58 [00:02<00:00, 31.40it/s]Capturing num tokens (num_tokens=24 avail_mem=71.57 GB):  86%|████████▌ | 50/58 [00:02<00:00, 31.40it/s]Capturing num tokens (num_tokens=20 avail_mem=71.56 GB):  86%|████████▌ | 50/58 [00:02<00:00, 31.40it/s]Capturing num tokens (num_tokens=20 avail_mem=71.56 GB):  93%|█████████▎| 54/58 [00:02<00:00, 32.50it/s]Capturing num tokens (num_tokens=16 avail_mem=71.55 GB):  93%|█████████▎| 54/58 [00:02<00:00, 32.50it/s]Capturing num tokens (num_tokens=12 avail_mem=71.54 GB):  93%|█████████▎| 54/58 [00:02<00:00, 32.50it/s]

    Capturing num tokens (num_tokens=8 avail_mem=71.52 GB):  93%|█████████▎| 54/58 [00:02<00:00, 32.50it/s] Capturing num tokens (num_tokens=4 avail_mem=71.51 GB):  93%|█████████▎| 54/58 [00:02<00:00, 32.50it/s]Capturing num tokens (num_tokens=4 avail_mem=71.51 GB): 100%|██████████| 58/58 [00:02<00:00, 33.21it/s]Capturing num tokens (num_tokens=4 avail_mem=71.51 GB): 100%|██████████| 58/58 [00:02<00:00, 24.00it/s]


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
    Generated text:  Nemo and I am a 22 year old British boy who was asked to be the Secretary of the British Hearts Foundation in 2016. I do work with a group of young people who are passionate about raising funds for the charity to help homeless people. They have been working on the project since 2017 and have received over £250,000 since then, which is completely in line with the goals of the charity. I am also fluent in English and the language of the heart. I am excited to help the charity achieve its goals. The British Hearts Foundation was established in 1
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the Vice President. In 2022, the Vice President was elected to a three-year term. If the Vice President served for 3.5 years and had 2.5 years left to serve, how many years in total did the Vice President serve in office? To determine the total number of years the Vice President served in office, we need to consider both the years he had left to serve and the years he served for in the current term.
    
    1. Identify the years the Vice President had left to serve:
       \[
       \text{Years left to serve} = 2.5 \text
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    a) Paris
    
    b) Marseille
    
    c) Paris, Luxembourg, or Luxembourg City
    
    d) Brussels
    
    e) Lille
    
    Answer: a
    
    To determine the capital of France, we need to recall which city is located in France and is a major political and economic center. Let's analyze each option step by step:
    
    1. **Paris**:
       - Paris is the capital of France.
       - It is also the largest city in France, with a population of approximately 2.7 million people (as of 2021).
    
    2. **Marseille**:
       - Marseille is the capital of France
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the companies that innovate and lead. This is the case of NextAI, a company that is pioneering machine learning and AI technology in the medical field.
    We are excited to announce that NextAI has just launched a fresh version of our AI tool, which is called NextAI AI. This is the first time in the history of AI that NextAI is launching a new AI tool. We believe it will be a significant milestone in the growth and development of NextAI.
    This new AI tool is an AI-driven tool that is designed to help patients with diseases. It is called NextAI AI and it is a powerful tool that


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character]. What do you do for a living? I'm a [insert a short description of your job]. What do you enjoy doing? I enjoy [insert a short description of what you enjoy doing]. What do you like to do for fun? I like to [insert a short description of what you like to do for fun]. What do you like to do for a hobby? I like to [insert
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the seat of the French government and the country's cultural and political center. Paris is a bustling metropolis with a rich history and a diverse population of over 2.5 million people. The city is known for its fashion, art, and cuisine, and is a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant culture. The city is also home to many famous museums, including the Musée d'
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from feedback and improve their performance over time.
    
    2. Greater emphasis on ethical and social implications: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social implications. This could lead to more rigorous testing and evaluation of AI systems to ensure they are safe and ethical.
    
    3. Increased focus on privacy and security: As AI becomes
    


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
    Generated text:  [Name]. I am a [gender] with [background information] and I enjoy [what I like to do] with [what I like to do]. I am [age] years old and I am currently [occupation]. I am [any name] and I love [any hobbies or interests]. How are you today? I hope you are well. I am [any name] and I am a [gender] with [background information]. I enjoy [what I like to do] with [what I like to do]. I am [age] years old and I am currently [occupation]. I am [any name] and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris, officially known as the Île-de-France region, is the largest city in France and the most populous city in Europe. It is located on the River Seine and forms the northernmost part of the Île-de-France region. The city is also one of the world's most populous metropolitan areas. Paris is a cultural and economic center, known for its art, architecture, and fashion. It is also famous for its opera, cinema, and music. The city is home to many landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is a major tourist
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a combination of advancements, breakthroughs, and challenges. Here are some potential trends we can expect to see in the future:
    
    1. Increased AI integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This integration could lead to new applications and industries, such as autonomous vehicles, smart home technology, and healthcare.
    
    2. AI as a weapon: As AI becomes more advanced, there is a potential for AI to be used as a weapon, with unintended consequences. This could include ethical issues, such as bias in AI algorithms,


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

    your

     profession

     or

     title

    ].

     I

     have

     [

    number

     of

     years

     in

     my

     professional

     career

    ]

     years

     of

     experience

     in

     [

    your

     profession

    /

    industry

    ].

     I

     currently

     hold

     a

     [

    degree

     or

     certification

    ]

     in

     [

    your

     area

     of

     expertise

    ].

     I

     thrive

     on

     learning

     new

     things

     and

     trying

     new

     things

    .

     I

     enjoy

     taking

     on

     challenges

     and

     pushing

     myself

     to

     grow

     professionally

    .

     What

     do

     you

     think

     you

    're

     best

     at

    ?

     My

     strengths

     include

     [

    mention

     three

     specific

     strengths

    ].

     I

    ’m

     looking

     for

     a

     new

     opportunity

     to

     grow

     and

     develop

    .

     What

     do

     you

     think

     your

     skills

     set

     you

     apart

     from

     others

    ?

     My

     skills

     set

     me

     apart

     from

     others

     are

     my

     ability

     to

     work

     with

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Sure

    ,

     Paris

     is

     the

     capital

     of

     France

    .

     It

     is

     the

     largest

     city

     in

     Europe

     and

     is

     known

     as

     the

     "

    City

     of

     Light

    "

     due

     to

     its

     historical

     significance

     in

     art

    ,

     music

    ,

     and

     literature

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     is

     home

     to

     many

     notable

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

     It

     is

     also

     known

     for

     its

     architecture

    ,

     cuisine

    ,

     and

     cultural

     events

    .

     Paris

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

     and

     continues

     to

     be

     a

     major

     cultural

     and

     economic

     center

     in

     Europe

    .

     It

     is

     often

     referred

     to

     as

     the

     "

    King

    dom

     of

     Flowers

    "

     and

     is

     recognized

     as

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     complex

     and

     rapidly

     evolving

     field

     with

     potential

     to

     revolution

    ize

     various

     industries

    .

     Here

     are

     some

     possible

     trends

     to

     consider

    :
    


    1

    .

     Increased

     integration

     with

     human

     behavior

    :

     As

     AI

     continues

     to

     improve

     and

     become

     more

     sophisticated

    ,

     it

    's

     possible

     that

     it

     will

     become

     more

     integrated

     with

     human

     behavior

    .

     This

     could

     lead

     to

     more

     personalized

     and

     context

    -specific

     AI

     systems

     that

     can

     better

     understand

     and

     respond

     to

     individual

     users

    .
    


    2

    .

     Enhanced

     natural

     language

     processing

    :

     Natural

     language

     processing

     (

    N

    LP

    )

     is

     becoming

     increasingly

     important

     as

     AI

     evolves

    .

     As

     N

    LP

     improves

    ,

     it

    's

     possible

     that

     AI

    -powered

     assistants

     will

     be

     able

     to

     understand

     and

     respond

     to

     human

     language

     more

     accurately

     and

     naturally

    .
    


    3

    .

    



```python
llm.shutdown()
```
