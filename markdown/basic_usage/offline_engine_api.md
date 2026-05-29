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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.87it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.36s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:24,  2.09it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:24,  2.09it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:24,  2.09it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:24,  2.09it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:24,  2.09it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:11,  4.26it/s]

    Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:05,  8.40it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:05,  8.40it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:05,  8.40it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:05,  8.40it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:05,  8.40it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:04<00:05,  8.40it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:04<00:05,  8.40it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:04<00:05,  8.40it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 14.25it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 14.25it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 14.25it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 14.25it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 14.25it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 14.25it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 14.25it/s]

    Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:02, 14.25it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 20.92it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 29.20it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 29.20it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 37.64it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 37.64it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 37.64it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 37.64it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 37.64it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 37.64it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 37.64it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 37.64it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 37.64it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 37.64it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 47.35it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 47.35it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 47.35it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 47.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.36 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=74.36 GB):   3%|▎         | 2/58 [00:00<00:07,  7.41it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.37 GB):   3%|▎         | 2/58 [00:00<00:07,  7.41it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.18 GB):   3%|▎         | 2/58 [00:00<00:07,  7.41it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.18 GB):   7%|▋         | 4/58 [00:00<00:05,  9.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.35 GB):   7%|▋         | 4/58 [00:00<00:05,  9.17it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.34 GB):   7%|▋         | 4/58 [00:00<00:05,  9.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.34 GB):  10%|█         | 6/58 [00:00<00:04, 11.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.23 GB):  10%|█         | 6/58 [00:00<00:04, 11.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.32 GB):  10%|█         | 6/58 [00:00<00:04, 11.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.32 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.32 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.33it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.31 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.30 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.30 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.30 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.29 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.29 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.29 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.28 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.74it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.27 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.25 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.74it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.26 GB):  24%|██▍       | 14/58 [00:01<00:02, 19.74it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.26 GB):  31%|███       | 18/58 [00:01<00:01, 23.25it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.24 GB):  31%|███       | 18/58 [00:01<00:01, 23.25it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.25 GB):  31%|███       | 18/58 [00:01<00:01, 23.25it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.22 GB):  31%|███       | 18/58 [00:01<00:01, 23.25it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.22 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.26it/s]Capturing num tokens (num_tokens=960 avail_mem=74.23 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.26it/s] Capturing num tokens (num_tokens=896 avail_mem=74.22 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.26it/s]Capturing num tokens (num_tokens=832 avail_mem=74.23 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.26it/s]Capturing num tokens (num_tokens=768 avail_mem=74.22 GB):  36%|███▌      | 21/58 [00:01<00:01, 24.26it/s]Capturing num tokens (num_tokens=768 avail_mem=74.22 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.60it/s]Capturing num tokens (num_tokens=704 avail_mem=74.22 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.60it/s]Capturing num tokens (num_tokens=640 avail_mem=74.21 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.60it/s]Capturing num tokens (num_tokens=576 avail_mem=74.21 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.60it/s]Capturing num tokens (num_tokens=512 avail_mem=74.16 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.60it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.16 GB):  50%|█████     | 29/58 [00:01<00:00, 29.72it/s]Capturing num tokens (num_tokens=480 avail_mem=74.20 GB):  50%|█████     | 29/58 [00:01<00:00, 29.72it/s]Capturing num tokens (num_tokens=448 avail_mem=74.19 GB):  50%|█████     | 29/58 [00:01<00:00, 29.72it/s]Capturing num tokens (num_tokens=416 avail_mem=74.18 GB):  50%|█████     | 29/58 [00:01<00:00, 29.72it/s]Capturing num tokens (num_tokens=384 avail_mem=74.18 GB):  50%|█████     | 29/58 [00:01<00:00, 29.72it/s]Capturing num tokens (num_tokens=384 avail_mem=74.18 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.51it/s]Capturing num tokens (num_tokens=352 avail_mem=74.17 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.51it/s]Capturing num tokens (num_tokens=320 avail_mem=74.14 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.51it/s]Capturing num tokens (num_tokens=288 avail_mem=74.14 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.51it/s]Capturing num tokens (num_tokens=256 avail_mem=74.13 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.51it/s]

    Capturing num tokens (num_tokens=256 avail_mem=74.13 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=240 avail_mem=74.15 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=224 avail_mem=74.14 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=208 avail_mem=74.13 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=192 avail_mem=74.13 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=192 avail_mem=74.13 GB):  71%|███████   | 41/58 [00:01<00:00, 34.06it/s]Capturing num tokens (num_tokens=176 avail_mem=74.12 GB):  71%|███████   | 41/58 [00:01<00:00, 34.06it/s]Capturing num tokens (num_tokens=160 avail_mem=74.11 GB):  71%|███████   | 41/58 [00:01<00:00, 34.06it/s]Capturing num tokens (num_tokens=144 avail_mem=74.10 GB):  71%|███████   | 41/58 [00:01<00:00, 34.06it/s]Capturing num tokens (num_tokens=128 avail_mem=74.10 GB):  71%|███████   | 41/58 [00:01<00:00, 34.06it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.10 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.99it/s]Capturing num tokens (num_tokens=112 avail_mem=74.09 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.99it/s]Capturing num tokens (num_tokens=96 avail_mem=74.09 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.99it/s] Capturing num tokens (num_tokens=80 avail_mem=74.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.99it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.99it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=48 avail_mem=74.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=32 avail_mem=74.06 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=24 avail_mem=74.05 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.10it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:02<00:00, 39.01it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:02<00:00, 39.01it/s]Capturing num tokens (num_tokens=12 avail_mem=74.04 GB):  93%|█████████▎| 54/58 [00:02<00:00, 39.01it/s]Capturing num tokens (num_tokens=8 avail_mem=74.04 GB):  93%|█████████▎| 54/58 [00:02<00:00, 39.01it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  93%|█████████▎| 54/58 [00:02<00:00, 39.01it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:02<00:00, 27.24it/s]


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
    Generated text:  Adam, I'm 17 years old and I live in the USA. As you can imagine, I'm quite popular in my school and I have lots of friends here. I've been learning Spanish for about a year now and I'm not sure how to pronounce the words correctly. I feel like I'm not speaking Spanish. I want to learn and I'm really interested in it. I have been doing the wordlists and I've been practicing speaking in the language every day. I hope to make progress by the time I finish school. What are some ways I can improve my pronunciation and improve my overall speaking skills in Spanish?
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. The United States has a president. A president is a position. The position of president is a formal one. What does a president do? A president is the head of state and the head of government. The United States has a president that holds the office of president and the head of state. The president of the United States is the Commander-in-Chief. The commander-in-chief is the head of the United States military.
    What does a commander-in-chief do? A commander-in-chief is the head of the United States military. The commander-in-chief has the power to direct and command the military forces of the United States.
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Rome
    C. Vienna
    D. Beijing
    Answer:
    
    A
    
    The airport and runway are aligned with the airport's runway centerline and have a clearance distance of ____.
    A. 100m
    B. 150m
    C. 200m
    D. 250m
    Answer:
    
    A
    
    When a commercial bank grants a loan, if the borrower fails to repay the principal and interest on time, the commercial bank can accept the mortgaged assets as collateral. The mortgagor has the right to receive compensation. Which of the
    ===============================
    Prompt: The future of AI is
    Generated text:  on the rise, and the challenges that the world faces today are even more complex than ever before. With the continuous advancement of technology, AI is becoming increasingly important for solving complex problems that we are facing today.
    AI, which stands for Artificial Intelligence, is a technology that uses algorithms and machine learning to process, analyze, and generate information from data. It is used in a wide range of industries, including healthcare, finance, transportation, and more. AI can analyze vast amounts of data in real-time, making it an excellent tool for solving complex problems and making informed decisions.
    However, AI is not without its challenges. One of the biggest


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Favorite Activity] lover, and I enjoy spending time with my family and friends. I'm always up for a good laugh and love to make people smile. I'm a [Favorite Book or Movie] fan, and I love to read or watch it whenever I have free time. I'm a [Favorite Music] lover, and I love to listen to music that makes me feel happy and inspired. I'm a [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville-Marie" and "La Ville de Paris". It is the largest city in France and the second-largest city in the European Union. The city is located on the Seine River and is home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is known for its rich history, art, and culture, and is a popular tourist destination. The city is also home to many famous landmarks and attractions, including the Champs-Élysées, the Eiffel Tower, and the Louvre Museum. Paris is a vibrant and diverse city with a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced ethical considerations: As AI becomes more advanced, there will be increased scrutiny of its ethical implications, including issues such as bias, privacy, and accountability.
    
    3. Greater reliance on data: AI will continue to rely more heavily on data to learn and make decisions, and there will be an increased focus on data privacy and security.
    
    4. Increased use of AI in healthcare: AI is
    


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
    Generated text:  [Your Name], and I am [Your Profession]. I have always been an avid reader of various genres and have a strong love for literature. I enjoy writing short stories and poetry, as well as engaging in creative writing workshops and speaking at book clubs. I believe in the power of storytelling and I am always eager to learn new writing techniques and share my knowledge with others. I am also a huge fan of travel and have spent countless hours exploring different countries and cultures. I have a passion for creating characters and developing narratives, and I am always on the lookout for new stories and opportunities to grow as a writer. In short, I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city where the Eiffel Tower, the Louvre Museum, and many other landmarks are located. The city has been the capital of France since 1804, and it is known for its rich cultural heritage, its grand architecture, and its diverse population. Paris is also famous for its fashion, its art, and its cuisine, and it is a major tourist destination worldwide. Its economic significance has made it one of the world's most important cities, and it continues to be a major center of learning and innovation. Overall, Paris is a vibrant and important city that is a reflection of France's rich cultural and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising and will continue to evolve in many ways. Here are some possible trends in AI in the next few years:
    
    1. Increased automation: AI will continue to become more integrated into nearly every industry and will automate many of the tasks that we currently do. This will mean more people will be able to focus on creative or intellectual work, and more companies will be able to focus on making profits.
    
    2. Enhanced personalization: AI will become more capable of understanding and personalizing our lives. This could include things like recommending movies or music, predicting our buying behavior, and even helping us find a job.
    
    3. Autonomous vehicles: AI


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

     experienced

     [

    occupation

    ].

     I

    'm

     passionate

     about

     [

    why

     you

    're

     passionate

     about

     it

    ].

     I

    'm

     always

     up

     for

     a

     challenge

     and

     enjoy

     [

    why

     you

    're

     passionate

     about

     it

    ].

     I

    'm

     [

    number

     of

     years

     working

     in

     this

     field

    ]

     years

     of

     experience

    ,

     and

     I

    'm

     always

     eager

     to

     learn

     and

     grow

    .

     I

    'm

     [

    how

     you

    're

     different

     from

     others

     in

     this

     field

    ].

     And

     I

     believe

     in

     [

    why

     you

     believe

     in

     it

    ].

     Whatever

     you

    're

     passionate

     about

    ,

     I

    'm

     always

     here

     to

     help

     you

    !

     [

    Include

     your

     qualifications

     or

     accomplishments

     in

     this

     role

    ].

     [

    Include

     any

     relevant

     experiences

    ,

     projects

    ,

     or

     accomplishments

    ].

     And

     I

    'm

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    What

     is

     the

     answer

    ?

     Paris

     is

     the

     capital

     of

     France

    .
    


    The

     answer

     is

     Paris

    .

     The

     capital

     of

     France

     is

     Paris

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     constantly

     evolving

    ,

     and

     there

     are

     many

     potential

     trends

     that

     could

     shape

     the

     future

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

     in

     AI

    :
    


    1

    .

     Improved

     Explain

    ability

    :

     AI

     systems

     are

     becoming

     more

     and

     more

     accurate

    ,

     but

     they

     are

     still

     complex

     and

     difficult

     to

     understand

    .

     As

     AI

     systems

     become

     more

     complex

    ,

     it

     may

     become

     easier

     to

     determine

     why

     a

     particular

     decision

     was

     made

    .

     One

     way

     to

     improve

     explain

    ability

     is

     to

     increase

     the

     amount

     of

     information

     that

     is

     available

     to

     users

     of

     the

     AI

     system

    .
    


    2

    .

     Personal

    ization

    :

     As

     AI

     systems

     become

     more

     accurate

     and

     complex

    ,

     they

     may

     become

     more

     personalized

    .

     Personal

    ization

     can

     help

     to

     improve

     user

     satisfaction

     and

     engagement

     with

     AI

    



```python
llm.shutdown()
```
