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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:02,  1.13s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.73it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.73it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.73it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:30,  1.73it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.33it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.33it/s]

    Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.33it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.33it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.35it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.35it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.35it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.35it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:08,  5.35it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:05,  8.52it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:05,  8.52it/s]

    Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:05,  8.52it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:05,  8.52it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:05,  8.52it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:03, 12.24it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:03, 12.24it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:03, 12.24it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:03, 12.24it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:03, 12.24it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 16.15it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 16.15it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 16.15it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 16.15it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 16.15it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 16.15it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 21.53it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 21.53it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 21.53it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 21.53it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 21.53it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 21.53it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:00, 26.74it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:00, 26.74it/s]

    Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:00, 26.74it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:00, 26.74it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:00, 26.74it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:00, 26.74it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 31.44it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 31.44it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 31.44it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 31.44it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 31.44it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 31.44it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 31.44it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 37.19it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 37.19it/s]

    Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 37.19it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 37.19it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 37.19it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 37.19it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 37.19it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 41.17it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 41.17it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 41.17it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 41.17it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 41.17it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 41.17it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 41.17it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 45.31it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 45.31it/s] 

    Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 45.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.76 GB):   2%|▏         | 1/58 [00:00<00:07,  7.28it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.73 GB):   2%|▏         | 1/58 [00:00<00:07,  7.28it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.73 GB):   3%|▎         | 2/58 [00:00<00:07,  7.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.73 GB):   3%|▎         | 2/58 [00:00<00:07,  7.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.73 GB):   5%|▌         | 3/58 [00:00<00:07,  7.46it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.73 GB):   5%|▌         | 3/58 [00:00<00:07,  7.46it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.73 GB):   7%|▋         | 4/58 [00:00<00:07,  7.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.73 GB):   7%|▋         | 4/58 [00:00<00:07,  7.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.73 GB):   9%|▊         | 5/58 [00:00<00:06,  7.86it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.72 GB):   9%|▊         | 5/58 [00:00<00:06,  7.86it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=42.72 GB):  10%|█         | 6/58 [00:00<00:06,  8.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.71 GB):  10%|█         | 6/58 [00:00<00:06,  8.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.71 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.71 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.51it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=42.71 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.71 GB):  14%|█▍        | 8/58 [00:00<00:05,  8.87it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.70 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.70 GB):  14%|█▍        | 8/58 [00:01<00:05,  8.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.70 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.77it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.69 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.77it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.69 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.69 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.68 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.68 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.77it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=42.68 GB):  28%|██▊       | 16/58 [00:01<00:01, 24.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.68 GB):  28%|██▊       | 16/58 [00:01<00:01, 24.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.67 GB):  28%|██▊       | 16/58 [00:01<00:01, 24.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.67 GB):  28%|██▊       | 16/58 [00:01<00:01, 24.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.67 GB):  28%|██▊       | 16/58 [00:01<00:01, 24.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.65 GB):  28%|██▊       | 16/58 [00:01<00:01, 24.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.65 GB):  36%|███▌      | 21/58 [00:01<00:01, 31.33it/s]Capturing num tokens (num_tokens=960 avail_mem=42.66 GB):  36%|███▌      | 21/58 [00:01<00:01, 31.33it/s] Capturing num tokens (num_tokens=896 avail_mem=42.66 GB):  36%|███▌      | 21/58 [00:01<00:01, 31.33it/s]Capturing num tokens (num_tokens=832 avail_mem=42.66 GB):  36%|███▌      | 21/58 [00:01<00:01, 31.33it/s]Capturing num tokens (num_tokens=768 avail_mem=42.65 GB):  36%|███▌      | 21/58 [00:01<00:01, 31.33it/s]Capturing num tokens (num_tokens=704 avail_mem=42.65 GB):  36%|███▌      | 21/58 [00:01<00:01, 31.33it/s]

    Capturing num tokens (num_tokens=704 avail_mem=42.65 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=640 avail_mem=42.65 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=576 avail_mem=42.65 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=512 avail_mem=42.63 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=480 avail_mem=42.65 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=448 avail_mem=42.65 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.64it/s]Capturing num tokens (num_tokens=448 avail_mem=42.65 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=416 avail_mem=42.65 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=384 avail_mem=42.64 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=352 avail_mem=42.64 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=320 avail_mem=42.63 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=288 avail_mem=42.63 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.99it/s]

    Capturing num tokens (num_tokens=288 avail_mem=42.63 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=256 avail_mem=42.63 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=240 avail_mem=42.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=224 avail_mem=42.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=208 avail_mem=42.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=192 avail_mem=42.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=192 avail_mem=42.62 GB):  71%|███████   | 41/58 [00:01<00:00, 44.26it/s]Capturing num tokens (num_tokens=176 avail_mem=42.61 GB):  71%|███████   | 41/58 [00:01<00:00, 44.26it/s]Capturing num tokens (num_tokens=160 avail_mem=42.61 GB):  71%|███████   | 41/58 [00:01<00:00, 44.26it/s]Capturing num tokens (num_tokens=144 avail_mem=42.61 GB):  71%|███████   | 41/58 [00:01<00:00, 44.26it/s]Capturing num tokens (num_tokens=128 avail_mem=42.60 GB):  71%|███████   | 41/58 [00:01<00:00, 44.26it/s]Capturing num tokens (num_tokens=112 avail_mem=42.60 GB):  71%|███████   | 41/58 [00:01<00:00, 44.26it/s]

    Capturing num tokens (num_tokens=112 avail_mem=42.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=96 avail_mem=42.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.26it/s] Capturing num tokens (num_tokens=80 avail_mem=42.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=64 avail_mem=42.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=48 avail_mem=42.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=32 avail_mem=39.80 GB):  79%|███████▉  | 46/58 [00:02<00:00, 43.26it/s]

    Capturing num tokens (num_tokens=32 avail_mem=39.80 GB):  88%|████████▊ | 51/58 [00:02<00:00, 34.08it/s]Capturing num tokens (num_tokens=28 avail_mem=39.79 GB):  88%|████████▊ | 51/58 [00:02<00:00, 34.08it/s]Capturing num tokens (num_tokens=24 avail_mem=39.79 GB):  88%|████████▊ | 51/58 [00:02<00:00, 34.08it/s]Capturing num tokens (num_tokens=20 avail_mem=39.78 GB):  88%|████████▊ | 51/58 [00:02<00:00, 34.08it/s]Capturing num tokens (num_tokens=16 avail_mem=39.78 GB):  88%|████████▊ | 51/58 [00:02<00:00, 34.08it/s]Capturing num tokens (num_tokens=16 avail_mem=39.78 GB):  95%|█████████▍| 55/58 [00:02<00:00, 30.32it/s]Capturing num tokens (num_tokens=12 avail_mem=39.78 GB):  95%|█████████▍| 55/58 [00:02<00:00, 30.32it/s]

    Capturing num tokens (num_tokens=8 avail_mem=39.78 GB):  95%|█████████▍| 55/58 [00:02<00:00, 30.32it/s] Capturing num tokens (num_tokens=4 avail_mem=39.77 GB):  95%|█████████▍| 55/58 [00:02<00:00, 30.32it/s]Capturing num tokens (num_tokens=4 avail_mem=39.77 GB): 100%|██████████| 58/58 [00:02<00:00, 24.78it/s]


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
    Generated text:  Seth. I have no idea what I want to do. I'm a sophomore and I'm in the middle school. I'm in the gym with my friends and it is quite boring. What should I do? Your help would be greatly appreciated. Thanks! Seth
    
    I am happy to help you get out of your current situation. First, I would suggest that you consider the following:
    
      1. Talk to your parents or guardians about your concerns. It is important to have open and honest communication with your parents or guardians to address your concerns and get guidance on how to address them.
      2. Take some time for yourself
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the government of the United States. He is responsible for making major decisions and running the country. Can you tell me what the president of the United States does? The president of the United States is responsible for making decisions on issues of national importance, such as foreign policy, national security, and regulating the economy. He also has the power to appoint and remove federal judges and members of Congress, and is responsible for the day-to-day operations of the government. Can you tell me what the president of the United States does? The president of the United States is responsible for making major decisions on issues of national importance, such
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A) Paris
    B) London
    C) Berlin
    D) Moscow
    The capital of France is Paris. 
    
    To verify:
    - Paris is the capital of France.
    - It is located in the North-East of France, on the Île de la Cité.
    - It is the largest city in France by population. 
    
    Now, let's analyze the options:
    A) Paris - Correct.
    B) London - Incorrect, London is the capital of the United Kingdom.
    C) Berlin - Incorrect, Berlin is the capital of Germany.
    D) Moscow - Incorrect, Moscow is the capital of Russia.
    
    Thus, the correct answer
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, and it is now crucial to understand the ethical implications of AI. In a recent survey, 20% of the respondents agreed with the following statement: "AI should be used as a tool for good, not as a tool for evil." 
    
    In a class of 50 students, how many students would agree with this statement?
    To determine how many students in a class of 50 agree with the statement that AI should be used as a tool for good, not as a tool for evil, we can follow these steps:
    
    1. Identify the percentage of the class that agrees with the statement.
    2. Convert that


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the field]. I enjoy [reason for interest in the field]. I'm a [reason for interest in the field] and I'm always looking for [reason for interest in the field]. I'm a [reason for interest in the field] and I'm always looking for [reason for interest in the field]. I'm a [reason for interest in the field] and I'm always looking for [reason for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a cultural and historical center with a rich history dating back to the Middle Ages. Paris is a popular tourist destination and is home to many world-renowned museums, art galleries, and restaurants. The city is also known for its fashion industry, with many famous fashion houses and boutiques located in the city center. Paris is a vibrant and dynamic city with a diverse population and a rich cultural heritage. It is a major transportation hub and a major economic center in Europe. The city is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased integration with human intelligence: One of the most significant trends in AI is the increasing integration of AI with human intelligence. This could lead to more sophisticated and personalized AI that can better understand and respond to the needs of humans.
    
    2. Enhanced ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development
    


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
    Generated text:  [Name] and I am a [Occupation] [Role]. My greatest strength is [Strength or Talent], and my greatest weakness is [Weakness or Liability]. What kind of person am I?
    
    Hello, my name is [Name] and I am a [Occupation] [Role]. My greatest strength is [Strength or Talent], and my greatest weakness is [Weakness or Liability]. What kind of person am I? As a fictional character, I am a versatile AI designed to assist and provide helpful responses. My name is [Name] and I am a [Occupation] [Role], but I am not a real
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, commonly referred to as "La Parisienne" and also known as the "City of Love". It is a historical and cultural city with a rich heritage of art, history, and cuisine, known for its beautiful architecture, museums, and theaters. Paris is also home to many of the world's most iconic landmarks, including the Eiffel Tower and the Louvre Museum. It is a cosmopolitan city with a diverse population that has played a significant role in shaping French culture and identity. The city is renowned for its annual Eiffel Tower World Heritage Site, as well as the annual Carnaval. Paris is also the second
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising, with potential applications in areas such as healthcare, finance, transportation, and security. Here are some possible future trends in artificial intelligence:
    
    1. Increased autonomy and self-awareness: As AI technology continues to evolve, we may see more autonomous machines taking on more tasks that require decision-making and problem-solving, potentially leading to a more integrated and autonomous society.
    
    2. Greater integration with human emotions and social contexts: AI systems are increasingly becoming more "intelligent" and capable of understanding and empathizing with human emotions and social contexts. This could lead to more nuanced and adaptive AI that can better understand and respond to human social dynamics.
    
    


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

    job

     title

    ]

     at

     [

    company

     name

    ].

     I

    've

     been

     working

     here

     for

     [

    number

     of

     years

    ]

     years

     and

     I

    'm

     passionate

     about

     [

    reason

     for

     your

     interest

     in

     the

     company

    ].

     I

    'm

     always

     eager

     to

     learn

     new

     things

     and

     get

     involved

     in

     community

     events

    .

     My

     goal

     is

     to

     be

     a

     valuable

     addition

     to

     your

     team

     and

     help

     you

     achieve

     your

     goals

    .

     


    [

    Name

    ]:

     "

    Hello

    ,

     my

     name

     is

     [

    Name

    ]

     and

     I

    'm

     a

     [

    job

     title

    ]

     at

     [

    company

     name

    ].

     I

    've

     been

     working

     here

     for

     [

    number

     of

     years

    ]

     years

     and

     I

    'm

     passionate

     about

     [

    reason

     for

     your

     interest

     in

     the

     company

    ].

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     as

     the

     City

     of

     Light

     and

     the

     City

     of

     Vers

    ailles

    .

     It

     is

     an

     important

     cultural

     and

     economic

     center

    ,

     with

     its

     skyline

     featuring

     many

     landmarks

     and

     monuments

    .

     The

     city

     is

     also

     home

     to

     the

     headquarters

     of

     many

     major

     French

     companies

    .

     In

     addition

    ,

     Paris

     has

     a

     rich

     culinary

     tradition

    ,

     with

     its

     famous

     dishes

    ,

     such

     as

     fo

    ie

     gras

     and

     co

    q

     au

     vin

    .

     The

     city

     is

     also

     home

     to

     many

     museums

     and

     galleries

    ,

     and

     is

     known

     for

     its

     fine

     arts

     scene

    .

     Overall

    ,

     Paris

     is

     a

     popular

     tourist

     destination

     and

     an

     important

     cultural

     hub

     in

     France

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     may

     not

     have

     a

     clear

     direction

    ,

     but

     here

     are

     some

     possible

     trends

     we

     can

     expect

     to

     see

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     Integration

     with

     Human

     Conscious

    ness

    :

     AI

     systems

     will

     become

     more

     integrated

     with

     human

     consciousness

    ,

     allowing

     them

     to

     learn

     from

     and

     adapt

     to

     human

     emotions

    ,

     preferences

    ,

     and

     behaviors

    .
    


    2

    .

     Advanced

     Natural

     Language

     Processing

    :

     AI

     systems

     will

     become

     more

     capable

     of

     understanding

     and

     generating

     natural

     language

    ,

     allowing

     for

     more

     sophisticated

     language

     understanding

     and

     interaction

    .
    


    3

    .

     Greater

     Use

     of

     Predict

    ive

     Analytics

    :

     AI

     systems

     will

     become

     more

     adept

     at

     predicting

     future

     outcomes

     and

     trends

    ,

     allowing

     for

     more

     proactive

     and

     predictive

     approaches

     to

     decision

    -making

    .
    


    4

    .

     Enhanced

     Human

    -A

    



```python
llm.shutdown()
```
