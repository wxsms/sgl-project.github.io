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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.80it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:08,  1.24s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:08,  1.24s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:08,  1.24s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:16,  3.10it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:16,  3.10it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:16,  3.10it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:16,  3.10it/s]

    Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:16,  3.10it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:08,  5.67it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:08,  5.67it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:08,  5.67it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:08,  5.67it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:08,  5.67it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:08,  5.67it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:04,  9.62it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:04,  9.62it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:04,  9.62it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:04,  9.62it/s]

    Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:04,  9.62it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:05<00:04,  9.62it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:05<00:04,  9.62it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 15.11it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 15.11it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 15.11it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 15.11it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 15.11it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 15.11it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:02, 15.11it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:02, 15.11it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 22.22it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 22.22it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 22.22it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 22.22it/s]

    Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 22.22it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 22.22it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 22.22it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 22.22it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 29.65it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 29.65it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 29.65it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 29.65it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 29.65it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 29.65it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 29.65it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 29.65it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 29.65it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 38.05it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 38.05it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 38.05it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 38.05it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 38.05it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 38.05it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 38.05it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 38.05it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 44.60it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 44.60it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 44.60it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 44.60it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 44.60it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 44.60it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 44.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.69 GB):   2%|▏         | 1/58 [00:00<00:06,  8.71it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.66 GB):   2%|▏         | 1/58 [00:00<00:06,  8.71it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.66 GB):   3%|▎         | 2/58 [00:00<00:06,  9.16it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.65 GB):   3%|▎         | 2/58 [00:00<00:06,  9.16it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.65 GB):   3%|▎         | 2/58 [00:00<00:06,  9.16it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.65 GB):   7%|▋         | 4/58 [00:00<00:05, 10.26it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.65 GB):   7%|▋         | 4/58 [00:00<00:05, 10.26it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.64 GB):   7%|▋         | 4/58 [00:00<00:05, 10.26it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.64 GB):  10%|█         | 6/58 [00:00<00:04, 11.47it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.63 GB):  10%|█         | 6/58 [00:00<00:04, 11.47it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.63 GB):  10%|█         | 6/58 [00:00<00:04, 11.47it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.63 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.86it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.63 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.86it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=55.63 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.86it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.62 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.86it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.62 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.53it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.62 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.53it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.61 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.53it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.61 GB):  19%|█▉        | 11/58 [00:00<00:03, 15.53it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=55.61 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.61 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.73it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.61 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.73it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.60 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.73it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.60 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.60 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.60 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.99it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.59 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.99it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=55.59 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.58 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.41it/s]Capturing num tokens (num_tokens=960 avail_mem=55.59 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.41it/s] Capturing num tokens (num_tokens=896 avail_mem=55.59 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.41it/s]Capturing num tokens (num_tokens=832 avail_mem=55.58 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.41it/s]Capturing num tokens (num_tokens=832 avail_mem=55.58 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.63it/s]Capturing num tokens (num_tokens=768 avail_mem=55.58 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.63it/s]Capturing num tokens (num_tokens=704 avail_mem=55.58 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.63it/s]Capturing num tokens (num_tokens=640 avail_mem=55.57 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.63it/s]

    Capturing num tokens (num_tokens=576 avail_mem=55.57 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.63it/s]Capturing num tokens (num_tokens=576 avail_mem=55.57 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.05it/s]Capturing num tokens (num_tokens=512 avail_mem=55.56 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.05it/s]Capturing num tokens (num_tokens=480 avail_mem=55.57 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.05it/s]Capturing num tokens (num_tokens=448 avail_mem=55.57 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.05it/s]Capturing num tokens (num_tokens=416 avail_mem=55.57 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.05it/s]Capturing num tokens (num_tokens=416 avail_mem=55.57 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=384 avail_mem=55.57 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=352 avail_mem=55.56 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.25it/s]

    Capturing num tokens (num_tokens=320 avail_mem=55.56 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=288 avail_mem=55.55 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=288 avail_mem=55.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=256 avail_mem=55.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=240 avail_mem=55.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=224 avail_mem=55.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=208 avail_mem=55.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=208 avail_mem=55.54 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=192 avail_mem=55.54 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.35it/s]

    Capturing num tokens (num_tokens=176 avail_mem=55.54 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=160 avail_mem=55.53 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=144 avail_mem=55.53 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=144 avail_mem=55.53 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.06it/s]Capturing num tokens (num_tokens=128 avail_mem=55.53 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.06it/s]Capturing num tokens (num_tokens=112 avail_mem=55.53 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.06it/s]Capturing num tokens (num_tokens=96 avail_mem=55.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.06it/s] Capturing num tokens (num_tokens=80 avail_mem=55.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.06it/s]

    Capturing num tokens (num_tokens=80 avail_mem=55.52 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=64 avail_mem=55.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=48 avail_mem=55.51 GB):  83%|████████▎ | 48/58 [00:02<00:00, 33.34it/s]Capturing num tokens (num_tokens=32 avail_mem=55.51 GB):  83%|████████▎ | 48/58 [00:02<00:00, 33.34it/s]Capturing num tokens (num_tokens=28 avail_mem=55.50 GB):  83%|████████▎ | 48/58 [00:02<00:00, 33.34it/s]Capturing num tokens (num_tokens=28 avail_mem=55.50 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.83it/s]Capturing num tokens (num_tokens=24 avail_mem=55.50 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.83it/s]Capturing num tokens (num_tokens=20 avail_mem=55.50 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.83it/s]Capturing num tokens (num_tokens=16 avail_mem=55.50 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.83it/s]Capturing num tokens (num_tokens=12 avail_mem=55.49 GB):  90%|████████▉ | 52/58 [00:02<00:00, 33.83it/s]

    Capturing num tokens (num_tokens=12 avail_mem=55.49 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.27it/s]Capturing num tokens (num_tokens=8 avail_mem=55.49 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.27it/s] Capturing num tokens (num_tokens=4 avail_mem=55.49 GB):  97%|█████████▋| 56/58 [00:02<00:00, 34.27it/s]Capturing num tokens (num_tokens=4 avail_mem=55.49 GB): 100%|██████████| 58/58 [00:02<00:00, 25.42it/s]


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
    Generated text:  Jilin and I am a middle school student. I am very smart and I am very kind. My best friend is a girl named Lily. She is also very smart and likes to tell jokes. The two of us like to play together every Saturday. We often talk about different subjects and enjoy listening to each other talk. We also like to help each other in our studies. I love being with Lily and I am happy when she is with me. What can we infer from the passage? A) Jilin and Lily are very common. B) Jilin is very smart. C) Jilin is very kind
    ===============================
    Prompt: The president of the United States is
    Generated text:  a kind man. He has many qualities, but what are they? 
    
    A) His body is large and muscular.
    
    B) He is a very energetic person and always enjoys activities.
    
    C) He speaks with a friendly and approachable manner.
    
    D) He is a vegetarian and does not eat meat.
    
    Answer the question based on the following text.
    
    To explain the situation, the text provided mentions that the president is a kind man. This can be deduced from the statement that he "has many qualities, but what are they? " In this case, the qualities he possesses are not directly mentioned in the text. However, the text
    ===============================
    Prompt: The capital of France is
    Generated text:  located on the right bank of the Rhine. It is located in the center of the country and is situated in the north-western region of France. The capital of France is Paris. The administrative centre of the capital is the City Hall. It is the heart of Paris.
    The people of Paris speak French. They speak many other languages which they learn at school. They speak French, English, German, Italian, Dutch, and Spanish as their native languages. They also speak Spanish, French, and other languages such as Italian, Portuguese, and Arabic.
    The capital of France is the world’s largest city. Its population is 2
    ===============================
    Prompt: The future of AI is
    Generated text:  mostly defined by the skill of the designer and the individuals that are coding the next iteration of this technology. While the technology itself is in its infancy, there is a growing trend towards designing and coding AI in a way that allows it to consider the ethical implications of its actions and output. 
    
    As developers continue to refine and expand on this technology, it's likely to be used for a wide range of applications in fields such as healthcare, finance, and transportation. However, it's important to note that the ethical implications of AI are multifaceted, and are often beyond the scope of just one set of guidelines or ethical principles.
    
    To help


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short description of your favorite activity]. I'm always looking for ways to improve myself and make the world a better place. What's your favorite book or movie? I love [insert a short
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also known for its cuisine, including French cuisine, and its fashion industry. The city is home to many international organizations and institutions, including the French Academy of Sciences and the French National Library. Paris is a city that is a true reflection of France's rich history and culture. It is a city that is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced ethical considerations: As AI becomes more advanced, there will be a growing need for ethical guidelines and regulations to ensure that AI systems are used in a responsible and ethical manner.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in more complex and personalized ways, with the goal of
    


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
    Generated text:  [Your Name], and I am an [occupation] in the [substance] industry. I have been in this industry for [number] years, and I have always been passionate about [description of your occupation]. I enjoy meeting new people, learning about different cultures, and creating unique experiences for those I meet. I strive to be a positive and supportive member of the community, and I believe in the power of [description of your occupation]. I hope you find my character engaging and relatable. 
    Your name: [Your Name] Your occupation: [Your Occupation] Your interest in the industry: [Interest in the industry]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    This statement accurately captures the essence of the capital city of France, providing a clear and concise overview of its location and significance. 
    
    To elaborate further, Paris is the largest city in France, located in the south of the country, on the banks of the Seine river. It is home to many of the country's notable landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also known as the "City of Light" and is home to numerous museums, such as the Louvre and the Musée d'Orsay. The city's historical importance is reflected in its numerous
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a topic of great interest and debate. Here are some possible trends that may emerge in the next few years:
    
    1. Increased specialization and expertise: AI is becoming increasingly specialized, with more and more companies focusing on specific tasks and applications. This may lead to a greater emphasis on expertise and skills in certain areas, such as natural language processing or image recognition.
    
    2. AI becoming more human-like: As AI becomes more advanced, it may become more like humans, with the ability to learn from feedback and improve its performance over time. This could lead to a more natural and interactive AI experience.
    
    3. AI becoming more personal: AI is already


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

    ],

     and

     I

    'm

     a

     

    3

    0

    -year

    -old

     [

    occupation

    ].

     I

    'm

     an

     energetic

     and

     driven

     individual

     who

     loves

     nothing

     more

     than

     to

     challenge

     the

     status

     quo

    .

     My

     goal

     is

     to

     push

     the

     boundaries

     of

     what

    's

     possible

     and

     make

     a

     positive

     impact

     on

     the

     world

    .

     I

    'm

     a

     problem

    -s

    olver

     who

     thr

    ives

     in

     fast

    -paced

     environments

     and

     enjoys

     taking

     on

     new

     challenges

    .

     I

    've

     had

     the

     privilege

     of

     working

     in

     various

     fields

    ,

     including

     technology

    ,

     healthcare

    ,

     and

     marketing

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

     value

     open

    -minded

    ness

     and

     a

     proactive

     mindset

    ,

     and

     I

     strive

     to

     make

     a

     difference

     in

     the

     world

    .

     What

     better

     way

     to

     start

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Paris

     is

     the

     largest

     city

     in

     France

     and

     one

     of

     the

     largest

     cities

     in

     the

     European

     Union

    .

     It

     is

     a

     major

     cultural

    ,

     economic

    ,

     and

     political

     center

    .

     The

     city

     has

     a

     rich

     history

    ,

     and

     is

     known

     for

     its

     artistic

    ,

     literary

    ,

     and

     culinary

     traditions

    .

     Paris

     is

     also

     the

     second

    -largest

     city

     in

     terms

     of

     population

    ,

     after

     London

    ,

     and

     has

     a

     high

     population

     density

    .

     It

     is

     also

     the

     birth

    place

     of

     numerous

     famous

     historical

     figures

    ,

     including

     Napoleon

     and

     Victor

     Hugo

    .

     
    


    Paris

     is

     famous

     for

     its

     architecture

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

     Arc

     de

     Tri

    omp

    he

    .

     It

     is

     also

     known

     for

     its

     cuisine

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     not

     yet

     clear

    ,

     but

     some

     potential

     trends

     that

     may

     emerge

     include

    :
    


     

     

    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     may

     see

     more

     applications

     of

     AI

     in

     other

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    .


     

     

    2

    .

     Greater

     automation

    :

     AI

     is

     already

     being

     used

     to

     automate

     tasks

     in

     industries

     such

     as

     manufacturing

    ,

     but

     there

     is

     potential

     for

     even

     more

     automation

     in

     the

     future

    ,

     including

     the

     use

     of

     AI

    -powered

     robots

     and

     autonomous

     vehicles

    .


     

     

    3

    .

     Enhanced

     intelligence

    :

     AI

     is

     getting

     better

     at

     performing

     tasks

     that

     were

     previously

     considered

     impossible

    ,

     and

     with

     further

     advances

     in

     AI

    ,

     it

     is

     possible

     that

     we

    



```python
llm.shutdown()
```
