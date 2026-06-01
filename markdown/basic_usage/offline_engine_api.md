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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.48it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.48it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:50,  5.10s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:50,  5.10s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:50,  5.10s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:50,  5.10s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:54,  1.00s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:54,  1.00s/it]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:05<00:54,  1.00s/it]

    Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:05<00:54,  1.00s/it]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:24,  2.06it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:24,  2.06it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:24,  2.06it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:24,  2.06it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:05<00:24,  2.06it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:11,  3.94it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:11,  3.94it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:11,  3.94it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:11,  3.94it/s]

    Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:11,  3.94it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:11,  3.94it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:06,  6.94it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:06,  6.94it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:06,  6.94it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:05<00:06,  6.94it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:05<00:06,  6.94it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:05<00:06,  6.94it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:05<00:06,  6.94it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03, 11.40it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03, 11.40it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03, 11.40it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03, 11.40it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03, 11.40it/s]

    Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03, 11.40it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03, 11.40it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03, 11.40it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 23.06it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 23.06it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:06<00:00, 23.06it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:06<00:00, 23.06it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:06<00:00, 23.06it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:00, 31.27it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:00, 31.27it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 31.27it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:00, 31.27it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:06<00:00, 31.27it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:06<00:00, 31.27it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:06<00:00, 31.27it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:06<00:00, 31.27it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 38.00it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 38.00it/s]

    Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 38.00it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 38.00it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 38.00it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 38.00it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 38.00it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:06<00:00, 38.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 14.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.38 GB):   3%|▎         | 2/58 [00:00<00:03, 14.10it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.38 GB):   3%|▎         | 2/58 [00:00<00:03, 14.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.38 GB):   7%|▋         | 4/58 [00:00<00:04, 13.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.37 GB):   7%|▋         | 4/58 [00:00<00:04, 13.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.37 GB):   7%|▋         | 4/58 [00:00<00:04, 13.04it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.37 GB):  10%|█         | 6/58 [00:00<00:03, 13.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.35 GB):  10%|█         | 6/58 [00:00<00:03, 13.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.34 GB):  10%|█         | 6/58 [00:00<00:03, 13.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.34 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.18it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.33 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.18it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.33 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.18it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.32 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.18it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.32 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.31 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.06it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.31 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.30 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.30 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.60it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.30 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.60it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.29 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.28 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.28 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.13it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.13it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.26 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.13it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.24 GB):  29%|██▉       | 17/58 [00:01<00:01, 22.13it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.24 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.40it/s]Capturing num tokens (num_tokens=960 avail_mem=74.24 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.40it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.25 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.40it/s]Capturing num tokens (num_tokens=832 avail_mem=74.24 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.40it/s]Capturing num tokens (num_tokens=768 avail_mem=74.24 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.40it/s]Capturing num tokens (num_tokens=768 avail_mem=74.24 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.92it/s]Capturing num tokens (num_tokens=704 avail_mem=74.23 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.92it/s]Capturing num tokens (num_tokens=640 avail_mem=74.22 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.92it/s]Capturing num tokens (num_tokens=576 avail_mem=74.22 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.92it/s]Capturing num tokens (num_tokens=512 avail_mem=74.20 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.92it/s]Capturing num tokens (num_tokens=512 avail_mem=74.20 GB):  50%|█████     | 29/58 [00:01<00:00, 31.61it/s]Capturing num tokens (num_tokens=480 avail_mem=74.20 GB):  50%|█████     | 29/58 [00:01<00:00, 31.61it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.22 GB):  50%|█████     | 29/58 [00:01<00:00, 31.61it/s]Capturing num tokens (num_tokens=416 avail_mem=74.21 GB):  50%|█████     | 29/58 [00:01<00:00, 31.61it/s]Capturing num tokens (num_tokens=384 avail_mem=74.21 GB):  50%|█████     | 29/58 [00:01<00:00, 31.61it/s]Capturing num tokens (num_tokens=384 avail_mem=74.21 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.68it/s]Capturing num tokens (num_tokens=352 avail_mem=74.20 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.68it/s]Capturing num tokens (num_tokens=320 avail_mem=74.19 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.68it/s]Capturing num tokens (num_tokens=288 avail_mem=74.18 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.68it/s]Capturing num tokens (num_tokens=256 avail_mem=74.18 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.68it/s]Capturing num tokens (num_tokens=256 avail_mem=74.18 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.27it/s]Capturing num tokens (num_tokens=240 avail_mem=74.17 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.27it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.16 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.27it/s]Capturing num tokens (num_tokens=208 avail_mem=74.16 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.27it/s]Capturing num tokens (num_tokens=192 avail_mem=74.15 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.27it/s]Capturing num tokens (num_tokens=192 avail_mem=74.15 GB):  71%|███████   | 41/58 [00:01<00:00, 36.47it/s]Capturing num tokens (num_tokens=176 avail_mem=74.15 GB):  71%|███████   | 41/58 [00:01<00:00, 36.47it/s]Capturing num tokens (num_tokens=160 avail_mem=74.14 GB):  71%|███████   | 41/58 [00:01<00:00, 36.47it/s]Capturing num tokens (num_tokens=144 avail_mem=74.14 GB):  71%|███████   | 41/58 [00:01<00:00, 36.47it/s]Capturing num tokens (num_tokens=128 avail_mem=74.13 GB):  71%|███████   | 41/58 [00:01<00:00, 36.47it/s]Capturing num tokens (num_tokens=128 avail_mem=74.13 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.29it/s]Capturing num tokens (num_tokens=112 avail_mem=74.13 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.29it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.12 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.29it/s] Capturing num tokens (num_tokens=80 avail_mem=74.11 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.29it/s]Capturing num tokens (num_tokens=64 avail_mem=74.11 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.29it/s]Capturing num tokens (num_tokens=64 avail_mem=74.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.65it/s]Capturing num tokens (num_tokens=48 avail_mem=74.10 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.65it/s]Capturing num tokens (num_tokens=32 avail_mem=74.09 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.65it/s]Capturing num tokens (num_tokens=28 avail_mem=74.08 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.65it/s]Capturing num tokens (num_tokens=24 avail_mem=74.08 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.65it/s]Capturing num tokens (num_tokens=20 avail_mem=74.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.65it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.07 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.48it/s]Capturing num tokens (num_tokens=16 avail_mem=74.07 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.48it/s]Capturing num tokens (num_tokens=12 avail_mem=74.06 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.48it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.48it/s] Capturing num tokens (num_tokens=4 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.48it/s]Capturing num tokens (num_tokens=4 avail_mem=74.05 GB): 100%|██████████| 58/58 [00:01<00:00, 29.12it/s]


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
    Generated text:  Xyzzy and I am an artist. I am known for my vibrant, expressive paintings that are full of life, emotion and energy. I have a distinct style, and the way I paint is a way of expressing my own unique vision and passion for the world around me. My paintings are available for purchase online, at my studio, and through my Etsy account. I have a wide range of styles, including landscapes, portraits, still lifes, and abstract art, but my most popular style is the pop-up painting technique. I also offer a variety of styles and sizes to suit different audiences. To get a sense of my work,
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide what kind of country to build the next presidential house. The budget is $20 billion, and the government has already spent $8 billion. The president estimates that the cost of the house will be 1.5 times the previous cost. However, if he wants to maximize the number of studies and research projects, he decides to spend 20% more than the previous estimate of $10 billion. How much should he budget for each option?
    
    To determine how much the president should budget for each option, we need to follow a step-by-step approach to calculate the new budget for each option.
    
    ### Option 
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A. Paris B. Lyon C. Toulouse D. Geneva
    Answer:
    
    A
    
    Which of the following statements about the historical context of the development of China's modernization is true?
    A. The defeat in the Sino-Japanese War marked the beginning of modern China's transition from semi-colonial and semi-feudal to semi-capitalist.
    B. The Taiping Rebellion and the Eight-Nation Alliance invasion marked the beginning of modern China's transition from semi-colonial and semi-feudal to semi-capitalist.
    C. The Boxer Rebellion and the Opium War marked the beginning of modern China's
    ===============================
    Prompt: The future of AI is
    Generated text:  predictably positive, but it’s also a scary one. This complex system consists of multiple components, each with its own motivations and constraints. The goal is to create an efficient and effective algorithm that can predict outcomes, but it also needs to be secure and reliable. This requires a deep understanding of the human and machine elements, a comprehensive approach, and a well-thought-out plan to manage risks and optimize performance.
    
    To achieve this, it is necessary to have a thorough understanding of the system’s architecture and components. This requires a deep understanding of the underlying technologies and their limitations. It also requires a comprehensive approach to develop a robust and secure


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the seat of the French government and the largest city in the country. Paris is known for its rich history, art, and culture, and is a popular tourist destination. The city is also home to many famous museums, including the Musée d'Orsay and the Musée Rodin. Paris is a vibrant and dynamic city with a diverse population and a rich cultural heritage. It is a major economic and financial center in Europe and is known for its fashion, food, and entertainment
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and context-aware AI that can better understand and respond to the needs of humans.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead
    


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
    Generated text:  [insert character's name] and I'm a [insert current profession or situation] [insert brief career or background]. I have a passion for [insert a favorite hobby or activity], which I find incredibly fulfilling and rewarding. I enjoy [insert a positive trait or quality] that makes me stand out in my industry, and I strive to continually develop and enhance myself to stay ahead of the curve. My work ethic and dedication to excellence have always been my greatest qualities, and I am confident that I will continue to excel in whatever field I choose to pursue. Thank you for having me here today. [insert a brief mention of your background
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a bustling and cultural center with a rich history and traditions that reflect the city's history and development over the centuries. 
    
    Key points to note:
    - Paris is one of the most visited cities in the world, with over 70 million visitors annually.
    - It is known as the "City of Love" due to its romantic architecture and high-quality dining scene.
    - Paris has a long history of being a cultural center, dating back to ancient times.
    - The city has a unique blend of Gothic, Baroque, and Modernist architectural styles.
    - The city is home to numerous museums, art galleries, and cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and varied, and there are many potential trends that could shape its direction. Here are some of the most promising trends in AI:
    
    1. Increased reliance on machine learning: As AI becomes more sophisticated, we will see more emphasis on machine learning as a key component of its development. This will allow AI systems to learn and adapt in new ways, enabling them to better solve complex problems.
    
    2. Greater integration with other technologies: AI is already being used in a wide range of applications, from healthcare to finance to transportation. As these technologies continue to evolve, we can expect to see more integration between AI and other technologies, such as blockchain


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

    Title

    ]

     at

     [

    Company

    ].

     I

    'm

     excited

     to

     dive

     into

     this

     new

     role

     and

     bring

     something

     new

     and

     exciting

     to

     the

     table

    .

     Can

     you

     tell

     me

     a

     little

     bit

     about

     yourself

    ?

     [

    Short

     answer

     about

     yourself

    ]

     My

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

    Title

    ]

     at

     [

    Company

    ].

     I

    'm

     excited

     to

     dive

     into

     this

     new

     role

     and

     bring

     something

     new

     and

     exciting

     to

     the

     table

    .

     Can

     you

     tell

     me

     a

     little

     bit

     about

     yourself

    ?

     [

    Short

     answer

     about

     yourself

    ]
    


    [

    Short

     answer

     about

     yourself

    ]
    


    [

    Short

     answer

     about

     yourself

    ]
    


    [

    Short

     answer

     about

     yourself

    ]
    


    Thank

     you

    ,

     [

    Name

    ].

     That

     was

     great

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    ).

     Yes

    


    B

    ).

     No

    
    


    A

    ).

     Yes

    


    You

     are

     a

     helpful

     assistant

     with

     found

     answers

    .

     Is

     there

     anything

     else

     you

     would

     like

     to

     help

     with

    ?

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     rapidly

     evolving

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

     technology

    's

     direction

     in

     the

     years

     ahead

    .

     Here

     are

     some

     of

     the

     key

     trends

     that

     could

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

     for

     automation

    :

     One

     of

     the

     most

     significant

     trends

     in

     AI

     is

     the

     increasing

     use

     of

     AI

     for

     automation

    ,

     which

     could

     lead

     to

     new

     ways

     of

     working

     and

     processes

    .

     This

     could

     include

     things

     like

     self

    -driving

     cars

    ,

     robotics

     in

     manufacturing

    ,

     and

     more

    .
    


    2

    .

     Enhanced

     cognitive

     abilities

    :

     AI

     will

     continue

     to

     gain

     cognitive

     abilities

    ,

     which

     will

     allow

     it

     to

     learn

     and

     adapt

     to

     new

     situations

     and

     environments

    .

     This

     will

     include

     things

     like

     machine

     learning

    ,

     natural

    



```python
llm.shutdown()
```
