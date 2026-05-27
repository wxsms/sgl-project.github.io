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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:57,  1.05s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:57,  1.05s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:57,  1.05s/it]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<00:57,  1.05s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.63it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.63it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.63it/s]

    Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.63it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:10,  4.63it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:10,  4.63it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:05,  8.26it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:05,  8.26it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:05,  8.26it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:05,  8.26it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:05,  8.26it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:05,  8.26it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:05,  8.26it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:05,  8.26it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:02, 14.52it/s]

    Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 24.02it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 24.02it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 24.02it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 24.02it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 24.02it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 24.02it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 24.02it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 24.02it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 24.02it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 24.02it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 24.02it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 35.52it/s]

    Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 35.52it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 46.47it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 46.47it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 46.47it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 46.47it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 46.47it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 46.47it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 46.47it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 46.47it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.53 GB):   2%|▏         | 1/58 [00:00<00:06,  8.83it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.50 GB):   2%|▏         | 1/58 [00:00<00:06,  8.83it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=57.50 GB):   3%|▎         | 2/58 [00:00<00:06,  9.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.49 GB):   3%|▎         | 2/58 [00:00<00:06,  9.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.49 GB):   3%|▎         | 2/58 [00:00<00:06,  9.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.49 GB):   7%|▋         | 4/58 [00:00<00:05,  9.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.49 GB):   7%|▋         | 4/58 [00:00<00:05,  9.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.49 GB):   9%|▊         | 5/58 [00:00<00:05,  9.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.49 GB):   9%|▊         | 5/58 [00:00<00:05,  9.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.48 GB):   9%|▊         | 5/58 [00:00<00:05,  9.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.48 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.94it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.47 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.94it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=57.47 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.47 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.47 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.46 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.71it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  17%|█▋        | 10/58 [00:01<00:03, 15.71it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:01<00:03, 11.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:01<00:03, 11.93it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:01<00:03, 11.93it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:01<00:03, 11.93it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:01<00:03, 11.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  21%|██        | 12/58 [00:01<00:03, 11.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.85it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.85it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.85it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.85it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.85it/s] Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.67it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.67it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.67it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.67it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.67it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:01<00:01, 26.67it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.84it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.84it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.84it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.84it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.84it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.84it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.59it/s]

    Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.59it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.22it/s] Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  81%|████████  | 47/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 43.35it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.66it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:02<00:00, 28.98it/s]


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
    Generated text:  Patrick and I'm a 25 year old male living in San Francisco. I had my first shot of sunscreen yesterday (4th August) and I'm worried about a rash/dermatitis after I sunbathe in the sun. Is it normal to have a rash/dermatitis after sun exposure? If so, can I avoid this rash/dermatitis by not using a sunscreen? I already have aloe vera on my skin. What is the best way to prevent a rash/dermatitis after sun exposure? Thank you.
    Yes, it is normal for a rash or dermatitis to occur after sun exposure
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to use the 2020 presidential election results to change the rules of the country. He likes the idea of keeping the 50 states as they are, but wants to have a smaller number of major parties. He wants to know the winner of each election race.
    
    There are 141 total races involving 50 states and 50 races are going on at the same time. 
    
    What is the probability of the outcome for the race with the winner being Hillary Clinton?
    
    To determine the probability of the outcome for the race with the winner being Hillary Clinton, we need to follow these steps:
    
    
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located on the left bank of the Seine river and is surrounded by hills. It is famous for its architecture and for being a center of culture and philosophy. Paris is a historical city, and the great buildings of the city have a historical significance. The most famous landmark of Paris is the Eiffel Tower. The Eiffel Tower is the tallest structure in the world, standing at 324 feet tall. It was constructed in 1889 and was first used in 1889 for the opening of the 1889 World's Fair.
    
    The Eiffel Tower was
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s going to have a profound impact on how we live, work, and play. As we continue to develop AI and its applications, it is essential to ensure that the technology is designed and implemented in a way that is ethical, responsible, and beneficial for all stakeholders involved. This involves implementing ethical principles and values that guide the development and use of AI, and fostering a culture of transparency, accountability, and ethical considerations throughout the entire lifecycle of AI development. By doing so, we can ensure that AI is used to create positive and sustainable outcomes for all.
    The benefits of AI can be immense, including improved efficiency, reduced


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. Let's chat! [Name] [Job Title] [Company Name] [Company Address] [City, State, ZIP Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major transportation hub, with many major highways and railroads connecting the city to other parts of France and the world. Paris is a popular tourist destination, with millions of visitors each year. It is a cultural and artistic center, with many museums, theaters, and art galleries. The city is also known for its food and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and experiences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Enhanced machine learning capabilities: AI is likely to become even more powerful and capable, with the ability to learn from vast amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective use of resources, as well as more accurate predictions of human behavior and outcomes.
    
    3.
    


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
    Generated text:  [insert your name here] and I'm a [insert your occupation] who has always been passionate about learning and exploring the world around us. I have a keen curiosity and love to challenge myself with new knowledge and experiences. My main hobby is learning how to code and am always up for sharing my knowledge with others. I am passionate about promoting technology and innovation and believe that everyone should have access to the latest tools and resources to help them grow in their career. My main goal is to help others achieve their dreams by sharing my expertise and passion. I am [insert your age and position] and I believe that having a good sense of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the 19th and 21st century cultural and intellectual hub. The city, located in the south of France, has an impressive historical architecture, and is known for its annual fashion and gastronomic festivals. In addition, Paris has a large international community and is one of the largest cities in the world by population. The city is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Its many landmarks and museums make it a popular tourist destination. The capital of France is Paris, an influential city that has long been a center of culture, politics, and commerce.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  shaped by a variety of trends, including:
    
    1. Increased Integration with Human Work: With more and more of the world's work being automated, AI is likely to become even more integrated with human workers, creating a more efficient and effective workforce.
    
    2. Increased Use of AI for Predictive Maintenance: AI can be used to predict equipment failure and to optimize maintenance schedules, reducing the need for expensive repairs and downtime.
    
    3. AI for Personalization: AI can be used to personalize the experience of customers, from recommendations for products or services to personalized marketing campaigns.
    
    4. AI for Fraud Detection: AI can be used to identify fraudulent activity,


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

     [

    Country

    ].

     I

    'm

     a

     [

    Prof

    ession

    ]

     with

     [

    Position

    ]

     at

     [

    Company

    ].

     I

    'm

     a

     [

    Role

    ]

     for

     [

    Title

    ].

     I

    'm

     [

    About

     the

     role

    ]

     [

    Character

    ].

     I

    'm

     a

     [

    Other

     Skills

     or

     Abilities

    ]

     that

     make

     me

     [

    Some

     characteristic

    ].

     I

    'm

     [

    Person

    ality

    ].

     And

     [

    Character

    istics

     of

     your

     personality

    ].

     I

    'm

     an

     [

    Summary

     of

     your

     personality

    ].

     And

     I

     love

     [

    Your

     primary

     passion

     or

     interest

    ].

     I

    'm

     a

     [

    Your

     Passion

    /

    Interest

    ].

     I

    'm

     also

     [

    Your

     Family

     or

     Last

     Name

    ].

     And

     [

    Your

     Last

     Name

    ].

     I

    'm

     [

    Your

     Family

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     a

     city

     located

     in

     the

     north

     of

     the

     country

    .
    


    Paris

     is

     home

     to

     the

     iconic

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     landmarks

     and

     attractions

    ,

     making

     it

     the

     cultural

     and

     economic

     center

     of

     France

    .

     It

     is

     also

     known

     for

     its

     vibrant

     street

     food

     culture

    ,

     French

     cooking

    ,

     and

     its

     status

     as

     a

     world

    -ren

    owned

     cultural

     and

     artistic

     city

    .

     
    


    Paris

     is

     home

     to

     the

     Paris

     Review

    ,

     a

     well

    -known

     literary

     magazine

    ,

     and

     the

     French

     Wikipedia

    ,

     the

     largest

     encyclopedia

     in

     the

     world

    .

     It

     has

     also

     been

     the

     site

     of

     major

     political

     events

     such

     as

     the

     July

     

    1

    4

    th

     Pand

    emon

    ium

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     dynamic

    ,

     with

     a

     wide

     range

     of

     potential

     developments

     and

     applications

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

     Increased

     Use

     of

     AI

     in

     Healthcare

    :

     AI

     is

     becoming

     more

     integrated

     into

     healthcare

    ,

     from

     medical

     imaging

     to

     personalized

     treatment

     plans

    .

     As

     AI

     becomes

     better

     at

     recognizing

     patterns

     and

     making

     decisions

    ,

     it

     is

     likely

     to

     become

     a

     more

     integral

     part

     of

     healthcare

    ,

     potentially

     leading

     to

     more

     accurate

     diagnoses

     and

     more

     efficient

     use

     of

     resources

    .
    


    2

    .

     Enhanced

     AI

     for

     Manufacturing

    :

     AI

     is

     being

     used

     in

     manufacturing

     to

     optimize

     processes

    ,

     identify

     defects

     and

     improve

     quality

     control

    .

     For

     example

    ,

     AI

     can

     be

     used

     to

     analyze

     sensor

     data

     from

     machines

     and

     identify

     potential

     issues

     before

     they

     occur

    .

    



```python
llm.shutdown()
```
