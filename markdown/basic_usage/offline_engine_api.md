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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.11it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.11it/s]


    2026-05-21 02:11:09,910 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-21 02:11:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:45,  1.18it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 12.34it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 12.34it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 12.34it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 12.34it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 12.34it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 12.34it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 12.34it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 12.34it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 12.34it/s]Compiling num tokens (num_tokens=416):  40%|███▉      | 23/58 [00:04<00:02, 12.34it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 19.90it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 19.90it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 19.90it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 19.90it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:01, 19.90it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:04<00:01, 19.90it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:04<00:01, 19.90it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:04<00:01, 19.90it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:04<00:01, 19.90it/s]

    Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:04<00:01, 19.90it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 28.39it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 28.39it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 28.39it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 28.39it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 28.39it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 28.39it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 28.39it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 28.39it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 28.39it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 28.39it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 37.23it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 37.23it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 37.23it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 37.23it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 37.23it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 37.23it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 37.23it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 37.23it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 37.23it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.04 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.04 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.69it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.01 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.69it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=71.00 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.00 GB):  21%|██        | 12/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.00 GB):  21%|██        | 12/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.00 GB):  21%|██        | 12/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.97 GB):  21%|██        | 12/58 [00:00<00:01, 29.29it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.97 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.78it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.97 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.78it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.96 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.78it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=70.96 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.78it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.96 GB):  31%|███       | 18/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.96 GB):  31%|███       | 18/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.94 GB):  31%|███       | 18/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.44 GB):  31%|███       | 18/58 [00:00<00:01, 27.42it/s]Capturing num tokens (num_tokens=960 avail_mem=70.46 GB):  31%|███       | 18/58 [00:00<00:01, 27.42it/s] Capturing num tokens (num_tokens=960 avail_mem=70.46 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=896 avail_mem=70.46 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=832 avail_mem=70.45 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=768 avail_mem=70.45 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=704 avail_mem=70.45 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.57it/s]

    Capturing num tokens (num_tokens=640 avail_mem=70.31 GB):  38%|███▊      | 22/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=640 avail_mem=70.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.60it/s]Capturing num tokens (num_tokens=576 avail_mem=70.28 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.60it/s]Capturing num tokens (num_tokens=512 avail_mem=70.27 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.60it/s]Capturing num tokens (num_tokens=480 avail_mem=70.28 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.60it/s]Capturing num tokens (num_tokens=448 avail_mem=70.28 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.60it/s]Capturing num tokens (num_tokens=416 avail_mem=70.28 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.60it/s]Capturing num tokens (num_tokens=416 avail_mem=70.28 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.92it/s]Capturing num tokens (num_tokens=384 avail_mem=70.28 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.92it/s]Capturing num tokens (num_tokens=352 avail_mem=70.27 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.92it/s]Capturing num tokens (num_tokens=320 avail_mem=70.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.92it/s]Capturing num tokens (num_tokens=288 avail_mem=70.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.92it/s]

    Capturing num tokens (num_tokens=256 avail_mem=70.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.92it/s]Capturing num tokens (num_tokens=256 avail_mem=70.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=240 avail_mem=70.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=224 avail_mem=70.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=208 avail_mem=70.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=192 avail_mem=70.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=176 avail_mem=70.24 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=176 avail_mem=70.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.23it/s]Capturing num tokens (num_tokens=160 avail_mem=70.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.23it/s]Capturing num tokens (num_tokens=144 avail_mem=70.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.23it/s]Capturing num tokens (num_tokens=128 avail_mem=70.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.23it/s]Capturing num tokens (num_tokens=112 avail_mem=70.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.23it/s]

    Capturing num tokens (num_tokens=96 avail_mem=70.23 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.23it/s] Capturing num tokens (num_tokens=96 avail_mem=70.23 GB):  81%|████████  | 47/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=80 avail_mem=70.23 GB):  81%|████████  | 47/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=64 avail_mem=70.22 GB):  81%|████████  | 47/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=48 avail_mem=70.22 GB):  81%|████████  | 47/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=32 avail_mem=70.22 GB):  81%|████████  | 47/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=28 avail_mem=70.21 GB):  81%|████████  | 47/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=28 avail_mem=70.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.28it/s]Capturing num tokens (num_tokens=24 avail_mem=70.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.28it/s]Capturing num tokens (num_tokens=20 avail_mem=70.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.28it/s]Capturing num tokens (num_tokens=16 avail_mem=70.21 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.28it/s]

    Capturing num tokens (num_tokens=12 avail_mem=70.20 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.28it/s]Capturing num tokens (num_tokens=8 avail_mem=70.20 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.28it/s] Capturing num tokens (num_tokens=8 avail_mem=70.20 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=4 avail_mem=70.19 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=4 avail_mem=70.19 GB): 100%|██████████| 58/58 [00:01<00:00, 36.87it/s]


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
    Generated text:  Alex, and I am an advanced computer scientist. I am interested in exploring the possibilities of artificial intelligence and machine learning. I am also a passionate advocate for the use of technology in a responsible and ethical manner. Can you please provide some insights on the future of technology and AI? To do this, I would like to know your perspective on the impact of AI on the future of work, education, and healthcare. Additionally, please share your thoughts on the ethical considerations surrounding AI and how to ensure that it is used in a responsible and beneficial way. Finally, can you provide some examples of how artificial intelligence has been applied in the past,
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to executive a policy to increase taxes. The president will randomly select 500 people to survey. The government has recorded that the average tax rate in 100 randomly selected countries is 18.8%. Suppose that this trend is consistent across countries.
    
    a) What is the probability that at least one person selected from the 500 surveyed will have a tax rate above the average?
    
    b) What is the probability that at least one person selected from the 500 surveyed will have a tax rate above the average if the president decides to conduct a survey on a different group of 5
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Rome
    C. Athens
    D. Berlin
    Answer:
    A
    
    A certain non-metallic element has a half-life of 75 years. If you want to prepare 100g of a solution with a solute concentration of 0.1 mol/L, the amount of the solution should be ____
    A. 1/8
    B. 2/3
    C. 2
    D. 3/2
    Answer:
    D
    
    In the case of a firm implementing a cost-plus pricing strategy, the cost of production is given by the following formula: C
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, with new technologies poised to redefine the world. Here are some of the most exciting AI trends we've seen in recent years:
    1. Generative AI
    Generative AI is a type of AI that can generate new text, images, and audio based on existing data. This is a game-changer for natural language processing, as it allows machines to generate human-like text, music, and more.
    2. Reinforcement Learning
    Reinforcement learning is a type of AI that involves interacting with the environment to learn how to perform tasks. This is particularly useful for games and robotics, as it allows machines to learn how to make


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm a [job title] at [company name], and I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm a [job title] at [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire. Paris is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. The city is also known for its fashion industry, with many famous designers and boutiques. Paris is a popular tourist destination, with millions of visitors annually. The city is also home to many important universities, including the University of Paris-Sorbonne and the University of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud detection
    


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
    Generated text:  [Your Name]. I am [Your Age] years old and I am a [Your Profession] with a background in [Your Major/Field of Study] and a keen interest in [Your Interests/Professions/Books/…]. Whether it's technology, creativity, history, philosophy, or literature, I am passionate about exploring the depths of these subjects and I am eager to learn from the people around me. What's your name, what's your profession, and what kind of hobbies do you have? Let me know if there's anything else you'd like me to know about me. Hello, my name is [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, which is a historic city with a rich history and many landmarks such as the Eiffel Tower and Louvre Museum. It is located in the Paris region, about 118 km west of Paris. It is the second-largest city in France, after Paris. The city is known for its beautiful architecture, its vibrant nightlife, and its annual Le Seine River Festival. The city is home to many universities and cultural institutions, and has a strong cultural and artistic tradition. It is an important economic center and the seat of the French government and the country's political capital. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and changing rapidly. Here are some possible trends to expect:
    
    1. Increased AI integration with other technologies: AI is becoming more integrated with other technologies, such as machine learning, big data, and edge computing. This integration could lead to a more seamless and integrated experience for users.
    
    2. AI becoming more intelligent: AI is getting smarter and more capable of understanding and solving problems on its own. This could lead to more personalized and efficient services for users.
    
    3. AI becoming more ethical: As AI becomes more advanced, there will be a growing emphasis on ethical considerations in the development of AI. This could lead to more responsible and accountable


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

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    _.

     And

     my

     favorite

     hobby

     is

     __

    ________

    _.

     I

     work

     as

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     My

     greatest

     achievement

     is

     __

    ________

    .

     I

     like

     to

     __

    ________

    .

     I

     have

     a

    /an

     __

    ________

    .


    S

    incerely

    ,

     __

    ________

    .

     (

    Write

     down

     your

     name

    ,

     character

     type

    ,

     occupation

    ,

     hobby

    ,

     workplace

    ,

     greatest

     achievement

    ,

     favorite

     hobby

    ,

     likes

    ,

     and

     any

     other

     relevant

     details

    )

     Dear

     reader

    ,

     I

     am

     a

     fictional

     character

    ,

     Jane

     Doe

    ,

     a

    /an

     graphic

     designer

    .

     I

     specialize

     in

     creating

     beautiful

     and

     professional

     graphic

     designs

    ,

     and

     I

     enjoy

     designing

     websites

     and

     graphics

     for

     clients

     who

     are

     looking

     to

     boost

     their

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

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

     It

     is

     also

     home

     to

     many

     world

    -ren

    owned

     art

     and

     music

     venues

    ,

     such

     as

     the

     Palace

     of

     Vers

    ailles

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    .

     Paris

     is

     considered

     one

     of

     the

     most

     cosm

    opolitan

     cities

     in

     the

     world

     and

     is

     a

     significant

     center

     of

     science

    ,

     culture

    ,

     and

     politics

    .

     As

     a

     result

    ,

     it

     has

     a

     rich

     and

     diverse

     cultural

     scene

    ,

     with

     many

     cultural

     events

    ,

     festivals

    ,

     and

     festivals

     throughout

     the

     year

    .

     The

     city

     has

     a

     vibrant

     nightlife

     and

     a

     sophisticated

     dining

     scene

    ,

     with

     many

     restaurants

     and

     cafes

     offering

     delicious

     cuisine

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     trends

     and

     developments

    ,

     including

    :
    


    1

    .

     Improved

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     complex

     and

     widespread

    ,

     there

     will

     be

     an

     increasing

     need

     for

     measures

     to

     protect

     user

     privacy

     and

     security

    .

     This

     could

     include

     encryption

    ,

     AI

    -powered

     cybersecurity

    ,

     and

     data

     anonym

    ization

    .
    


    2

    .

     Increased

     automation

     and

     efficiency

    :

     AI

     is

     becoming

     increasingly

     integrated

     into

     many

     industries

    ,

     from

     manufacturing

     to

     healthcare

     to

     transportation

    .

     As

     this

     automation

     increases

    ,

     there

     will

     be

     greater

     efficiency

     and

     productivity

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     help

     diagnose

     and

     treat

     diseases

    .

     As

     the

     technology

     continues

     to

     improve

    ,

     we

     may

     see

     even

     more

     sophisticated

    



```python
llm.shutdown()
```
