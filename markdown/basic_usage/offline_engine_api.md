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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:10,  4.33it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  6.11it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  6.11it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  6.11it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  6.11it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:04,  8.27it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:04,  8.27it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:04,  8.27it/s]

    Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:04,  8.27it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:03, 10.20it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:03, 10.20it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:03, 10.20it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:03, 10.20it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 12.09it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 12.09it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 12.09it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 12.09it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 14.01it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 14.01it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 15.36it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 16.80it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 16.80it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 16.80it/s]

    Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 16.80it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 18.08it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 18.08it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 18.08it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 18.08it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:06<00:01, 18.43it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:06<00:01, 18.43it/s]

    Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:06<00:01, 18.43it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:06<00:01, 18.43it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:06<00:01, 18.43it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:06<00:00, 22.16it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:06<00:00, 22.16it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:06<00:00, 22.16it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:06<00:00, 22.16it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:06<00:00, 22.16it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:06<00:00, 26.03it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:06<00:00, 26.03it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:06<00:00, 26.03it/s]

    Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:06<00:00, 26.03it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:06<00:00, 26.03it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:06<00:00, 26.03it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:06<00:00, 26.03it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:06<00:00, 26.03it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:06<00:00, 26.03it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:06<00:00, 26.03it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:06<00:00, 26.03it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:06<00:00, 43.22it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:06<00:00, 43.22it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:06<00:00, 43.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.39it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 20.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 20.30it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 20.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 20.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.94it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  21%|██        | 12/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  21%|██        | 12/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  21%|██        | 12/58 [00:00<00:01, 28.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.82it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.82it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.82it/s] Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.77it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.26it/s]

    Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.79it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.79it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.79it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.79it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.79it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.79it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=208 avail_mem=72.45 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.90it/s]

    Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.29it/s] Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  81%|████████  | 47/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  81%|████████  | 47/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  81%|████████  | 47/58 [00:01<00:00, 42.12it/s]

    Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  81%|████████  | 47/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  81%|████████  | 47/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  81%|████████  | 47/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.86it/s] Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.28it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.28it/s]

    Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 38.40it/s]


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
    Generated text:  Lillian. I am a three-year-old who has been adopted from the UK. I was adopted by a family that believes in the importance of a nurturing, loving, and supportive environment. They hope that their adopted child will be able to grow up to be a loving, loving, loving human being. They believe that childhood is a time to develop the character traits and values that will form the basis of their adult lives. I am a compassionate child, friendly, loving, and kind to all who approach me. I have three siblings and am a very good friend to my two younger brothers. We enjoy playing all sorts of games together,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking executive office, with the first lady being the vice president. The president is elected by ________.
    
    To determine the answer to the question "The president of the United States is a high-ranking executive office, with the first lady being the vice president. The president is elected by ________," we can follow these steps:
    
    1. Identify the relevant information: The president of the United States is a high-ranking executive office.
    2. Identify the first lady: The first lady is the vice president.
    3. Determine the elected office: The president is elected by the United States Congress.
    
    Since the president is elected by the United States
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. Nice
    C. Bordeaux
    D. Lille
    Answer:
    A
    
    The internal environment of a cell refers to
    A. The sum of the extracellular fluid
    B. The external environment
    C. The internal environment of the cell
    D. The external environment of the cell
    Answer:
    C
    
    The most common cause of death in patients with acute myocardial infarction is
    A. Arrhythmia
    B. Cardiogenic shock
    C. Cardiac rupture
    D. Cardiac tamponade
    E. Cardiac rupture
    Answer:
    A
    
    When the daily urine output
    ===============================
    Prompt: The future of AI is
    Generated text:  here. It’s here in the form of virtual assistants, AI assistants, and even an AI that can walk.
    There is a small but important caveat: we aren’t quite there yet. The world is changing rapidly and we are still a long way off. But with the right innovation, we can ensure that we have a future where AI is used for the betterment of all.
    So, what is AI?
    It is a subset of technology that uses computers and algorithms to mimic the way that humans learn, think, and make decisions.
    Imagine an AI system that can understand language and emotion, and learn to respond to people in a way


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


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. Paris is also home to numerous museums, theaters, and restaurants, making it a popular tourist destination. The city is known for its fashion industry, with iconic fashion houses like Chanel and Louis Vuitton, and its cuisine, including the famous Parisian dishes like croissants and crêpes. Paris is a vibrant and dynamic city that continues to thrive as a major global city.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into various industries, from manufacturing to healthcare to transportation. This will lead to the automation of tasks that are currently performed by humans, such as data entry, routine maintenance, and routine decision-making.
    
    2. Improved privacy and security: As AI becomes more integrated into our lives, there will be a growing concern about the privacy and security of personal data. This will lead to the development of new technologies and regulations to
    


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
    Generated text:  [insert your full name], and I'm a [insert your occupation] at [insert your workplace or location]. I'm a [insert your profession] because I enjoy [insert an interesting fact about your profession]. I've always been fascinated by the world of [insert your favorite hobby or interest], and I'm always learning new things. I'm always looking for ways to [insert your hobbies or interests], and I'm always trying to make a positive impact in the world. I'm always excited to talk to new people and share my knowledge and interests with them. So, please tell me what you think about me. Sure, here
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and serves as the capital of the country. It is also the most populous city and the second-largest metropolitan area in the European Union, with over 11 million inhabitants. Paris is known for its art, history, and music, and is a major hub for French culture, politics, and economy. It is located on the left bank of the Seine River, facing the medieval city of Paris and the Champs-Élysées. Paris is home to iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and Arc de Triomphe,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of different trends, including:
    
    1. Increased use of AI in healthcare: With the increasing amount of health data being generated by people all over the world, AI has the potential to be used to improve healthcare outcomes. This could include developing more effective treatments for diseases, improving the accuracy of medical diagnosis, and analyzing medical imaging data to help doctors make more informed decisions.
    
    2. Improved transparency and explainability of AI systems: As AI systems become more complex, there is a growing emphasis on making them more transparent and explainable. This means that AI models should be designed in a way that makes it easy for humans to


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

    age

    ]

     year

     old

     [

    gender

    ]

     [

    occupation

    ].


    I

     come

     from

     [

    city

    ]

     and

     I

    've

     always

     been

     fascinated

     by

     [

    h

    obby

    /

    interest

    ],

     which

     has

     been

     my

     constant

     desire

     for

     a

     long

     time

    .

     My

     goal

     is

     to

     pursue

     my

     passion

     and

     become

     a

     [

    high

    ly

     skilled

    ]

     professional

     in

     this

     field

    .

     I

    'm

     committed

     to

     using

     my

     skills

     to

     make

     a

     positive

     impact

     in

     the

     world

    .


    I

    'm

     always

     looking

     for

     new

     opportunities

     to

     learn

    ,

     grow

    ,

     and

     achieve

     my

     goals

    .

     I

    'm

     ready

     to

     explore

     new

     challenges

     and

     try

     new

     things

    ,

     and

     I

    'm

     open

     to

     learning

     from

     others

    .

     I

    'm

     also

     a

     [

    person

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Does

     this

     statement

     correctly

     understand

     the

     relationship

     between

     the

     city

     and

     France

    ?


    Yes

    ,

     the

     statement

     accurately

     describes

     the

     relationship

     between

     the

     city

     and

     France

    .

     Paris

     is

     the

     capital

     city

     of

     France

    .

     Here

     are

     some

     key

     points

     about

     Paris

    :
    


    1

    .

     The

     city

     is

     located

     in

     the

     Prov

    ence

     region

     of

     the

     country

    ,

     which

     is

     part

     of

     the

     region

     of

     the

     Western

     Mediterranean

    .


    2

    .

     Paris

     is

     the

     largest

     city

     in

     France

     and

     has

     a

     population

     of

     over

     

    1

    1

     million

     people

    .


    3

    .

     It

     is

     the

     seat

     of

     the

     French

     government

    ,

     the

     French

     parliament

    ,

     and

     the

     French

     state

    ,

     as

     well

     as

     the

     headquarters

     of

     several

     major

     French

     companies

     and

     institutions

    .


    4

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

     and

     will

     likely

     involve

     a

     multitude

     of

     changes

    .

     Here

     are

     some

     of

     the

     key

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

     Use

     of

     AI

     for

     Healthcare

    :

     AI

     will

     play

     an

     increasing

     role

     in

     the

     healthcare

     industry

    ,

     including

     diagnosis

    ,

     treatment

    ,

     and

     drug

     discovery

    .

     Machine

     learning

     algorithms

     will

     be

     used

     to

     analyze

     vast

     amounts

     of

     medical

     data

     to

     identify

     patterns

     and

     improve

     diagnostic

     accuracy

    .
    


    2

    .

     Integration

     with

     IoT

    :

     AI

     will

     be

     more

     integrated

     with

     Internet

     of

     Things

     (

    Io

    T

    )

     devices

    ,

     enabling

     them

     to

     collect

     and

     process

     data

     in

     real

    -time

    .

     This

     will

     allow

     for

     more

     efficient

     management

     of

     assets

    ,

     personalized

     recommendations

    ,

     and

     predictive

     maintenance

    .
    


    3

    



```python
llm.shutdown()
```
