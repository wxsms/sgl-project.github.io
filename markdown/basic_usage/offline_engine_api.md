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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.25it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:07,  1.23s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:26,  2.00it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:26,  2.00it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:26,  2.00it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:26,  2.00it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:26,  2.00it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:11,  4.07it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:11,  4.07it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:11,  4.07it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:11,  4.07it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:11,  4.07it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:05<00:11,  4.07it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:05,  7.41it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:05,  7.41it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:05,  7.41it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:05,  7.41it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:05,  7.41it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:05<00:05,  7.41it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:05<00:05,  7.41it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:05<00:05,  7.41it/s] Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:05<00:05,  7.41it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 14.08it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 14.08it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 14.08it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 14.08it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 14.08it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 14.08it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:02, 14.08it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:02, 14.08it/s]

    Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:02, 14.08it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 21.68it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 21.68it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 21.68it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 21.68it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 21.68it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 21.68it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 21.68it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 21.68it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 21.68it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 21.68it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 31.26it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 40.72it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 40.72it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 40.72it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 40.72it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 40.72it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 40.72it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 40.72it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 40.72it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 40.72it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 40.72it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.46it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.40 GB):   2%|▏         | 1/58 [00:00<00:05,  9.83it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.13 GB):   2%|▏         | 1/58 [00:00<00:05,  9.83it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=74.14 GB):   2%|▏         | 1/58 [00:00<00:05,  9.83it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.14 GB):   5%|▌         | 3/58 [00:00<00:05, 10.89it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.15 GB):   5%|▌         | 3/58 [00:00<00:05, 10.89it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   5%|▌         | 3/58 [00:00<00:05, 10.89it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:04, 12.74it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:04, 12.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:04, 12.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.17 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.18 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.20 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.77it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.20 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.77it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.20 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.20 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.20 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.20 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.20 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.20 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.22 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.22 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.88it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.22 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.22 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.22 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.27 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.22 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.21 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.21 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.61it/s]Capturing num tokens (num_tokens=960 avail_mem=74.23 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.61it/s] Capturing num tokens (num_tokens=896 avail_mem=74.23 GB):  36%|███▌      | 21/58 [00:01<00:01, 27.61it/s]Capturing num tokens (num_tokens=832 avail_mem=74.23 GB):  36%|███▌      | 21/58 [00:01<00:01, 27.61it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.22 GB):  36%|███▌      | 21/58 [00:01<00:01, 27.61it/s]Capturing num tokens (num_tokens=704 avail_mem=74.22 GB):  36%|███▌      | 21/58 [00:01<00:01, 27.61it/s]Capturing num tokens (num_tokens=704 avail_mem=74.22 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.67it/s]Capturing num tokens (num_tokens=640 avail_mem=74.20 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.67it/s]Capturing num tokens (num_tokens=576 avail_mem=74.20 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.67it/s]Capturing num tokens (num_tokens=512 avail_mem=74.19 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.67it/s]Capturing num tokens (num_tokens=480 avail_mem=74.20 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.67it/s]Capturing num tokens (num_tokens=448 avail_mem=74.20 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.67it/s]Capturing num tokens (num_tokens=448 avail_mem=74.20 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=416 avail_mem=74.19 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=384 avail_mem=74.19 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.98it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.18 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=320 avail_mem=74.17 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=288 avail_mem=74.16 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=288 avail_mem=74.16 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=256 avail_mem=74.16 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=240 avail_mem=74.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=224 avail_mem=74.14 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=208 avail_mem=74.14 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=192 avail_mem=74.14 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=192 avail_mem=74.14 GB):  71%|███████   | 41/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=176 avail_mem=74.13 GB):  71%|███████   | 41/58 [00:01<00:00, 38.99it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.12 GB):  71%|███████   | 41/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=144 avail_mem=74.11 GB):  71%|███████   | 41/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=128 avail_mem=74.11 GB):  71%|███████   | 41/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=112 avail_mem=74.11 GB):  71%|███████   | 41/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=112 avail_mem=74.11 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.76it/s]Capturing num tokens (num_tokens=96 avail_mem=74.10 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.76it/s] Capturing num tokens (num_tokens=80 avail_mem=74.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.76it/s]Capturing num tokens (num_tokens=64 avail_mem=74.08 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.76it/s]Capturing num tokens (num_tokens=48 avail_mem=74.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.76it/s]Capturing num tokens (num_tokens=32 avail_mem=74.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.76it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.09 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=28 avail_mem=74.08 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=24 avail_mem=74.08 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=20 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=16 avail_mem=74.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=12 avail_mem=74.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=12 avail_mem=74.06 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.64it/s] Capturing num tokens (num_tokens=4 avail_mem=74.05 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.64it/s]Capturing num tokens (num_tokens=4 avail_mem=74.05 GB): 100%|██████████| 58/58 [00:01<00:00, 30.91it/s]


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
    Generated text:  Deniz. I'm a 13-year-old girl who was born in Italy. I used to take medication to help me grow taller. I was born with a very low hair growth rate. My parents took my case to the doctor. After a few tests, they said that it was normal. Now, I'm 18 years old and still a teenager. What is your favorite thing to do on the weekend? I can’t talk about drugs or alcohol. However, I enjoy spending time with friends, reading books, and cooking. Can you suggest a way to help me get the hair growth back? I think I might
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the Vice President. In how many ways can the vice president be chosen?
    To determine the number of ways the vice president can be chosen from the 50 different members of the executive branch, we need to consider that each choice of a vice president is independent of the others. This means that if the first person chosen is any one of the 50 members, the second person chosen will also be any one of the 50 members, and so on. Therefore, we can calculate the total number of ways to choose a vice president by multiplying the number of choices for the first vice president by the number of choices for
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Rome
    C. Washington
    D. Jerusalem
    
    A. Paris is the capital of France. It is located in the south of France, on the Mediterranean coast, and is one of the most famous cities in the world. The city is known for its rich history, culture, art, and cuisine. Some of the most famous landmarks in Paris include the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and Montmartre. The French Parliament is also located in Paris, as well as the French Parliament Building, the Palace of Versailles, and other important historical sites. The
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain. But in this uncertain future, the future of AI is not entirely one-dimensional. It is a multi-dimensional future, with a lot of possibilities for the future, where the world is filled with hopes and the world is filled with fears. In the future of AI, there are no absolutes, there are no guarantees. In the future of AI, there is a lot of uncertainty. In the future of AI, there is a lot of uncertainty, and there is a lot of uncertainty. The future of AI is a multifaceted future, filled with possibilities and uncertainties.
    In the future of AI, there are no absolutes


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] with [Number] years of experience in [Field]. I'm [Gender] and [Race]. I'm [Name] and I'm [Occupation]. I'm [Name] and I'm [Age]. I'm [Name] and I'm [Age]. I'm [Name] and I'm [Age]. I'm [Name] and I'm [Age]. I'm [Name] and I'm [Age]. I'm [Name] and I'm [Age]. I'm [Name] and I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Middle Ages and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for art, culture, and fashion, and is home to many world-renowned museums, theaters, and restaurants. The city is also known for its vibrant nightlife and is a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. Its status as the world's most populous city is also reflected
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives through devices such as smartphones, smart homes, and self-driving cars. As AI technology continues to improve, we can expect to see even more integration into our daily lives, from virtual assistants to self-driving cars.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more use
    


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
    Generated text:  [insert your name] and I'm here to help you today. Please tell me a little bit about yourself. Sure, I'm [insert your name] and I love to help people. I'm always ready to assist with anything you need, no matter the task. What would you like to talk about? Feel free to ask me anything, and I'll do my best to provide you with helpful advice and guidance. And don't forget to ask me any questions you might have about the world of work and life. Whatever you need, I'm here to help. How can I assist you today? What would you like to do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. The city is known for its rich history, stunning architecture, and vibrant culture. Its Old Town district is a UNESCO World Heritage site, while the city is home to numerous museums, art galleries, and festivals. Paris is also known for its food, fashion, and nightlife. It's a popular tourist destination for its annual Le Week End and annual Eiffel Tower climb. Paris is a unique city with a rich history and is a great place to visit for anyone interested in French culture and history. (Note: the capital city of France is actually Lyon, not Paris.) 
    
    The statement is: **Paris is the capital city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating, and it is likely to continue evolving and changing in numerous ways. Here are some possible trends in AI that could emerge in the next few years:
    
    1. Increased integration with human intelligence: One of the most exciting possibilities for AI is its integration with human intelligence. As AI becomes more powerful, it could potentially learn from the experiences of humans and improve its performance accordingly. This could lead to more intelligent and adaptive machines that can make better decisions and respond to more complex situations.
    
    2. Greater emphasis on ethics and transparency: As AI systems become more complex and integrated into our daily lives, there is a growing need for ethical considerations and transparency


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

    First

     Name

    ]

     and

     I

     am

     a

     [

    First

     Name

    ]

     [

    Last

     Name

    ],

     a

     [

    Type

     of

     Work

    ].

     I

     have

     a

     passion

     for

     [

    what

     you

     do

     best

    ].

     I

     aim

     to

     create

     [

    a

     product

     or

     service

    ],

     [

    what

     you

     do

     best

    ]

     [

    a

     way

     to

     help

     people

    ],

     [

    what

     you

     do

     best

    ]

     [

    a

     solution

    ].

     I

     am

     always

     looking

     to

     learn

     new

     things

     and

     improve

     [

    what

     you

     do

     best

    ]

     in

     order

     to

     grow

     as

     a

     professional

    .

     I

     strive

     to

     be

     [

    what

     you

     do

     best

    ]

     [

    someone

     with

     confidence

    ,

     a

     positive

     attitude

    ,

     etc

    .]

     and

     to

     make

     a

     difference

     [

    impact

     on

     the

     world

    ].

     I

     believe

     that

     teamwork

     and

     communication

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     major

     French

     city

     located

     in

     the

     south

     of

     the

     country

    ,

     and

     is

     the

     seat

     of

     government

    ,

     capital

     of

     the

     country

    ,

     and

     largest

     city

     by

     population

    .

     The

     city

     is

     home

     to

     many

     world

    -ren

    owned

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     known

     for

     its

     cultural

     scene

    ,

     cuisine

    ,

     and

     fashion

    ,

     and

     is

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     The

     city

     is

     also

     home

     to

     the

     French

     Riv

    iera

     and

     the

     Euro

    vision

     Song

     Contest

    .

     It

     is

     also

     the

     official

     residence

     of

     the

     President

     of

     the

     French

     Republic

    .

     The

     city

     is

     a

     major

     cultural

     and

     economic

     hub

     in

     Europe

     and

     plays

     a

     significant

     role

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

     and

     potential

     challenges

    .

     Some

     of

     the

     key

     trends

     we

     can

     expect

     to

     see

     in

     the

     years

     ahead

     include

    :
    


    1

    .

     Improved

     hardware

    :

     Moore

    's

     Law

     is

     expected

     to

     continue

     to

     power

     the

     rise

     of

     AI

    ,

     as

     chip

    -making

     technology

     will

     continue

     to

     improve

     and

     become

     more

     efficient

    .

     This

     means

     that

     AI

     systems

     will

     become

     faster

    ,

     more

     powerful

    ,

     and

     more

     affordable

    ,

     making

     them

     more

     accessible

     to

     a

     wider

     range

     of

     users

    .
    


    2

    .

     Artificial

     intelligence

     will

     become

     more

     diverse

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     will

     be

     able

     to

     learn

     from

     and

     adapt

     to

     new

     information

     and

     data

    .

     This

     will

     make

     it

     easier

     for

     them

     to

     handle

     increasingly

     complex

     and

     varied

     tasks

    



```python
llm.shutdown()
```
