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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.31it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:04,  5.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:04,  5.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:04,  5.35s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:04,  5.35s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:04,  5.35s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:43,  1.22it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:14,  3.31it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.01it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.01it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.01it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.01it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.01it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.01it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.01it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.01it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 11.03it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 11.03it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 11.03it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 11.03it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 11.03it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 11.03it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 11.03it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 11.03it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:05<00:02, 11.03it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:01, 16.77it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:01, 16.77it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:01, 16.77it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:01, 16.77it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:01, 16.77it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:01, 16.77it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:01, 16.77it/s]

    Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:05<00:01, 16.77it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.42it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.42it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.42it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:06<00:00, 22.42it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:06<00:00, 22.42it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:06<00:00, 22.42it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:06<00:00, 22.42it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:06<00:00, 22.42it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:06<00:00, 22.42it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:06<00:00, 29.74it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:06<00:00, 29.74it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:06<00:00, 29.74it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:06<00:00, 29.74it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:06<00:00, 29.74it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:06<00:00, 29.74it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:06<00:00, 29.74it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:06<00:00, 29.74it/s]

    Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:06<00:00, 29.74it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:06<00:00, 29.74it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 14.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 14.10it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 14.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   7%|▋         | 4/58 [00:00<00:03, 15.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   7%|▋         | 4/58 [00:00<00:03, 15.75it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   7%|▋         | 4/58 [00:00<00:03, 15.75it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):  10%|█         | 6/58 [00:00<00:03, 14.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):  10%|█         | 6/58 [00:00<00:03, 14.89it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  10%|█         | 6/58 [00:00<00:03, 14.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.36it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.36it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.36it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.85it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.85it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.85it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.32it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.38it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.38it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.38it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.38it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.38it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.85it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.85it/s]

    Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.85it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.85it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.85it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.95it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.95it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.95it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.95it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.95it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.95it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.04it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.04it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.07it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.07it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.07it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.07it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.07it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  71%|███████   | 41/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 34.64it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.70it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.70it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.70it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.70it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.70it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.87it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.87it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.87it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.87it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.87it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.87it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.76it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.76it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 29.18it/s]


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
    Generated text:  Liam. I am the founder of this website. It is named "The Best Book Lover in the World". I have over 6 years of experience in the field of publishing, and I have a PhD in literature. As such, I specialize in creating and promoting books that will appeal to my target audience. I believe that a good book is a book that is well written, well illustrated, and that touches the reader on a personal level. I believe that it is important for readers to feel like they have become part of the story, and that by doing so, they will feel a sense of connection to the author, and their own
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to pick a new logo for the country. He has chosen to make the logo a circle with the diameter of the Earth. The diameter of the Earth is $8,000$ miles. The scale of the map he is using is $1$ inch to $3$ miles. How many square inches will the circle occupy in his map?
    To determine how many square inches the circle will occupy in the map, we need to follow these steps:
    
    1. **Find the radius of the circle in inches.**
    2. **Use the scale to find the area of the circle in square inches.**
    
    First, let's
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]
    A. Lille
    B. Paris
    C. Brest
    D. Lyon
    
    To determine the capital of France, let's analyze the options step by step:
    
    1. **Lille**: This is a city in southwestern France, known for its historical importance and cultural attractions. While it is a city, it is not the capital of France.
    
    2. **Paris**: Paris is the capital of France. It is the largest city in the European Union and the seat of the French government. Paris is famous for its architecture, culture, and world-class museums, such as the Louvre.
    
    3. **Brest**:
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, with companies investing billions of dollars each year in developing and deploying new algorithms and technologies. However, as with any emerging field, there are some important questions to consider when developing AI systems. One of the most important questions is the ethical implications of AI. Here are a few ways AI is being used to make ethical problems even more complex:
    1. Facial recognition technology: Facial recognition technology is increasingly being used for identity theft, cyberbullying, and other malicious activities. It is also being used for security purposes, such as biometric authentication in passports and other documents.
    2. Autonomous vehicles: Autonomous vehicles are being developed with the goal


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a few details about your personality or background]. And what do you do for a living? I'm a [insert a few details about your job or profession]. And what do you enjoy doing? I enjoy [insert a few details about your hobbies or interests]. And what do you like to do in your free time? I like to [insert a few details about your hobbies or interests]. And what do you like to do when you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, diverse culture, and vibrant nightlife. It is a major transportation hub, with many major highways and rail lines connecting the city to other parts of France and the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are already being used in a wide range of applications, from self-driving cars to personalized medicine. As these technologies continue to evolve, we can expect to see even more innovative applications and improvements in AI. Additionally, there is a growing emphasis on ethical considerations and the development of responsible AI that can be used to benefit society as a whole. This includes efforts to ensure that AI is developed and used in a way that is fair, transparent, and accountable. Overall, the future of AI is likely to be one of continued innovation
    


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
    Generated text:  [Your Name] and I'm a [Your Occupation] who specializes in [Your Expertise]. I have a passion for [Your Hobby, Passion, or Interest] and I am always looking for ways to [Your Life Objective or Goal]. I believe in the importance of [Your Character Trait or Quality], and I strive to make the world a better place by [Your Characteristic]. I'm confident in my abilities and I'm excited to bring my knowledge and skills to [Your Field or Profession]. How can I help you today? Let's set up a [Your Intention for the Meeting or Activity]. Best regards, [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is located in the center of the country and is the political, economic, cultural, and cultural center of France. It is also the world's most populous city and the largest metropolitan area in the European Union. The city is famous for its historical landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is known for its art, music, fashion, and gastronomy, and it is a major center of the arts and sciences. The city is also home to the Parliament of France and the European Parliament. Paris is also famous for its fashion industry and for being the birth
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a number of different trends, including:
    
    1. Increased sophistication of AI: The development of more complex and sophisticated AI systems will likely continue, with further advancements in areas such as natural language processing, computer vision, and robotics.
    
    2. Integration with human experience: AI systems will increasingly be integrated into human interfaces, making them more intuitive and seamless for humans to use.
    
    3. Personalization and augmentation: AI will be used to improve the accuracy and efficiency of personalized and augmented customer experiences, such as those offered by chatbots and virtual assistants.
    
    4. Ethical considerations: There will be increased emphasis on ethical considerations in


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

    insert

     name

    ].

     I

     am

     a

     [

    insert

     profession

     or

     title

    ]

     with

     a

     passion

     for

     [

    insert

     what

     they

     love

     to

     do

     or

     do

     for

     a

     living

    ].

     I

    'm

     always

     up

     for

     learning

     new

     things

     and

     trying

     new

     things

    ,

     whether

     it

    's

     cooking

    ,

     playing

     sports

    ,

     or

     exploring

     the

     world

     through

     my

     computer

    .

     I

    'm

     also

     a

     huge

     fan

     of

     [

    insert

     a

     genre

     of

     book

     or

     film

    ].

     
    


    In

     my

     spare

     time

    ,

     I

     enjoy

     [

    insert

     a

     hobby

     or

     activity

     of

     mine

    ].

     I

     also

     have

     a

     friend

     who

     is

     a

     [

    insert

     a

     profession

     or

     title

    ].

     I

     have

     a

     lot

     of

     energy

     and

     can

     often

     be

     found

     in

     different

     parts

     of

     the

     world

     to

     meet

     new

     people

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     the

     European

     Union

    ,

     the

     second

    -largest

     city

     in

     the

     Americas

    ,

     and

     the

     second

    -largest

     city

     in

     the

     world

     by

     population

    .

     It

     is

     also

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     romantic

     architecture

     and

     cultural

     diversity

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

    ,

     famous

     for

     its

     art

    ,

     music

    ,

     cuisine

    ,

     and

     fashion

    .

     It

     is

     home

     to

     the

     E

    iff

    el

     Tower

    ,

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

     historic

     landmarks

    .

     Paris

     is

     also

     the

     seat

     of

     France

    's

     government

     and

     the

     seat

     of

     the

     French

     parliament

    .

     The

     city

     is

     known

     for

     its

     fashion

    ,

     art

    ,

     food

    ,

     and

     nightlife

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     a

     significant

     game

    -ch

    anger

    ,

     with

     the

     potential

     to

     revolution

    ize

     many

     aspects

     of

     our

     daily

     lives

    .

     Some

     of

     the

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     Integration

     of

     AI

     into

     all

     aspects

     of

     society

    ,

     including

     healthcare

    ,

     transportation

    ,

     and

     education

    .
    


    2

    .

     Artificial

     intelligence

     systems

     becoming

     more

     advanced

     and

     capable

     of

     performing

     tasks

     that

     were

     previously

     thought

     impossible

    .
    


    3

    .

     The

     development

     of

     more

     sophisticated

     and

     autonomous

     AI

    ,

     able

     to

     learn

     and

     adapt

     to

     new

     information

     and

     situations

    .
    


    4

    .

     The

     integration

     of

     AI

     into

     human

    -machine

     interfaces

    ,

     allowing

     humans

     to

     interact

     with

     AI

     systems

     more

     effectively

     and

     efficiently

    .
    


    5

    .

     The

     potential

     for

     AI

     to

     play

     a

     more

     significant

     role

    



```python
llm.shutdown()
```
