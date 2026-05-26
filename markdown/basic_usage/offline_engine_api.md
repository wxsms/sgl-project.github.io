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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.90it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:40,  4.93s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  7.91it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 12.83it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 18.82it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 18.82it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 18.82it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 18.82it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 18.82it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 18.82it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 18.82it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 18.82it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 18.82it/s]

    Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 18.82it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 26.71it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 26.71it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 26.71it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 26.71it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 26.71it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 26.71it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 26.71it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 26.71it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 26.71it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 26.71it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 35.37it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 35.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:03, 16.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:03, 16.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:03, 16.38it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.02 GB):   7%|▋         | 4/58 [00:00<00:03, 17.59it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   7%|▋         | 4/58 [00:00<00:03, 17.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.02 GB):   7%|▋         | 4/58 [00:00<00:03, 17.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.02 GB):  10%|█         | 6/58 [00:00<00:02, 17.93it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.01 GB):  10%|█         | 6/58 [00:00<00:02, 17.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.00 GB):  10%|█         | 6/58 [00:00<00:02, 17.93it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.00 GB):  10%|█         | 6/58 [00:00<00:02, 17.93it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.00 GB):  10%|█         | 6/58 [00:00<00:02, 17.93it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.00 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.99 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.99 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.99 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.99 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.46it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.71it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=73.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.68 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.68 GB):  31%|███       | 18/58 [00:00<00:01, 28.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.99 GB):  31%|███       | 18/58 [00:00<00:01, 28.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.99 GB):  31%|███       | 18/58 [00:00<00:01, 28.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.97 GB):  31%|███       | 18/58 [00:00<00:01, 28.54it/s]Capturing num tokens (num_tokens=960 avail_mem=72.99 GB):  31%|███       | 18/58 [00:00<00:01, 28.54it/s] Capturing num tokens (num_tokens=896 avail_mem=72.98 GB):  31%|███       | 18/58 [00:00<00:01, 28.54it/s]Capturing num tokens (num_tokens=896 avail_mem=72.98 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=832 avail_mem=72.98 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.93it/s]

    Capturing num tokens (num_tokens=768 avail_mem=72.98 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=704 avail_mem=72.98 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=640 avail_mem=72.97 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=576 avail_mem=72.97 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=576 avail_mem=72.97 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.77it/s]Capturing num tokens (num_tokens=512 avail_mem=72.96 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.77it/s]Capturing num tokens (num_tokens=480 avail_mem=72.97 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.77it/s]Capturing num tokens (num_tokens=448 avail_mem=72.97 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.77it/s]Capturing num tokens (num_tokens=416 avail_mem=72.97 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.77it/s]Capturing num tokens (num_tokens=384 avail_mem=72.97 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.77it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.97 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=352 avail_mem=72.96 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=320 avail_mem=72.95 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=288 avail_mem=72.95 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=256 avail_mem=72.95 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=240 avail_mem=72.95 GB):  57%|█████▋    | 33/58 [00:01<00:00, 38.30it/s]Capturing num tokens (num_tokens=240 avail_mem=72.95 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=224 avail_mem=72.94 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=208 avail_mem=72.94 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=192 avail_mem=72.94 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=176 avail_mem=72.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=160 avail_mem=72.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.34it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.93 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=144 avail_mem=72.93 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=128 avail_mem=72.93 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=112 avail_mem=72.92 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.93it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=64 avail_mem=72.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.30it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.30it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.33it/s] Capturing num tokens (num_tokens=4 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=4 avail_mem=72.46 GB): 100%|██████████| 58/58 [00:01<00:00, 43.03it/s]Capturing num tokens (num_tokens=4 avail_mem=72.46 GB): 100%|██████████| 58/58 [00:01<00:00, 35.30it/s]


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
    Generated text:  Tim, and I'm a graduate student in developmental psychology at the University of Washington. I'm a lifelong learner and I always try to find new ways to improve my skills and knowledge. I have a strong interest in developmental psychology and in particular, my field of research, which is the origins of human emotion and social behavior. I am deeply interested in how one's early experiences shape and mold the way they behave and think.
    I am studying the origin of social and emotional development, but I am also interested in the origins of emotions and our understanding of how they develop and change over time. I am interested in how these emotions are shaped by
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. It is known that he is also a person. Could it be that the president of the United States is a person who is also a person? Yes, it is possible. The president of the United States is a representative of the country, and the president of a country is a person who is also a person. However, it is also possible that the president of the United States is a person who is not a person. In this case, the president would be a person who is an individual. 
    
    Therefore, it is not possible for the president of the United States to be both a person and a person who is also
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the seat of the French government. It is located in the center of the French department of the same name, which is about in the south of the region of the same name in the center of France. It is one of the five largest cities of the world and the second largest in metropolitan area of France, after Paris. It is in the vicinity of the English Channel and is located at an elevation of about in the region of the same name. It is a city situated in the heart of the French countryside. It is also a city surrounded by a high wall built by the ancients. The local dialect of the
    ===============================
    Prompt: The future of AI is
    Generated text:  a bit like a thermometer, on one end of the scale is confidence and in the other, confidence is not a function of the data; it’s a function of the algorithm. In other words, if the data is good, the algorithm will produce good results, and the algorithm is a function of the data. If the data is bad, the algorithm will produce bad results, and the algorithm is a function of the data. In the real world, that is, in the real world where we have data, we need to have confidence.
    One of the things I’m working on at work at the moment is to teach machine learning algorithms


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name],
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" or simply "Paris". It is the largest city in France and the third-largest city in the world by population. The city is located on the Seine River and is home to many of France's most famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its rich cultural heritage, including its art, music, and cuisine. It is a major center for business, finance, and tourism, and is a popular destination for tourists and locals alike. Paris is a vibrant and dynamic city with a rich history and a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve the quality of care. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance
    


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
    Generated text:  [Your Name], and I am a [insert your profession, such as "entrepreneur", "doctor", "teacher", etc.] with [insert a relevant field, such as "software engineer", "lawyer", etc.].
    
    I am currently [insert your current occupation, such as "engineer", "teacher", "doctor", "entrepreneur", "lawyer", etc.]. Throughout my career, I have been [insert a relevant accomplishment, such as "co-founded a successful startup", "received a patent", "achieved a 30% increase in revenue", etc.].
    
    I am a passionate [insert a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France. It is located on the Seine river and is the largest city in the country. The city has a rich history dating back over 2,000 years, and it is known for its beautiful architecture, delicious food, and lively culture. Paris is a major cultural hub and home to many famous landmarks, including the Louvre and Notre-Dame Cathedral. The city is also famous for its annual Eiffel Tower celebration, which attracts millions of tourists each year. Paris is a vibrant and dynamic city, with a diverse population and a rich history. It is one of the most
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly volatile and uncertain. Some possible trends include:
    
    1. AI will continue to advance rapidly, with more sophisticated algorithms and models that can better understand and interpret complex data.
    
    2. AI will become increasingly integrated into our daily lives, from smart home devices to self-driving cars and personal assistants like Siri and Alexa.
    
    3. AI will also continue to become more ethical and responsible, with greater emphasis on fairness, transparency, and accountability in how AI is used and developed.
    
    4. AI will likely continue to evolve and change more rapidly than we can predict, as new technologies and patterns emerge that challenge our existing understanding of AI.
    
    5


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

    occupation

    ]

     from

     [

    location

    ].

     I

    'm

     always

     [

    character

    istic

    ],

     and

     I

     love

     [

    reason

     for

     love

    /b

    el

    ief

    ].

     I

     believe

     that

     [

    reason

     for

     belief

    ]

     has

     led

     me

     to

     this

     character

    .

     What

     do

     you

     think

     of

     me

    ?

     Let

    's

     chat

    !

     [

    Name

    ]

    ...

     [

    Tell

     your

     story

     in

     a

     few

     sentences

     that

     give

     us

     a

     good

     idea

     of

     who

     you

     are

    ].

     [

    Name

    ]

    ...

     [

    Tell

     your

     story

     in

     a

     few

     sentences

     that

     give

     us

     a

     good

     idea

     of

     who

     you

     are

    ].

     [

    Name

    ]

    ...

     [

    Tell

     your

     story

     in

     a

     few

     sentences

     that

     give

     us

     a

     good

     idea

     of

     who

     you

     are

    ].

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     and

     largest

     city

     of

     France

    ,

     and

     is

     located

     on

     the

     left

     bank

     of

     the

     Se

    ine

     river

     in

     the

     Mos

    elle

     department

     in

     north

    western

     France

    .

     The

     city

     was

     founded

     in

     

    7

    8

    7

     as

     the

     “

    New

     Rome

    ”

     and

     is

     one

     of

     the

     most

     important

     French

     cities

     in

     terms

     of

     economy

    ,

     culture

    ,

     and

     politics

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     charming

     architecture

    ,

     beautiful

     parks

    ,

     and

     world

    -class

     museums

    .

     It

     is

     also

     home

     to

     the

     Lou

    vre

     Museum

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     many

     other

     landmarks

    .

     Paris

     is

     a

     symbol

     of

     French

     culture

     and

     a

     major

     tourist

     destination

    .

     The

     city

     has

     a

     population

     of

     over

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     several

     key

     trends

    ,

     including

    :
    


    1

    .

     Increased

     integration

    :

     AI

     is

     expected

     to

     become

     more

     integrated

     into

     various

     industries

     and

     applications

    ,

     from

     healthcare

     and

     finance

     to

     transportation

     and

     manufacturing

    .

     As

     more

     systems

     learn

     and

     adapt

     to

     new

     data

    ,

     they

     will

     become

     more

     efficient

     and

     effective

    .
    


    2

    .

     Real

    -time

     learning

    :

     AI

     will

     become

     more

     adept

     at

     learning

     and

     adapting

     to

     new

     data

     in

     real

    -time

    ,

     rather

     than

     waiting

     for

     data

     to

     be

     collected

     and

     analyzed

    .

     This

     will

     allow

     for

     faster

    ,

     more

     efficient

     decision

    -making

     and

     improvement

     of

     systems

    .
    


    3

    .

     Personal

    ization

    :

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     personalize

     user

     experiences

    ,

     from

     recommendations

     for

     products

     and

     services

     to

     personalized

     marketing

    



```python
llm.shutdown()
```
