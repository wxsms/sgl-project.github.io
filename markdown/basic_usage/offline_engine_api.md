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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.12it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.70it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.70it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.70it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.70it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:12,  3.70it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:12,  3.70it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:12,  3.70it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:12,  3.70it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.70it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:12,  3.70it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.33it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.33it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.33it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 14.04it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 14.04it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 14.04it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 14.04it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 14.04it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.04it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.04it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.04it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.04it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.04it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.86it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.86it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.86it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.86it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.86it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.86it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.86it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.86it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.86it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.86it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.65it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 35.86it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 35.86it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 35.86it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 35.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.02it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.02it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.02it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.03it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.03it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.52it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.52it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 29.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 29.25it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 29.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 29.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 29.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 29.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.40it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.40it/s] Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.52it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.52it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.52it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.52it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.52it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.52it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.66it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.66it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.66it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.66it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.66it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.66it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.89it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.03it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.03it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.79it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.79it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.79it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.79it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.79it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.79it/s] Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  81%|████████  | 47/58 [00:01<00:00, 39.18it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 39.18it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 39.18it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 39.18it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 39.18it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 39.18it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.35it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.35it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.35it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.35it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.35it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.35it/s] Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.46it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.46it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 37.36it/s]


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
    Generated text:  Volodymyr and I'm a computer programmer. I have a passion for solving problems. My current projects include developing the Android phone app for Proton, a feature-rich, app-based online social networking and messaging system.
    I am interested in both mathematics and computer science. I'm particularly interested in game programming, algorithm design and systems, and I am currently working on a project on wireless networks and multi-hop routing.
    I like to solve problems, learn new things and think in a constructive way. I enjoy working with other people, especially people who are also interested in technology, and I like to work on projects that are both interesting and
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. The president has a lot of different jobs, but the one that is most important is being the leader of the country. The president is usually elected by the people in the country, and they usually have to get a lot of votes before they get to be the leader of the country. The president has a lot of power, but he can't do anything without the help of the other branches of the government. The president has a lot of other jobs too, like being in charge of the military and being the leader of the economy. But the one job that is most important to the country is being the leader of
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Brussels
    C. London
    D. Rome
    Answer:
    A
    
    The fundamental purpose of a company's business strategy is to ____.
    A. ensure competitiveness
    B. ensure profit
    C. ensure safety
    D. ensure efficiency
    Answer:
    B
    
    In a certain city, the monthly average temperature is approximately: 10°C, 11°C, 9°C, 12°C, 13°C, and 14°C. What is the mode of this data set?
    A. 9°C
    B. 10°C
    C. 11
    ===============================
    Prompt: The future of AI is
    Generated text:  coming true and it will be an essential part of the way we live and work, with applications in fields ranging from healthcare to finance.
    And while we’re still a long way from that day, the folks at the Allen Institute for AI have already started to build the tools that will allow researchers to build powerful artificial intelligence models.
    The Allen Institute for AI has been set up in Minnesota to accelerate AI research, and the team has already made progress building out a number of AI models.
    One of the most exciting things about AI is that we can build models that can learn from large amounts of data and make predictions on new data.
    That's something


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [Age], [Gender], [Nationality], [Occupation], and I have [Number of Years] years of experience in [Field of Work]. I'm always looking for new opportunities to grow and learn, and I'm always eager to learn more about the world around me. What's your favorite hobby or activity? I enjoy [Favorite Activity], and I'm always looking for new ways to explore and discover new things. What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a cultural and economic hub, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. Paris is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin, as well as the Notre-Dame Cathedral, which is a symbol of French culture and history. The city is also known for its food, fashion, and music scenes, and is a popular tourist destination for visitors from
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud
    


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
    Generated text:  [Name], and I'm a [job title] for [company name]. I'm passionate about [insert a quote about your work or hobby]. My ultimate goal is to make a positive impact in the world and create meaningful change. I'm excited to contribute to [insert the reason why you want to work at [company name]]. Would you like to know more about me? 
    
    Remember, I am not here to evaluate, but to connect with you in a way that we can both grow together. How can I help you today? 
    
    [Your Name]   
    [Your Position]  
    [Company Name]  
    [Company Address]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Grande Seine". It is the largest city in the country and is a major cultural, economic and political center. Paris is known for its iconic architecture, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also famous for its annual summer Olympics and cultural festivals. It is a popular destination for tourists and locals alike. Paris is a bustling and exciting metropolis that is part of the European Union. Its status as a major global city is reflected in its impressive skyline, numerous museums and attractions, and its extensive transportation network. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be exciting and diverse. Here are some possible trends that we can expect to see in the coming years:
    
    1. Advancements in machine learning and deep learning: We are already seeing amazing progress in machine learning and deep learning, with algorithms that can analyze and understand complex patterns in large amounts of data. We may see even more breakthroughs in this area in the coming years.
    
    2. Increased integration of AI into different industries: AI is already being used in a wide range of industries, from healthcare and finance to retail and transportation. We may see even more integration in the coming years as AI becomes more widely used and integrated into various


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

     [

    Type

     of

     Character

    ].

     I

    'm

     very

     good

     at

     [

    What

     I

    'm

     Good

     At

    ],

     and

     I

     have

     a

     lot

     of

     [

    What

     I

     Have

    ].

     I

     like

     [

    What

     I

     Enjoy

     Doing

    ],

     and

     I

    'm

     always

     ready

     to

     [

    What

     I

    'm

     Ready

     To

     Do

    ].

     I

    'm

     confident

    ,

     hard

    working

    ,

     and

     always

     up

     for

     [

    What

     I

    'm

     Up

     For

    ].

     I

    'm

     always

     looking

     for

     new

     experiences

     and

     challenges

     to

     try

     and

     grow

     as

     a

     person

    .

     I

    'm

     a

     [

    Type

     of

     Character

    ],

     and

     I

     look

     forward

     to

     [

    What

     I

    'd

     Like

     to

     See

     From

    ]

     when

     we

     meet

    .
    


    I

     hope

     this

     intro

     is

     helpful

    ,

     and

     if

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Paris

     is

     the

     capital

     of

     France

     and

     is

     a

     large

     city

     on

     the

     banks

     of

     the

     Se

    ine

     River

     in

     the

     northern

     part

     of

     the

     country

    .

     The

     city

     is

     also

     famous

     for

     its

     rich

     history

    ,

     arts

    ,

     culture

    ,

     and

     architecture

    .

     Paris

     has

     a

     unique

     blend

     of

     European

     and

     Mediterranean

     influences

     and

     has

     been

     a

     hub

     of

     culture

     and

     politics

     for

     centuries

    .

     It

     is

     home

     to

     many

     world

    -f

    amous

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

     The

     city

     has

     been

     a

     center

     of

     intellectual

     and

     artistic

     life

     since

     the

     Middle

     Ages

     and

     has

     been

     a

     major

     hub

     of

     government

    ,

     trade

    ,

     and

     commerce

     for

     centuries

    .

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

    ,

     with

     many

     possibilities

     and

     potential

     applications

    .

     Some

     of

     the

     most

     promising

     trends

     include

    :
    


    1

    .

     Advanced

     machine

     learning

     and

     deep

     learning

    :

     AI

     systems

     will

     continue

     to

     develop

     at

     a

     rapid

     pace

    ,

     with

     more

     sophisticated

     algorithms

     and

     models

    .

     This

     means

     that

     AI

     solutions

     will

     become

     increasingly

     accurate

     and

     capable

     of

     handling

     complex

     tasks

    .
    


    2

    .

     Increased

     integration

     with

     other

     technologies

    :

     AI

     systems

     will

     likely

     become

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     blockchain

    ,

     IoT

    ,

     and

     edge

     computing

    .

     This

     will

     enable

     AI

     systems

     to

     work

     more

     seamlessly

     with

     other

     systems

    ,

     reducing

     the

     complexity

     and

     cost

     of

     integration

    .
    


    3

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

     sophisticated

    ,

     there

     will

     be

    



```python
llm.shutdown()
```
