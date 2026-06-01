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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.04it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.03it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.07it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.07it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.65it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 12.93it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 12.93it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 12.93it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 12.93it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 12.93it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 12.93it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 12.93it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 12.93it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 12.93it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:05<00:02, 12.93it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 19.87it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 19.87it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 19.87it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 19.87it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 19.87it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 19.87it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 19.87it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 19.87it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 19.87it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 19.87it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:05<00:01, 19.87it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.85it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.85it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.85it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.85it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.85it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.85it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.85it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.85it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.85it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 28.85it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 28.85it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 38.55it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 38.55it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.45it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.45it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.45it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.82 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.81 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.98it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.98it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.98it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.98it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 28.75it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 28.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.09 GB):  21%|██        | 12/58 [00:00<00:01, 28.75it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 28.75it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 28.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.08 GB):  21%|██        | 12/58 [00:00<00:01, 28.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=1792 avail_mem=75.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=1280 avail_mem=75.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=75.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.62it/s]

    Capturing num tokens (num_tokens=960 avail_mem=75.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.62it/s] Capturing num tokens (num_tokens=960 avail_mem=75.07 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.39it/s]Capturing num tokens (num_tokens=896 avail_mem=75.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.39it/s]Capturing num tokens (num_tokens=832 avail_mem=75.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.39it/s]Capturing num tokens (num_tokens=768 avail_mem=75.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.39it/s]Capturing num tokens (num_tokens=704 avail_mem=75.05 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.39it/s]Capturing num tokens (num_tokens=640 avail_mem=75.05 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.39it/s]Capturing num tokens (num_tokens=640 avail_mem=75.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=576 avail_mem=75.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=512 avail_mem=75.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=480 avail_mem=75.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.50it/s]

    Capturing num tokens (num_tokens=448 avail_mem=75.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=416 avail_mem=75.05 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.50it/s]Capturing num tokens (num_tokens=416 avail_mem=75.05 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.67it/s]Capturing num tokens (num_tokens=384 avail_mem=75.04 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.67it/s]Capturing num tokens (num_tokens=352 avail_mem=74.90 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.67it/s]Capturing num tokens (num_tokens=320 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.67it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.36it/s]

    Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.19it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.19it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.19it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.19it/s]Capturing num tokens (num_tokens=112 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.19it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.19it/s] Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 40.33it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 40.33it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 40.33it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 40.33it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 40.33it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  81%|████████  | 47/58 [00:01<00:00, 40.33it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=20 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=16 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.72it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.72it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 37.13it/s]


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
    Generated text:  Sam and I have a computer science background and I am interested in the field of robotics. I am looking for a project in Python that could be used in a robotics project. Can you please provide me with some Python libraries that could be used for this purpose? Additionally, could you please share any best practices or tips that I can follow while working with the libraries? Sure, here are some Python libraries that could be useful in robotics projects:
    
    1. ROS (Robot Operating System): ROS is a set of open-source software tools for building robotics software. It provides a set of tools and protocols for creating a robust and scalable robotic ecosystem. It
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to ensure that their administration will accomplish its goals. It is important for the president to communicate his or her understanding of the issues to the general public so that people can understand the issues. An effective communication strategy would include using plain language and avoiding complex jargon and terminology that could confuse or overwhelm the general public.
    One such strategy is to use analogies and metaphors that are easy to understand. Analogies and metaphors can be used to convey complex ideas in a clear and memorable way. They can also be used to show the interconnectedness of different issues, making it easier for people to see the bigger picture.
    One example of an
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The city is situated on the Île-de-France region, which is located in the southern part of France. The Île-de-France region is part of the historic Paris Basin, which is a region of France that stretches from the north to the south. The city of Paris is situated on the Île de France, which is located at the eastern part of the Paris Basin. As a result, Paris is situated in the southern part of the Paris Basin, which is part of the historic Paris Basin. The Paris Basin is a basin that is located in the northern part of France and stretches from the north to the south
    ===============================
    Prompt: The future of AI is
    Generated text:  more complex and diverse than ever before. AI is constantly evolving, with new technologies being developed and new applications being created at an unprecedented rate. As a result, it is important to stay updated on the latest trends and developments in the field of AI.
    One of the key areas of focus in the field of AI is natural language processing (NLP). This is a subset of AI that involves the use of algorithms and machine learning techniques to process and understand human language. NLP has a wide range of applications, from language translation and speech recognition to text summarization and sentiment analysis.
    Another area of focus in the field of AI is computer vision


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Character] who has always been [What motivates you to be who you are]. I'm passionate about [What you enjoy doing that makes you happy]. I'm [What you're known for]. I'm [What you're most proud of]. I'm [What you're most proud of]. I'm [What you're most proud of]. I'm [What you're most proud of]. I'm [What you're most proud of]. I'm [What you're most proud of]. I'm [What you're
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Monument. Paris is a bustling metropolis with a diverse population and is a major tourist destination. It is home to many famous French artists, writers, and musicians. The city is also known for its cuisine, including its famous croissants and its famous cheese, Camembert. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a unique and fascinating city. It is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced interactions between humans and machines. This could lead to more personalized and adaptive AI systems.
    
    3. Increased use of AI in healthcare: AI is already
    


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
    Generated text:  [Name], and I am a [Age] year old [Gender] person. I am an [Occupation] with [Skill or Expertise] in my field. I have [Number] years of experience in this field, and I have always been [Positive or Neutral] about my work. I am a [Type of Person], and I am always [Positive or Neutral] about my life. I enjoy [What I Am Good At], and I believe that [Why I Love My Work or Life]. My personality is [Positive or Neutral], and I am always [Positive or Neutral] about my relationships. I believe in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the 6th largest city in the world by population and the largest city by area.
    
    The statement provided accurately reflects the factual statement about the capital city of France, which is Paris. The complete statement is: "The capital of France is Paris, the 6th largest city in the world by population and the largest city by area." 
    
    To elaborate further:
    1. Paris, known as the "City of Light," is the main city of France, located on the Seine River in the heart of Paris, a UNESCO World Heritage site.
    2. It is France's most populous city (3.7 million residents as
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be a rapidly evolving field. Some potential future trends in AI include:
    
    1. Increased use of AI in autonomous vehicles: With the growing concern for road safety and environmental impact, there is a growing demand for autonomous vehicles that can operate without human intervention. This trend is expected to drive the development of AI algorithms and technologies that can analyze data, make decisions, and drive autonomous vehicles.
    
    2. Greater integration of AI into healthcare: AI is already being used in healthcare to improve diagnosis and treatment, predict disease progression, and personalize medicine. As the technology advances, we can expect to see even more integration of AI into healthcare, such as


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

    occupation

    ]

     [

    Gender

    ].

     I

    'm

     currently

     in

     [

    location

    ]

     [

    city

    ,

     state

    ,

     country

    ]

     [

    country

    ].

     I

    'm

     a

     [

    occupation

    ],

     known

     for

     [

    something

     you

    're

     famous

     for

    ].

     I

     live

     [

    location

    ]

     [

    city

    ,

     state

    ,

     country

    ]

     [

    country

    ].

     I

     love

     [

    reason

     for

     passion

     or

     love

    ].

     I

     believe

     in

     [

    what

     you

     believe

     in

    ],

     which

     is

     [

    reason

     for

     belief

    ].

     I

    'm

     always

     [

    how

     you

    're

     organized

    ,

     organized

    ,

     etc

    .

    ].

     My

     favorite

     [

    food

    ]

     is

     [

    eat

    /

    love

    /

    like

    ].

     I

     enjoy

     [

    an

     activity

     you

     love

     or

     dislike

    ],

     and

    
    
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

     the

     third

     largest

     city

     in

     the

     European

     Union

    .

     Its

     population

     is

     approximately

     

    2

    .

    7

     million

     people

    ,

     making

     it

     the

     most

     populous

     city

     in

     France

     and

     the

     United

     Kingdom

     combined

    .

     Paris

     is

     known

     for

     its

     art

    ,

     fashion

    ,

     music

    ,

     and

     cuisine

    .

     The

     city

    's

     skyline

     is

     famous

     for

     its

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

     E

    iff

    el

     River

    ,

     and

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     theaters

    .

     Paris

     is

     also

     home

     to

     the

     European

     Parliament

     and

     the

     French

     Parliament

    .

     The

     city

     is

     known

     for

     its

     annual

     Carn

    aval

    ,

     a

     parade

     that

     attracts

     thousands

     of

     tourists

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     continued

     advancement

    ,

     but

     also

     by

     significant

     changes

     in

     how

     it

     is

     used

     and

     regulated

    .

     Some

     possible

     trends

     include

    :
    


    1

    .

     Increased

     AI

     in

     healthcare

    :

     AI

     will

     be

     used

     in

     healthcare

     to

     improve

     diagnosis

    ,

     treatment

    ,

     and

     prevention

     of

     disease

    .

     For

     example

    ,

     AI

     algorithms

     will

     be

     used

     to

     analyze

     medical

     images

     and

     identify

     patterns

     that

     may

     be

     missing

     from

     current

     diagnostic

     methods

    .

     AI

    -powered

     chat

    bots

     will

     also

     be

     used

     to

     provide

     patients

     with

     advice

     and

     support

    .
    


    2.

     Autonomous

     vehicles

    :

     The

     development

     of

     autonomous

     vehicles

     will

     continue

     to

     advance

    ,

     with

     more

     vehicles

     being

     able

     to

     self

    -p

    ilot

     and

     navigate

     roads

    .

     AI

     will

     be

     used

     to

     optimize

     vehicle

     performance

    ,

     improve

    



```python
llm.shutdown()
```
