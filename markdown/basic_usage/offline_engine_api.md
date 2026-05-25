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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.27it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:43,  3.92s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.88it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.84it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.28it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.28it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.28it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.28it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.28it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.28it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.28it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.28it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.28it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.28it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.28it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.83it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.83it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.83it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.83it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.83it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.83it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.83it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.83it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.83it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.83it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.48it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.48it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.48it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.48it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.48it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.48it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.48it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.48it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.96 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.93 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.93 GB):   3%|▎         | 2/58 [00:00<00:03, 18.61it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.92 GB):   3%|▎         | 2/58 [00:00<00:03, 18.61it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.92 GB):   3%|▎         | 2/58 [00:00<00:03, 18.61it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.92 GB):   3%|▎         | 2/58 [00:00<00:03, 18.61it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.92 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.92 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.91 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.90 GB):   9%|▊         | 5/58 [00:00<00:02, 21.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.90 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.90 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.90 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.89 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.17it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.89 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.17it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=71.89 GB):  21%|██        | 12/58 [00:00<00:01, 30.18it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.89 GB):  21%|██        | 12/58 [00:00<00:01, 30.18it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.89 GB):  21%|██        | 12/58 [00:00<00:01, 30.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.38 GB):  21%|██        | 12/58 [00:00<00:01, 30.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.06 GB):  21%|██        | 12/58 [00:00<00:01, 30.18it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.06 GB):  21%|██        | 12/58 [00:00<00:01, 30.18it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.66it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.42 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.66it/s]

    Capturing num tokens (num_tokens=960 avail_mem=67.53 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.66it/s] Capturing num tokens (num_tokens=960 avail_mem=67.53 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.01it/s]Capturing num tokens (num_tokens=896 avail_mem=67.53 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.01it/s]Capturing num tokens (num_tokens=832 avail_mem=67.53 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.01it/s]Capturing num tokens (num_tokens=768 avail_mem=67.52 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.01it/s]Capturing num tokens (num_tokens=704 avail_mem=67.52 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.01it/s]Capturing num tokens (num_tokens=640 avail_mem=67.52 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.01it/s]Capturing num tokens (num_tokens=640 avail_mem=67.52 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.90it/s]Capturing num tokens (num_tokens=576 avail_mem=67.52 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.90it/s]Capturing num tokens (num_tokens=512 avail_mem=67.50 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.90it/s]Capturing num tokens (num_tokens=480 avail_mem=67.52 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.90it/s]Capturing num tokens (num_tokens=448 avail_mem=67.52 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.90it/s]

    Capturing num tokens (num_tokens=416 avail_mem=67.51 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.90it/s]Capturing num tokens (num_tokens=416 avail_mem=67.51 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=384 avail_mem=67.51 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=352 avail_mem=67.51 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=320 avail_mem=67.50 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=288 avail_mem=67.50 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=256 avail_mem=67.50 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.87it/s]Capturing num tokens (num_tokens=256 avail_mem=67.50 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.30it/s]Capturing num tokens (num_tokens=240 avail_mem=67.49 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.30it/s]Capturing num tokens (num_tokens=224 avail_mem=67.49 GB):  64%|██████▍   | 37/58 [00:00<00:00, 45.30it/s]Capturing num tokens (num_tokens=208 avail_mem=67.48 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.30it/s]Capturing num tokens (num_tokens=192 avail_mem=67.48 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.30it/s]

    Capturing num tokens (num_tokens=176 avail_mem=67.48 GB):  64%|██████▍   | 37/58 [00:01<00:00, 45.30it/s]Capturing num tokens (num_tokens=176 avail_mem=67.48 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=160 avail_mem=67.48 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=144 avail_mem=67.48 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=128 avail_mem=67.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=112 avail_mem=67.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=96 avail_mem=67.47 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.05it/s] Capturing num tokens (num_tokens=96 avail_mem=67.47 GB):  81%|████████  | 47/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=80 avail_mem=67.46 GB):  81%|████████  | 47/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=64 avail_mem=67.46 GB):  81%|████████  | 47/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=48 avail_mem=67.46 GB):  81%|████████  | 47/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=32 avail_mem=67.45 GB):  81%|████████  | 47/58 [00:01<00:00, 46.41it/s]

    Capturing num tokens (num_tokens=28 avail_mem=67.45 GB):  81%|████████  | 47/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=28 avail_mem=67.45 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=24 avail_mem=67.45 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=20 avail_mem=67.44 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=16 avail_mem=67.44 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=12 avail_mem=67.44 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=8 avail_mem=67.43 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.78it/s] Capturing num tokens (num_tokens=8 avail_mem=67.43 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.27it/s]Capturing num tokens (num_tokens=4 avail_mem=67.43 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.27it/s]Capturing num tokens (num_tokens=4 avail_mem=67.43 GB): 100%|██████████| 58/58 [00:01<00:00, 41.15it/s]


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
    Generated text:  Jane, a 22-year-old single woman. I'm from the UK. I graduated from a prestigious university. I'm quite well educated. My education has a strong academic base. I have the ability to do research in a broad range of subjects, and I've got the ability to communicate in English. I'm a very strong communicator. I work as an IT professional and have experience in the fields of software development and project management. I'm interested in learning about AI and AI ethics. I have a passion for science and I love to travel. I'm now looking to expand my horizons. I'm keen to explore more
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 3 inches tall. The vice president is 3 feet 7 inches tall. If the height of the building they are standing in is 30 stories, with each floor being 12 inches high, how much taller is the vice president relative to the height of the building?
    To determine how much taller the vice president is relative to the height of the building, we need to follow these steps:
    
    1. Convert the height of the vice president from inches to feet.
    2. Calculate the total height of the vice president, including the height of the building.
    3. Compare the height of the vice president to the
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Lille
    B. Paris
    C. Paris
    D. Lille
    Answer:
    
    B
    
    Which of the following are examples of legitimate proprietary rights? ① Patent rights ② Trademark rights ③ Copyrights ④ Business trademark rights
    A. ①②③
    B. ①②④
    C. ①③④
    D. ①②③④
    Answer:
    
    A
    
    The proportion of enterprises that have entered the central and eastern regions in China's total number of enterprises is ____.
    A.
    ===============================
    Prompt: The future of AI is
    Generated text:  fast moving and technology is rapidly evolving with a lot of progress happening. However, there are still many areas that need to be solved and improved for the future to be fully realized. One such area is the issue of bias in AI algorithms.
    Bias in AI refers to the unfairness or discrimination in the algorithms that are created to make decisions based on data. Bias can be due to a variety of factors such as race, gender, religion, age, disability, and more. The issue of bias in AI has been recognized and has been a topic of debate for quite some time. The impact of bias in AI on various industries such as finance


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


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a cultural and economic hub, with a rich history dating back to the Middle Ages and a modern city that has undergone significant development over the years. Paris is a popular tourist destination, with millions of visitors annually, and is home to many world-renowned museums, theaters, and restaurants. The city is also known for its vibrant nightlife, with many bars and clubs offering a wide range of entertainment options. Overall, Paris is a city of art, culture, and history that is a must
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing concerns about its impact on society. This includes questions about privacy, bias, and the potential for AI to be
    


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
    Generated text:  [Name] and I am a [Name] who recently graduated from [University/College]. I have been [short answer here about what you have achieved in your academic and personal life]. What can you tell me about yourself? As a [Name], I have been a passionate [Name] and have always [state why you are passionate about your chosen profession, interests, hobbies, or other aspects of your life]. I have also been [State how you plan to continue your journey and what you are looking forward to in the future].
    [Name] brings a unique combination of expertise and a genuine passion for [Name], which will make
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    In addition, please provide a brief explanation of the significance of Paris in France's cultural and political history. Paris is the seat of the French government, and has been a significant cultural and economic center since ancient times. The city is also famous for its distinctive architecture, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The city's rich history, including the influence of the French Revolution and Romanticism, has also played a role in shaping its modern identity. 
    
    The modern capital of France is currently located in the city of Paris, which is a historic city with a rich history of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising and will continue to develop rapidly. Here are some possible future trends in AI:
    
    1. AI will become more ubiquitous: As AI technology continues to advance, we expect that more and more devices and systems will be equipped with AI capabilities. This will lead to a more ubiquitous presence of AI in our daily lives.
    
    2. AI will enable more personalized experiences: As AI technology continues to advance, we expect that it will become more effective at understanding and personalizing the user experience. This will lead to more personalized experiences, from the way we interact with virtual assistants to the products and services we buy online.
    
    3. AI will be used


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

    Career

    ]

     with

     [

    Number

     of

     Years

     Experience

    ]

     years

     of

     experience

     in

     [

    Industry

    /

    Field

    ].

     I

    've

     been

     working

     in

     [

    Position

    ]

     for

     [

    Number

     of

     Years

    ]

     years

     now

    .

     I

    'm

     a

     [

    Role

    ]

     that

     specializes

     in

     [

    Skill

    ],

     and

     I

     enjoy

     [

    Experience

    ]

     in

     [

    Skill

    ].

     I

    'm

     passionate

     about

     [

    Reason

     for

     Passion

    ],

     and

     I

    'm

     always

     looking

     for

     [

    Action

     or

     Goal

    ]

     to

     achieve

    .

     I

    'm

     a

     [

    A

    val

    iable

     Role

    ]

     that

    's

     always

     eager

     to

     learn

     and

     grow

    .

     I

    'm

     confident

     in

     [

    Skill

    ]

     and

     I

     believe

     in

     [

    Value

    ].

     I

     love

     [

    Ad

    jective

    ],

     and

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     renowned

     for

     its

     beautiful

     architecture

    ,

     vibrant

     culture

    ,

     and

     historical

     significance

    .

     It

     is

     the

     political

    ,

     economic

    ,

     and

     cultural

     center

     of

     France

     and

     is

     home

     to

     the

     French

     Parliament

    ,

     the

     presidential

     palace

    ,

     and

     the

     E

    iff

    el

     Tower

    ,

     among

     other

     iconic

     landmarks

    .

     With

     its

     rolling

     hills

    ,

     medieval

     architecture

    ,

     and

     French

     cuisine

    ,

     Paris

     has

     become

     a

     global

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

     As

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

    ,

     Paris

     has

     played

     a

     vital

     role

     in

     shaping

     French

     identity

     and

     contributing

     to

     its

     global

     prominence

    .

     Its

     history

    ,

     including

     the

     Roman

     Empire

    ,

     Gothic

     period

    ,

     Renaissance

    ,

     and

     

    1

    8

    th

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     rapid

     growth

    ,

     development

    ,

     and

     integration

     into

     various

     sectors

     of

     society

    .

     Some

     possible

     trends

     include

    :
    


    1

    .

     Increased

     automation

    :

     AI

     is

     already

     transforming

     many

     industries

     by

     autom

    ating

     routine

     tasks

     and

     freeing

     workers

     from

     repetitive

     tasks

    .

     As

     AI

     technology

     continues

     to

     evolve

    ,

     we

     can

     expect

     to

     see

     increased

     automation

     in

     various

     sectors

    ,

     including

     manufacturing

    ,

     transportation

    ,

     healthcare

    ,

     and

     agriculture

    .
    


    2

    .

     AI

     ethics

     and

     governance

    :

     As

     AI

     technology

     becomes

     more

     integrated

     into

     daily

     life

    ,

     there

     is

     a

     growing

     concern

     about

     its

     ethical

     implications

    .

     As

     AI

     systems

     become

     more

     advanced

    ,

     we

     will

     need

     to

     develop

     regulations

     and

     guidelines

     to

     ensure

     that

     AI

     is

     used

     responsibly

     and

     in

     ways

     that

     benefit

    



```python
llm.shutdown()
```
