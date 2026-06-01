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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.19it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:43,  4.97s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:43,  4.97s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.93it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.93it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.93it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.93it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.93it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.93it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.93it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.93it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.93it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.93it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.34it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.34it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.34it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.34it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.34it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.34it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.34it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.34it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.34it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.19it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:01, 19.81it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:01, 19.81it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:01, 19.81it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:01, 19.81it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:01, 19.81it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:01, 19.81it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:01, 19.81it/s]

    Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:01, 19.81it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:01, 19.81it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:01, 19.81it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 27.42it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 27.42it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 27.42it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 27.42it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 27.42it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 27.42it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 27.42it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 27.42it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 27.42it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 27.42it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 27.42it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 36.96it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 36.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.47it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.47it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.47it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.23it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.23it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.26it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.26it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.26it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.26it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.26it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.26it/s] Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.13it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.13it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.13it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.13it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.13it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.13it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.46it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.46it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.46it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.46it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.46it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.46it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.18it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.43it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.43it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.43it/s]

    Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.43it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.43it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.43it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.72it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.72it/s] Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  81%|████████  | 47/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 43.02it/s]

    Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 43.02it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.35it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.83it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 38.55it/s]


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
    Generated text:  Emilia and I’m a student at the University of California, Davis in the Natural Sciences Department. I’m a neuroscientist and a neurologist.
    My research examines the role of the female brain in cognitive functions such as memory, learning, memory retention, and attention. My lab also investigates how the female brain responds to stress and anxiety. My goal is to understand how the brain adapts and learns during and after pregnancy.
    I teach classes in all sciences, and also work with underrepresented populations on campus. My mentor is Dr. Jeremy Gonczarowski, who is a senior assistant professor of the Department of Neurology and
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 3 inches tall. The vice president of the United States is 5 feet 5 inches tall. How many more inches tall is the vice president than the president? To determine how many more inches tall the vice president is compared to the president, we need to follow these steps:
    
    1. Convert the heights of both individuals to inches.
    2. Subtract the height of the president from the height of the vice president.
    
    First, let's convert the heights of the presidents and the vice president to inches. We know that there are 12 inches in a foot. Therefore, we can convert the heights as follows:
    
    The
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is an administrative center of the country. Besides it is a very popular destination for tourists and one of the top destinations for Paris tourists. Each year, more than 8 million tourists visit the city of Paris. Paris is also one of the top cities for French culture and cuisine, and is the birthplace of the French language.
    The French language is used in the city and throughout the country. Paris has been called the most important language in the world.
    French cuisine is a very popular type of food in Paris. The best known French dishes are croissants, fish and chips, and some versions of bouden and
    ===============================
    Prompt: The future of AI is
    Generated text:  bright
    
    The world is going to be forever changed by the rise of Artificial Intelligence (AI) over the next 15 years. Many have been predicting the rise of AI for years. But many have also argued that AI will be ignored in favor of human leadership. But do we really need to ignore AI? Here are some of the reasons why we should embrace AI.
    
    Artificial Intelligence
    
    Artificial Intelligence is the most recent and most disruptive technology in the world today. It has the potential to change the way we live our lives, the way we work, and the way we communicate. AI can perform tasks that are currently the domain


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm a [job title] at [company name], and I love [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite book or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also famous for its cuisine, fashion, and music. Paris is a popular tourist destination and a major economic center in France. It is home to many world-renowned museums, art galleries, and theaters. The city is also known for its fashion industry, with many famous designers and boutiques. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the potential for AI to be used for malicious purposes.
    
    2. Greater integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This will allow for more sophisticated
    


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
    Generated text:  [Name] and I'm a [职业] who has been in [职业] for [number] years. I'm always up for learning and growing, and always ready to help others. I believe in the power of empathy and compassion, and I'm always trying to make the world a better place. I'm excited to meet you! Let's do this! [Name]: Welcome! How are you today? How did you get to where you are today? [Name]: I've always been interested in the world around me, and I've been working towards becoming an empathetic and compassionate person for as long as I can remember
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its rich history, stunning architecture, and vibrant culture. The city is home to numerous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its annual festivals and events, including the famous 2017 Eiffel Tower Torch Festival and the annual Les Diurnes market. It is one of the most visited cities in the world and is a cultural and historical destination for millions of visitors each year.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be dominated by three trends: big data, deep learning, and quantum computing. Big data will play a key role in the growth of AI. As more data is created, the ability to analyze and interpret that data will become more advanced. Deep learning, a subset of AI, is also likely to continue to advance, with more sophisticated models being developed. Finally, quantum computing is also expected to play a role in AI, with the development of quantum computers expected to revolutionize certain types of AI tasks. These trends are likely to continue as the technology advances and more resources are dedicated to researching and developing these technologies. Overall, the


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

     an

     AI

     assistant

     designed

     to

     assist

     individuals

     in

     finding

     information

     and

     answering

     their

     questions

    .

     I

    'm

     here

     to

     help

     people

     with

     their

     inquiries

    ,

     whether

     it

    's

     answering

     questions

     about

     science

    ,

     history

    ,

     literature

    ,

     or

     general

     knowledge

    .

     If

     you

     need

     help

     or

     have

     questions

    ,

     please

     feel

     free

     to

     ask

    .

     I

    'm

     here

     to

     help

     you

     with

     whatever

     you

     need

    .

     Take

     a

     look

     around

     and

     I

    'll

     show

     you

     the

     world

     around

     you

    !

     How

     can

     I

     help

     you

     today

    ?

     As

     an

     AI

     assistant

    ,

     I

    'm

     always

     ready

     to

     assist

     individuals

     with

     their

     inquiries

    .

     I

     have

     access

     to

     a

     vast

     amount

     of

     information

     and

     can

     provide

     answers

     to

     a

     wide

     range

     of

     questions

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     The

     city

     is

     located

     in

     the

     Lo

    ire

     Valley

     region

     of

     the

     north

    western

     part

     of

     the

     country

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     the

     second

     most

     populous

     city

    ,

     after

     London

    .

     Paris

     is

     known

     for

     its

     rich

     cultural

     and

     artistic

     heritage

    ,

     fashion

     industry

    ,

     gastr

    onomy

    ,

     and

     numerous

     landmarks

     such

     as

     Notre

     Dame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     and

     Lou

    vre

     Museum

    .

     It

     is

     a

     major

     transportation

     hub

     and

     an

     important

     economic

     center

     in

     Europe

    .

     The

     city

     is

     famous

     for

     its

     op

    ulent

     architecture

    ,

     including

     the

     Lou

    vre

     and

     the

     Tu

    il

    eries

     Garden

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

     President

     of

     the

     Senate

    ,

     and

     the

     President

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     increasingly

     focused

     on

     developing

     more

     advanced

     models

     that

     can

     handle

     complex

    ,

     real

    -world

     problems

    .

     Some

     possible

     future

     trends

     include

    :
    


    1

    .

     Deep

     learning

    :

     As

     the

     size

     and

     complexity

     of

     computer

     systems

     continue

     to

     increase

    ,

     so

     does

     the

     power

     and

     accuracy

     of

     deep

     learning

     models

    .

     These

     models

     can

     now

     handle

     a

     wider

     range

     of

     tasks

    ,

     including

     image

     recognition

    ,

     natural

     language

     processing

    ,

     and

     speech

     recognition

    .
    


    2

    .

     Natural

     language

     processing

    :

     AI

     that

     can

     understand

    ,

     interpret

    ,

     and

     generate

     human

     language

     is

     becoming

     increasingly

     common

    .

     This

     will

     likely

     lead

     to

     the

     development

     of

     more

     advanced

     AI

     systems

     that

     can

     understand

     and

     generate

     human

    -like

     language

    .
    


    3

    .

     Autonomous

     vehicles

    :

     AI

     is

     becoming

     more

     advanced

    ,

     and

    



```python
llm.shutdown()
```
