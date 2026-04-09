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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.10it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.10it/s]


    2026-04-09 04:07:49,050 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 04:07:49] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.61it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.61it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.61it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.61it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.61it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.61it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.61it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.61it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.61it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.90it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.90it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.90it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.90it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.90it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.90it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.90it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.90it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.97it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.66it/s]

    Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.74it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.17it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.17it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.17it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.17it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.17it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.17it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.17it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   9%|▊         | 5/58 [00:00<00:02, 20.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 20.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 20.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 20.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=3072 avail_mem=131.61 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.61 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.61 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.60 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.43it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.43it/s]Capturing num tokens (num_tokens=1280 avail_mem=131.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.43it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.43it/s]Capturing num tokens (num_tokens=960 avail_mem=131.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.43it/s] Capturing num tokens (num_tokens=960 avail_mem=131.58 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.66it/s]Capturing num tokens (num_tokens=896 avail_mem=131.58 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.66it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.66it/s]Capturing num tokens (num_tokens=768 avail_mem=131.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.66it/s]Capturing num tokens (num_tokens=704 avail_mem=131.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.66it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.66it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=480 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.37it/s]

    Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=320 avail_mem=131.56 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.21it/s]Capturing num tokens (num_tokens=288 avail_mem=131.55 GB):  55%|█████▌    | 32/58 [00:01<00:00, 42.21it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  55%|█████▌    | 32/58 [00:01<00:00, 42.21it/s]

    Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.11it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.11it/s]Capturing num tokens (num_tokens=224 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.11it/s]Capturing num tokens (num_tokens=208 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.11it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.11it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  71%|███████   | 41/58 [00:01<00:00, 36.25it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  71%|███████   | 41/58 [00:01<00:00, 36.25it/s]Capturing num tokens (num_tokens=160 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 36.25it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 36.25it/s]Capturing num tokens (num_tokens=128 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 36.25it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 36.25it/s]

    Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.64it/s] Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=64 avail_mem=131.51 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=48 avail_mem=131.51 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.43it/s]Capturing num tokens (num_tokens=28 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.43it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.43it/s]Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.43it/s]Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.43it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.43it/s]

    Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.36it/s]Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.36it/s] Capturing num tokens (num_tokens=4 avail_mem=131.48 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.36it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 37.18it/s]


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
    Generated text:  Ali, and I'm a senior at the University of Utah. I'm studying Information and Data Science. As a computer science major, I have been working as a software developer and data analyst for the past three years. I have a strong background in programming languages such as Python, Java, and C++, and I have a love for data analysis and business intelligence. I'm also passionate about coding in a variety of languages, such as JavaScript, HTML, CSS, and SQL. My favorite hobby is to take photos, especially of people and places. I am interested in learning more about technology, especially in the areas of machine learning, natural
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 45 years old. How old will the president be in 6 years? To determine how old the president of the United States will be in 6 years, we need to add 6 years to his current age.
    
    The president's current age is 45 years. Adding 6 years to 45, we get:
    
    \[ 45 + 6 = 51 \]
    
    Therefore, the president will be \boxed{51} years old in 6 years.
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A) Paris
    B) London
    C) Berlin
    D) Rome
    The capital of France is:
    
    A) Paris
    
    Paris is the capital city of France, located in the south of the country. It is the largest city in France by population and is home to the Paris Museum of Science, one of the oldest science museums in the world. Paris is known for its rich history, art, fashion, and food, making it a popular destination for tourists and visitors from all over the world. 
    
    Let me know if you need any other information about Paris or the French capital! Let me know if you need help with anything
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the business of the day, the purpose of which is to fully understand customers, their needs, desires, and behaviors and generate the results that satisfy them. Of course, this is a very broad and ambitious goal and there are very many challenges that still need to be overcome.
    Nevertheless, the business of tomorrow will be more and more influenced by the results of AI, and the industry will be more and more interested in the work of the company that can apply the latest in machine learning and data science to improve their results.
    Every company has its own data, and the most important thing is to be able to extract the most


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Skill] who has been [Number of Years] years in the field of [Field of Interest]. I am a [Skill] who has been [Number of Years] years in the field of [Field of Interest]. I am a [Skill] who has been [Number of Years] years in the field of [Field of Interest]. I am a [Skill] who has been [Number of Years] years in the field of [Field of Interest]. I am a [Skill] who has been [Number of Years] years in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Fluviale" or "La Ville Blanche" (White City). It is the largest city in Europe and the second-largest city in the world by population. Paris is known for its rich history, art, and cuisine, and is a major center for politics, culture, and fashion. The city is also home to many famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a popular tourist destination and a major economic center in Europe. It is also home to many important institutions of higher education, including the University of Paris and the Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is likely to play a greater role in healthcare, with more personalized and accurate diagnoses and treatments being developed.
    
    4. Greater use of AI in manufacturing: AI is likely to be used more
    


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
    Generated text:  [Name] and I am an AI assistant. I am programmed to be helpful and patient, and I am always ready to assist with any questions or concerns you may have. Whether you need help with a problem, want to learn about a topic, or just want to chat with me, I am here to assist you. Feel free to ask me anything, and I will do my best to provide you with the information you need. Let's get started! [Name] (insert your name here) [Name] (insert your name here) [Name] (insert your name here) [Name] (insert your name here)
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Make it into a multi-choice question with options:
    
    Which of the following statements is incorrect about Paris?
    
    a) It is the largest city in Europe.
    b) It has the world's most famous Eiffel Tower.
    c) It is the home of French cuisine.
    d) It is home to many historical landmarks such as Notre-Dame Cathedral. 
    e) It is home to the Louvre museum.
    
    The correct answer is:
    c) It is the home of French cuisine.
    
    Explanation: The Eiffel Tower is not the most famous landmark in Paris, but it is very popular and is considered one of the city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  promising and exciting, with many possibilities and opportunities for growth. Here are some possible future trends in AI:
    
    1. Increased integration with human AI: With the increasing reliance on AI for automation and decision-making, we can expect to see more integration of AI with human AI. This could involve the development of more advanced forms of AI that can learn from human behavior and decision-making.
    
    2. Autonomous AI: As AI becomes more integrated with humans, we may see the development of autonomous AI. This would involve machines that can operate autonomously, without human intervention, in certain tasks such as driving or managing household tasks.
    
    3. Enhanced data analysis and


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

    Your

     Name

    ].

     I

     am

     a

     [

    Your

     Profession

    ]

     with

     [

    Your

     Education

    ]

     and

     [

    Your

     Experience

    ].

     I

     am

     passionate

     about

     [

    Your

     Hobby

    /

    Interest

    ].

     I

    'm

     excited

     to

     meet

     you

     and

     learn

     more

     about

     you

    .

     [

    Your

     Name

    ]

     [

    Your

     Job

    /

    Position

    ]

     [

    Your

     Education

    ]

     [

    Your

     Experience

    ]

     [

    Your

     Hobby

    /

    Interest

    ]

     [

    Your

     Goals

    /

     aspirations

    ]

     [

    Your

     Future

     Goals

    ]

     [

    Your

     Challenges

    /

     Challenges

    ]

     [

    Your

     Accom

    pl

    ishments

    /

     Accom

    pl

    ishments

    ]

     [

    Your

     Future

     Accom

    pl

    ishments

    ]

     [

    Your

     Strength

    s

    /

     Strength

    s

    ]

     [

    Your

     Weak

    ness

    es

    /

     Weak

    ness

    es

    ]

     [

    Your

     Personality

    ]

     [

    Your

     Work

     Style

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    What

     is

     the

     capital

     city

     of

     France

    ?

     Paris

    .

     
    


    Explanation

     for

     an

     AI

     system

    :

     


    The

     capital

     of

     France

     is

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     serves

     as

     the

     country

    's

     political

    ,

     cultural

    ,

     and

     economic

     center

    .

     The

     city

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

     and

     continues

     to

     thrive

     in

     a

     rapidly

     changing

     modern

     society

    .

     Paris

     is

     famous

     for

     its

     stunning

     architecture

    ,

     vibrant

     cultural

     scene

    ,

     and

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

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     As

     the

     main

     seat

     of

     government

    ,

     the

     city

     is

     home

     to

     the

     French

     Parliament

    ,

     the

     President

     of

     the

     Senate

    ,

     and

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     several

     trends

     are

     likely

     to

     shape

     the

     technology

    's

     development

     in

     the

     coming

     years

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     systems

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     increasing

     pressure

     to

     address

     issues

     such

     as

     bias

    ,

     transparency

    ,

     accountability

    ,

     and

     security

    .

     These

     issues

     will

     likely

     receive

     more

     attention

     from

     policymakers

    ,

     industry

     leaders

    ,

     and

     the

     public

    .
    


    2

    .

     Greater

     integration

     with

     human

    -com

    puter

     interaction

    :

     AI

     will

     continue

     to

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     including

     through

     the

     integration

     of

     voice

     assistants

    ,

     smart

     homes

    ,

     and

     more

    .

     This

     will

     require

     more

     research

     and

     development

     in

     areas

    



```python
llm.shutdown()
```
