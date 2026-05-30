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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:09,  4.39s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:09,  4.39s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.36it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.72it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.72it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.72it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.72it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.72it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.72it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.72it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.72it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.72it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.46it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.46it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.46it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   3%|▎         | 2/58 [00:00<00:03, 17.46it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.05 GB):   9%|▊         | 5/58 [00:00<00:02, 20.74it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.04 GB):   9%|▊         | 5/58 [00:00<00:02, 20.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 20.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 20.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.03 GB):   9%|▊         | 5/58 [00:00<00:02, 20.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.03 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=73.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.59 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.59 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.80it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.80it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.80it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.80it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.80it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  31%|███       | 18/58 [00:00<00:01, 36.16it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 36.16it/s] Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.34it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.44it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.44it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.44it/s]

    Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.44it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.44it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.44it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=320 avail_mem=72.53 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.59it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.24it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.24it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.24it/s]

    Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.24it/s]Capturing num tokens (num_tokens=176 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.24it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.24it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.38it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.38it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.38it/s]Capturing num tokens (num_tokens=112 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.38it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.38it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.38it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.21it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.21it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.21it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.21it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.21it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.21it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.54it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.54it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.54it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.54it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.54it/s] Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.54it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 46.26it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 40.22it/s]


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
    Generated text:  **Aleksei** and I am a professional tutor, my main focus is to help you achieve your educational goals. You can find me at the top of the page and the following links are my professional credentials: 
    
    [Linkedin](https://www.linkedin.com/in/aleksey-nikitin-812a12240/) 
    
    [GitHub](https://github.com/aleksey-nikitin) 
    
    [Stack Overflow](https://stackoverflow.com/users/1622679/aleksey-nikitin) 
    
    [Google Scholar](https://scholar.google.com/citations?
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a country in which a local official has to create a list of all the countries that are located on the same continent as the United States. If there are 4 continents in total and the president has a list of 5 countries to list, what is the probability that the president selects the correct continent for each of the 5 countries listed? Express your answer as a common fraction.
    
    To determine the probability that the president selects the correct continent for each of the 5 countries listed, we need to follow these steps:
    
    1. Identify the total number of continents.
    2. Determine the number of ways to arrange 5 different continents for
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the region of the same name. It is located in the north-western part of the country, near the border with Switzerland. The capital is located in the Île-de-France department of the north of France. The city has a population of around 532,000, and has a population density of 1,300 people per square kilometer. It is the capital of the Île-de-France metropolitan area and the largest city in the French region of Île-de-France. It is the main city of the Île-de-France department. It is the administrative center of
    ===============================
    Prompt: The future of AI is
    Generated text:  in motion, but it is also set by regulations. This guide will help you navigate the regulations you should be aware of, and the benefits and risks associated with each.
    
    Regulations on AI Ethics and Privacy
    
    Regulations and guidelines that govern the use of AI, and the ethical and privacy considerations surrounding its use, have been in place since the early 2000s. While there is no one overarching set of regulations, there are a number of different laws and standards that can be considered when developing, designing, and implementing AI applications. These include:
    
      1. The General Data Protection Regulation (GDPR) in Europe


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the Louvre Museum. It is the largest city in France and the third-largest in the world by population. Paris is also known for its rich history, including the French Revolution and the French Revolution Museum. The city is home to many famous landmarks, including the Notre-Dame Cathedral and the Arc de Triomphe. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. It is a popular tourist destination and a major economic center in Europe. The city is also known for its fashion industry, with many famous designers and boutiques. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Greater reliance on data: AI systems will become increasingly data-driven, with more data being collected and analyzed to improve their performance. This could lead to more efficient and effective AI systems that can make better decisions based on data.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to
    


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
    Generated text:  [Your Name] and I'm a [Your Age] year old female superhero. I'm a space marine, the most dangerous hero in the galaxy. I wear a space suit, a cape, and a helmet. I have amazing endurance, incredible combat skills, and can take on any enemy. I'm not afraid of anything and always ready to defend my home planet. I love being a space marine and I'm passionate about the galaxy. I believe in unity and teamwork, and I'm always looking for new opportunities to serve my people. Thank you for choosing to meet me, [Your Name]. [Your Name] is a superhero
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a sprawling and bustling metropolis, with a rich history and vibrant culture. The city is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also home to the Louvre Museum, the most important art and history museum in the world. In addition, the city is home to many other renowned attractions, including the Palace of Versailles and the Champs-Élysées. Paris is a major cultural hub, with many renowned universities and art galleries, as well as its own cuisine and local traditions. Overall, Paris is a fascinating and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and several trends are likely to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased precision and accuracy: AI is becoming more accurate and precise in its predictions and decision-making processes. This is likely to lead to even more reliable and accurate applications in various industries.
    
    2. Augmented and virtual reality: AI-powered augmented and virtual reality technologies are likely to become more integrated into our daily lives. This could lead to new applications in fields such as education, entertainment, and manufacturing.
    
    3. AI is becoming more accessible: As AI technology becomes more sophisticated, it will become more accessible to a wider


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

     your

     name

    ].

     I

     am

     [

    insert

     your

     age

     and

     gender

    ].

     I

     have

     [

    insert

     your

     occupation

     or

     profession

    ].

     I

     like

     to

     [

    insert

     something

     specific

     about

     your

     personality

    ].

     I

     am

     an

     [

    insert

     your

     favorite

     color

     or

     hobby

    ]

     person

     and

     I

     love

     [

    insert

     something

     specific

     about

     your

     hobby

    ].

     I

     also

     enjoy

     [

    insert

     something

     specific

     about

     your

     hobby

    ]

     and

     I

     believe

     that

     [

    insert

     a

     statement

     about

     your

     beliefs

     or

     values

    ].

     I

     am

     passionate

     about

     [

    insert

     something

     specific

     about

     your

     hobby

    ],

     and

     I

     am

     determined

     to

     achieve

     my

     goals

    .

     I

     also

     enjoy

     [

    insert

     something

     specific

     about

     your

     hobby

    ]

     and

     I

     am

     a

     [

    insert

     your

     favorite

     type

     of

     music

     or

     sport

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     has

     a

     population

     of

     over

     

    2

     million

     people

     and

     is

     the

     largest

     city

     in

     the

     country

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     vibrant

     culture

    ,

     and

     is

     home

     to

     many

     of

     France

    's

     most

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     a

     popular

     tourist

     destination

    ,

     drawing

     millions

     of

     visitors

     each

     year

     and

     making

     it

     one

     of

     the

     most

     famous

     cities

     in

     the

     world

    .

     It

     is

     often

     referred

     to

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

     and

     sens

    uous

     atmosphere

    .

     Overall

    ,

     Paris

     is

     a

     fascinating

     city

     with

     a

     rich

     cultural

     and

     historical

     heritage

    ,

     and

     continues

     to

     be

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

     trends

    ,

     including

    :
    


    1

    .

     Increased

     AI

     integration

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     and

     robotics

    ,

     we

     can

     expect

     to

     see

     even

     more

     sophisticated

     applications

     of

     AI

    .

     This

     could

     lead

     to

     breakthrough

    s

     in

     areas

     such

     as

     personalized

     medicine

    ,

     autonomous

     vehicles

    ,

     and

     intelligent

     transportation

     systems

    .
    


    2

    .

     Greater

     use

     of

     AI

     in

     automation

    :

     AI

     is

     increasingly

     being

     used

     to

     automate

     repetitive

     tasks

    ,

     from

     manufacturing

     to

     customer

     service

    ,

     which

     could

     lead

     to

     significant

     job

     displacement

     but

     also

     potential

     job

     creation

     in

     areas

     such

     as

     data

     analysis

     and

     machine

     learning

    .
    


    3

    .

     Greater

     emphasis

     on

     ethical

     AI

    



```python
llm.shutdown()
```
