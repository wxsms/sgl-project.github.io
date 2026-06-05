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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.11it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:18,  2.70it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:18,  2.70it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:18,  2.70it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:18,  2.70it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:18,  2.70it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:18,  2.70it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:18,  2.70it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:18,  2.70it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.37it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.37it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.37it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.37it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.37it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.37it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:05<00:06,  6.37it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:05<00:06,  6.37it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:05<00:06,  6.37it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03, 11.67it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03, 11.67it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03, 11.67it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03, 11.67it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03, 11.67it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03, 11.67it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03, 11.67it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03, 11.67it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 17.04it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 17.04it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 17.04it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 17.04it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 17.04it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 17.04it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 17.04it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 17.04it/s]

    Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 17.04it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 17.04it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 25.40it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 25.40it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 25.40it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 25.40it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 25.40it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 25.40it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 25.40it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 25.40it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 25.40it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 33.02it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 33.02it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 33.02it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 33.02it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 33.02it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 33.02it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 33.02it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 33.02it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 33.02it/s]

    Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 33.02it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 33.02it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 43.77it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 43.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.57 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.54 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.54 GB):   3%|▎         | 2/58 [00:00<00:03, 18.62it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.54 GB):   3%|▎         | 2/58 [00:00<00:03, 18.62it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.53 GB):   3%|▎         | 2/58 [00:00<00:03, 18.62it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.53 GB):   3%|▎         | 2/58 [00:00<00:03, 18.62it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.53 GB):   9%|▊         | 5/58 [00:00<00:02, 19.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.53 GB):   9%|▊         | 5/58 [00:00<00:02, 19.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.52 GB):   9%|▊         | 5/58 [00:00<00:02, 19.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.51 GB):   9%|▊         | 5/58 [00:00<00:02, 19.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.51 GB):   9%|▊         | 5/58 [00:00<00:02, 19.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.51 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.51 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.92it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=73.50 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.50 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.50 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.50 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.85it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.50 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.85it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.49 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.49 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.85it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.49 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.85it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.48 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.85it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.48 GB):  31%|███       | 18/58 [00:00<00:01, 34.68it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.48 GB):  31%|███       | 18/58 [00:00<00:01, 34.68it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=73.48 GB):  31%|███       | 18/58 [00:00<00:01, 34.68it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.46 GB):  31%|███       | 18/58 [00:00<00:01, 34.68it/s]Capturing num tokens (num_tokens=960 avail_mem=73.47 GB):  31%|███       | 18/58 [00:00<00:01, 34.68it/s] Capturing num tokens (num_tokens=896 avail_mem=73.47 GB):  31%|███       | 18/58 [00:00<00:01, 34.68it/s]Capturing num tokens (num_tokens=896 avail_mem=73.47 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=832 avail_mem=73.47 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=768 avail_mem=73.46 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=704 avail_mem=73.46 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=640 avail_mem=73.46 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=576 avail_mem=73.46 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.71it/s]Capturing num tokens (num_tokens=576 avail_mem=73.46 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=512 avail_mem=73.44 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.34it/s]

    Capturing num tokens (num_tokens=480 avail_mem=73.46 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=448 avail_mem=73.46 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=416 avail_mem=73.45 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=384 avail_mem=73.45 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=384 avail_mem=73.45 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.35it/s]Capturing num tokens (num_tokens=352 avail_mem=73.45 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.35it/s]Capturing num tokens (num_tokens=320 avail_mem=73.44 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.35it/s]Capturing num tokens (num_tokens=288 avail_mem=73.44 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.35it/s]Capturing num tokens (num_tokens=256 avail_mem=73.44 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.35it/s]Capturing num tokens (num_tokens=240 avail_mem=73.43 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.35it/s]Capturing num tokens (num_tokens=240 avail_mem=73.43 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=224 avail_mem=73.43 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.61it/s]

    Capturing num tokens (num_tokens=208 avail_mem=73.42 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=192 avail_mem=73.42 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=176 avail_mem=73.42 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=160 avail_mem=73.42 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.61it/s]Capturing num tokens (num_tokens=160 avail_mem=73.42 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.71it/s]Capturing num tokens (num_tokens=144 avail_mem=73.41 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.71it/s]Capturing num tokens (num_tokens=128 avail_mem=73.41 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.71it/s]Capturing num tokens (num_tokens=112 avail_mem=73.41 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.71it/s]Capturing num tokens (num_tokens=96 avail_mem=73.41 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.71it/s] Capturing num tokens (num_tokens=80 avail_mem=73.40 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.71it/s]Capturing num tokens (num_tokens=80 avail_mem=73.40 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.04it/s]Capturing num tokens (num_tokens=64 avail_mem=73.40 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.04it/s]

    Capturing num tokens (num_tokens=48 avail_mem=73.40 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.04it/s]Capturing num tokens (num_tokens=32 avail_mem=73.39 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.04it/s]Capturing num tokens (num_tokens=28 avail_mem=73.39 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.04it/s]Capturing num tokens (num_tokens=24 avail_mem=73.39 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.04it/s]Capturing num tokens (num_tokens=24 avail_mem=73.39 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=20 avail_mem=73.38 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=16 avail_mem=73.38 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=12 avail_mem=73.38 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=8 avail_mem=73.37 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.47it/s] Capturing num tokens (num_tokens=4 avail_mem=73.37 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.47it/s]Capturing num tokens (num_tokens=4 avail_mem=73.37 GB): 100%|██████████| 58/58 [00:01<00:00, 46.88it/s]Capturing num tokens (num_tokens=4 avail_mem=73.37 GB): 100%|██████████| 58/58 [00:01<00:00, 40.25it/s]


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
    Generated text:  Steve. I'm the founder of MyFitnessPal. I'm here to tell you all about your body, how to get in shape and how to lose it. I was born in Finland and I had some incredible life experiences there. I've been in the food business for over 20 years. I've traveled the world and I've lived in all 50 states. I've owned and run my own food truck since I was 19. I've been in a lot of situations where I could have easily gotten in trouble. I never did. That's my secret. I've always believed in sharing and giving out
    ===============================
    Prompt: The president of the United States is
    Generated text:  worth $83 million. How many years would it take to make the president of the United States the oldest person on Earth? The president of the United States currently has a lifespan of 8 years.
    
    If the president is worth $83 million, it would take 83 million years to increase to the age of an 80-year-old person.
    
    However, there is a significant chance that the president will live to be 80, which would be 80 million years after the 8-year lifespan of their current life.
    
    Therefore, it would take approximately 83 million years plus 80 million years,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Which of the following statements is true?
    A. France is the only country with the same name as Paris.
    B. France is the capital of Asia.
    C. France is the largest country in the world by area.
    D. France is the oldest country in Europe.
    Answer:
    
    C
    
    In a high-rise apartment, Person A was unfortunately attacked by Person B. It is known that Person A is 30 years old, while Person B is 20 years old. The legal age for declaring war is 18 years old. Based on the above information, which of the following statements is correct?
    A. Person
    ===============================
    Prompt: The future of AI is
    Generated text:  here
    
    New research has shown that the future of artificial intelligence is here.
    
    The future of AI is here. While it is still very early, the amount of AI that is being used to solve business and societal problems is growing exponentially. In a recent report, AI research firm McKinsey found that 40% of all major challenges facing businesses today are the result of AI. This is the biggest wave of AI innovation that has ever occurred. Since the 1950s, AI has progressed from rudimentary algorithms to sophisticated models capable of understanding and interpreting the world around us.
    
    This is a huge step forward in the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic hub, with a rich history dating back to the Roman Empire and the French Revolution. It is a popular tourist destination, with millions of visitors annually. The city is also known for its cuisine, including French cuisine, and its fashion industry. Paris is a vibrant and dynamic city with a strong sense of French identity and culture. The city is also home to many international organizations and institutions
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced decision-making. This could lead to more personalized and context-aware AI systems that can better understand and respond to the needs of users.
    
    2. Enhanced machine learning capabilities: AI is likely to become even more powerful and capable, with the ability to learn from vast amounts of data and adapt to new situations. This could lead to more efficient and effective AI systems that can handle a wider range of tasks.
    
    3. Greater emphasis on ethical and social implications:
    


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
    Generated text:  [Your Name] and I am a 27-year-old [insert age] who currently works as a [insert occupation] in [insert location]. I am known for my [insert relevant experience or skills] and have always been passionate about [insert something that represents my personality]. I am also a member of the [insert name of a social group or club], and I enjoy [insert something that represents my interests]. If you need any more information about me, please ask and I'll be happy to provide it. I look forward to meeting you! So, what's your name? And who are you? 
    
    Remember, I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich history, beautiful architecture, and famous landmarks such as the Eiffel Tower and Louvre Museum. Paris is also famous for its vibrant culture, including its famous fashion scene and annual celebrity parties. The city is home to many international organizations, including the United Nations and the European Parliament. Paris is a bustling metropolis with a diverse population of people of all races and nationalities. It is a popular tourist destination, with millions of visitors each year.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, with an array of exciting and transformative possibilities shaping our world. Here are some potential trends that could shape the future of AI in the next few decades:
    
    1. Natural Language Processing: As AI continues to advance, we can expect to see further breakthroughs in natural language processing. This will allow machines to understand and respond to human language more accurately and efficiently, which could lead to more effective customer service, improved healthcare outcomes, and even more advanced virtual assistants.
    
    2. Autonomous vehicles: Autonomous vehicles are already being developed and could become a major part of our future. As technology improves, we can expect to see further advancements in autonomous vehicle


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

    ]

     and

     I

     am

     a

     [

    insert

     occupation

     or

     profession

    ].

     I

     am

     a

     [

    insert

     age

     or

     graduation

     year

    ],

     [

    insert

     occupation

    ]

     and

     I

     have

     always

     been

     passionate

     about

     [

    insert

     reason

     or

     hobby

    ].

     I

     enjoy

     [

    insert

     hobby

    ]

     because

     it

     has

     taught

     me

     [

    insert

     point

     in

     life

    ]

     and

     I

     am

     always

     eager

     to

     learn

     more

    .

     I

     have

     a

     keen

     interest

     in

     [

    insert

     field

     of

     study

     or

     hobby

    ],

     and

     I

     am

     always

     looking

     for

     ways

     to

     [

    insert

     point

     in

     life

    ]

     and

     achieve

     my

     goals

    .

     I

     am

     a

     [

    insert

     character

     trait

     or

     personality

    ]

     who

     is

     always

     [

    insert

     an

     action

     or

     statement

    ].

     I

     am

     [

    insert

     age

     or

     education

     level

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     the

     historic

     and

     cultural

     capital

     of

     France

    ,

     is

     renowned

     for

     its

     iconic

     landmarks

    ,

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     the

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

     the

     Palace

     of

     Vers

    ailles

    ,

     as

     well

     as

     its

     vibrant

     arts

     scene

     and

     renowned

     cuisine

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

     including

     the

     Romantic

     era

     and

     the

     French

     Revolution

    ,

     and

     its

     role

     in

     the

     French

     Revolution

     and

     World

     War

     II

    .

     Today

    ,

     Paris

     is

     a

     globally

     recognized

     cultural

    ,

     economic

    ,

     and

     political

     center

    ,

     with

     a

     population

     of

     over

     

    1

    1

     million

     people

     and

     a

     rich

     cultural

     heritage

    .

     The

     city

     continues

     to

     be

     a

     major

     center

     of

     global

     affairs

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     an

     increasing

     focus

     on

     privacy

    ,

     ethics

    ,

     and

     transparency

    ,

     as

     well

     as

     a

     continued

     evolution

     towards

     more

     advanced

     technologies

     such

     as

     machine

     learning

     and

     natural

     language

     processing

    .

     AI

     may

     also

     become

     more

     integrated

     with

     human

     behavior

     and

     emotions

    ,

     leading

     to

     a

     more

     human

    -like

     experience

     for

     machines

    .
    


    AI

     is

     also

     likely

     to

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     from

     the

     devices

     we

     use

     to

     the

     services

     we

     rely

     on

    .

     This

     could

     lead

     to

     a

     more

     connected

     world

     where

     machines

     are

     more

     integrated

     into

     the

     fabric

     of

     our

     lives

    .
    


    Additionally

    ,

     AI

     may

     become

     more

     dependent

     on

     human

     oversight

     and

     feedback

    ,

     which

     could

     lead

     to

     a

     more

     complex

     and

     nuanced

     understanding

     of

     AI

     systems

    



```python
llm.shutdown()
```
