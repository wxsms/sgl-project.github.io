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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:07,  4.34s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.44it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.48it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 22.61it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 22.61it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 22.61it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 22.61it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 22.61it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 22.61it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 22.61it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 22.61it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 22.61it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 22.61it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 30.78it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 30.78it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 30.78it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.78it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.33it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.33it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.33it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.33it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.68it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.68it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.68it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.70 GB):  31%|███       | 18/58 [00:00<00:01, 36.68it/s]Capturing num tokens (num_tokens=960 avail_mem=71.71 GB):  31%|███       | 18/58 [00:00<00:01, 36.68it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  31%|███       | 18/58 [00:00<00:01, 36.68it/s]Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=832 avail_mem=71.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=640 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=576 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=576 avail_mem=71.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.61it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.61it/s]Capturing num tokens (num_tokens=480 avail_mem=71.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.61it/s]Capturing num tokens (num_tokens=448 avail_mem=71.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.61it/s]Capturing num tokens (num_tokens=416 avail_mem=71.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.61it/s]

    Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.61it/s]Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=352 avail_mem=71.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=320 avail_mem=71.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=288 avail_mem=71.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=256 avail_mem=71.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 44.07it/s]Capturing num tokens (num_tokens=224 avail_mem=71.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 44.07it/s]Capturing num tokens (num_tokens=208 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=192 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.07it/s]

    Capturing num tokens (num_tokens=176 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=144 avail_mem=71.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=128 avail_mem=71.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=112 avail_mem=71.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=96 avail_mem=71.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.17it/s] Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=64 avail_mem=71.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=48 avail_mem=71.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=32 avail_mem=71.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.05it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=24 avail_mem=71.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=24 avail_mem=71.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.56it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.56it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.56it/s]Capturing num tokens (num_tokens=12 avail_mem=71.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.56it/s]Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.56it/s] Capturing num tokens (num_tokens=4 avail_mem=71.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.56it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 45.40it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 40.35it/s]


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
    Generated text:  Alex. I'm a student at a university in Germany. My hobby is to play the guitar. Since I can't go to the music school, I have to take piano lessons. I love music and I want to make a music video. It's not for a big event, just to record myself playing and to show my talent. What should I do? I don't want to waste time taking piano lessons, I want to make a music video but the problems I have with taking piano lessons is that I don't know how to play any guitar, my teacher doesn't teach me how to play guitar, and the most important thing
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. The president has 8 bases to choose from and the government wants to have at least 1 base. The president wants to have at most 4 bases with a particular candidate. What is the probability that the president will have at least one base with the particular candidate and no bases with the other candidate?
    
    To determine the probability that the president will have at least one base with the particular candidate and no bases with the other candidate, we need to follow these steps:
    
    1. **Identify the total number of bases and the candidate preferences:**
       - The president has 8 bases to
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. What is the location of Paris?
    
    a. Europe
    
    b. Asia
    
    c. North America
    
    d. South America
    
    e. Africa
    
    e. Africa
    
    Paris is located in France, which is situated in Europe. Europe is a continent, so the correct answer is:
    
    a. Europe
    
    The location of Paris is on the European continent. 
    
    Options b, c, d, and e are not correct because:
    b. Asia is a continent, but Paris is located in Europe.
    c. North America is a continent, but Paris is located in Europe.
    d. South America is a continent, but Paris is located
    ===============================
    Prompt: The future of AI is
    Generated text:  in the future. What is the role of AI in the modern world?
    
    AI has revolutionized many aspects of modern life, including healthcare, transportation, and education. It has improved the efficiency and accuracy of processes, and has made it possible for people to work from home and remote work. In addition, AI has also transformed the way we communicate and access information. It has made it easier for people to learn new skills and stay up-to-date with the latest developments.
    
    In addition, AI is becoming more and more integrated into our daily lives, from the devices we use to the services we access. AI-powered assistants, such as Siri and


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of many famous French artists, writers, and composers. Paris is a bustling metropolis with a rich cultural heritage and a diverse population of over 2 million people. The city is known for its romantic atmosphere and is a popular tourist destination. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. Its status as the capital
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries. This could lead to increased efficiency and productivity, but it could also lead to job displacement for some workers.
    
    2. Enhanced privacy and security: As AI technology becomes more advanced, we can expect to see more privacy and security concerns. This could lead to increased regulation and enforcement of privacy laws
    


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
    Generated text:  [Name] and I'm a [type of character] with [age]. I'm originally from [location], but I've moved to [new location] recently. I love [job or hobby]. I'm [job or hobby] at heart, but I'm also [ability or interest]. I enjoy [reason for being]. If you're interested in learning more about me, please let me know. [Name] always ready to learn and grow! [Name] am looking forward to meeting you! [Name] - [Type of character] - [Age]
    
    This is a great introduction! Could you maybe add some background information
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France by population and has a rich history, with several UNESCO World Heritage sites. Paris has a strong cultural and artistic heritage, with numerous museums, theaters, and other cultural institutions. The city is also known for its fashion, art, and wine industries. It is a popular tourist destination, with a bustling street food market, beautiful parks, and many historic landmarks. Paris is often referred to as the "City of Light" and is a major center for finance, media, and entertainment. It is the world's most populous city, home to millions of people and a major economic and political center.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be one of rapid progress and innovation, driven by advancements in both hardware and software. Here are some potential future trends in AI:
    
    1. Increased focus on ethical considerations: AI systems will need to be designed and programmed with ethical considerations in mind. This includes things like bias in algorithms, privacy concerns, and transparency about how data is used.
    
    2. Integration of AI with other technologies: AI will become more integrated with other technologies, such as IoT (Internet of Things) and blockchain, to create more efficient and reliable systems.
    
    3. Greater emphasis on machine learning: Machine learning will become more prominent as AI systems become more complex and


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

    ].

     I

     am

     [

    Age

    ]

     years

     old

    ,

     and

     I

     am

     [

    Gender

    ]

     [

    Gender

     Identity

    ].

     I

     am

     from

     [

    Location

    ],

     and

     I

     have

     [

    Favorite

     Activity

    ]

     [

    Favorite

     Food

    ].

     I

     am

     [

    Current

     Location

    ]

     and

     I

     am

     [

    Current

     Mood

    ].

     I

     have

     [

    Number

     of

     Friends

    ]

     friends

    ,

     and

     I

     am

     [

    Number

     of

     Projects

    ]

     projects

    .

     I

     am

     [

    Current

     Goal

    ]

     [

    Current

     Goal

    ].

     I

     am

     [

    Current

     Profession

    ].

     What

     is

     your

     name

    ?

     [

    Tell

     the

     reader

     something

     interesting

     about

     yourself

    ].

     My

     name

     is

    ...

     [

    Name

    ].

     I

     am

     a

     [

    Gender

    ]

     [

    Gender

     Identity

    ].

     I

     am

     from

     [

    Location

    ],

     and

     I

     have

     been

     living

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     that

     was

     the

     center

     of

     the

     ancient

     Roman

     world

     and

     the

     birth

    place

     of

     the

     French

     Revolution

    .

     It

     is

     also

     one

     of

     the

     oldest

     continuously

     inhabited

     cities

     in

     the

     world

     and

     has

     been

     home

     to

     many

     famous

     artists

    ,

     writers

    ,

     and

     philosophers

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     is

     known

     for

     its

     famous

     landmarks

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

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     home

     to

     many

     important

     museums

    ,

     including

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     the

     Lou

    vre

    ,

     and

     the

     Mus

    ée

     National

     d

    '

    Art

     Moder

    ne

    .

     Overall

    ,

     Paris

     is

     a

     vibrant

     and

     culturally

     rich

     city

     that

     is

     well

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     number

     of

     different

     trends

    ,

     including

    :
    


    1

    .

     Increased

     development

     of

     AI

     ethics

     and

     ethical

     standards

    :

     As

     more

     companies

     and

     governments

     develop

     AI

     systems

    ,

     there

     will

     be

     greater

     pressure

     to

     ensure

     that

     they

     are

     developed

     and

     used

     in

     ways

     that

     are

     ethical

     and

     beneficial

     for

     society

     as

     a

     whole

    .

     This

     may

     lead

     to

     more

     stringent

     regulations

     and

     standards

     to

     govern

     AI

     development

     and

     use

    .
    


    2

    .

     Rise

     of

     AI

    -powered

     "

    super

    int

    elligent

    "

     AI

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     there

     is

     a

     possibility

     that

     AI

     may

     become

     more

     capable

     than

     humans

     in

     some

     areas

    ,

     such

     as

     creating

     art

    ,

     music

    ,

     or

     literature

    .

     This

     could

     lead

     to

     a

     future

     where

    



```python
llm.shutdown()
```
