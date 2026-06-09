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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.17it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:41,  4.94s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.41it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.41it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.41it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.41it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.41it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.41it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.41it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.41it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.41it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.41it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 18.58it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 18.58it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 18.58it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 18.58it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 18.58it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 18.58it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 18.58it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 18.58it/s]

    Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 18.58it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 18.58it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:05<00:01, 18.58it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 27.30it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 27.30it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 27.30it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 27.30it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 27.30it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 27.30it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 27.30it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 27.30it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 27.30it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 27.30it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 27.30it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 27.30it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 38.36it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.16it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.62it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.62it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.62it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.62it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.17it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.17it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.94it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.94it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.94it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.96it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.35it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.35it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.35it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.35it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:01, 30.75it/s]

    Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.02it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.02it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.02it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.02it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.02it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.02it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.00it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.41it/s]

    Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.16it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.16it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.16it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.16it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.16it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.16it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.72it/s]

    Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.74it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 33.76it/s]


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
    Generated text:  James Mitchell, a fourth year nursing student at the University of Texas at Dallas. I am a graduate of the International Medical Corps program and I am currently employed by the Dallas Area Health Education Center (DALHEC). I have always been passionate about health and wanted to use my education to make a difference in the lives of others.
    
    I enjoy working with small groups of patients, to understand how they react to various health issues and develop a plan of care for each one. I also enjoy seeing how a new patient relates to a particular group of patients and to the overall health and well-being of the population.
    
    What is one way that you can
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person, and he is also the leader of the country. On this important day, he is going to give a speech in the United States. The president is going to speak in the country for the first time. He is looking forward to speaking in front of all the people there.
    
    I have never seen him before, and I am sure that I will be the first to see him in the country. I had a great time at the last presidential inauguration, but I have not been to the inauguration before. I hope that I will be able to experience the feeling of being at the inauguration like I have never experienced it before
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. According to the latest population data, the population in 2014 was 2.6 million. The population has been growing steadily since 2014, and the growth rate is 1% per year. If the population is expected to grow by 1% every two years, what will be the population in 2025?
    
    To determine the population in 2025, we need to account for the growth rate over the 2-year period from 2014 to 2024. The population growth rate is 1% per year, but every two years
    ===============================
    Prompt: The future of AI is
    Generated text:  very promising. But, it’s not without some challenges. One of these challenges is the issue of accuracy, which is really important. It’s a huge issue, and it’s not going away any time soon.
    At the moment, most AI systems are not as accurate as they can be. This can happen for several reasons:
    - Overfitting: Overfitting is a situation where the AI model is too detailed and accurate for the dataset it was trained on, and can thus make poor predictions on new data. This is a major issue in the current AI system, and it can be difficult to deal with.
    - Underfitting


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


    Generated text:  [Name] and I am a [occupation] who has been [number of years] in the industry. I am passionate about [reason for passion], and I am always looking for ways to [action or achievement]. I am a [type of person] who is [character trait or quality] and I am always [character trait or quality]. I am [character trait or quality] and I am always [character trait or quality]. I am [character trait or quality] and I am always [character trait or quality]. I am [character trait or quality] and I am always [character trait or quality]. I am [character
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    A. True
    B. False
    A. True
    
    Paris is the capital city of France, located on the Île de la Cité, a small island in the Seine River. It is the largest city in France and the second-largest city in the European Union, with a population of over 2.1 million people. Paris is known for its rich history, art, and culture, and is a major tourist destination. The city is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a popular tourist destination, with millions of visitors each year
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Enhanced privacy and security: As AI systems become more integrated with human intelligence, there will be an increased need for privacy and security measures to protect the data and information that is generated and processed by these systems. This could
    


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
    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am an [X] who has always been a [X] at heart. I'm known for [X], and I strive to [X] daily. I love [X] and I'm [X] in my life. I'm always [X] and I believe in [X]. What's your name? What's your occupation? What's your age? What's your occupation? What's your age? What's your occupation? What's your age? What's your occupation? What's your age? What's your occupation? What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, often referred to as the "City of Love." 
    
    To elaborate:
    - Paris is the largest city in France, with an estimated population of over 2.8 million.
    - It is the capital of France, known for its rich history, art, music, cuisine, and fashion.
    - The city's architecture, particularly its cathedrals and landmarks, is celebrated worldwide for its elegance and beauty.
    - Paris is also renowned for its luxury goods and fashion, making it a global fashion capital.
    - The city is home to many famous landmarks, including the Eiffel Tower and the Louvre Museum. 
    
    This fact
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be marked by significant changes and advancements, and here are some potential trends that are likely to shape the AI landscape:
    
    1. Increased focus on ethical AI: There is a growing awareness of the potential for AI to be used for unethical purposes, such as the misuse of AI to perpetuate discrimination or human rights abuses. As a result, there is increasing focus on developing AI that is designed to be ethical and transparent, and that is used for positive purposes.
    
    2. The rise of AI-powered autonomous vehicles: As autonomous vehicles become more advanced, we are likely to see a significant increase in the use of AI in transportation. This could


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

    Job

     Title

    ]

     at

     [

    Company

     Name

    ].

     I

     have

     a

     passion

     for

     [

    what

     you

     do

     best

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

    how

     you

     can

     improve

     yourself

    ].

     What

    's

     your

     favorite

     hobby

     or

     activity

     to

     do

    ?

     What

     are

     your

     passions

    ?

     What

    's

     something

     you

    've

     always

     wanted

     to

     try

     but

     haven

    't

    ?

     And

     what

    's

     your

     favorite

     book

     or

     movie

     to

     read

    ?

     And

     what

    's

     your

     favorite

     food

    ?

     And

     what

    's

     your

     favorite

     place

     to

     visit

    ?

     And

     what

    's

     your

     favorite

     thing

     to

     do

    ?

     And

     what

    's

     your

     favorite

     thing

     to

     do

    ?

     And

     what

    's

     your

     favorite

     thing

     to

     do

    ?

     And

     what

    's

     your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     bustling

     met

    ropolis

     and

     the

     country

    's

     cultural

     and

     political

     center

    .

     The

     city

     is

     known

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

     and

     the

     Lou

    vre

     Museum

    ,

     as

     well

     as

     its

     rich

     history

     and

     diverse

     population

    .

     Despite

     its

     size

     and

     population

    ,

     Paris

     is

     a

     highly

     walk

    able

     and

     pedestrian

    -friendly

     city

    ,

     with

     many

     well

    -develop

    ed

     neighborhoods

     and

     a

     reputation

     for

     being

     a

     favorite

     place

     for

     locals

     and

     tourists

     to

     visit

    .

     Its

     annual

     F

    ête

     de

     la

     S

    aison

    ,

     which

     celebrates

     the

     changing

     of

     the

     seasons

    ,

     is

     a

     significant

     event

     that

     draws

     visitors

     from

     all

     over

     the

     world

    .

     Paris

     is

     also

     known

     for

     its

     gastr

    onomy

    ,

     which

     includes

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     much

     up

     in

     the

     air

    ,

     but

     there

     are

     several

     trends

     that

     seem

     to

     be

     growing

     and

     becoming

     increasingly

     important

    :
    


    1

    .

     Autonomous

     vehicles

    :

     As

     the

     need

     for

     safe

     and

     efficient

     transportation

     increases

    ,

     autonomous

     vehicles

     are

     becoming

     more

     common

    .

     This

     will

     require

     sophisticated

     AI

     algorithms

     and

     machine

     learning

     to

     navigate

     and

     navigate

     around

     obstacles

    ,

     and

     to

     make

     decisions

     about

     the

     safest

     routes

     for

     vehicles

    .
    


    2

    .

     Smart

     homes

    :

     AI

     is

     also

     playing

     a

     role

     in

     the

     development

     of

     smart

     homes

    .

     Smart

     homes

     are

     becoming

     more

     sophisticated

    ,

     with

     built

    -in

     AI

     that

     can

     manage

     energy

     usage

    ,

     automate

     tasks

    ,

     and

     even

     make

     decisions

     on

     our

     behalf

    .
    


    3

    .

     Healthcare

    :

     AI

     is

     also

     expanding

     its

     reach

     in

    



```python
llm.shutdown()
```
