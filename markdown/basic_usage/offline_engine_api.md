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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.96it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.95it/s]


    2026-05-16 00:42:55,273 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 00:42:55] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:07<07:33,  7.95s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:07<07:33,  7.95s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:07<07:33,  7.95s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:08<07:33,  7.95s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:08<07:33,  7.95s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:08<01:03,  1.21s/it]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:08<01:03,  1.21s/it]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:08<01:03,  1.21s/it]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:08<01:03,  1.21s/it]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:08<01:03,  1.21s/it]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:08<01:03,  1.21s/it]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:08<01:03,  1.21s/it]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:08<01:03,  1.21s/it]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:18,  2.52it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:18,  2.52it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:08<00:18,  2.52it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:08<00:18,  2.52it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:08<00:18,  2.52it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:08<00:18,  2.52it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:08<00:18,  2.52it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:08<00:18,  2.52it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:08<00:18,  2.52it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:08<00:18,  2.52it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:08<00:06,  5.48it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:08<00:06,  5.48it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:08<00:06,  5.48it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:08<00:06,  5.48it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:08<00:06,  5.48it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:08<00:06,  5.48it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:08<00:06,  5.48it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:08<00:06,  5.48it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:08<00:06,  5.48it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:08<00:06,  5.48it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:08<00:02,  9.34it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:08<00:02,  9.34it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:08<00:02,  9.34it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:08<00:02,  9.34it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:08<00:02,  9.34it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:08<00:02,  9.34it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:08<00:02,  9.34it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:08<00:02,  9.34it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:08<00:02,  9.34it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:08<00:02,  9.34it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:08<00:02,  9.34it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:08<00:01, 14.88it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:08<00:01, 14.88it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:08<00:01, 14.88it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:08<00:01, 14.88it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:08<00:01, 14.88it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:08<00:01, 14.88it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:08<00:01, 14.88it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:08<00:01, 14.88it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:08<00:01, 14.88it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:08<00:01, 14.88it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:08<00:00, 20.88it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:08<00:00, 20.88it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:08<00:00, 20.88it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:08<00:00, 20.88it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:08<00:00, 20.88it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:08<00:00, 20.88it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:08<00:00, 20.88it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:08<00:00, 20.88it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:08<00:00, 20.88it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:08<00:00, 20.88it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.69it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.49 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.46 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.46 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.45 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.45 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.45 GB):   9%|▊         | 5/58 [00:00<00:02, 20.49it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.45 GB):   9%|▊         | 5/58 [00:00<00:02, 20.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.44 GB):   9%|▊         | 5/58 [00:00<00:02, 20.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.43 GB):   9%|▊         | 5/58 [00:00<00:02, 20.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.43 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.43 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.56it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.43 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.56it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.42 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.56it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.42 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.42 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.18it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.42 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.18it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.42 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.41 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.41 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.41 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.72it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=73.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.73it/s]Capturing num tokens (num_tokens=960 avail_mem=73.97 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.73it/s] Capturing num tokens (num_tokens=896 avail_mem=73.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.73it/s]Capturing num tokens (num_tokens=832 avail_mem=73.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.73it/s]Capturing num tokens (num_tokens=768 avail_mem=73.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.73it/s]Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.73it/s]Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=576 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=512 avail_mem=73.93 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.66it/s]

    Capturing num tokens (num_tokens=480 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.66it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.07it/s]Capturing num tokens (num_tokens=416 avail_mem=73.95 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.07it/s]Capturing num tokens (num_tokens=384 avail_mem=73.94 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.07it/s]Capturing num tokens (num_tokens=352 avail_mem=73.94 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.07it/s]Capturing num tokens (num_tokens=320 avail_mem=73.93 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.07it/s]Capturing num tokens (num_tokens=288 avail_mem=73.93 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.07it/s]Capturing num tokens (num_tokens=288 avail_mem=73.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=256 avail_mem=73.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=240 avail_mem=73.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=224 avail_mem=73.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.27it/s]

    Capturing num tokens (num_tokens=208 avail_mem=73.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=192 avail_mem=73.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.27it/s]Capturing num tokens (num_tokens=192 avail_mem=73.92 GB):  71%|███████   | 41/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  71%|███████   | 41/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=160 avail_mem=73.91 GB):  71%|███████   | 41/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=144 avail_mem=73.91 GB):  71%|███████   | 41/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=128 avail_mem=73.90 GB):  71%|███████   | 41/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=112 avail_mem=73.90 GB):  71%|███████   | 41/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=112 avail_mem=73.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.19it/s]Capturing num tokens (num_tokens=96 avail_mem=73.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.19it/s] Capturing num tokens (num_tokens=80 avail_mem=73.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.19it/s]Capturing num tokens (num_tokens=64 avail_mem=73.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.19it/s]

    Capturing num tokens (num_tokens=48 avail_mem=73.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.19it/s]Capturing num tokens (num_tokens=32 avail_mem=73.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.19it/s]Capturing num tokens (num_tokens=32 avail_mem=73.89 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=24 avail_mem=73.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=20 avail_mem=73.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=16 avail_mem=73.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=12 avail_mem=73.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.34it/s]Capturing num tokens (num_tokens=12 avail_mem=73.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=8 avail_mem=73.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.68it/s] Capturing num tokens (num_tokens=4 avail_mem=73.86 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.68it/s]Capturing num tokens (num_tokens=4 avail_mem=73.86 GB): 100%|██████████| 58/58 [00:01<00:00, 39.17it/s]


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
    Generated text:  Chrissie Vock, a 3rd degree master of jewellery. I come from a family of designers, and have been a jewellery designer for over 35 years. My main focus in my life is on creating stunning, innovative, and highly versatile jewellery. My work is inspired by the beauty of the human form, and I combine that with my passion for design and storytelling. I hold a Master of Arts in Design and an Associate of Arts in the Fashion and Product design from the University of Glasgow.
    What is your background and what is your main focus in your career?
    I have always loved jewelry and fashion design since I was
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to go to war with another country. To make his decision, he uses a unique scoring system where the score is calculated as follows: if the president wins, he gets 3 points; if he loses, he gets 1 point. He has a total of 30 points to score. If the president wins, his score will double. Given that the president has already lost 5 of the 10 games he has played, how many games does he need to win to reach 40 points?
    To determine how many games the president needs to win to reach a total of 40 points,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the capital of China is Beijing, and the capital of Italy is Rome. Which of the following statements is incorrect? 
    A. France is a country with a long history and complex cultural background, so Paris is a city with a long and rich history.
    B. Rome is the capital of the Italian country, so Rome is a city with a long and rich history.
    C. Beijing is the capital of the Chinese country, so Beijing is a city with a long and rich history.
    D. Paris is the capital of France, so Paris is a city with a long and rich history.
    Answer:
    
    C
    
    It is required that
    ===============================
    Prompt: The future of AI is
    Generated text:  here now, but it’s not here yet.
    
    AI is a constantly evolving field that requires a lot of data, computing power, and time. As we continue to build and deploy AI technology, we will need to constantly adapt and improve our approach to ensuring that AI is used for the benefit of society.
    
    What is AI?
    
    AI is a broad category of computer science that focuses on building intelligent machines that can think, learn, and make decisions. It is used in a variety of industries, including healthcare, finance, and education, to automate and improve processes and outcomes.
    
    One of the key characteristics of AI is its ability to learn from data


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, interesting fact about yourself]. And what's your favorite hobby? I love [insert a short, interesting fact about your hobby]. And what's your favorite book? I love [insert a short, interesting fact about your favorite book]. And what's your favorite color? I love [insert a short, interesting fact about your favorite color]. And what's your favorite food? I love [insert a short, interesting fact
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and art scene. Paris is a major cultural and economic center in Europe and is home to many world-renowned museums, theaters, and landmarks. It is also a popular tourist destination, with millions of visitors annually. The city is known for its vibrant nightlife, fashion, and food scene, and is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is likely to become more prevalent in various industries, including manufacturing, healthcare, transportation, and finance. Automation will likely lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. AI ethics and privacy concerns: As AI becomes more advanced, there will be increasing concerns about its ethical implications and potential privacy violations. This will likely lead to more regulations and standards being put
    


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
    Generated text:  [insert character's name]. I am a [insert occupation] with a passion for [insert hobby or interest]. I am [insert the number of years since graduating from [insert graduation school] and my current location]. [insert character's occupation] and I am here to bring my unique skills to the table for anyone interested. Whether you're in need of a helping hand or just a fun conversation, I am always happy to assist. What brings you to this conversation? [insert any relevant information or details you would like to include in your introduction]. [insert character's name] is [insert name of the fictional character]. It's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its rich history, beautiful architecture, and lively culture. Paris is home to iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral, as well as a diverse range of museums, theaters, and restaurants. The city is also known for its cosmopolitan atmosphere and its role as the center of France's political and cultural life. The French government's decision to move the capital to Paris in 1940 has had a significant impact on the city's development and identity. However, the city has struggled with issues such as traffic congestion and aging infrastructure. Despite these challenges, Paris continues to be
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, with many possibilities for growth and improvement in the years to come. Some potential trends include:
    
    1. Increased AI efficiency: As AI continues to get more powerful, there is a higher likelihood that it will become even more efficient and effective in its tasks. This could lead to a more streamlined and productive world.
    
    2. AI integration with other technologies: As AI technology becomes more advanced, it is likely to become more integrated with other technologies, such as quantum computing and machine learning. This could lead to new possibilities for AI applications, such as more powerful autonomous vehicles or smarter medical treatments.
    
    3. AI ethics and privacy concerns: With


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

     professional

     [

    job

     title

    ].

     I

    'm

     excited

     to

     meet

     you

     and

     let

    's

     talk

     about

     what

     we

     can

     do

     together

    !

     Let

    's

     make

     this

     conversation

     productive

     and

     helpful

    .

     How

     can

     I

     assist

     you

     today

    ?

     [

    Name

    ]

     can

     help

     you

     find

     the

     most

     cost

    -effective

     options

     for

     your

     business

     needs

    .

     [

    Name

    ]

     has

     helped

     [

    company

     name

    ]

     improve

     [

    company

    's

     product

    /service

    ].

     Are

     you

     tired

     of

     repetitive

     work

    ,

     stuck

     in

     a

     rut

    ,

     and

     frustrated

     by

     your

     work

    ?

     If

     you

    're

     ready

     to

     turn

     your

     work

     into

     success

    ,

     [

    Name

    ]

     can

     help

     you

     break

     through

    .

     How

     can

     I

     assist

     you

     today

    ?


    [

    Name

    ]

     can

     help

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    I

    'll

     provide

     you

     with

     the

     answer

     before

     you

     go

    .


    Paris

    


    Paris

     is

     the

     capital

     city

     of

     France

    .

     It

     is

     the

     largest

     city

     in

     France

     by

     population

     and

     is

     the

     second

    -largest

     city

     in

     the

     European

     Union

     by

     population

    ,

     after

     Brussels

    .

     The

     city

     is

     home

     to

     numerous

     museums

    ,

     art

     galleries

    ,

     and

     museums

    ,

     and

     is

     known

     for

     its

     vibrant

     street

     life

    ,

     art

     museums

    ,

     and

     fashion

    .

     Paris

     is

     also

     a

     cultural

     hub

     with

     many

     theaters

    ,

     cafes

    ,

     and

     theaters

     as

     well

     as

     opera

     houses

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     a

     major

     economic

     and

     financial

     center

     in

     Europe

    .

     Paris

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     Rome

    ,

     and

     it

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     and

     it

    's

     likely

     that

     we

     will

     see

     many

     new

     trends

     and

     developments

     in

     the

     coming

     years

    .

     Here

     are

     some

     potential

     areas

     of

     growth

     and

     development

    :
    


    1

    .

     AI

     Ethics

     and

     Responsibility

    :

     With

     the

     rise

     of

     automation

     and

     artificial

     intelligence

    ,

     the

     ethical

     implications

     of

     AI

     are

     becoming

     more

     pressing

    .

     We

     will

     see

     a

     greater

     focus

     on

     responsible

     AI

    ,

     with

     more

     consideration

     given

     to

     issues

     such

     as

     bias

    ,

     transparency

    ,

     and

     accountability

    .
    


    2

    .

     AI

     Personal

    ization

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     we

     may

     see

     an

     increase

     in

     personal

    ization

    ,

     where

     machines

     learn

     to

     better

     understand

     and

     adapt

     to

     individual

     users

    .

     This

     could

     lead

     to

     more

     personalized

     experiences

     for

     users

    ,

     including

     more

     targeted

    



```python
llm.shutdown()
```
