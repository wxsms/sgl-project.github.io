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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.91it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.67s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:25,  4.67s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.81it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.81it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.81it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.54it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 21.52it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.74it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.03it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.03it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.03it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.03it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.29it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  21%|██        | 12/58 [00:00<00:01, 28.82it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.82it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.82it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.82it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  21%|██        | 12/58 [00:00<00:01, 28.82it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.29it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.29it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.29it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.29it/s] Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.34it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.34it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.34it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.34it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.34it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.34it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.31it/s]

    Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.31it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.21it/s]

    Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.05it/s] Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  81%|████████  | 47/58 [00:01<00:00, 44.19it/s]

    Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  81%|████████  | 47/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  81%|████████  | 47/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.85it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.85it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.85it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.85it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.85it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.85it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 39.15it/s]


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
    Generated text:  Jimmy, and I live in the city centre. I'm a young man who goes to school and has a dream of becoming a doctor. Recently, my parents gave me a gift in the form of a red envelope. Inside the envelope was a ticket for the World Cup football match. My parents also told me about the ticket and how it was a lottery. I told my friends and asked them to buy tickets too. I also told my classmates to encourage them to buy tickets too. They all bought tickets to the World Cup football match. I bought a lottery ticket. I even took my friends to buy lottery tickets. I was very
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by a 4-person committee, and each member of this committee serves for 6 months. The vice president is considered the "veteran" of the presidency, and his term of office is different from that of the other committee members. If each committee member is represented by a different person from the current president to the vice president, how many different ways can the vice president be chosen? To determine the number of different ways to choose the vice president, we start by noting that the president is represented by a 4-person committee, and each member of this committee serves for 6 months. Therefore, there are 4 different
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and the capital of England is London. France is located in Europe and England in the east of Europe. The reason for the difference in the capital is ____.
    
    A: Shape of the land
    
    B: Level of the land
    
    C: Size of the land
    
    D: Climate of the land To determine the correct answer, let's analyze the given information step by step:
    
    1. **Identify the given facts:**
       - The capital of France is Paris.
       - The capital of England is London.
       - France is located in Europe.
       - England is located in the east of Europe.
    
    2. **Under
    ===============================
    Prompt: The future of AI is
    Generated text:  ( ) and ( ), making people's lives more convenient and colorful.
    A. Fast
    B. Smart
    C. Beautiful
    D. Fast and smart
    Answer:
    
    BD
    
    Which of the following accurately describes the conditions for a startup's success?
    A. Having a solid business plan and a solid business team
    B. Possessing advanced technology and industry knowledge
    C. Having a strong management team
    D. The team's management level should be as high as possible
    Answer:
    
    AB
    
    Which of the following descriptions about the composition of the Shanghai Youth Science and Technology Museum are correct?
    A. The Shanghai Science and Technology Museum has


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a brief description of your job or experience here]. I enjoy [insert a brief description of your hobbies or interests here]. What's your favorite hobby or activity? I love [insert a brief description of your favorite activity here]. What's your favorite book or movie? I love [insert a brief description of your favorite book or movie here]. What's your favorite color? I love [insert a brief description of your favorite color here
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and a vibrant culture. It is located in the south of France and is the largest city in the country. Paris is known for its beautiful architecture, world-renowned museums, and annual festivals such as the Eiffel Tower and the Louvre. The city is also home to many famous landmarks such as Notre-Dame Cathedral and the Palace of Versailles. Paris is a major cultural and economic center in France and a popular tourist destination. It is also home to many important institutions such as the French Academy of Sciences and the French National Library. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more sophisticated, it is likely to be used in more areas of healthcare, including diagnosis, treatment, and patient care.
    
    2. AI in manufacturing: AI is already being used in manufacturing to improve efficiency and reduce costs. As AI technology continues to improve, it is likely to be used in more areas of manufacturing
    


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
    Generated text:  Sarah and I am a freelance writer. I work in the tech industry and have over a decade of experience in writing and editing software. My specialties include technical writing, web development, and data visualization. I am a big fan of using simple language to explain complex concepts and am always eager to learn new things. I enjoy collaborating with other writers and being in a creative environment. I have a passion for providing a user-friendly writing experience for my clients. I hope you can meet me! ✨✨✨
    
    Sarah, a freelance writer with a decade of experience in technology, specializing in technical writing, web development, and data visualization. A
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Ville d'Or" or simply "La République." It is a major city and the largest metropolitan area in Europe, with a population of about 1. 348 million people. Paris is renowned for its artistic culture, historic architecture, and cuisine, and is a popular tourist destination. It is also a center for political and economic activities. The French capital is a global financial center and is known for its fashion, music, and literature scenes. Paris is home to the Eiffel Tower, the Louvre Museum, and the Arc de Triomphe, among other iconic landmarks
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and variable, but here are some possible trends that could impact the field in the coming years:
    
    1. Increased use of AI in healthcare: With advancements in AI, we may see a greater use of AI in healthcare, such as improved diagnosis and treatment of diseases, personalized medicine, and predictive analytics for patient care.
    
    2. AI in transportation: AI may play a bigger role in transportation, particularly in reducing traffic congestion and emissions by optimizing traffic flow and improving route routing.
    
    3. AI in entertainment: AI is already being used in the gaming industry, but it may also have a significant impact on the entertainment industry. For example, AI


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

     am

     [

    Age

    ].

     I

     am

     a

     [

    Occup

    ation

    ]

     who

     enjoys

     [

    Favorite

     Activity

    ].

     I

     love

     [

    X

    -factor

    ]

     because

     [

    X

    -factor

     reason

    ].

     I

     am

     a

     [

    General

     Summary

    ],

     and

     I

     am

     excited

     to

     meet

     you

    .


    I

     look

     forward

     to

     meeting

     you

     and

     learning

     about

     you

    .

     What

     is

     the

     character

    's

     name

    ,

     occupation

    ,

     and

     the

     specific

     activity

     they

     enjoy

    ?


    Hi

    ,

     I

    'm

     [

    Name

    ]

     from

     [

    Your

     Town

    ].

     I

    'm

     a

     [

    Occup

    ation

    ]

     who

     enjoys

     [

    Favorite

     Activity

    ].

     I

     love

     [

    X

    -factor

    ]

     because

     [

    X

    -factor

     reason

    ].

     I

     am

     a

     [

    General

     Summary

    ],

     and

     I

     am

     excited

     to

     meet

     you

    .


    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     city

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     vibrant

     culture

    .

     It

     is

     home

     to

     the

     iconic

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

     numerous

     museums

     and

     galleries

    ,

     as

     well

     as

     its

     famous

     fashion

     district

    ,

     the

     Se

    ine

     River

    ,

     and

     numerous

     parks

     and

     gardens

    .

     Paris

     is

     a

     bustling

     city

     with

     a

     diverse

     population

     and

     is

     an

     important

     economic

     center

     in

     Europe

    .

     Its

     status

     as

     a

     global

     cultural

     and

     political

     hub

     has

     made

     it

     a

     popular

     destination

     for

     tourists

     and

     residents

     alike

    .

     
    


    Given

     the

     above

     information

    ,

     how

     would

     you

     describe

     the

     location

     of

     Paris

     in

     relation

     to

     its

     geographical

     location

     in

     Europe

    ?


    The

     location

     of

     Paris

     in

     relation

     to

     its

    
    
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

     that

     will

     continue

     to

     influence

     the

     development

     and

     evolution

     of

     artificial

     intelligence

     systems

    .

     Here

     are

     some

     of

     the

     most

     notable

     trends

     that

     are

     expected

     to

     shape

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     **

    Increased

     Focus

     on

     Explain

    able

     AI

     (

    X

    AI

    )**

    :

     The

     rise

     of

     machine

     learning

     models

     that

     can

     generate

     human

    -like

     explanations

     for

     their

     predictions

     is

     likely

     to

     become

     more

     common

    .

     This

     trend

     is

     driven

     by

     concerns

     about

     transparency

     and

     accountability

     in

     AI

     systems

    ,

     as

     well

     as

     the

     need

     to

     ensure

     that

     AI

     decisions

     are

     understandable

     and

     not

     just

     deterministic

    .
    


    2

    .

     **

    Enh

    anced

     Privacy

     and

     Security

    **:

     As

     AI

     systems

     become

     more

     integrated

     into

     everyday

     life

    ,

     there

    



```python
llm.shutdown()
```
