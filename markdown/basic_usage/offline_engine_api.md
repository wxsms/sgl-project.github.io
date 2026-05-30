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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.60it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:37,  4.87s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:37,  4.87s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:37,  4.87s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:37,  4.87s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:37,  4.87s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.12it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.12it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.12it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.12it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.12it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.12it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.12it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.12it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.12it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.12it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 20.96it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 20.96it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 20.96it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 20.96it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 20.96it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 20.96it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 20.96it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 20.96it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 20.96it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 20.96it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 20.96it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 29.62it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 29.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.78it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.78it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.68it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.68it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.68it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.68it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 30.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 30.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 30.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 30.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 30.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 30.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.55it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.55it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.55it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.55it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.54it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.54it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.54it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.54it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.54it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.54it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.69it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.69it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.69it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.69it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.69it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.69it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.58it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.58it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.58it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.58it/s]Capturing num tokens (num_tokens=288 avail_mem=76.66 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.58it/s]Capturing num tokens (num_tokens=256 avail_mem=76.66 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.58it/s]Capturing num tokens (num_tokens=256 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=240 avail_mem=76.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=224 avail_mem=76.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.02it/s]

    Capturing num tokens (num_tokens=208 avail_mem=76.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=192 avail_mem=76.56 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=176 avail_mem=76.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.02it/s]Capturing num tokens (num_tokens=176 avail_mem=76.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.45it/s]Capturing num tokens (num_tokens=160 avail_mem=76.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.45it/s]Capturing num tokens (num_tokens=144 avail_mem=76.14 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.45it/s]Capturing num tokens (num_tokens=128 avail_mem=76.01 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.45it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.45it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.45it/s] Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  81%|████████  | 47/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 40.74it/s]

    Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.88it/s] Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.37it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.37it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 38.59it/s]


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
    Generated text:  Nicole. I'm a senior at the University of Florida, where I'm pursuing a major in music. I also have a minor in music production. Currently, I'm a performance assistant at the University of Florida in the Vocal Performance and Music Therapy programs. I'm the Music Therapy Manager for the Vocal Performance, and I also manage the University of Florida's music therapy program. I also perform in the band, the Windy City Wind Band. I play the alto flute. I also play the saxophone. I am a member of the university's student orchestra. I currently serve on the music education committee for the University of Florida.
    I
    ===============================
    Prompt: The president of the United States is
    Generated text:  not elected by the citizens of the United States. Where does this statement come from? I believe that the statement is not accurate. The president of the United States is elected by the citizens of the United States, not by the citizens of another country. The United States is a federal republic, which means that the government is divided into three branches - the executive, legislative, and judicial branches - and the president is appointed by the president of the United States from among their own chosen candidates.
    The president's term of office is for four years, during which time they must complete their term by voting in the Electoral College and selecting a vice president and
    ===============================
    Prompt: The capital of France is
    Generated text: : ____
    A. Paris
    B. London
    C. Berlin
    D. Tokyo
    Answer:
    
    A
    
    After an earthquake, the maximum speed at which the police can conduct an investigation is ____
    A. 10 km/h
    B. 20 km/h
    C. 30 km/h
    D. 40 km/h
    Answer:
    
    B
    
    The Yangtze River Delta region is one of the world's four major river Delta regions, primarily located in _______.
    A. Jiangsu, Zhejiang, and Shanghai
    B. Jiangsu, Shanghai, and Anhui
    C.
    ===============================
    Prompt: The future of AI is
    Generated text:  emerging as an essential technology driving innovation across industries and sectors. How can AI be optimized to ensure a sustainable future for both humans and the environment?
    
    One approach to optimize AI for a sustainable future is to incorporate ethical considerations into the development process. This means ensuring that AI systems are designed and implemented in a way that takes into account the impact on the environment and society as a whole. For example, AI systems that generate or process data should be designed to minimize the use of resources and the generation of waste. This could involve reducing the amount of data that is generated, using energy-efficient algorithms, and incorporating measures to prevent data breaches and other


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. Let's chat! [Name] [Company Name] [Company Address] [Company Phone Number] [Company Email] [Company Website] [Company LinkedIn Profile] [Company Twitter Profile] [Company Facebook Profile] [Company Instagram Profile] [Company Github Profile] [Company LinkedIn Profile] [Company Twitter Profile] [Company Facebook Profile] [Company Instagram Profile] [Company Github Profile] [Company LinkedIn Profile] [Company Twitter Profile] [Company Facebook Profile] [Company Instagram Profile
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination and a significant cultural hub for Europe. The city is known for its rich history, art, and cuisine, and is a major hub for international business and diplomacy. It is also home to the French Parliament and the French Parliament building. Paris is a vibrant and dynamic city that is a major cultural and economic center in France. It is also a popular tourist destination and a significant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI will continue to automate many tasks, from manufacturing and transportation to customer service and healthcare. This will lead to increased efficiency and productivity, but it will also create new jobs and require new skills.
    
    2. Enhanced human interaction: AI will continue to enhance human interaction, making it easier to communicate and collaborate with others. This will lead to more social and emotional intelligence, as well as increased empathy and understanding.
    
    3. AI will become more integrated with the physical world: AI will become more integrated with the physical world, such as through the use of sensors and cameras. This
    


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
    Generated text:  [Name] and I'm a [age] year old girl with [occupation]. I've always been fascinated by the outdoors and have always wanted to experience nature firsthand. I've always enjoyed hiking, camping, and exploring the great outdoors. I've been active since I was a little kid, and I've always dreamed of going on a trip to the mountains. I've always wanted to learn how to take care of myself and take care of the earth. I love animals and I'm always ready to help them. I'm going to be a [career] and [education] with [major]. I'm excited to start my adventure
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city located in the south of the country. It is a cosmopolitan and historic city with many historical sites and landmarks, including the Eiffel Tower. French cuisine is also popular, and it has been praised for its unique mix of regional and international flavors. The city is known for its vibrant nightlife and has a long history of art and cultural expression. Paris is also home to many renowned museums, including the Louvre and the Musée d'Orsay. The city is a major center for culture, including theater, music, and the arts, and its economy is heavily reliant on tourism and the hospitality industry. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve significant advancements in areas such as machine learning, natural language processing, robotics, and computer vision. Some possible future trends include:
    
    1. Increased sophistication of AI: As technology continues to advance, we can expect to see even more sophisticated and advanced AI systems emerge. This may include better understanding of human language, better image recognition, and even more advanced forms of AI that can perform tasks that are currently only possible with human intelligence.
    
    2. Increased collaboration between humans and AI: We may see more significant collaboration between humans and AI in the future. This may involve developing AI that can assist humans in performing tasks that are currently difficult or


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

    type

     of

     character

    ]

     who

     has

     spent

     [

    number

     of

     years

    ]

     years

     in

     this

     world

    ,

     [

    Age

    ].

     I

    'm

     known

     for

     my

     [

    unique

     trait

    ]

     that

     has

     made

     me

     a

     legend

    ,

     and

     I

    'm

     always

     ready

     to

     help

     those

     in

     need

    .

     My

     name

     is

     [

    Name

    ],

     and

     I

    'm

     a

     [

    type

     of

     character

    ]

     who

     has

     spent

     [

    number

     of

     years

    ]

     years

     in

     this

     world

    ,

     [

    Age

    ].

     I

    'm

     known

     for

     my

     [

    unique

     trait

    ]

     that

     has

     made

     me

     a

     legend

    ,

     and

     I

    'm

     always

     ready

     to

     help

     those

     in

     need

    .

     I

    'm

     [

    Name

    ],

     a

     [

    Type

     of

     Character

    ]

     who

     has

     spent

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    ,

     a

     historic

     city

     with

     a

     rich

     history

    ,

     including

     the

     birth

    place

     of

     the

     French

     Revolution

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     
    


    French

     history

     and

     art

     are

     deeply

     intertwined

     in

     the

     capital

    's

     architecture

    ,

     art

     museums

    ,

     and

     cultural

     institutions

    .

     Paris

     is

     also

     known

     for

     its

     thriving

     nightlife

    ,

     food

     scene

    ,

     and

     annual

     festivals

     like

     the

     E

    iff

    el

     Tower

     Festival

     and

     the

     Carn

    aval

    .
    


    As

     the

     world

    's

     cultural

     and

     political

     center

    ,

     Paris

     is

     a

     hub

     for

     many

     international

     affairs

     and

     offers

     a

     diverse

     range

     of

     cultural

     experiences

     for

     both

     locals

     and

     tourists

    .

     It

    's

     also

     home

     to

     many

     notable

     French

     restaurants

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     anticipated

     and

     dynamic

    ,

     with

     advancements

     in

     multiple

     areas

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

     AI

    -driven

     decision

    -making

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

     help

     diagnose

     and

     treat

     diseases

    ,

     predict

     patient

     outcomes

    ,

     and

     improve

     treatment

     plans

    .

     This

     will

     require

     the

     development

     of

     more

     sophisticated

     machine

     learning

     algorithms

     that

     can

     interpret

     medical

     data

     and

     make

     informed

     decisions

    .
    


    2

    .

     AI

     in

     manufacturing

    :

     AI

     will

     enable

     manufacturing

     plants

     to

     monitor

     and

     optimize

     their

     processes

    ,

     reducing

     waste

     and

     improving

     efficiency

    .

     This

     will

     require

     the

     development

     of

     more

     advanced

     algorithms

     that

     can

     analyze

     large

     amounts

     of

     data

     and

     make

     real

    -time

     decisions

    .
    


    3

    .

     AI

     in

     education

    :

     AI

     will

     be

     used

    



```python
llm.shutdown()
```
