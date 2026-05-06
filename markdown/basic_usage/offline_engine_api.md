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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.31it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.30it/s]


    2026-05-06 06:58:58,161 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 06:58:58] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.23it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.23it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.49it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.49it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.30it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.12it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.12it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.12it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.12it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.12it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.12it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.12it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.12it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.12it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.12it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.12it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 31.76it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:03, 15.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 15.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 15.80it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 15.80it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 19.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 19.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 19.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 19.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.64it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.64it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.64it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  21%|██        | 12/58 [00:00<00:01, 28.81it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.81it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:01, 28.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.51it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s] Capturing num tokens (num_tokens=896 avail_mem=72.28 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=768 avail_mem=72.27 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=704 avail_mem=72.27 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.09it/s]Capturing num tokens (num_tokens=704 avail_mem=72.27 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.42it/s]

    Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=448 avail_mem=72.26 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=416 avail_mem=72.26 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=352 avail_mem=72.25 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.35it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.35it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=256 avail_mem=72.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.31it/s]

    Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.08it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.08it/s]

    Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.08it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.22it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.22it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.22it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.22it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.22it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.22it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.44it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.44it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.44it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 38.31it/s]


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
    Generated text:  José Antonio. I am a Professor of Mathematics at the Universidad Carlos III de Madrid and a member of the Spanish Academy of Sciences. I am also an independent and consulting mathematics professor at the Universidad San Carlos de Madrid. I have been a guest lecturer at the University of Edinburgh and the London School of Economics. I have been a student fellow at the University of Cambridge. I have been a visiting professor at the University of Toronto. I have been a research professor at the University of Texas, Austin. I have been a research professor at the University of Chicago, Urbana-Champaign. I have been a research professor at the University of Michigan,
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy and he has to make so many speeches every year. The president is happy when he is able to speak to the nation about the most important issues. 
    
    A few years ago, the president did not make any speeches. So he had to borrow a speech writer to help him. The speech writer cost the president $100,000. 
    
    The president then had the following expenses in his speech:
    
    1. Printing the speeches: $1,500 per speech
    2. Distributing the speeches: $2,000 per speech
    3. Selling the speeches: $3,000
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of ______
    A. Parigi
    B. Paris
    C. Milan
    D. Moscow
    Answer:
    
    B
    
    The atomic structure and type of molecules in the following substances are all different. 
    A. Ethanol
    B. Ethylene
    C. Acetone
    D. Acetic acid
    Answer:
    
    B
    
    The atomic structure and type of molecules in the following substances are all different. 
    A. Ethanol 
    B. Ethylene 
    C. Acetone 
    D. Acetic acid 
    Answer:
    
    B
    
    Which of the following substances can be used to prepare sodium hypochlorite?
    A.
    ===============================
    Prompt: The future of AI is
    Generated text:  on the horizon. According to a report by the Bureau of Labor Statistics (BLS), employment in the field of artificial intelligence is expected to grow at an annual rate of 24.2% from 2021 to 2027. By contrast, the BLS predicts that the field of physical therapy will experience a 6.1% growth over the same period.
    According to a study published in the Journal of the American Medical Association, AI has the potential to revolutionize the field of physical therapy. The study shows that the use of AI in physical therapy can improve the accuracy of treatment, reduce the time


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your profession or role]. I enjoy [insert a short description of your hobbies or interests]. What brings you to this company? I'm drawn to [insert a short description of why you're interested in this position]. What do you like to do in your free time? I enjoy [insert a short description of what you like to do in your free time]. What's your favorite thing about working at [company name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in Europe by population. It is located on the Seine River and is the seat of government, administration, and culture for the French Republic. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. The city is also home to many famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a vibrant and dynamic city that continues to be a major center of culture and commerce in Europe. The city is also home to many important organizations and institutions, including the French Academy
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This includes issues such as bias, transparency, and accountability.
    
    2. Integration with other technologies: AI will continue to be integrated with other technologies such as blockchain, quantum computing, and quantum cryptography. This will create new opportunities for AI to be used in new and innovative ways.
    
    3. Increased use of AI in healthcare: AI will be used to
    


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
    Generated text:  [Name] and I'm a [job title or profession] with a [number] of years of experience in [related field]. What can you tell me about yourself? I'm always looking for new challenges and opportunities to grow and learn, and I'm eager to try something new and exciting every day. I'm passionate about [career objective or interest], and I'm always looking for ways to help others achieve their goals and reach their potential. I value hard work, dedication, and perseverance, and I'm always on the lookout for opportunities to learn and grow within my field. I'm confident in my abilities and will do my best
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Paille", the "City of Lights", and "The City of Light". It is the largest city in Europe and the largest city in the world, home to over 10 million people. Paris is known for its rich history, art, and culture, as well as its stunning architecture and modern technology. The city is also home to many famous landmarks, such as the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. Paris is a major international financial center and plays a crucial role in the global economy. In addition to its status as the capital of France, Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a rapid and continuous expansion. Here are some possible trends that could shape this future:
    
    1. Increased transparency and accountability: As AI systems become more complex and sophisticated, there will be a need for greater transparency and accountability in their design, development, and operation. This will require increased investment in research and development to improve the accuracy and reliability of AI systems, as well as in creating a culture of ethical AI practices.
    
    2. Multi-modal and multimodal AI: AI systems will become more capable of handling a wider range of inputs, including visual, auditory, and haptic data. This will require improvements in data generation


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

    ]

     and

     I

     am

     a

     [

    Type

    ]

     [

    You

    're

     a

     professional

    ,

     a

     teacher

    ,

     or

     something

     else

    ]

     at

     [

    Your

     Company

    ].

     I

     specialize

     in

     [

    Your

     Expert

    ise

    ],

     [

    Your

     Special

    ity

    ],

     [

    Your

     Profession

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

     our

     company

    .

     [

    Your

     Name

    ]

     [

    Your

     Company

    ]

     Welcome

     to

     our

     office

    ,

     and

     thank

     you

     for

     stopping

     by

    .

     What

     can

     I

     do

     for

     you

     today

    ?

     [

    Your

     Name

    ]

     [

    Your

     Company

    ]


    Hey

    ,

     can

     you

     tell

     me

     a

     bit

     more

     about

     yourself

    ?

     I

    'm

     really

     interested

     in

     knowing

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

     Company

    ]


    Hello

    ,

     my

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     largest

     city

     and

     the

     political

    ,

     economic

    ,

     cultural

    ,

     and

     financial

     center

     of

     the

     country

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     located

     on

     the

     Î

    le

     de

     la

     C

    ité

    ,

     which

     is

     situated

     in

     the

     Gar

    onne

     River

     and

     the

     Atlantic

     Ocean

    ,

     making

     it

     a

     famous

     European

     city

    .

     It

    's

     home

     to

     the

     Lou

    vre

     Museum

    ,

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     landmarks

    .

     Paris

     is

     also

     famous

     for

     its

     rich

     history

    ,

     art

    ,

     and

     culture

    ,

     and

     is

     a

     popular

     tourist

     destination

    .

     The

     city

     is

     also

     known

     for

     its

     fashion

     industry

    ,

     and

     has

     produced

     many

     iconic

     fashion

     designers

     and

     models

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     driven

     by

     the

     rapid

     advancements

     in

     machine

     learning

     and

     deep

     learning

    .

     With

     the

     increasing

     availability

     of

     large

     datasets

     and

     the

     increasing

     power

     of

     computing

    ,

     AI

     systems

     are

     expected

     to

     become

     increasingly

     complex

     and

     capable

     of

     performing

     increasingly

     sophisticated

     tasks

    .

     Some

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     has

     the

     potential

     to

     revolution

    ize

     the

     healthcare

     industry

     by

     improving

     diagnostics

    ,

     personal

    izing

     treatment

     plans

    ,

     and

     more

    .

     AI

    -powered

     diagnostic

     tools

     could

     help

     doctors

     make

     better

     diagnoses

     and

     prescribe

     more

     effective

     treatments

    .
    


    2

    .

     Autonomous

     vehicles

    :

     With

     the

     increasing

     demand

     for

     self

    -driving

     cars

    ,

     AI

     is

     expected

     to

     play

     an

     increasingly

     important

     role

     in

     the

     development

     of

     autonomous

    



```python
llm.shutdown()
```
