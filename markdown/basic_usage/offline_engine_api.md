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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.46it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.31s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.51it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.94it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.94it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.94it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.94it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.94it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.94it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.94it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.94it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.94it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.94it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 32.10it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 32.10it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 32.10it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 32.10it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 32.10it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 32.10it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 32.10it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 32.10it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 32.10it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 32.10it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.34it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  31%|███       | 18/58 [00:00<00:01, 35.21it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  31%|███       | 18/58 [00:00<00:01, 35.21it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.72 GB):  31%|███       | 18/58 [00:00<00:01, 35.21it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.70 GB):  31%|███       | 18/58 [00:00<00:01, 35.21it/s]Capturing num tokens (num_tokens=960 avail_mem=71.71 GB):  31%|███       | 18/58 [00:00<00:01, 35.21it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  31%|███       | 18/58 [00:00<00:01, 35.21it/s]Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.23it/s]Capturing num tokens (num_tokens=832 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.23it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.23it/s]Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.23it/s]Capturing num tokens (num_tokens=640 avail_mem=71.69 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.23it/s]Capturing num tokens (num_tokens=576 avail_mem=71.69 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.23it/s]Capturing num tokens (num_tokens=576 avail_mem=71.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.99it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.99it/s]Capturing num tokens (num_tokens=480 avail_mem=71.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.99it/s]Capturing num tokens (num_tokens=448 avail_mem=71.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.99it/s]

    Capturing num tokens (num_tokens=416 avail_mem=71.67 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.99it/s]Capturing num tokens (num_tokens=384 avail_mem=71.67 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.99it/s]Capturing num tokens (num_tokens=384 avail_mem=71.67 GB):  57%|█████▋    | 33/58 [00:00<00:00, 35.86it/s]Capturing num tokens (num_tokens=352 avail_mem=71.66 GB):  57%|█████▋    | 33/58 [00:00<00:00, 35.86it/s]Capturing num tokens (num_tokens=320 avail_mem=71.66 GB):  57%|█████▋    | 33/58 [00:00<00:00, 35.86it/s]Capturing num tokens (num_tokens=288 avail_mem=71.65 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=256 avail_mem=71.64 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.86it/s]Capturing num tokens (num_tokens=256 avail_mem=71.64 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.60it/s]Capturing num tokens (num_tokens=240 avail_mem=71.16 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.60it/s]

    Capturing num tokens (num_tokens=224 avail_mem=70.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.60it/s]Capturing num tokens (num_tokens=208 avail_mem=70.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.60it/s]Capturing num tokens (num_tokens=192 avail_mem=70.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.60it/s]Capturing num tokens (num_tokens=176 avail_mem=70.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.60it/s]Capturing num tokens (num_tokens=176 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.08it/s]Capturing num tokens (num_tokens=160 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.08it/s]Capturing num tokens (num_tokens=144 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.08it/s]Capturing num tokens (num_tokens=128 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.08it/s]Capturing num tokens (num_tokens=112 avail_mem=70.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.08it/s]Capturing num tokens (num_tokens=96 avail_mem=70.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.08it/s] Capturing num tokens (num_tokens=96 avail_mem=70.97 GB):  81%|████████  | 47/58 [00:01<00:00, 40.67it/s]Capturing num tokens (num_tokens=80 avail_mem=70.97 GB):  81%|████████  | 47/58 [00:01<00:00, 40.67it/s]

    Capturing num tokens (num_tokens=64 avail_mem=70.96 GB):  81%|████████  | 47/58 [00:01<00:00, 40.67it/s]Capturing num tokens (num_tokens=48 avail_mem=70.96 GB):  81%|████████  | 47/58 [00:01<00:00, 40.67it/s]Capturing num tokens (num_tokens=32 avail_mem=70.96 GB):  81%|████████  | 47/58 [00:01<00:00, 40.67it/s]Capturing num tokens (num_tokens=28 avail_mem=70.95 GB):  81%|████████  | 47/58 [00:01<00:00, 40.67it/s]Capturing num tokens (num_tokens=28 avail_mem=70.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=24 avail_mem=70.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=20 avail_mem=70.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=16 avail_mem=70.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=12 avail_mem=70.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=8 avail_mem=70.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.10it/s] Capturing num tokens (num_tokens=8 avail_mem=70.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=4 avail_mem=70.93 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.62it/s]

    Capturing num tokens (num_tokens=4 avail_mem=70.93 GB): 100%|██████████| 58/58 [00:01<00:00, 37.78it/s]


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
    Generated text:  Thomas Green and I'm an experienced experimenter in the field of bioinformatics. I have been involved in the development of a variety of bioinformatics projects and have a background in computer programming, statistics, and machine learning. I have worked on a wide range of projects including genome-wide association studies, transcriptomics, proteomics, and metabolomics. I am passionate about using bioinformatics to inform scientific discovery and can guide small biologists in the use of software, databases, and statistical techniques. I am committed to providing practical and relevant guidance to students and researchers in the field of bioinformatics, and am excited about the potential of bioinformatics
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. (Judge true or false)
    
    To determine whether the statement "The president of the United States is a person" is true or false, we need to break down the statement and analyze its components.
    
    1. Identify the subject: The subject of the statement is "the president of the United States."
    2. Identify the predicate: The predicate is "is a person."
    3. Analyze the predicate:
       - The president of the United States is a person. This is a clear and unambiguous statement.
       - The president of the United States is a specific individual.
       - The president of the United States is a political
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the most populous city is ______.____
    A. Marseille
    B. Lyon
    C. Nice
    D. Marseille
    Answer:
    
    D
    
    During the Renaissance period, the four major works of Leonardo da Vinci include: A. Mona Lisa B. The Vitruvian Man C. The Last Supper D. The Chessboard
    Answer:
    
    B
    
    In Traditional Chinese Medicine (TCM), which acupoint is used to treat abdominal pain?
    A. Qihai
    B. Taichong
    C. Taixi
    D. Zusanli
    E. Taiyuan
    Answer:
    
    B
    
    In
    ===============================
    Prompt: The future of AI is
    Generated text:  already here, it’s just not yet here. As AI technology advances, it is becoming more and more pervasive in nearly every industry and has the potential to transform the way we live, work, and learn. While AI has the potential to be incredibly beneficial for businesses, the hype around it can be overwhelming. In this blog, we’ll explore the basics of artificial intelligence and how it is advancing in the digital age.
    What is AI?
    Artificial Intelligence (AI) refers to the development of computer systems that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. The term


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old [Gender] [Race] [Nationality] [Hometown]. I'm a [Number] year old [Gender] [Race] [Nationality] [Hometown]. I'm a [Number] year old [Gender] [Race] [Nationality] [Hometown]. I'm a [Number] year old [Gender] [Race] [Nationality] [Hometown]. I'm a [Number] year old [Gender] [Race] [Nationality] [Hometown]. I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. The city is home to many famous French artists, writers, and musicians, and is a major tourist destination. It is also a center of politics and government, with the French Parliament located in the city. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is known
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives, from smart home devices to self-driving cars. As AI technology continues to improve, we can expect to see even more integration into our daily lives.
    
    2. AI becoming more autonomous: As AI technology continues to improve, we can expect to see more autonomous vehicles on the roads. This could lead to a reduction in accidents and a decrease in the need for human drivers.
    
    3. AI becoming more ethical and
    


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
    Generated text:  [Name] and I'm a [character] enthusiast! I'm a [character] who enjoys [reason for love/hobby/intellect] and I've always been drawn to [character's hobbies or interests]. I enjoy [reason for pursuing my passion]. If you'd like to know more about me, please feel free to ask me anything! [Start your self-introduction with a brief introduction that sets a positive tone]. That's my [name]. [End]. Good night! That's all for now. Take care! [End of message].
    Based on the information provided in the text, here is a neutral self-int
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France, with an estimated population of over 1, 375, 000 people. Paris is known for its iconic landmarks, such as the Eiffel Tower, the Louvre Museum, and the Notre Dame Cathedral. The city is also known for its diverse and eclectic cuisine, as well as its rich cultural heritage, including its role in the French Revolution and its influence on modern art and literature. Paris is a global cultural and political center, known for its artistic, literary, and political institutions, and for its role as a melting pot of different cultures and ideas. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid progress, innovation, and transformation in several key areas. Here are some possible trends in AI that could shape the future:
    
    1. Increased automation: As AI becomes more capable, it is likely to automate a growing number of tasks, freeing up time for humans to focus on more strategic and creative activities. This could lead to a shift from manual to automated processes, and could also result in the creation of new jobs that are no longer available.
    
    2. Greater emphasis on ethics and fairness: As AI becomes more integrated into our daily lives, there will be a growing focus on ethical considerations and fairness. This could lead


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

    ],

     a

     [

    Your

     Profession

    ].

     I

     am

     a

     [

    X

    ]

     who

     [

    X

    ].

     I

     have

     [

    X

    ]

     years

     of

     experience

     in

     [

    Your

     Area

     of

     Expert

    ise

    ].

     I

    'm

     always

     ready

     to

     learn

     new

     things

     and

     keep

     up

     with

     the

     latest

     trends

     in

     [

    Your

     Field

     of

     Interest

    ].

     I

     enjoy

     [

    X

    ]

     and

     am

     always

     up

     for

     [

    X

    ].

     What

    's

     your

     favorite

     hobby

     or

     activity

    ?

     What

     is

     the

     most

     interesting

     thing

     you

    've

     read

     recently

    ?

     What

    's

     something

     that

     you

    've

     done

     that

     was

     a

     challenge

     but

     also

     a

     source

     of

     personal

     growth

    ?

     What

    's

     the

     best

     piece

     of

     advice

     you

    've

     ever

     received

    ?

     What

    's

     your

     favorite

     way

     to

     spend

     your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    To

     elaborate

    ,

     Paris

     is

     the

     largest

     city

     in

     France

     and

     has

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     times

    .

     It

     is

     known

     for

     its

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     also

     home

     to

     many

     museums

    ,

     restaurants

    ,

     and

     theaters

    .

     Paris

     is

     a

     lively

     and

     dynamic

     city

     with

     a

     rich

     cultural

     and

     historical

     heritage

     that

     attracts

     millions

     of

     visitors

     annually

    .

     Additionally

    ,

     it

     is

     home

     to

     many

     international

     organizations

    ,

     including

     the

     European

     Parliament

    ,

     the

     European

     Central

     Bank

    ,

     and

     the

     European

     Union

    .

     
    


    Overall

    ,

     Paris

     is

     a

     unique

     and

     vibrant

     city

     that

     offers

     a

     unique

     blend

     of

     history

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     a

     rapidly

     evolving

     field

    ,

     with

     new

     trends

     and

     innovations

     constantly

     emerging

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

     Deep

     learning

    :

     Deep

     learning

     is

     a

     subset

     of

     AI

     that

     involves

     training

     neural

     networks

     to

     learn

     complex

     patterns

     and

     relationships

     from

     large

     datasets

    .

     In

     the

     future

    ,

     we

     may

     see

     advancements

     in

     this

     area

    ,

     including

     faster

     training

     speeds

    ,

     more

     efficient

     algorithms

    ,

     and

     better

     performance

     on

     a

     wide

     range

     of

     tasks

    .
    


    2

    .

     Explain

    ability

     and

     fairness

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

     it

    's

     becoming

     increasingly

     important

     to

     ensure

     that

     they

     are

     explain

    able

     and

     fair

    .

     Future

     AI

     systems

     may

     incorporate

     more

     sophisticated

     explanations

     and

     fairness

     mechanisms

    ,

     such

     as

     probability

    



```python
llm.shutdown()
```
