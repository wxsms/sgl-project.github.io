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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.38it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.34it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.73it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.68it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.75it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.75it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.75it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.75it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.75it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.75it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.75it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.75it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.75it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.75it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.75it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.67 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.67 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.66 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=56.66 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.66 GB):   9%|▊         | 5/58 [00:00<00:02, 20.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.66 GB):   9%|▊         | 5/58 [00:00<00:02, 20.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.65 GB):   9%|▊         | 5/58 [00:00<00:02, 20.04it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.65 GB):   9%|▊         | 5/58 [00:00<00:02, 20.04it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.65 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.65 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.90it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=56.64 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.64 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.64 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.63 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.63it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.63 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.63it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.63 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.63it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.62 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.63it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.62 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.62 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.74it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=56.62 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.61 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.61 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.61 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.61 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.59 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=960 avail_mem=56.60 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.96it/s] Capturing num tokens (num_tokens=896 avail_mem=56.60 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=896 avail_mem=56.60 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=832 avail_mem=56.60 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.23it/s]

    Capturing num tokens (num_tokens=768 avail_mem=56.59 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=704 avail_mem=56.59 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=640 avail_mem=56.59 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=640 avail_mem=56.59 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.86it/s]Capturing num tokens (num_tokens=576 avail_mem=56.59 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.86it/s]Capturing num tokens (num_tokens=512 avail_mem=56.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.86it/s]Capturing num tokens (num_tokens=480 avail_mem=56.59 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.86it/s]Capturing num tokens (num_tokens=448 avail_mem=56.59 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=448 avail_mem=56.59 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.01it/s]Capturing num tokens (num_tokens=416 avail_mem=56.58 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.01it/s]

    Capturing num tokens (num_tokens=384 avail_mem=56.58 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.01it/s]Capturing num tokens (num_tokens=352 avail_mem=56.58 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.01it/s]Capturing num tokens (num_tokens=320 avail_mem=56.57 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.01it/s]Capturing num tokens (num_tokens=320 avail_mem=56.57 GB):  60%|██████    | 35/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=288 avail_mem=56.57 GB):  60%|██████    | 35/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=256 avail_mem=56.57 GB):  60%|██████    | 35/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=240 avail_mem=56.56 GB):  60%|██████    | 35/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=224 avail_mem=56.56 GB):  60%|██████    | 35/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=224 avail_mem=56.56 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.82it/s]Capturing num tokens (num_tokens=208 avail_mem=56.56 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.82it/s]

    Capturing num tokens (num_tokens=192 avail_mem=56.56 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.82it/s]Capturing num tokens (num_tokens=176 avail_mem=56.55 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.82it/s]Capturing num tokens (num_tokens=160 avail_mem=56.55 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.82it/s]Capturing num tokens (num_tokens=160 avail_mem=56.55 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=144 avail_mem=56.55 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=128 avail_mem=56.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=112 avail_mem=56.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=96 avail_mem=56.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.53it/s] Capturing num tokens (num_tokens=80 avail_mem=56.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=80 avail_mem=56.53 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.87it/s]Capturing num tokens (num_tokens=64 avail_mem=56.53 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.87it/s]

    Capturing num tokens (num_tokens=48 avail_mem=56.53 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.87it/s]Capturing num tokens (num_tokens=32 avail_mem=59.13 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.87it/s]Capturing num tokens (num_tokens=28 avail_mem=59.13 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.87it/s]Capturing num tokens (num_tokens=28 avail_mem=59.13 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.10it/s]Capturing num tokens (num_tokens=24 avail_mem=59.12 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.10it/s]Capturing num tokens (num_tokens=20 avail_mem=59.12 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.10it/s]Capturing num tokens (num_tokens=16 avail_mem=59.12 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.10it/s]

    Capturing num tokens (num_tokens=12 avail_mem=59.11 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.10it/s]Capturing num tokens (num_tokens=8 avail_mem=59.11 GB):  90%|████████▉ | 52/58 [00:01<00:00, 32.10it/s] Capturing num tokens (num_tokens=8 avail_mem=59.11 GB):  98%|█████████▊| 57/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=4 avail_mem=59.11 GB):  98%|█████████▊| 57/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=4 avail_mem=59.11 GB): 100%|██████████| 58/58 [00:01<00:00, 32.81it/s]


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
    Generated text:  Cristina. I'm from Uruguay and I'm the youngest of two daughters. I have a great family! My parents are both well-educated and they read a lot. My mother is a teacher, and my father is a doctor. I have lots of friends. I like to play sports and I love to visit museums. I have been to many countries and I've been to the United States before.
    A: 2113
    B: 2114
    C: 2115
    D: 2116
    What is the name of the oldest of the three daughters mentioned in the text?
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  a relatively young, experienced man by the standards of political leadership in most countries. By contrast, the president of Poland is much younger, and he is a fairly young man of middle age. Which of the following, if true, most effectively supports the conclusion that the president of Poland is younger than the president of the United States?
    
    A: The president of Poland was born in 1934, while the president of the United States was born in 1885.
    
    B: The president of the United States has been president seven times, while the president of Poland has been president only once.
    
    C: The president of Poland
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Rome
    C. Washington
    D. London
    Answer: A
    
    The capital of France is:
    
    A. Paris
    B. Rome
    C. Washington
    D. London
    Answer: A
    
    Which of the following is NOT a common characteristic of lysosomes?
    A. They contain enzymes that break down and destroy other substances.
    B. They contain various organelles such as ribosomes.
    C. They contain various organelles such as centrioles.
    D. They contain various organelles such as mitochondria.
    Answer: C
    
    Which of the following is NOT a characteristic of lysosomes
    ===============================
    Prompt: The future of AI is
    Generated text:  bright and its impact on society is enormous, but there are certain ethical issues that we need to consider. One of the most pressing issues is the ethical treatment of AI-generated images. Some people believe that AI-generated images should be treated as an extension of the human mind, and should therefore be protected and protected. However, this approach has some limitations and does not always achieve its intended goal.
    AI-generated images can be difficult to understand and interpret, and therefore it can be challenging to determine if they are genuine or fake. This makes it difficult to establish a clear distinction between real and fake images, which can lead to misunderstandings and potential ethical


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Gender] and [Country]. I'm a [Skill] with [Number] years of experience in [Field]. I'm passionate about [What I Love About My Job], and I'm always looking for ways to [What I Want to Improve]. I'm a [What I Like to Do] and I'm always looking for ways to [What I Want to Learn]. I'm a [What I Like to Do] and I'm always looking for ways to [What I Want to Learn]. I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on modern French culture and politics. Paris is also home to many famous French artists, writers, and musicians. The city is known for its cuisine, including its famous Parisian dishes such as croissants
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the impact of AI on society.
    
    2. Integration of AI with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This will allow for more sophisticated and intelligent applications of AI.
    
    3.
    


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
    Generated text:  [Name]. I'm an experienced [job title or hobby]. I've been working in this field for [number of years] years. I'm currently [current position]. 
    
    Now, to put it into context, I've always been passionate about [specific interest or field of interest]. I believe that [reason or reason for interest]. I enjoy [way I enjoy working in my field]. I am always looking to learn new things and I thrive on challenges. I am a [attitude or personality]. I'm confident in my ability to succeed and thrive in any situation. I thrive on opportunities to grow and learn.
    
    If you could
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    What is the answer? Paris is the capital city of France. Yes, that's correct! Paris is the capital of France, located on the Seine River in the heart of the country. It is also one of the most important cities in the world, known for its rich history, stunning architecture, and vibrant culture. Paris is home to many iconic landmarks like the Eiffel Tower, Notre Dame Cathedral, and Louvre Museum. The city is also known for its annual festivals like the New Year's Eve celebrations, the Eiffel Tower Open Day, and the Mardi Gras celebrations. Paris is a city that
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and unpredictable, and there is no guarantee that AI will continue to evolve in the same way it has in the past. However, there are some possible future trends that may be expected to influence AI development.
    
    1. Increased focus on ethical AI: As AI systems become more sophisticated and complex, there will be increasing pressure to ensure that they are being used ethically and responsibly. This could lead to more rigorous testing and evaluation of AI systems, as well as increased scrutiny of their deployment and application in different contexts.
    
    2. Advancements in quantum computing: As the technology advances, there may be significant breakthroughs in quantum computing, which


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

     and

     I

    ’m

     a

     [

    Your

     Profession

    /

    Role

    ].

     I

    ’m

     passionate

     about

     [

    Your

     Area

     of

     Expert

    ise

     or

     Passion

    ],

     and

     I

    ’m

     excited

     to

     learn

     and

     grow

     in

     this

     field

    .

     Thank

     you

     for

     the

     opportunity

     to

     share

     my

     story

     with

     you

     today

    .

     [

    Your

     Name

    ]

      


    That

    ’s

     all

     from

     [

    Your

     Name

    ].

     If

     you

     have

     any

     questions

     or

     need

     further

     information

    ,

     feel

     free

     to

     reach

     out

    .

     [

    Your

     Name

    ]

      


    (Note

    :

     The

     above

     text

     is

     designed

     to

     be

     a

     neutral

     self

    -int

    roduction

    ,

     not

     a

     personal

     introduction

    .

     It

    ’s

     designed

     to

     be

     a

     general

     introduction

     to

     the

     character

     without

     using

     any

     personal

     or

     emotional

     language

     or

     references

    .)

      


    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     city

     of

     France

     and

     is

     the

     largest

     city

     in

     the

     European

     Union

    .

     It

     is

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

     The

     city

     is

     also

     famous

     for

     its

     annual

     É

    to

    ile

     de

     Paris

    ,

     a

     festival

     of

     light

     and

     fireworks

     that

     takes

     place

     annually

     in

     November

    .

     Paris

     is

     also

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     other

     cultural

     institutions

    .

     Despite

     its

     size

    ,

     Paris

     has

     a

     small

     population

     of

     about

     

    2

    .

    7

     million

     people

    .

     The

     city

     is

     a

     major

     economic

     and

     political

     center

     of

     France

     and

     plays

     a

     significant

     role

     in

     its

     history

     and

     culture

    .

     The

     city

     has

     been

     continuously

     inhabited

     for

     over

     

    5

    ,

    0

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     increasingly

     complex

     and

     multid

    imensional

    .

     Some

     of

     the

     key

     trends

     that

     are

     likely

     to

     shape

     the

     AI

     landscape

     include

    :
    


    1

    .

     Increased

     reliance

     on

     AI

     for

     decision

    -making

     and

     automation

    :

     AI

     is

     increasingly

     being

     used

     for

     tasks

     that

     were

     previously

     performed

     by

     humans

    ,

     such

     as

     autonomous

     vehicles

    ,

     customer

     service

     chat

    bots

    ,

     and

     predictive

     analytics

    .

     These

     technologies

     are

     expected

     to

     become

     more

     ubiquitous

     and

     integrated

     into

     our

     daily

     lives

    .
    


    2

    .

     Greater

     emphasis

     on

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     integrated

     into

     various

     aspects

     of

     society

    ,

     there

     will

     be

     a

     greater

     focus

     on

     ethical

     considerations

    ,

     including

     issues

     related

     to

     bias

    ,

     transparency

    ,

     accountability

    ,

     and

     privacy

    .
    


    3

    .

     Continued

     integration

     with

     human

     cognitive

    



```python
llm.shutdown()
```
