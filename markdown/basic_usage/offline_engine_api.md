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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.52it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.52it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.52it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.52it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.52it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.52it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.52it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.52it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.52it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.52it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.56it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.56it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.56it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.56it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.56it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.56it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.56it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 22.63it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 22.63it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 22.63it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 22.63it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 22.63it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 22.63it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 22.63it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 22.63it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 22.63it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 22.63it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 22.63it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 31.77it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 31.77it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 31.77it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 31.77it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 31.77it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 31.77it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 31.77it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 31.77it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 31.77it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 31.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.39 GB):   3%|▎         | 2/58 [00:00<00:04, 12.86it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.38 GB):   3%|▎         | 2/58 [00:00<00:04, 12.86it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.38 GB):   3%|▎         | 2/58 [00:00<00:04, 12.86it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.38 GB):   3%|▎         | 2/58 [00:00<00:04, 12.86it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.38 GB):   9%|▊         | 5/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.38 GB):   9%|▊         | 5/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.37 GB):   9%|▊         | 5/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.36 GB):   9%|▊         | 5/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.36 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.36 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.62it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=59.36 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.35 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.35 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.35 GB):  21%|██        | 12/58 [00:00<00:01, 28.28it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.35 GB):  21%|██        | 12/58 [00:00<00:01, 28.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.35 GB):  21%|██        | 12/58 [00:00<00:01, 28.28it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.34 GB):  21%|██        | 12/58 [00:00<00:01, 28.28it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.34 GB):  21%|██        | 12/58 [00:00<00:01, 28.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.33 GB):  21%|██        | 12/58 [00:00<00:01, 28.28it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.52it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=59.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.33 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.31 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.52it/s]Capturing num tokens (num_tokens=960 avail_mem=59.32 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.52it/s] Capturing num tokens (num_tokens=960 avail_mem=59.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=896 avail_mem=59.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=832 avail_mem=59.32 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=768 avail_mem=59.31 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=704 avail_mem=59.31 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=640 avail_mem=59.31 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=640 avail_mem=59.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.17it/s]Capturing num tokens (num_tokens=576 avail_mem=59.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.17it/s]

    Capturing num tokens (num_tokens=512 avail_mem=59.29 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.17it/s]Capturing num tokens (num_tokens=480 avail_mem=59.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.17it/s]Capturing num tokens (num_tokens=448 avail_mem=59.30 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.17it/s]Capturing num tokens (num_tokens=416 avail_mem=59.30 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.17it/s]Capturing num tokens (num_tokens=416 avail_mem=59.30 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.23it/s]Capturing num tokens (num_tokens=384 avail_mem=59.30 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.23it/s]Capturing num tokens (num_tokens=352 avail_mem=59.29 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.23it/s]Capturing num tokens (num_tokens=320 avail_mem=59.29 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.23it/s]Capturing num tokens (num_tokens=288 avail_mem=59.29 GB):  55%|█████▌    | 32/58 [00:01<00:00, 42.23it/s]

    Capturing num tokens (num_tokens=256 avail_mem=59.28 GB):  55%|█████▌    | 32/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=256 avail_mem=59.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=240 avail_mem=59.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=224 avail_mem=59.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=208 avail_mem=59.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=192 avail_mem=59.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=176 avail_mem=59.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=176 avail_mem=59.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=160 avail_mem=59.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=144 avail_mem=59.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=128 avail_mem=59.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.35it/s]

    Capturing num tokens (num_tokens=112 avail_mem=59.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.35it/s]Capturing num tokens (num_tokens=96 avail_mem=59.24 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.35it/s] Capturing num tokens (num_tokens=96 avail_mem=59.24 GB):  81%|████████  | 47/58 [00:01<00:00, 26.69it/s]Capturing num tokens (num_tokens=80 avail_mem=59.23 GB):  81%|████████  | 47/58 [00:01<00:00, 26.69it/s]

    Capturing num tokens (num_tokens=64 avail_mem=59.14 GB):  81%|████████  | 47/58 [00:01<00:00, 26.69it/s]Capturing num tokens (num_tokens=48 avail_mem=58.57 GB):  81%|████████  | 47/58 [00:01<00:00, 26.69it/s]Capturing num tokens (num_tokens=32 avail_mem=58.57 GB):  81%|████████  | 47/58 [00:01<00:00, 26.69it/s]

    Capturing num tokens (num_tokens=32 avail_mem=58.57 GB):  88%|████████▊ | 51/58 [00:01<00:00, 19.45it/s]Capturing num tokens (num_tokens=28 avail_mem=58.56 GB):  88%|████████▊ | 51/58 [00:01<00:00, 19.45it/s]Capturing num tokens (num_tokens=24 avail_mem=58.56 GB):  88%|████████▊ | 51/58 [00:01<00:00, 19.45it/s]Capturing num tokens (num_tokens=20 avail_mem=58.56 GB):  88%|████████▊ | 51/58 [00:02<00:00, 19.45it/s]

    Capturing num tokens (num_tokens=20 avail_mem=58.56 GB):  93%|█████████▎| 54/58 [00:02<00:00, 17.29it/s]Capturing num tokens (num_tokens=16 avail_mem=58.56 GB):  93%|█████████▎| 54/58 [00:02<00:00, 17.29it/s]Capturing num tokens (num_tokens=12 avail_mem=58.55 GB):  93%|█████████▎| 54/58 [00:02<00:00, 17.29it/s]Capturing num tokens (num_tokens=8 avail_mem=58.55 GB):  93%|█████████▎| 54/58 [00:02<00:00, 17.29it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=58.55 GB):  98%|█████████▊| 57/58 [00:02<00:00, 15.80it/s]Capturing num tokens (num_tokens=4 avail_mem=58.55 GB):  98%|█████████▊| 57/58 [00:02<00:00, 15.80it/s]Capturing num tokens (num_tokens=4 avail_mem=58.55 GB): 100%|██████████| 58/58 [00:02<00:00, 23.87it/s]


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
    Generated text:  Rafael Cipriano. I am a student of the Computational Mathematics and Mathematical Physics department at the University of California, Los Angeles (UCLA). I am a Ph.D. student in Computational Mathematics. My advisor is Professor Boris Hassig. My current research interests are in the area of mathematical physics, particularly in the area of lattice integrals and Riemannian geometry. I am especially interested in the connection between the geometry of the space of solutions to the Dirac equation and the geometry of the space of solutions to the Laplacian in four dimensions. This research project is part of my PhD thesis and I am currently working on
    ===============================
    Prompt: The president of the United States is
    Generated text:  a noble man, he is a man of integrity, he always does what is right, he is always honest. This statement refers to which of the following? 
    A: Constitution of the United States
    B: The Declaration of Independence
    C: The Bill of Rights
    D: The Constitution of the United States
    
    To determine which statement refers to the Constitution of the United States, let's analyze the context and the options given.
    
    The statement "The president of the United States is a noble man, he is a man of integrity, he always does what is right, he is always honest" is comparing the characteristics of a president to
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. 500 years ago, there was a "pig" on the beach. The King and the Princess were very happy. But it was wrong to be happy. There was a problem. The king wanted to kill the pig so that his children would have to work. The princess wanted to eat the pig. The King ordered his men to chop the pig in half. But before they could do it, the pig ate the King. 
    
    What is the capital of France? The capital of France is Paris. 
    
    To provide a more detailed explanation:
    
    1. Paris, the capital of France, was founded in 78
    ===============================
    Prompt: The future of AI is
    Generated text:  the future of the human race.
    The emergence of AI has been unprecedented, causing many to feel overwhelmed by its effects on society. As the technology continues to advance, AI will continue to change how we live, work, and interact with each other. While there are many benefits to AI, including increased efficiency, productivity, and improved quality of life, there are also concerns and risks that need to be addressed.
    AI is a rapidly advancing technology that has the potential to revolutionize various fields, from healthcare to transportation to finance. However, there are also challenges and risks associated with AI, such as privacy and security concerns, biases in AI models


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I have a [job title] at [company name] and I enjoy [job title] work. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie? I love
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    A. True
    B. False
    A. True
    
    Paris is the capital city of France and is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. The city is known for its rich history, including the influence of French colonialism and the impact of the French Revolution. Paris is also home to many famous French artists, writers, and musicians. The city is a major tourist destination, with millions of visitors annually. Despite its size, Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends that are expected to shape the development of AI in the coming years:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare in the coming years.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management and fraud detection. As AI technology continues to improve, we can
    


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
    Generated text:  [Name]. I'm a self-proclaimed amateur photographer, and I've been experimenting with photography for as long as I can remember. My work has often been used in advertising campaigns and other creative projects, and I love exploring new techniques and themes in the medium. I also enjoy creating art, and often find myself collaborating with other artists to bring their unique visions to life. My goal is to use my skills to inspire and uplift those around me, whether that's through my photography, art, or other creative pursuits. Thank you for taking the time to meet me. How about you, [Name]? Hi there! I'm a self
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    To answer this question in Spanish, you would say:
    
    ¿Qué ciudad capital de Francia es?
    
    El capital de Francia es París. 
    
    Confección del pliego:
    1. Identificar la ciudad capital de Francia.
    2. Representarla en español utilizando el uso de símbolos o caracteres de habla española (por ejemplo, "Paris").
    3. Reiterar la información en español para asegurarse de que sea clara y precisa.
    
    Finalmente, poder presentarla en su contexto preciso y sin error, como "El capital de Francia es París".
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be a blend of three trends: development in machine learning, increased focus on ethical considerations, and the integration of AI with human decision-making.
    
    First, we can expect AI to continue developing in machine learning. As more data is collected and algorithms are improved, we may see more sophisticated models and applications emerge. This trend is likely to create new opportunities for companies to develop AI solutions that can solve complex problems, from medical diagnosis to autonomous vehicles.
    
    Second, ethical considerations will continue to play a significant role in AI development. As AI systems are integrated into human systems, there may be unintended consequences that require careful consideration. For example, if


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

     Alex

    ,

     and

     I

    ’m

     a

     software

     developer

     with

     a

     passion

     for

     creating

     dynamic

     and

     intuitive

     user

     interfaces

    .

     I

     have

     a

     knack

     for

     turning

     complex

     problems

     into

     simple

     solutions

     and

     have

     built

     my

     skills

     on

     a

     combination

     of

     theoretical

     knowledge

     and

     hands

    -on

     experience

    .

     Outside

     of

     coding

    ,

     I

     enjoy

     spending

     time

     outdoors

    ,

     reading

     books

    ,

     and

     practicing

     my

     guitar

    .

     My

     ultimate

     goal

     is

     to

     create

     software

     that

     makes

     my

     users

    '

     lives

     easier

     and

     happier

    .

     How

     would

     you

     introduce

     yourself

     to

     someone

     who

     is

     unfamiliar

     with

     my

     work

    ?

     You

     might

     say

    ,

     "

    Hi

    ,

     I

    'm

     Alex

    .

     I

    'm

     a

     software

     developer

     with

     a

     passion

     for

     creating

     dynamic

     and

     intuitive

     user

     interfaces

    ."

     Would

     you

     like

     me

     to

     elaborate

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     political

    ,

     economic

    ,

     and

     cultural

     center

     of

     France

    .

     It

     is

     located

     in

     the

     north

     of

     the

     country

     on

     the

     Î

    le

     de

     la

     C

    ité

    ,

     in

     the

     Se

    ine

     River

    .

     Paris

     is

     the

     most

     populous

     city

     in

     France

    ,

     with

     a

     population

     of

     

    2

    .

     

    7

     million

     people

    .

     The

     city

     is

     home

     to

     numerous

     landmarks

    ,

     including

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

     the

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

     Paris

     is

     also

     known

     for

     its

     delicious

     cuisine

    ,

     nightlife

    ,

     and

     fashion

     scene

    ,

     attracting

     tourists

     from

     all

     over

     the

     world

    .

     With

     its

     rich

     history

     and

     beautiful

     architecture

    ,

     Paris

     is

     a

     popular

     destination

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     number

     of

     trends

     that

     are

     currently

     being

     explored

     and

     developed

    :
    


    1

    .

     Natural

     language

     processing

     (

    N

    LP

    ):

     N

    LP

     is

     expected

     to

     continue

     to

     advance

     in

     terms

     of

     both

     the

     accuracy

     and

     the

     variety

     of

     natural

     language

     processing

     techniques

     that

     can

     be

     used

    .

     Some

     potential

     applications

     of

     N

    LP

     include

     virtual

     assistants

    ,

     language

     translation

    ,

     sentiment

     analysis

    ,

     and

     automated

     customer

     service

    .
    


    2

    .

     Robotics

     and

     automation

    :

     As

     more

     and

     more

     industries

     adopt

     automation

    ,

     there

     is

     likely

     to

     be

     an

     increased

     demand

     for

     robots

     and

     other

     automated

     systems

    .

     This

     could

     lead

     to

     a

     number

     of

     new

     job

     roles

     being

     created

    ,

     as

     well

     as

     new

     industries

     being

     developed

     to

     support

     these

     tasks

    .
    


    3

    .

    



```python
llm.shutdown()
```
