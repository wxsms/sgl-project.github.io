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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.38it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.39it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.39it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.39it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.39it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.39it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.39it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.39it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:04<00:02, 13.23it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 20.13it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 20.13it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 20.13it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 20.13it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:01, 20.13it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:01, 20.13it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:01, 20.13it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:01, 20.13it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:01, 20.13it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:04<00:01, 20.13it/s]Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 29.47it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 29.47it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 29.47it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 29.47it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 29.47it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 29.47it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 29.47it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 29.47it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 29.47it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 29.47it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 29.47it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 39.42it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 39.42it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 39.42it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.52it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.87it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.87it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.14it/s]Capturing num tokens (num_tokens=960 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.14it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.14it/s]Capturing num tokens (num_tokens=832 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.14it/s]Capturing num tokens (num_tokens=832 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.96it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.96it/s]Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.96it/s]Capturing num tokens (num_tokens=640 avail_mem=71.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.96it/s]Capturing num tokens (num_tokens=576 avail_mem=71.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.96it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.96it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  50%|█████     | 29/58 [00:00<00:00, 41.21it/s]Capturing num tokens (num_tokens=480 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 41.21it/s]Capturing num tokens (num_tokens=448 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 41.21it/s]Capturing num tokens (num_tokens=416 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 41.21it/s]

    Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 41.21it/s]Capturing num tokens (num_tokens=352 avail_mem=71.68 GB):  50%|█████     | 29/58 [00:00<00:00, 41.21it/s]Capturing num tokens (num_tokens=352 avail_mem=71.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=320 avail_mem=71.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=288 avail_mem=71.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=256 avail_mem=71.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=224 avail_mem=71.66 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.39it/s]Capturing num tokens (num_tokens=224 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.19it/s]Capturing num tokens (num_tokens=208 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.19it/s]Capturing num tokens (num_tokens=192 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.19it/s]Capturing num tokens (num_tokens=176 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.19it/s]

    Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.19it/s]Capturing num tokens (num_tokens=144 avail_mem=71.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.19it/s]Capturing num tokens (num_tokens=144 avail_mem=71.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.28it/s]Capturing num tokens (num_tokens=128 avail_mem=71.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.28it/s]Capturing num tokens (num_tokens=112 avail_mem=71.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.28it/s]Capturing num tokens (num_tokens=96 avail_mem=71.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.28it/s] Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.28it/s]Capturing num tokens (num_tokens=64 avail_mem=71.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.28it/s]Capturing num tokens (num_tokens=64 avail_mem=71.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.40it/s]Capturing num tokens (num_tokens=48 avail_mem=71.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.40it/s]Capturing num tokens (num_tokens=32 avail_mem=71.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.40it/s]Capturing num tokens (num_tokens=28 avail_mem=71.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.40it/s]

    Capturing num tokens (num_tokens=24 avail_mem=71.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.40it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.40it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.89it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.89it/s]Capturing num tokens (num_tokens=12 avail_mem=71.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.89it/s]Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.89it/s] Capturing num tokens (num_tokens=4 avail_mem=71.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.89it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 40.94it/s]


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
    Generated text:  Ivo. I am a young professional looking for a job in my field of expertise. I have a Bachelor's degree in Computer Science and am passionate about using technology to solve complex problems in the field of data science. I enjoy working on real-world projects and collaborating with teams to solve complex problems. I am currently seeking a job in a company with a focus on data science, and have experience in using Python and R for data analysis and machine learning. I am open to remote work opportunities and would like to work in a team-oriented environment. I also have experience with web development and have created a website for a local business that I'm
    ===============================
    Prompt: The president of the United States is
    Generated text:  a noble title. A senator is a higher rank than a president. There are two senators per state. There is a senator in the state of New York. Is it possible that the senator in New York has a lower rank than the president of the United States? To determine if it is possible for a senator in New York to have a lower rank than the president of the United States, let's break down the information provided and analyze it step by step.
    
    1. **Identify the rank of the president of the United States:**
       - The president of the United States is a noble title.
       - A senator is a higher
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population of Paris is 1.6 million people. On a certain day, the population of Paris increased by 4% of its population. On that same day, the population of Paris decreased by 2% of its population. What is the population of Paris on that day? 
    
    A) 1.63 million
    B) 1.58 million
    C) 1.55 million
    D) 1.52 million
    E) 1.50 million To determine the population of Paris on the given day, we need to follow these steps:
    
    1. Calculate the population
    ===============================
    Prompt: The future of AI is
    Generated text:  very promising. For many years now, researchers have been exploring the use of artificial intelligence (AI) to automate complex, repetitive tasks and improve the efficiency and accuracy of existing processes. By leveraging the power of AI, organizations can automate processes that were previously done manually, and this can lead to increased productivity, reduced costs, and improved customer satisfaction.
    One area where AI can have a significant impact is in the field of healthcare. AI can help healthcare providers to analyze large amounts of data, identify patterns and trends, and provide insights that can be used to make better decisions.
    For example, AI can be used to analyze medical imaging data, such


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and art scene. Paris is a popular tourist destination and a major economic center in France. It is home to many world-renowned museums, theaters, and other cultural institutions. The city is also known for its vibrant nightlife and its role in the French Revolution and the French Revolution. Paris is a city of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations and tasks. This integration could lead to more efficient and effective AI systems that can perform tasks that are currently beyond the capabilities of humans.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    


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
    Generated text:  [Your Name] and I'm a [Your occupation] who is passionate about [Your hobby or interest]. I enjoy [Your hobby or interest] because it [explain why it's your hobby or interest]. I believe in [Your belief or value system], which is [explain your belief or value system]. I believe in the power of [Your belief or value system] and I strive to [explain why you believe in this value or belief]. I believe in [Your belief or value system] because [explain why you believe in this value or belief]. I believe in the power of [Your belief or value system] and I strive
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "la Garde", and is a historic center of the French Republic. It is a major city in the north of France and is home to the European Parliament, the Rive Gauche, and the Louvre Museum. It is also a center of art, music, literature, and cuisine. Paris is often referred to as "la Belle Epoque" because of its unique architecture and art style of the early 20th century. The city is a world-renowned cultural hub and a popular tourist destination. Paris is also known as "la Sorority Capital of the World" due to the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly innovative, rapidly changing, and driven by new technologies and developments. Here are some potential trends in AI that could shape the future of the field:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies, such as the internet of things (IoT) and the cloud, as these technologies become more widespread. This integration could lead to more sophisticated AI systems that can learn from and interact with existing technologies.
    
    2. More advanced models: AI models are likely to become more advanced, with better representation of real-world data and ability to handle more complex tasks. This could lead to more accurate


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

     Sarah

    ,

     and

     I

    'm

     a

     writer

    .

     I

    've

     always

     been

     a

     fan

     of

     storytelling

     and

     have

     had

     a

     passion

     for

     writing

     since

     I

     was

     a

     child

    .

     I

     love

     exploring

     different

     genres

     and

     trying

     to

     create

     characters

     that

     are

     both

     exciting

     and

     rel

    atable

    .

     I

    'm

     always

     looking

     for

     new

     opportunities

     to

     write

     and

     I

    'm

     excited

     to

     share

     my

     love

     of

     storytelling

     with

     others

    .


    Sarah

    's

     short

    ,

     neutral

     self

    -int

    roduction

     is

    :


    "

    Hello

    !

     My

     name

     is

     Sarah

    ,

     a

     writer

     with

     a

     passion

     for

     storytelling

    .

     I

    've

     loved

     exploring

     different

     genres

     and

     trying

     to

     create

     characters

     that

     are

     both

     exciting

     and

     rel

    atable

    .

     I

    'm

     always

     looking

     for

     new

     opportunities

     to

     write

     and

     excited

     to

     share

     my

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Ville

     Fl

    ott

    ante

    "

     and

     "

    La

     Ville

     de

     la

     Renaissance

    ".

     The

     city

     is

     a

     bustling

     met

    ropolis

     with

     a

     population

     of

     over

     

    2

    .

    7

     million

     people

     and

     is

     the

     cultural

    ,

     political

    ,

     and

     economic

     center

     of

     France

    .

     Paris

     is

     a

     world

    -ren

    owned

     historical

     and

     architectural

     city

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

     Ch

    amps

    -

    É

    lys

    ées

    .

     It

     is

     also

     the

     birth

    place

     of

     many

     famous

     French

     artists

    ,

     writers

    ,

     and

     musicians

    ,

     and

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     theaters

    .

     Paris

     is

     also

     known

     for

     its

     cuisine

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     potential

    ,

     exciting

     developments

     that

     are

     shaping

     the

     technology

     landscape

    .

     Here

     are

     some

     of

     the

     most

     likely

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     understanding

     and

     control

     of

     AI

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     become

     even

     more

     sophisticated

     and

     able

     to

     make

     decisions

     on

     its

     own

    .

     This

     will

     require

     more

     control

     over

     AI

     systems

    ,

     which

     will

     be

     an

     area

     of

     research

     and

     development

    .
    


    2

    .

     Integration

     with

     human

     emotions

    :

     AI

     will

     become

     more

     sophisticated

     and

     able

     to

     understand

     and

     respond

     to

     human

     emotions

    .

     This

     will

     enable

     more

     personalized

     and

     empath

    etic

     interactions

     between

     humans

     and

     machines

    .
    


    3

    .

     Increased

     automation

     of

     human

     jobs

    :

     AI

     is

     likely

     to

     automate

     many

     of

     the

     tasks

    



```python
llm.shutdown()
```
