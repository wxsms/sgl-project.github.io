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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.89it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:57,  5.22s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:57,  5.22s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:57,  5.22s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:57,  5.22s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:57,  5.22s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:42,  1.25it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:12,  3.75it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  7.98it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  7.98it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  7.98it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.26it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.26it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.26it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.26it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.26it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.26it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:02, 13.26it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:02, 13.26it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:02, 13.26it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:02, 13.26it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 19.76it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 19.76it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 19.76it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 19.76it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 19.76it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 19.76it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 19.76it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 19.76it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 19.76it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 19.76it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 27.16it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 27.16it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 27.16it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 27.16it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 27.16it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 27.16it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 27.16it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 27.16it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 27.16it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 27.16it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 27.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 14.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 14.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 14.57it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 14.57it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 19.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  19%|█▉        | 11/58 [00:03<00:18,  2.49it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:03<00:18,  2.49it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:03<00:18,  2.49it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:03<00:18,  2.49it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  24%|██▍       | 14/58 [00:03<00:11,  3.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:03<00:11,  3.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:03<00:11,  3.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:03<00:11,  3.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:03<00:11,  3.77it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  31%|███       | 18/58 [00:03<00:06,  6.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  31%|███       | 18/58 [00:03<00:06,  6.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  31%|███       | 18/58 [00:03<00:06,  6.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  31%|███       | 18/58 [00:03<00:06,  6.00it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  31%|███       | 18/58 [00:03<00:06,  6.00it/s] Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:03<00:04,  8.71it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:03<00:04,  8.71it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:03<00:04,  8.71it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:03<00:04,  8.71it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:03<00:04,  8.71it/s]

    Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.93it/s]Capturing num tokens (num_tokens=640 avail_mem=74.33 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.93it/s]Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.93it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.93it/s]Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.93it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.93it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:03<00:01, 16.61it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:03<00:01, 16.61it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:03<00:01, 16.61it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:03<00:01, 16.61it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:03<00:01, 16.61it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:03<00:01, 16.61it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:03<00:01, 21.29it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:03<00:01, 21.29it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:03<00:01, 21.29it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:03<00:01, 21.29it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:03<00:01, 21.29it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:03<00:01, 21.29it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:03<00:00, 25.53it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:03<00:00, 25.53it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:03<00:00, 25.53it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:03<00:00, 25.53it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:03<00:00, 25.53it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.92it/s]Capturing num tokens (num_tokens=112 avail_mem=74.26 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.92it/s]Capturing num tokens (num_tokens=96 avail_mem=74.00 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.92it/s] Capturing num tokens (num_tokens=80 avail_mem=74.25 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.92it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.24 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.92it/s]Capturing num tokens (num_tokens=64 avail_mem=74.24 GB):  84%|████████▍ | 49/58 [00:04<00:00, 22.13it/s]Capturing num tokens (num_tokens=48 avail_mem=74.00 GB):  84%|████████▍ | 49/58 [00:04<00:00, 22.13it/s]Capturing num tokens (num_tokens=32 avail_mem=74.21 GB):  84%|████████▍ | 49/58 [00:04<00:00, 22.13it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:04<00:00, 22.13it/s]Capturing num tokens (num_tokens=28 avail_mem=74.03 GB):  90%|████████▉ | 52/58 [00:04<00:00, 19.51it/s]Capturing num tokens (num_tokens=24 avail_mem=74.00 GB):  90%|████████▉ | 52/58 [00:04<00:00, 19.51it/s]Capturing num tokens (num_tokens=20 avail_mem=74.10 GB):  90%|████████▉ | 52/58 [00:04<00:00, 19.51it/s]Capturing num tokens (num_tokens=16 avail_mem=73.69 GB):  90%|████████▉ | 52/58 [00:04<00:00, 19.51it/s]

    Capturing num tokens (num_tokens=16 avail_mem=73.69 GB):  95%|█████████▍| 55/58 [00:04<00:00, 19.44it/s]Capturing num tokens (num_tokens=12 avail_mem=73.68 GB):  95%|█████████▍| 55/58 [00:04<00:00, 19.44it/s]Capturing num tokens (num_tokens=8 avail_mem=73.52 GB):  95%|█████████▍| 55/58 [00:04<00:00, 19.44it/s] Capturing num tokens (num_tokens=4 avail_mem=73.51 GB):  95%|█████████▍| 55/58 [00:04<00:00, 19.44it/s]Capturing num tokens (num_tokens=4 avail_mem=73.51 GB): 100%|██████████| 58/58 [00:04<00:00, 20.32it/s]Capturing num tokens (num_tokens=4 avail_mem=73.51 GB): 100%|██████████| 58/58 [00:04<00:00, 12.18it/s]


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
    Generated text:  Luiggi Sosio and I am a freelance designer from Italy. My skills include graphic design, content marketing, web development, website design, and content strategy. I am proficient in designing, creating, and optimizing websites for SEO, social media, and email marketing.
    As a freelance graphic designer, my main focus is to create visually appealing and functional websites for businesses and individuals. I specialize in web design, creating responsive and user-friendly websites that are mobile-friendly, accessible, and optimized for search engines. I am also experienced in web development using frameworks such as Bootstrap, jQuery, and React.
    In addition to my design work, I
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to have a great big banana republic or a small banana republic. The bananas grow at a constant rate, but the small republic has a smaller population and thus less bananas per capita. If the president decides to have the small republic, how many bananas would he expect to have in one year? Assume that the bananas grow at a constant rate and the population does not change.
    To determine how many bananas the president of the United States would expect to have in one year if the small banana republic grows at a constant rate, we need to consider the following:
    
    1. **Banana Growth Rate**: The bananas grow at a constant
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the capital of Germany is Berlin. Let's find out what country Paris is located in. The capital of Germany is Berlin, so Paris is located in France.
    Therefore, the answer is France.
    ===============================
    Prompt: The future of AI is
    Generated text:  bright
    
    It’s a challenge and it’s a great opportunity for entrepreneurs and researchers.
    
    We must have a clear view of AI and the future of AI.
    
    What is AI?
    
    AI (Artificial Intelligence) is the ability of a computer or machine to mimic or simulate human-like functions or abilities.
    
    AI systems are created to mimic human cognition, perception and decision-making processes and so far, they have been trained on large amounts of data. An AI system is a computer or a machine that is built to work on tasks that are based on intelligence.
    
    AI systems can be defined as computers and machine learning software, such as neural networks, that simulate


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a bustling city with a rich history and culture, and is a popular tourist destination. The city is also known for its fashion industry, with many famous designers and boutiques. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a unique and fascinating city. It is a city that has been a center of French culture and politics for centuries, and continues to be a major cultural and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies will continue to improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks, from simple tasks like image recognition to complex tasks like autonomous driving and decision-making in healthcare and finance. Additionally, AI will continue to be integrated into everyday life, from smart home devices to virtual assistants and chatbots. As AI becomes more integrated into our daily lives, we may see a shift towards more personalized and adaptive AI systems that can learn and adapt to our needs over time. Overall, the future of AI
    


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
    Generated text:  [Name] and I am a software engineer with a passion for [specific interest or technology]. I have been working in the industry for [X] years and have honed my skills through [specific projects or experiences]. I am always eager to learn and continue to push boundaries. I am excited to share my knowledge with anyone who is interested in technology and programming. How can I get to know you better? [Optional] Can you share a story or project that illustrates your passion for [specific interest or technology]? [Optional] What are some of your favorite programming languages and frameworks? [Optional] Do you have any hobbies or interests outside
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as the City of Light and the Eternal City.
    Paris is a city with a rich history, renowned for its architecture, art, and cultural attractions. Here are some key points about Paris:
    
    1. Population: As of 2021, Paris has a population of approximately 2.3 million people.
    
    2. Elevation: Paris is one of the highest cities in Europe, with an average elevation of 330 meters (1,084 feet) above sea level.
    
    3. Climate: Paris is known for its warm, wet climate, with a Mediterranean climate.
    
    4. Languages: French is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and it is likely to continue to evolve in many different ways. Here are some possible trends in AI in the coming years:
    
    1. Increased Integration with Other Technologies: AI will continue to integrate more with other technologies like the Internet of Things (IoT), machine learning, deep learning, natural language processing, and computer vision. As these technologies advance, they will become more integrated and complementary.
    
    2. Enhanced Capabilities for Humans: AI will continue to evolve and become even more capable of performing tasks that require human intelligence, like creativity, empathy, and decision-making.
    
    3. More Privacy Concerns: As AI becomes more prevalent


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

     John

     and

     I

    'm

     a

     writer

    .

     I

     love

     to

     write

     about

     everything

     from

     science

     fiction

     to

     fantasy

     and

     I

     enjoy

     exploring

     the

     depths

     of

     imagination

    .

     I

    'm

     currently

     working

     on

     a

     new

     novel

     and

     I

    'm

     excited

     to

     share

     my

     ideas

     and

     come

     up

     with

     some

     unique

     stories

    .

     What

     kind

     of

     projects

     do

     you

     have

     in

     mind

     for

     the

     future

    ?

     I

    'm

     currently

     working

     on

     a

     novel

     set

     in

     a

     future

     world

     where

     space

     exploration

     is

     the new

     normal

    .

     I

     hope

     to

     explore

     the

     dangers

     and

     rewards

     of

     exploring

     other

     galaxies

     and

     I

    'm

     eager

     to

     see

     what

     happens

     to

     the

     human

     species

     in

     this

     new

     era

     of

     space

     exploration

    .

     What

    's

     your

     favorite

     part

     of

     writing

     your

     stories

    ?

     Writing

     is

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     in

     France

     and

     is

     the

     capital

     of

     the

     country

    .

     It

     is

     located

     in

     the

     center

     of

     the

     country

    ,

     near

     the

     Mediterranean

     Sea

    ,

     and

     is

     home

     to

     numerous

     historical

     and

     cultural

     landmarks

    ,

     including

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

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     known

     for

     its

     rich

     French

     culture

    ,

     including

     music

    ,

     food

    ,

     and

     art

    ,

     and

     has

     a

     bustling

     street

     life

     and

     vibrant

     nightlife

    .

     It

     is

     a

     global

     hub

     for

     business

    ,

     culture

    ,

     and

     politics

    .

     The

     city

     is

     also

     home

     to

     a

     diverse

     population

     of

     around

     

    2

    .

    2

     million

     people

    ,

     making

     it

     one

     of

     the

     most

     densely

     populated

     cities

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     continued

     rapid

     advancements

     in

     areas

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     robotics

    ,

     and

     cyber

     security

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

     Increased

     integration

     with

     other

     technologies

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     may

     see

     more

     seamless

     integration

     with

     other

     technologies

     such

     as

     the

     Internet

     of

     Things

     (

    Io

    T

    ),

     blockchain

    ,

     and

     quantum

     computing

    .
    


    2

    .

     Development

     of

     AI

     in

     new

     domains

    :

     AI

     is

     already

     being

     used

     in

     a

     wide

     range

     of

     industries

     such

     as

     healthcare

    ,

     finance

    ,

     transportation

    ,

     and

     agriculture

    .

     As

     AI

     continues

     to

     evolve

    ,

     we

     may

     see

     new

     applications

     in

     fields

     such

     as

     education

    ,

     entertainment

    ,

     and

     entertainment

    .
    


    3

    .

    



```python
llm.shutdown()
```
