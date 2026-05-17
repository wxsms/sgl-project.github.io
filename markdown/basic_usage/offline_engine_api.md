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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.33it/s]


    2026-05-17 00:18:47,489 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-17 00:18:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:17,  2.85it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.72it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.72it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.72it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.72it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.72it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.72it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:06,  6.72it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:06,  6.72it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:06,  6.72it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]

    Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 18.77it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 27.15it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 27.15it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 27.15it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 27.15it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 35.93it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 35.93it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 35.93it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 35.93it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 35.93it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 35.93it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 35.93it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 35.93it/s]

    Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 35.93it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 35.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.16it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.14 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   3%|▎         | 2/58 [00:00<00:02, 18.85it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.13 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.12 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.11 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.36it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.11 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.10 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.10 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.36it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.09 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.36it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.09 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.60it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.09 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.60it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.09 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.08 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.08 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.08 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.08 GB):  31%|███       | 18/58 [00:00<00:01, 35.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.07 GB):  31%|███       | 18/58 [00:00<00:01, 35.66it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.07 GB):  31%|███       | 18/58 [00:00<00:01, 35.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.05 GB):  31%|███       | 18/58 [00:00<00:01, 35.66it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.07 GB):  31%|███       | 18/58 [00:00<00:01, 35.66it/s] Capturing num tokens (num_tokens=896 avail_mem=72.07 GB):  31%|███       | 18/58 [00:00<00:01, 35.66it/s]Capturing num tokens (num_tokens=896 avail_mem=72.07 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.11it/s]Capturing num tokens (num_tokens=832 avail_mem=72.06 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.11it/s]Capturing num tokens (num_tokens=768 avail_mem=72.06 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.11it/s]Capturing num tokens (num_tokens=704 avail_mem=72.06 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.11it/s]Capturing num tokens (num_tokens=640 avail_mem=72.05 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.11it/s]Capturing num tokens (num_tokens=576 avail_mem=72.05 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.11it/s]Capturing num tokens (num_tokens=576 avail_mem=72.05 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=512 avail_mem=72.04 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=480 avail_mem=72.05 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=448 avail_mem=72.05 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.51it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.05 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=384 avail_mem=72.05 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.51it/s]Capturing num tokens (num_tokens=384 avail_mem=72.05 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=352 avail_mem=72.04 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=320 avail_mem=72.03 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=288 avail_mem=72.03 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=256 avail_mem=72.03 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=240 avail_mem=72.03 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=240 avail_mem=72.03 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.54it/s]Capturing num tokens (num_tokens=224 avail_mem=72.02 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.54it/s]Capturing num tokens (num_tokens=208 avail_mem=72.02 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.54it/s]Capturing num tokens (num_tokens=192 avail_mem=72.02 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.54it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.01 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.54it/s]Capturing num tokens (num_tokens=160 avail_mem=72.01 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.54it/s]Capturing num tokens (num_tokens=160 avail_mem=72.01 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=144 avail_mem=72.01 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=128 avail_mem=72.01 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=112 avail_mem=72.00 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=96 avail_mem=72.00 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.24it/s] Capturing num tokens (num_tokens=80 avail_mem=72.00 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=80 avail_mem=72.00 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=64 avail_mem=71.99 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=48 avail_mem=71.99 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=32 avail_mem=71.99 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.24it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.98 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=24 avail_mem=71.98 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.24it/s]Capturing num tokens (num_tokens=24 avail_mem=71.98 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.30it/s]Capturing num tokens (num_tokens=20 avail_mem=71.98 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.30it/s]Capturing num tokens (num_tokens=16 avail_mem=71.98 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.30it/s]Capturing num tokens (num_tokens=12 avail_mem=71.97 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.30it/s]Capturing num tokens (num_tokens=8 avail_mem=71.97 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.30it/s] Capturing num tokens (num_tokens=4 avail_mem=71.96 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.30it/s]Capturing num tokens (num_tokens=4 avail_mem=71.96 GB): 100%|██████████| 58/58 [00:01<00:00, 45.41it/s]Capturing num tokens (num_tokens=4 avail_mem=71.96 GB): 100%|██████████| 58/58 [00:01<00:00, 40.25it/s]


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
    Generated text:  Matthew, and I'm currently a Computer Science major at the University of Toronto. My primary area of interest is the field of machine learning. I am also an active member of various academic, professional, and community groups related to my field of study, such as the Canadian Association of Artificial Intelligence, the Society for Industrial and Applied Mathematics, and the Association for Computing Machinery. My current graduate studies are focused on unsupervised learning, with a particular emphasis on unsupervised clustering algorithms. I am also a member of the Machine Learning and Data Science Club, and I am involved in various outreach activities related to data science. Lastly, I have
    ===============================
    Prompt: The president of the United States is
    Generated text:  a powerful person. It is part of the United States constitution and is mentioned in the first sentence of the Constitution. The President is the head of the executive branch and the commander in chief of the armed forces. He has a lot of power to make decisions and carry out his duties. However, he also has a lot of duties to carry out. The first President, George Washington, was a commander in chief of the Continental Army, which was a big part of the United States. He had to be ready to defend the country against other countries that were invading it. The Second President, James Madison, was a very smart man who also
    ===============================
    Prompt: The capital of France is
    Generated text:  _______. A. London B. Paris C. Moscow D. Rome D. Rome
    
    The capital of France is Paris. London, Moscow, and Rome are capitals of different countries.
    
    The correct answer is: B. Paris. 
    
    The other options are not capitals of France:
    
    - London is the capital of the United Kingdom.
    - Moscow is the capital of Russia.
    - Rome is the capital of Italy. 
    
    However, Paris is the capital of France, which is a different country and language from the United Kingdom, Russia, and Italy. 
    
    Therefore, the correct answer is D. Rome. However, I see a typo in the
    ===============================
    Prompt: The future of AI is
    Generated text:  to create new types of chips. The high-end chips are currently being manufactured on the CPU. The "higher the price, the higher the quality" is an expression of the relationship between price and quality. The concept of "higher price, higher quality" is an expression of the relationship between price and technology. From this, we can infer that the relationship between price and the future of AI is ____. A. The higher the price, the higher the quality B. The higher the price, the higher the technology C. The higher the price, the lower the quality D. The higher the price, the lower the technology
    Answer:
    
    


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I have a [number] degree in [field of study], and I'm always looking for new challenges and opportunities to grow and learn. I'm a [character trait] person, and I'm always ready to learn and grow. What's your favorite hobby or activity? I love [mention a hobby or activity]. What's your favorite book or movie? I love [mention
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on modern France. Paris is also home to many famous French artists, writers, and musicians. The city is known for its cuisine, including its famous Parisian cuisine, and its fashion industry. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could emerge in the coming years:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence. This could lead to more sophisticated forms of AI that can learn from human behavior and adapt to new situations.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for the development and use of AI.
    
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
    Generated text:  [Name], and I'm a [insert occupation] who has been helping people for [insert number of years] years. I'm constantly learning and improving, always looking for ways to help and make a positive impact in the world. I'm available for consultations, coaching, and working on projects with you. Your success is my greatest goal, and I'm excited to help you achieve it. [Name] [Insert profession or title] [Name] [Insert profession or title]
    As an AI language model, I don't have a physical presence, but I'm here to assist you with any questions or tasks you have. How can
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city and most populous city in the country. Its origins trace back to the Roman Empire, where it was an important commercial and cultural hub. Paris is a major city and the economic, financial, and cultural center of France, and also the largest city in the world by population. It is home to many iconic landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is known for its cuisine, nightlife, and fashion, and is considered one of the most important cultural centers in the world. Its reputation as a melting pot of cultures and diverse social scene is reflected in its annual
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be diverse and innovative, driven by new technologies and advancements in data science, machine learning, and deep learning. Here are some possible future trends in AI:
    
    1. Increased integration with human cognition: AI is already showing signs of integrating more deeply with human cognition, allowing machines to learn and adapt more effectively to the world around us. This could lead to even greater integration in the future, with machines becoming more like humans in their ability to perceive, think, and act.
    
    2. Enhanced creativity: AI is already capable of generating creative art and literature, but it's still limited by human creativity. As AI becomes more capable of generating


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

     [

    Age

    ]

     year

    -old

     [

    Occup

    ation

    ].

     I

     recently

     graduated

     from

     [

    University

     or

     School

    ].

     In

     my

     free

     time

    ,

     I

     enjoy

     [

    or

     [

    or

     something

     else

     that

     interests

     you

    ]

    ].

     
    


    What

     would

     you

     like

     to

     know

     about

     me

    ?

     Or

     maybe

     I

     should

     change

     my

     name

     to

     [

    Name

    ]

     instead

    .

     How

     about

     it

    ?

     If

     you

     have

     a

     question

    ,

     feel

     free

     to

     ask

     me

     anything

    ,

     and

     I

    'll

     answer

     you

     politely

     and

     with

     confidence

    .

     I

    'm

     here

     for

     you

    ,

     always

    .

     

    😊

    
    


    This

     is

     a

     nice

     intro

    ,

     but

     I

     feel

     like

     it

     could

     be

     more

     polished

    .

     Can

     you

     suggest

     any

     tips

     on

     how

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     Se

    ine

     River

     and

     considered

     one

     of

     the

     most

     famous

     cities

     in

     the

     world

    .

     Paris

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

     the

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    ,

     as

     well

     as

     its

     vibrant

     street

     life

    ,

     fashion

     scene

    ,

     and

     French

     culture

    .

     Paris

     is

     also

     known

     for

     its

     annual

     fashion

     week

    ,

     film

     festival

    ,

     and

     numerous

     museums

     and

     historical

     sites

    .

     With

     its

     rich

     history

     and

     unique

     culture

    ,

     Paris

     continues

     to

     be

     a

     major

     hub

     for

     trade

    ,

     tourism

    ,

     and

     art

     in

     the

     world

    .

     It

    's

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    ,

     with

     millions

     of

     visitors

     annually

    .

     The

     capital

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

     and

     possibilities

     that

     are

     yet

     to

     be

     discovered

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

     Adv

    ancements

     in

     natural

     language

     processing

    :

     With

     advances

     in

     machine

     learning

     and

     artificial

     intelligence

    ,

     we

     may

     see

     improvements

     in

     natural

     language

     processing

    ,

     enabling

     machines

     to

     better

     understand

     and

     interpret

     human

     speech

    .

     This

     could

     lead

     to

     more

     human

    -like

     interactions

     and

     smarter

     personal

     assistants

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     With

     advancements

     in

     AI

    ,

     we

     may

     see

     increased

     use

     of

     AI

     in

     healthcare

     to

     help

     diagnose

     diseases

    ,

     predict

     health

     outcomes

    ,

     and

     improve

     patient

     care

    .

     This

     could

     lead

     to

     more

     personalized

     and

     effective

     treatments

     for

     patients

    .
    


    3

    .

     More

     ethical

     AI

    :

     As

    



```python
llm.shutdown()
```
