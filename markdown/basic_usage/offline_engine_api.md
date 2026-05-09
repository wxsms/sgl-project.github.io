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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.46it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.46it/s]


    2026-05-09 10:26:13,673 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 10:26:13] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:44,  3.93s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:44,  3.93s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:44,  3.93s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:44,  3.93s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:44,  3.93s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.88it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03, 10.27it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03, 10.27it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03, 10.27it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:01, 15.22it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 22.08it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 22.08it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:00, 22.08it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:00, 22.08it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:00, 22.08it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:00, 22.08it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:00, 22.08it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:00, 22.08it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:00, 22.08it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:04<00:00, 22.08it/s]Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:04<00:00, 22.08it/s]

    Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 31.90it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 31.90it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 31.90it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 31.90it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 31.90it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 31.90it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:04<00:00, 31.90it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:04<00:00, 31.90it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:04<00:00, 31.90it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:04<00:00, 31.90it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:04<00:00, 31.90it/s]Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:04<00:00, 31.90it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:04<00:00, 43.67it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:04<00:00, 43.67it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.88 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.85 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.85 GB):   3%|▎         | 2/58 [00:00<00:02, 19.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.85 GB):   3%|▎         | 2/58 [00:00<00:02, 19.08it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.85 GB):   3%|▎         | 2/58 [00:00<00:02, 19.08it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.85 GB):   3%|▎         | 2/58 [00:00<00:02, 19.08it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.85 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.84 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.83 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.83 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.83 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.83 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.35it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.82 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.35it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.82 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.82 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.35it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.81 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.35it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=71.81 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.35it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.81 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.81 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.80 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.80 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.80 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.79 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.79 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.79 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.77 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]Capturing num tokens (num_tokens=960 avail_mem=71.79 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s] Capturing num tokens (num_tokens=896 avail_mem=71.78 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]

    Capturing num tokens (num_tokens=832 avail_mem=71.78 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.32it/s]Capturing num tokens (num_tokens=832 avail_mem=71.78 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=768 avail_mem=71.78 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=704 avail_mem=71.77 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=640 avail_mem=71.77 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=576 avail_mem=71.77 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=512 avail_mem=71.75 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=512 avail_mem=71.75 GB):  50%|█████     | 29/58 [00:00<00:00, 44.36it/s]Capturing num tokens (num_tokens=480 avail_mem=71.77 GB):  50%|█████     | 29/58 [00:00<00:00, 44.36it/s]Capturing num tokens (num_tokens=448 avail_mem=71.77 GB):  50%|█████     | 29/58 [00:00<00:00, 44.36it/s]Capturing num tokens (num_tokens=416 avail_mem=71.77 GB):  50%|█████     | 29/58 [00:00<00:00, 44.36it/s]Capturing num tokens (num_tokens=384 avail_mem=71.76 GB):  50%|█████     | 29/58 [00:00<00:00, 44.36it/s]Capturing num tokens (num_tokens=352 avail_mem=71.76 GB):  50%|█████     | 29/58 [00:00<00:00, 44.36it/s]

    Capturing num tokens (num_tokens=320 avail_mem=71.75 GB):  50%|█████     | 29/58 [00:00<00:00, 44.36it/s]Capturing num tokens (num_tokens=320 avail_mem=71.75 GB):  60%|██████    | 35/58 [00:00<00:00, 46.40it/s]Capturing num tokens (num_tokens=288 avail_mem=71.75 GB):  60%|██████    | 35/58 [00:00<00:00, 46.40it/s]Capturing num tokens (num_tokens=256 avail_mem=71.75 GB):  60%|██████    | 35/58 [00:00<00:00, 46.40it/s]Capturing num tokens (num_tokens=240 avail_mem=71.74 GB):  60%|██████    | 35/58 [00:00<00:00, 46.40it/s]Capturing num tokens (num_tokens=224 avail_mem=71.74 GB):  60%|██████    | 35/58 [00:00<00:00, 46.40it/s]Capturing num tokens (num_tokens=208 avail_mem=71.74 GB):  60%|██████    | 35/58 [00:00<00:00, 46.40it/s]Capturing num tokens (num_tokens=192 avail_mem=71.74 GB):  60%|██████    | 35/58 [00:00<00:00, 46.40it/s]Capturing num tokens (num_tokens=192 avail_mem=71.74 GB):  71%|███████   | 41/58 [00:01<00:00, 47.68it/s]Capturing num tokens (num_tokens=176 avail_mem=71.73 GB):  71%|███████   | 41/58 [00:01<00:00, 47.68it/s]Capturing num tokens (num_tokens=160 avail_mem=71.73 GB):  71%|███████   | 41/58 [00:01<00:00, 47.68it/s]Capturing num tokens (num_tokens=144 avail_mem=71.73 GB):  71%|███████   | 41/58 [00:01<00:00, 47.68it/s]Capturing num tokens (num_tokens=128 avail_mem=71.73 GB):  71%|███████   | 41/58 [00:01<00:00, 47.68it/s]

    Capturing num tokens (num_tokens=112 avail_mem=71.72 GB):  71%|███████   | 41/58 [00:01<00:00, 47.68it/s]Capturing num tokens (num_tokens=112 avail_mem=71.72 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.96it/s]Capturing num tokens (num_tokens=96 avail_mem=71.72 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.96it/s] Capturing num tokens (num_tokens=80 avail_mem=71.72 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.96it/s]Capturing num tokens (num_tokens=64 avail_mem=71.71 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.96it/s]

    Capturing num tokens (num_tokens=48 avail_mem=71.71 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.96it/s]Capturing num tokens (num_tokens=32 avail_mem=71.71 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.96it/s]Capturing num tokens (num_tokens=32 avail_mem=71.71 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=28 avail_mem=71.70 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=24 avail_mem=71.70 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=20 avail_mem=71.69 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=16 avail_mem=71.69 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.72it/s]

    Capturing num tokens (num_tokens=16 avail_mem=71.69 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=12 avail_mem=71.69 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=8 avail_mem=71.69 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.57it/s] Capturing num tokens (num_tokens=4 avail_mem=71.68 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.57it/s]Capturing num tokens (num_tokens=4 avail_mem=71.68 GB): 100%|██████████| 58/58 [00:01<00:00, 37.04it/s]


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
    Generated text:  Matt and I'm a software engineer. In this role, I've worked with a variety of programming languages and platforms, from Node.js to Python to Java. I'm a big fan of Python and enjoy solving real-world problems using Python. I've also been a contributor to the Flask framework, a popular web framework. Currently, I'm working on a Python project called "DataScienceToolbox" that focuses on data science and machine learning. My current projects include setting up a machine learning pipeline using TensorFlow for a Python application and implementing data preprocessing and model training using TensorFlow 2.0. I'm also currently exploring the use of TensorFlow
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He has many important jobs to do. So, he always takes good care of himself and others. If he has an accident, he will be badly hurt. He needs to be in hospital for a long time. But, he is very kind to people. He always helps people and is always ready to lend a hand when needed. He likes to play sports and watch TV. But he is very good at studying. He is very important to his people. Many people think the president is very important. He makes all of the important decisions. He's also very good at running the country. He always takes good
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and the population is around 2.3 million people.
    
    What type of literary device is this sentence?
    
    a) Foreshadowing
    b) Implied
    c) Allusion
    d) Metaphor
    
    To determine the type of literary device in the sentence "The capital of France is Paris, and the population is around 2. 3 million people," let's analyze the sentence step by step.
    
    1. **Identify the key elements:**
       - The main element is "Paris" as the capital of France.
       - The secondary element is "population" as the population of France.
    
    2.
    ===============================
    Prompt: The future of AI is
    Generated text:  fast approaching, but we’ve got a lot of interesting research to do to understand it. But one part of the research that is really exciting is the “Emotional AI” component. As it is now, AI is only good at logical reasoning, but we’ll need it to be even better at emotion as well.
    A recent paper from DeepMind announced that they’ve created a language model which can understand and respond to emotional cues. The results are that they can actually recognize when someone is upset, happy, or neutral. The language model can also use the information to predict what kind of response it will get, which can then be used


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. Paris is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin, as well as a vibrant arts scene and a thriving food and fashion industry. The city is also known for its fashion and wine industries, and is a popular tourist destination for its beautiful architecture and cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt in ways that are difficult for humans to do. This could lead to more efficient and effective decision-making, as well as better human-computer interaction.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI becomes more advanced, it is likely to be used in even more areas, including personalized medicine, drug discovery, and patient monitoring.
    
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
    Generated text:  [Name] and I'm a [Job Title] at [Company Name]. I'm passionate about [Job Title] and love to [mention a specific skill or action that relates to your profession]. I'm always eager to learn new things and contribute to the success of [Company Name]. I strive to be [describe your character trait or personality], and I'm always up for challenges. I'm confident and always willing to work hard to achieve my goals. Thank you for considering me for a job at [Company Name]. Looking forward to hearing from you! [Job Title] at [Company Name] I'm a [Job Title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its historical importance, vibrant culture, and iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major transportation hub, home to several international airports and a bustling food and fashion scene. Its rich history and vibrant culture make it a city worth visiting for anyone interested in France. Paris is a city that brings together the best of Europe. Paris has a unique blend of old-world charm and modern innovation, making it a place of cultural richness and urban excitement. The city's famous landmarks and museums are a must-see for any visitor. Paris is a cultural hub
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid progress and innovation, with new technologies, advances in data analysis, and developments in machine learning techniques. Here are some possible trends in AI in the coming years:
    
    1. Increased use of AI for autonomous vehicles: Autonomous vehicles will become more common, with more and more self-driving cars on the road. AI will be used to develop algorithms that can interpret traffic patterns, identify pedestrians and other vehicles, and navigate roads with precision.
    
    2. AI will become more integrated into healthcare: AI will be used in healthcare to improve patient outcomes, reduce costs, and improve medical research. AI will be used to analyze medical records


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

    'm

     a

     [

    job

     title

    ]

     in

     [

    your

     current

     company

    ].

     I

     have

     a

     background

     in

     [

    relevant

     field

    ]

     and

     have

     been

     working

     in

     [

    mention

     the

     last

     few

     years

     of

     your

     career

    ]

     at

     [

    your

     previous

     company

    ].

     I

     am

     currently

     working

     as

     a

     [

    job

     title

    ]

     in

     [

    your

     current

     company

    ],

     and

     I

     specialize

     in

     [

    mention

     your

     expertise

     or

     specialty

    ].

     I

     enjoy

     [

    mention

     any

     personal

     interests

     or

     hobbies

     you

     have

    ].

     I

     believe

     in

     [

    mention

     your

     values

     or

     beliefs

    ],

     and

     I

     am

     always

     up

    -to

    -date

     with

     the

     latest

     trends

     and

     technologies

    .

     I

     am

     a

     [

    mention

     any

     qualifications

     or

     certifications

     you

     hold

    ],

     and

     I

     am

     always

     committed

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    P

    .S

    .

     If

     you

     need

     a

     summary

     of

     Paris

    ,

     I

     have

     included

     that

     information

     in

     my

     response

    .

     
    


    [

    summary

    ]:

     The

     capital

     of

     France

     is

     Paris

    .

     [

    summary

    ]
    


    P

    .S

    .

     If

     you

     need

     a

     summary

     of

     Paris

    ,

     please

     ask

    !

     I

    've

     included

     that

     information

     in

     my

     response

    .

     
    


    [

    summary

    ]:

     The

     capital

     of

     France

     is

     Paris

    .

     [

    summary

    ]

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     highly

     dependent

     on

     continued

     advancements

     in

     technology

     and

     research

    ,

     and

     it

     is

     likely

     to

     continue

     to

     evolve

     and

     change

     rapidly

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

     and

     deep

     learning

    :

     AI

     experts

     believe

     that

     machine

     learning

     and

     deep

     learning

     will

     continue

     to

     be

     the

     dominant

     technologies

     for

     AI

     in

     the

     coming

     years

    ,

     and

     that

     they

     will

     be

     applied

     to

     a

     wide

     range

     of

     tasks

    ,

     from

     image

     and

     speech

     recognition

     to

     natural

     language

     processing

     and

     predictive

     analytics

    .
    


    2

    .

     Greater

     focus

     on

     ethical

     and

     legal

     considerations

    :

     As

     AI

     becomes

     more

     prevalent

     in

     our

     daily

     lives

    ,

     there

     will

     be

     a

     growing

     emphasis

     on

     ensuring

     that

    



```python
llm.shutdown()
```
