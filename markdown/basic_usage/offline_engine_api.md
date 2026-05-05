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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.26it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.26it/s]


    2026-05-05 16:36:21,967 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 16:36:21] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.74it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.74it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.59it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.99it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.99it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.99it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.99it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.99it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.99it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.99it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.99it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.99it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.99it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.99it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:04<00:01, 16.99it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 26.38it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 26.38it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 26.38it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 26.38it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 26.38it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 26.38it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 26.38it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:04<00:00, 26.38it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:04<00:00, 26.38it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:04<00:00, 26.38it/s]

    Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:04<00:00, 26.38it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 35.67it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 35.67it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 35.67it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 35.67it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 35.67it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 35.67it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 35.67it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.24it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:02, 19.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:02, 19.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:02, 19.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.65 GB):   3%|▎         | 2/58 [00:00<00:02, 19.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=69.65 GB):   9%|▊         | 5/58 [00:00<00:02, 22.87it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.64 GB):   9%|▊         | 5/58 [00:00<00:02, 22.87it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.63 GB):   9%|▊         | 5/58 [00:00<00:02, 22.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=69.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=69.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.78it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.78it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=69.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.78it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=69.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.59 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=69.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=69.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.55it/s]Capturing num tokens (num_tokens=960 avail_mem=69.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.55it/s] Capturing num tokens (num_tokens=896 avail_mem=69.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.55it/s]

    Capturing num tokens (num_tokens=832 avail_mem=69.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.55it/s]Capturing num tokens (num_tokens=832 avail_mem=69.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.17it/s]Capturing num tokens (num_tokens=768 avail_mem=69.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.17it/s]Capturing num tokens (num_tokens=704 avail_mem=69.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.17it/s]Capturing num tokens (num_tokens=640 avail_mem=69.53 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.17it/s]Capturing num tokens (num_tokens=576 avail_mem=69.53 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.17it/s]Capturing num tokens (num_tokens=512 avail_mem=69.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.17it/s]Capturing num tokens (num_tokens=512 avail_mem=69.52 GB):  50%|█████     | 29/58 [00:00<00:00, 43.01it/s]Capturing num tokens (num_tokens=480 avail_mem=69.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.01it/s]Capturing num tokens (num_tokens=448 avail_mem=69.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.01it/s]Capturing num tokens (num_tokens=416 avail_mem=69.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.01it/s]Capturing num tokens (num_tokens=384 avail_mem=69.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.01it/s]

    Capturing num tokens (num_tokens=352 avail_mem=69.52 GB):  50%|█████     | 29/58 [00:00<00:00, 43.01it/s]Capturing num tokens (num_tokens=352 avail_mem=69.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=320 avail_mem=69.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=288 avail_mem=69.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=256 avail_mem=69.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=240 avail_mem=69.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=224 avail_mem=69.50 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=208 avail_mem=69.50 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.02it/s]Capturing num tokens (num_tokens=208 avail_mem=69.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=192 avail_mem=69.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=176 avail_mem=69.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=160 avail_mem=69.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.85it/s]

    Capturing num tokens (num_tokens=144 avail_mem=69.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=128 avail_mem=69.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=128 avail_mem=69.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.59it/s]Capturing num tokens (num_tokens=112 avail_mem=69.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.59it/s]Capturing num tokens (num_tokens=96 avail_mem=69.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.59it/s] Capturing num tokens (num_tokens=80 avail_mem=69.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.59it/s]Capturing num tokens (num_tokens=64 avail_mem=69.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.59it/s]Capturing num tokens (num_tokens=48 avail_mem=69.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.59it/s]Capturing num tokens (num_tokens=48 avail_mem=69.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.79it/s]Capturing num tokens (num_tokens=32 avail_mem=69.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.79it/s]Capturing num tokens (num_tokens=28 avail_mem=69.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.79it/s]Capturing num tokens (num_tokens=24 avail_mem=69.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.79it/s]

    Capturing num tokens (num_tokens=20 avail_mem=69.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.79it/s]Capturing num tokens (num_tokens=16 avail_mem=69.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.79it/s]Capturing num tokens (num_tokens=16 avail_mem=69.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.26it/s]Capturing num tokens (num_tokens=12 avail_mem=69.45 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.26it/s]Capturing num tokens (num_tokens=8 avail_mem=69.45 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.26it/s] Capturing num tokens (num_tokens=4 avail_mem=69.44 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.26it/s]Capturing num tokens (num_tokens=4 avail_mem=69.44 GB): 100%|██████████| 58/58 [00:01<00:00, 42.44it/s]


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
    Generated text:  Kate, I'm a middle-aged woman. I have been in the front line of earthquake and tsunami disasters for many years. A few years ago, I worked in a community center, the center was very quiet. But a year ago, one of the community center workers, Mr. Wang, was electrocuted by the power grid. Because there were no electrical wires outside the community center, we had no way of knowing where Mr. Wang was, so I was on the phone with the police to search for Mr. Wang. Several hours later, I went back to the community center and found Mr. Wang badly injured. He was
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking executive officer of the United States government. Among the following options, which one is NOT a government organ of the United States? 
    A. U.S. Congress
    B. U.S. Supreme Court
    C. U.S. Department of Defense
    D. U.S. Department of Education
    Answer:
    
    D
    
    It's very cold outside. The students need to be very ________ because they will need to wear their warmest clothes.
    A. warm
    B. cool
    C. hot
    D. cold
    Answer:
    
    A
    
    Based on the given theme, which of the following poems is most suitable for a lesson
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. Lyon
    C. Lyon
    D. Bordeaux
    A capital is the largest city and administrative centre of a country. The capital of France is Paris. Therefore, the answer is A. Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  highly uncertain, but researchers have found a way to create a large database of images that can then be used to train AI systems. This could lead to significant advancements in areas such as computer vision and robotics.
    One of the biggest challenges in creating a large database of images is the difficulty of labeling and categorizing the images. This is because most of the images are not labeled, which means that it is difficult to know what exactly the image is about.
    However, researchers have found a way to overcome this challenge. They have developed a new algorithm that can automatically label and categorize images. This algorithm uses machine learning to recognize patterns in the images


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] years in the industry. I'm passionate about [What I Love to Do], and I'm always looking for ways to [What I Want to Improve]. I'm a [What I Want to Be], and I'm always striving to [What I Want to Achieve]. I'm excited to meet you and learn more about you. What's your name? What's your occupation? What's your skill? What's your passion? What's your goal? What's your desired state of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is the largest city in France and the second-largest city in the world by population. The city is known for its diverse cuisine, fashion, and art scene. Paris is also home to the French Parliament, the French Academy of Sciences, and the French National Library. It is a bustling city with a rich
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into many aspects of our lives, from self-driving cars to personalized medicine. As AI becomes more integrated into our daily lives, we can expect to see even more widespread adoption of AI in the future.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This could include issues such as bias, privacy, and transparency.
    
    3. Increased use of
    


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
    Generated text:  [Your Name], and I'm a [Title/Job/Position] with [Number of Years Experience] years of experience in [Industry/Field]. I'm passionate about [Why You're Interesting and Why You're Suitable for the Job]. I enjoy learning new things and staying up to date with the latest trends and technologies. I'm a [Intangible Quality or Trait] in my work, and I believe in [Why You're Successful]. I strive to be a role model for others and I believe in [My Vision for the Future]. Lastly, I'm always looking for new challenges and opportunities to learn and grow. I hope
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Grande-Bretagne," the largest city and seat of government of the country. It is a historical and cultural center, renowned for its architecture, art, and gastronomy. Paris is also the capital of France, with a population of over 15 million people, making it the most populous city in the European Union and the world's 10th most populous city by population. It is home to many famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The city is known for its vibrant nightlife, fashion industry, and cultural events,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a combination of emerging technologies, innovative research, and changing societal needs. Here are some potential trends in AI that are currently predicted and have the potential to impact the industry:
    
    1. AI ethics and accountability: As AI systems become more advanced, there will be a growing concern about their impact on human morality and responsibility. This includes issues such as bias, transparency, and accountability. Future AI systems will be required to be transparent about their decision-making processes, and must be held accountable for their actions.
    
    2. AI-driven automation: AI will continue to play an increasingly important role in manufacturing, healthcare, and other industries,


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

    ].

     I

     am

     a

     passionate

     [

    occup

    ational

     interest

     or

     hobby

    ].

     And

     I

    'm

     excited

     to

     share

     my

     love

     for

     [

    occupation

    ]

     with

     you

    .

     Let

    's

     make

     this

     experience

     unforgettable

    !

     

    🌟

    ✨

    
    


    ---
    


    Tell

     me

     more

     about

     yourself

    .

     What

     drives

     your

     passion

     for

     [

    occupation

    ]?

     What

     does

     it

     mean

     to

     you

    ?

     

    🌟

    ✨

    
    


    ---
    


    Tell

     us

     about

     a

     moment

     when

     you

    've

     faced

     a

     challenge

     or

     obstacle

     in

     your

     career

    .

     What

     did

     you

     learn

     from

     it

    ,

     and

     how

     did

     it

     shape

     your

     career

     path

    ?

     

    🌟

    ✨

    
    


    ---
    


    Describe

     a

     time

     when

     you

     faced

     a

     difficult

     decision

     that

     tested

     your

     character

    .

     What

     did

     you

     decide

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

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

     and

     it

    's

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     famous

     landmarks

     such

     as

     the

     Lou

    vre

     Museum

    ,

     and

     delicious

     food

     like

     cro

    iss

    ants

     and

     cheeses

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

     including

     the

     famous

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

     It

    's

     a

     bustling

     met

    ropolis

     with

     a

     rich

     culture

     and

     history

    .

     It

    's

     a

     popular

     tourist

     destination

     and

     a

     major

     hub

     of

     politics

     and

     business

     in

     France

    .

     Paris

     is

     also

     home

     to

     many

     famous

     museums

    ,

     including

     the

     Lou

    vre

    ,

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     the

     Pal

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     combination

     of

     technological

     advances

    ,

     policy

     changes

    ,

     and

     social

     and

     cultural

     shifts

    .

     Here

     are

     some

     possible

     trends

     that

     could

     emerge

     in

     the

     next

     few

     years

    :
    


    1

    .

     Increased

     focus

     on

     ethics

     and

     privacy

    :

     With

     the

     increasing

     amount

     of

     personal

     data

     being

     collected

     and

     analyzed

     by

     AI

     systems

    ,

     there

     will

     be

     a

     growing

     demand

     for

     frameworks

     and

     regulations

     that

     address

     privacy

     concerns

    .

     This

     could

     lead

     to

     more

     stringent

     data

     protection

     laws

     and

     increased

     scrutiny

     of

     AI

     systems

    .
    


    2

    .

     Expansion

     of

     AI

     applications

    :

     AI

     is

     likely

     to

     have

     a

     more

     widespread

     impact

     on

     society

     in

     the

     coming

     years

    ,

     with

     applications

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

     expected

     to

     grow

    .

    



```python
llm.shutdown()
```
