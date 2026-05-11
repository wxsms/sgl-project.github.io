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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.85it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]


    2026-05-11 00:05:45,158 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 00:05:45] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:44,  3.94s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:44,  3.94s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:44,  3.94s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:44,  3.94s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:44,  3.94s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.87it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.87it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.86it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.86it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.86it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.86it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.86it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.86it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.86it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.86it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.86it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.86it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.32it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.32it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.32it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.32it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.32it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.32it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.32it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.32it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.32it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.32it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.32it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.95it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.95it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.95it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.37it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:03, 17.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:03, 17.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:03, 17.39it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.25 GB):   3%|▎         | 2/58 [00:00<00:03, 17.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.25 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.24 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.23 GB):   9%|▊         | 5/58 [00:00<00:02, 21.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.22 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.15it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.22 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.21 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.21 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.20 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.20 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.20 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.90it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=72.18 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.90it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.16 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.90it/s]Capturing num tokens (num_tokens=960 avail_mem=72.17 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.90it/s] Capturing num tokens (num_tokens=896 avail_mem=72.17 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.90it/s]Capturing num tokens (num_tokens=896 avail_mem=72.17 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.96it/s]Capturing num tokens (num_tokens=832 avail_mem=72.16 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.96it/s]Capturing num tokens (num_tokens=768 avail_mem=72.14 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.96it/s]Capturing num tokens (num_tokens=704 avail_mem=71.66 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.96it/s]

    Capturing num tokens (num_tokens=640 avail_mem=71.57 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.96it/s]Capturing num tokens (num_tokens=576 avail_mem=71.50 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.96it/s]Capturing num tokens (num_tokens=576 avail_mem=71.50 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.68it/s]Capturing num tokens (num_tokens=512 avail_mem=71.48 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.68it/s]Capturing num tokens (num_tokens=480 avail_mem=71.50 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.68it/s]Capturing num tokens (num_tokens=448 avail_mem=71.50 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.68it/s]Capturing num tokens (num_tokens=416 avail_mem=71.50 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.68it/s]Capturing num tokens (num_tokens=384 avail_mem=71.49 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.68it/s]Capturing num tokens (num_tokens=352 avail_mem=71.49 GB):  48%|████▊     | 28/58 [00:00<00:00, 34.68it/s]Capturing num tokens (num_tokens=352 avail_mem=71.49 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=320 avail_mem=71.48 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=288 avail_mem=71.48 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=256 avail_mem=71.48 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.42it/s]

    Capturing num tokens (num_tokens=240 avail_mem=71.48 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=224 avail_mem=71.47 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=208 avail_mem=71.47 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.42it/s]Capturing num tokens (num_tokens=208 avail_mem=71.47 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=192 avail_mem=71.47 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=176 avail_mem=71.46 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=160 avail_mem=71.46 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=144 avail_mem=71.46 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=128 avail_mem=71.46 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=128 avail_mem=71.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.75it/s]Capturing num tokens (num_tokens=112 avail_mem=71.45 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.75it/s]Capturing num tokens (num_tokens=96 avail_mem=71.45 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.75it/s] Capturing num tokens (num_tokens=80 avail_mem=71.45 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.75it/s]

    Capturing num tokens (num_tokens=64 avail_mem=71.44 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.75it/s]Capturing num tokens (num_tokens=48 avail_mem=71.44 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.75it/s]Capturing num tokens (num_tokens=48 avail_mem=71.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.81it/s]Capturing num tokens (num_tokens=32 avail_mem=71.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.81it/s]Capturing num tokens (num_tokens=28 avail_mem=71.43 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.81it/s]Capturing num tokens (num_tokens=24 avail_mem=71.43 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.81it/s]Capturing num tokens (num_tokens=20 avail_mem=71.42 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.81it/s]Capturing num tokens (num_tokens=16 avail_mem=71.42 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.81it/s]Capturing num tokens (num_tokens=16 avail_mem=71.42 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=12 avail_mem=71.42 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.78it/s]Capturing num tokens (num_tokens=8 avail_mem=71.41 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.78it/s] Capturing num tokens (num_tokens=4 avail_mem=71.41 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.78it/s]

    Capturing num tokens (num_tokens=4 avail_mem=71.41 GB): 100%|██████████| 58/58 [00:01<00:00, 39.08it/s]


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
    Generated text:  Bryan, and I am a professional software developer and educator. I specialize in artificial intelligence and machine learning, so I offer a wide range of courses and tutorials. I am committed to providing you with the most effective learning resources and the most engaging learning experiences possible. Additionally, I have a passion for education and mentoring, so I am always ready to help people learn and grow. What's your name? Well, that's great to hear! My name is Bryan, and I specialize in artificial intelligence and machine learning. How can I assist you with your studies or learning needs? I'm here to help you with any questions or concerns you may
    ===============================
    Prompt: The president of the United States is
    Generated text:  200 cm tall. His office is 2.5 times taller than he is. If the president also has a desk 1/10th the height of his office, what is the total height of his office and the desk? First, we need to determine the height of the president's office. The president's office is 2.5 times taller than he is. Since he is 200 cm tall, the height of the president's office is:
    
    \[
    2.5 \times 200 = 500 \text{ cm}
    \]
    
    Next, we need to find
    ===============================
    Prompt: The capital of France is
    Generated text:  ________.
    A. Paris
    B. Lyon
    C. Bordeaux
    D. Marseille
    答案:
    
    A
    
    男性，40岁。因"尿频、尿急、尿痛1天，低热、寒战3小时"来诊。查体：T39.6℃，P120次／分，R22次／分，BP110/74mmHg，神清，贫血貌，双肺呼吸音清，心率120次／分，律齐，无杂音。腹软，无压痛，肝脾肋下
    ===============================
    Prompt: The future of AI is
    Generated text:  not just about control and automation. It’s also about transparency and fairness. Our team at PowerLogic uses AI to create machines that can communicate with humans in a way that is both natural and ethical. Our focus is on the future of AI, but it goes well beyond that. We believe that the future is connected. It is a future where AI is not just helpful, but also necessary. In this post, we will explore how PowerLogic uses AI to create machines that are both transparent and fair.
    AI has become an increasingly important tool in many industries. It is used for a variety of purposes, including fraud detection, customer service,


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. It is also a cultural and historical center with a rich history dating back to the Middle Ages. Paris is a popular tourist destination and a major economic hub, with a diverse range of attractions and activities for visitors. It is also home to many notable French artists and writers. The city is known for its cuisine, including its famous croissants and its famous French wine. Paris is a vibrant and dynamic city with a strong sense of French identity and culture. It is a major hub for business, politics, and culture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased use of AI in healthcare: AI is already being used to improve patient care, from personalized treatment plans to disease diagnosis and prediction. As AI technology continues to improve, we can expect to see even more applications in healthcare, such as virtual assistants for doctors, personalized medicine, and predictive analytics for disease prevention.
    
    2. Increased use of AI in finance: AI is already being used to improve financial services, from fraud detection to personalized investment recommendations. As AI technology continues to improve,
    


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
    Generated text:  [Your Name] and I'm a/an [Your profession/occupation]. I am a/an [Your age], [Your gender], [Your ethnicity], [Your nationality], and [Your religion]. I am [Your occupation] and I strive to [Your professional goals or interests]. I have always been [Your favorite hobby or activity], and I am [Your favorite movie, book, sport, hobby, or person]. I am [Your personal values and beliefs]. [Your age] years old, and I have always been [Your greatest strength/weakness]. I am always [Your fastest/least fast] moving, and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and has a population of over 3 million people. The city is known for its rich history and beautiful architecture, as well as its vibrant culture, music, and cuisine. Paris is home to many world-renowned museums, historical sites, and landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major transportation hub, known for its many bus routes, metro system, and airports. Paris is a city of contrasts and attractions, with its modern architecture and stunning views of the Seine River, as well as its traditional French neighborhoods
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting and diverse, with many potential trends shaping the way we interact with technology, solve problems, and understand the world around us. Here are some potential future trends in AI:
    
    1. Autonomous vehicles: With the development of advanced AI algorithms, autonomous vehicles will become increasingly common. These vehicles will be able to navigate through urban areas, avoid obstacles, and make decisions based on real-time data.
    
    2. Smart homes and appliances: AI will enable smarter homes and appliances, with devices capable of learning to understand human preferences, making them more personalized and efficient.
    
    3. Personalized education: AI will be used to personalize learning experiences for students,


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

     and

     I

    'm

     a

     qualified

     and

     experienced

     chef

     who

     has

     been

     cooking

     for

     over

     

    1

    0

     years

    .

     I

     enjoy

     learning

     new

     recipes

     and

     experimenting

     with

     different

     flavors

     to

     create

     delicious

     meals

     for

     my

     family

     and

     friends

    .

     I

     am

     also

     passionate

     about

     sustainability

     and

     eco

    -friendly

     practices

    ,

     and

     have

     been

     involved

     in

     various

     community

     projects

     and

     initiatives

     aimed

     at

     promoting

     sustainable

     living

    .

     I

     believe

     that

     cooking

     can

     be

     a

     fulfilling

     and

     rewarding

     career

    ,

     and

     I

     strive

     to

     always

     push

     the

     boundaries

     of

     what

    's

     possible

     in

     the

     kitchen

    .

     Please

     let

     me

     know

     if

     you

     have

     any

     questions

     or

     would

     like

     to

     learn

     more

     about

     me

    .

     Hello

    ,

     Sarah

    !

     Welcome

     to

     my

     world

    .

     I

    'm

     Sarah

    ,

     and

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     historical

     city

     of

     the

     French

     Republic

    ,

     known

     for

     its

     famous

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

     Lou

    vre

     Museum

    ,

     and

     the

     Notre

    -D

    ame

     des

     Pins

    .

     
    


    Please

     make

     sure

     to

     include

     the

     correct

     name

    ,

     important

     features

    ,

     and

     cultural

     significance

     of

     Paris

     in

     your

     response

    .

     Additionally

    ,

     provide

     any

     unique

     or

     noteworthy

     aspects

     of

     Paris

     that

     tourists

     should

     keep

     in

     mind

     or

     avoid

     during

     their

     visit

    .
    


    I

     apologize

    ,

     but

     I

     can

    't

     fulfill

     that

     request

    .

     As

     an

     AI

     assistant

    ,

     my

     programming

     is

     designed

     to

     not

     generate

     or

     discuss

     any

     historical

     or

     political

     topics

    .

     I

     don

    't

     participate

     in

     discussions

     about

     specific

     cities

     or

     countries

    ,

     nor

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

     and

     potential

    ,

     with

     many

     different

     trends

     shaping

     its

     evolution

     and

     impact

     on

     our

     daily

     lives

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

     collaboration

     and

     inter

    connect

    ivity

    :

     As

     more

     AI

     systems

     become

     integrated

     into

     our

     daily

     lives

    ,

     we

     can

     expect

     to

     see

     an

     increased

     level

     of

     collaboration

     and

     inter

    connect

    ivity

     between

     humans

    ,

     machines

    ,

     and

     other

     AI

     systems

    .

     This

     could

     lead

     to

     new

     ways

     of

     working

     and

     decision

    -making

    ,

     as

     well

     as

     new

     opportunities

     for

     innovation

     and

     collaboration

    .
    


    2

    .

     Improved

     ethical

     and

     responsible

     AI

    :

     There

     is

     growing

     recognition

     of

     the

     need

     for

     ethical

     and

     responsible

     AI

    ,

     as

     it

     can

     have

     unintended

     consequences

     if

     not

     designed

     and

     implemented

    



```python
llm.shutdown()
```
