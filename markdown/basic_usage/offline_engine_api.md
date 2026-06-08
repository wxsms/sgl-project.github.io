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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.39it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:18,  2.73it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:18,  2.73it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:18,  2.73it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:18,  2.73it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:18,  2.73it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:18,  2.73it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:18,  2.73it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:18,  2.73it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.43it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.43it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.43it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.43it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.43it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.43it/s]

    Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:06,  6.43it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:06,  6.43it/s] Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:06,  6.43it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 11.77it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 11.77it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 11.77it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 11.77it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 11.77it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 11.77it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:02, 11.77it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:02, 11.77it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:02, 11.77it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 18.07it/s]

    Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 18.07it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 25.24it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 25.24it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 25.24it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 25.24it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 25.24it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 25.24it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 25.24it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 25.24it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 25.24it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 33.04it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 33.04it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 33.04it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 33.04it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 33.04it/s]

    Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 33.04it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 33.04it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 33.04it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 33.04it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 33.04it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 42.09it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 42.09it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 42.09it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.56 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.56 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.56 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=54.56 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.56 GB):   3%|▎         | 2/58 [00:00<00:02, 18.75it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=54.56 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.55 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.54 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.54 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.54 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.54 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.53 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.53 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.52 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=54.52 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.52 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.52 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.51 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.51 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.51 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.50 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.50 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.23it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.50 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.50 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.48 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=960 avail_mem=54.49 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.82it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=54.49 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=832 avail_mem=54.49 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=832 avail_mem=54.49 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=768 avail_mem=54.48 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=704 avail_mem=54.48 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=640 avail_mem=54.48 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=576 avail_mem=54.48 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=512 avail_mem=54.46 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.89it/s]Capturing num tokens (num_tokens=512 avail_mem=54.46 GB):  50%|█████     | 29/58 [00:00<00:00, 41.90it/s]Capturing num tokens (num_tokens=480 avail_mem=54.48 GB):  50%|█████     | 29/58 [00:00<00:00, 41.90it/s]Capturing num tokens (num_tokens=448 avail_mem=54.48 GB):  50%|█████     | 29/58 [00:00<00:00, 41.90it/s]Capturing num tokens (num_tokens=416 avail_mem=54.48 GB):  50%|█████     | 29/58 [00:00<00:00, 41.90it/s]

    Capturing num tokens (num_tokens=384 avail_mem=54.47 GB):  50%|█████     | 29/58 [00:00<00:00, 41.90it/s]Capturing num tokens (num_tokens=352 avail_mem=54.47 GB):  50%|█████     | 29/58 [00:00<00:00, 41.90it/s]Capturing num tokens (num_tokens=352 avail_mem=54.47 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=320 avail_mem=54.46 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=288 avail_mem=54.46 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=256 avail_mem=54.46 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=240 avail_mem=54.45 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=224 avail_mem=54.45 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.27it/s]Capturing num tokens (num_tokens=224 avail_mem=54.45 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=208 avail_mem=54.45 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=192 avail_mem=54.45 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=176 avail_mem=54.44 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.34it/s]

    Capturing num tokens (num_tokens=160 avail_mem=54.44 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=144 avail_mem=54.44 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=144 avail_mem=54.44 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=128 avail_mem=54.43 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=112 avail_mem=54.43 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=96 avail_mem=54.43 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.03it/s] Capturing num tokens (num_tokens=80 avail_mem=54.42 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=64 avail_mem=54.42 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=64 avail_mem=54.42 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=48 avail_mem=54.42 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=32 avail_mem=54.41 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=28 avail_mem=54.41 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.02it/s]

    Capturing num tokens (num_tokens=24 avail_mem=54.41 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=20 avail_mem=54.40 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.02it/s]Capturing num tokens (num_tokens=20 avail_mem=54.40 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.29it/s]Capturing num tokens (num_tokens=16 avail_mem=54.40 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.29it/s]Capturing num tokens (num_tokens=12 avail_mem=54.40 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.29it/s]Capturing num tokens (num_tokens=8 avail_mem=54.40 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.29it/s] Capturing num tokens (num_tokens=4 avail_mem=54.39 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.29it/s]Capturing num tokens (num_tokens=4 avail_mem=54.39 GB): 100%|██████████| 58/58 [00:01<00:00, 40.44it/s]


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
    Generated text:  123456 and I am a customer service rep. I need help with my account.
    
    I have a problem with my account and I would like to contact my account manager to request assistance. However, I am not familiar with the account manager's name. 
    
    Can you please help me find the account manager's name and the specific issues I am having with my account? 
    
    Additionally, I would like to ask if there are any tools or resources available to assist me in my account, such as a customer service chat or a live chat. If there are none, please advise me on how to find such resources.
    
    Please
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to improve the effectiveness of his administration by bringing together the best ideas from his own and other political parties to create a single, unified front. This approach is known as _____. ____
    A. Party Politics
    B. New Party Politics
    C. Official Party Politics
    D. Popular Party Politics
    Answer:
    B
    
    Which of the following is NOT a cause of high blood pressure? ____ 
    A. Rapid heart rate
    B. Excessive fluid intake
    C. Lack of physical activity
    D. Excessive sodium intake
    Answer:
    A
    
    The purpose of implementing the principles of Marxism is to ____.
    A. Realize the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. ____
    A. 正确
    B. 错误
    答案:
    B
    
    某销售部门的职工要按工作需要安排住宿，该部门所用的客房数是____。
    A. 常数
    B. 变量
    C. 两个变量
    D. 三个变量
    答案:
    B
    
    某单位为了扩大其在市场上的知名度，决定在2018年3月1日，向2018年3月1日之前出生的人中任选1名，以购买一种新型电子设备，这种行为不违背民事法律的行为是____
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it's going to change the way we work. As an AI system, what can you do? Can you provide the following information:
    
    1. Explain the concept of artificial intelligence (AI) and its role in modern society.
    2. Describe the key components of a typical AI system, such as the learning algorithm, data processing pipeline, and decision-making process.
    3. Explain the different types of AI systems and their applications.
    4. Discuss the ethical considerations and legal issues surrounding AI, including issues related to privacy, bias, and accountability.
    5. Discuss the role of AI in healthcare, finance, and other industries, and


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


    Generated text:  [Name] and I am a [job title] at [company name]. I am excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I am excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I am excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I am excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is the largest city in France by population and is a major economic and political center. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a unique and fascinating
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can better understand and respond to human emotions and preferences.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could
    


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
    Generated text:  [Name] and I am [Age]. I have [Number] years of experience in [occupation or profession] and have been in the field for [Number of years]. Throughout my career, I have always been passionate about [reason for interest or hobby]. I enjoy [reason for enjoyment/excitement], and I believe that my passion has helped me to become the person I am today. 
    
    What is your favorite hobby or activity? What are your greatest accomplishments in your career so far? And what is your dream job or career objective? I would love to hear about your experiences and passions. [Name] [Occupation or Profession
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Garde". 
    
    Paris is the 6th largest city in the world by population and the 3rd largest by land area. It is the most populous city in France and the second most populous in the European Union. It is a major center of learning and culture, and is known for its historical landmarks such as the Louvre, Notre-Dame Cathedral, Champs-Élysées, and the Eiffel Tower. Its cuisine is also famous for its dishes such as crepes, bouillabaisse, and escargot. Paris has been the home of the French monarchy and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by exponential growth and diversification, with multiple AI technologies emerging that have the potential to transform industries and improve society as a whole. Some potential trends include:
    
    1. AI in healthcare: AI is already being used to improve patient outcomes in various healthcare settings, such as telemedicine and personalized medicine. As AI technology continues to evolve, we can expect to see even more innovative applications in this field.
    
    2. AI in finance: AI is already being used to improve trading decisions and fraud detection in the financial industry. As AI technology continues to evolve, we can expect to see even more innovations in this space.
    
    3. AI


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

    'm

     a

     friendly

    ,

     lo

    vable

    ,

     and

     helpful

     AI

     assistant

     that

     you

     can

     trust

     with

     all

     your

     questions

     and

     concerns

    .

     I

    'm

     always

     here

     to

     help

     you

     out

    ,

     and

     I

    'll

     do

     my

     best

     to

     assist

     you

     in

     every

     way

    .

     What

     can

     I

     do

     for

     you

     today

    ?

     Let

    's

     make

     it

     a

     positive

     experience

     together

    .

     How

     can

     I

     assist

     you

     today

    ?

     Are

     there

     specific

     topics

     or

     areas

     you

    'd

     like

     to

     know

     more

     about

    ?

     Let

     me

     know

     and

     I

    'll

     do

     my

     best

     to

     provide

     you

     with

     the

     information

     you

     need

    .

     What

     would

     you

     like

     to

     know

    ?

     Let

    's

     chat

     and

     get

     to

     know

     each

     other

     better

    .

     How

     can

     I

     get

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     the

     country

     and

     the

     largest

     metropolitan

     area

     by

     population

    .

     The

     city

     is

     famous

     for

     its

     beautiful

     architecture

    ,

     art

    ,

     and

     culinary

     traditions

    .

     It

     is

     also

     home

     to

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

     the

     Palace

     of

     Vers

    ailles

    .

     Paris

     is

     a

     major

     cultural

     and

     economic

     hub

    ,

     hosting

     numerous

     festivals

     and

     events

     throughout

     the

     year

    .

     It

     is

     also

     known

     for

     its

     important

     role

     in

     French

     foreign

     policy

     and

     its

     historical

     significance

     in

     French

     history

    .

     The

     city

     is

     home

     to

     many

     renowned

     universities

    ,

     including

     the

     University

     of

     Paris

     and

     the

     Sor

    bon

    ne

    .

     Paris

     is

     also

     a

     popular

     tourist

     destination

    ,

     with

     many

     visitors

     coming

     to

     explore

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     will

     likely

     bring

     a

     host

     of

     exciting

     developments

    .

     Here

     are

     some

     possible

     trends

    :
    


    1

    .

     More

     personalized

     and

     contextual

     AI

    :

     With

     the

     increasing

     amount

     of

     data

     and

     the

     ability

     to

     analyze

     it

     more

     accurately

    ,

     AI

     will

     become

     more

     personalized

     and

     contextual

    ,

     so

     that

     machines

     can

     understand

     human

     behavior

     and

     preferences

     in

     a

     way

     that

     humans

     are

     not

    .

     For

     example

    ,

     AI

     could

     be

     used

     to

     create

     personalized

     recommendations

     for

     insurance

    ,

     banking

    ,

     or

     healthcare

    ,

     based

     on

     an

     individual

    's

     health

     and

     financial

     information

    .
    


    2

    .

     Increased

     transparency

     and

     accountability

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     we

     will

     need

     to

     make

     sure

     that

     they

     are

     transparent

     and

     accountable

     to

     their

     users

    .

     This

     means

     that

     we

    



```python
llm.shutdown()
```
