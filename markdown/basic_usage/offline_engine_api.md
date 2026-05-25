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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.33it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.33it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 13.51it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 13.51it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 13.51it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 13.51it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 13.51it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 13.51it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 13.51it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.51it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 13.51it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:05<00:02, 13.51it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 20.68it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:05<00:01, 20.68it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 30.04it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 30.04it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.69 GB):   3%|▎         | 2/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.69 GB):   3%|▎         | 2/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.68 GB):   3%|▎         | 2/58 [00:00<00:03, 14.07it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.68 GB):   7%|▋         | 4/58 [00:00<00:03, 16.19it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.68 GB):   7%|▋         | 4/58 [00:00<00:03, 16.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.68 GB):   7%|▋         | 4/58 [00:00<00:03, 16.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.67 GB):   7%|▋         | 4/58 [00:00<00:03, 16.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.66 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.66 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.18it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=56.66 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.66 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.65 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.65 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.65 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.65 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.20it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=56.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.63 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.63 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.61 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=960 avail_mem=56.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.48it/s] Capturing num tokens (num_tokens=896 avail_mem=56.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=832 avail_mem=56.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=832 avail_mem=56.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.35it/s]Capturing num tokens (num_tokens=768 avail_mem=56.61 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.35it/s]

    Capturing num tokens (num_tokens=704 avail_mem=56.61 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.35it/s]Capturing num tokens (num_tokens=640 avail_mem=56.61 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.35it/s]Capturing num tokens (num_tokens=576 avail_mem=56.61 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.35it/s]Capturing num tokens (num_tokens=576 avail_mem=56.61 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.40it/s]Capturing num tokens (num_tokens=512 avail_mem=56.59 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.40it/s]Capturing num tokens (num_tokens=480 avail_mem=56.61 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.40it/s]Capturing num tokens (num_tokens=448 avail_mem=56.60 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.40it/s]Capturing num tokens (num_tokens=416 avail_mem=56.60 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.40it/s]Capturing num tokens (num_tokens=384 avail_mem=56.60 GB):  48%|████▊     | 28/58 [00:01<00:00, 37.40it/s]Capturing num tokens (num_tokens=384 avail_mem=56.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=352 avail_mem=56.59 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.34it/s]

    Capturing num tokens (num_tokens=320 avail_mem=56.59 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=288 avail_mem=56.27 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=256 avail_mem=56.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=240 avail_mem=56.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=240 avail_mem=56.55 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.27it/s]Capturing num tokens (num_tokens=224 avail_mem=56.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.27it/s]

    Capturing num tokens (num_tokens=208 avail_mem=56.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.27it/s]Capturing num tokens (num_tokens=192 avail_mem=56.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.27it/s]Capturing num tokens (num_tokens=176 avail_mem=56.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.27it/s]Capturing num tokens (num_tokens=176 avail_mem=56.52 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.84it/s]Capturing num tokens (num_tokens=160 avail_mem=56.34 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.84it/s]Capturing num tokens (num_tokens=144 avail_mem=56.35 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.84it/s]Capturing num tokens (num_tokens=128 avail_mem=56.37 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.84it/s]

    Capturing num tokens (num_tokens=112 avail_mem=56.37 GB):  72%|███████▏  | 42/58 [00:01<00:00, 29.84it/s]Capturing num tokens (num_tokens=112 avail_mem=56.37 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.35it/s]Capturing num tokens (num_tokens=96 avail_mem=56.49 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.35it/s] Capturing num tokens (num_tokens=80 avail_mem=56.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.35it/s]Capturing num tokens (num_tokens=64 avail_mem=56.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.35it/s]Capturing num tokens (num_tokens=48 avail_mem=56.47 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.35it/s]Capturing num tokens (num_tokens=48 avail_mem=56.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.58it/s]Capturing num tokens (num_tokens=32 avail_mem=56.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.58it/s]Capturing num tokens (num_tokens=28 avail_mem=56.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.58it/s]

    Capturing num tokens (num_tokens=24 avail_mem=56.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.58it/s]Capturing num tokens (num_tokens=20 avail_mem=56.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.58it/s]Capturing num tokens (num_tokens=20 avail_mem=56.44 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=16 avail_mem=56.44 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=12 avail_mem=56.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=8 avail_mem=56.43 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.67it/s] Capturing num tokens (num_tokens=4 avail_mem=56.42 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=4 avail_mem=56.42 GB): 100%|██████████| 58/58 [00:01<00:00, 31.79it/s]Capturing num tokens (num_tokens=4 avail_mem=56.42 GB): 100%|██████████| 58/58 [00:01<00:00, 30.13it/s]


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
    Generated text:  Maria. I'm a middle school student in the second year. I have a strong will and I can be very active. I like to use my imagination and I like to try new things. I love to read books and I like to listen to music. I also like to watch movies and read newspapers. I like to be creative. I also like to help people. I have some interesting ideas to tell you about. How do you like to stay fit and healthy? I like to go swimming or riding a bicycle. I also like to play sports. I go to the gym often. I have a good habit of eating healthy food
    ===============================
    Prompt: The president of the United States is
    Generated text:  a wealthy and influential person who represents the interests of the country. Which of the following statements is incorrect? 
    A. The president is elected by the people.
    B. The president is elected directly by the people.
    C. The president is appointed by the president of the United States.
    D. The president is appointed by the president of the United States and the legislative branch of the federal government.
    Answer:
    
    C
    
    The three major campaigns of the War of Resistance Against Japanese Aggression were ____.
    A. Liaoshen, Pingjin, and Huaihai Campaigns
    B. Jinan, Xuzhou, and Wuhan Campaign
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Lisbon
    C. Moscow
    D. Tokyo
    Answer: A
    
    The purpose of China's current budget management system is ____
    A. Balance budgeting
    B. Dual-sum budgeting
    C. Cost management
    D. Budget preparation
    Answer: A
    
    In a market economy, the core of the 'Six Major' policies is ____
    A. Fiscal Policy
    B. Monetary Policy
    C. Industrial Policy
    D. Financial Policy
    Answer: A
    
    In the process of market economy development, the main basis for adjusting the supply and demand relationship in the market is ____
    A.
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it is not without risks.
    The first step in a 2014 report from the Center for Digital Government said that by 2022, 66% of jobs in government will be replaced by AI, an 85% chance that 100% of jobs will be replaced by AI in 2030 and a 95% chance that 50% of jobs will be replaced by AI by 2050.
    The report said that AI has the potential to transform government services, making the nation’s money and resources more efficiently spent, making more people eligible


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


    Generated text:  Paris, which is known for its rich history, beautiful architecture, and vibrant culture. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a popular tourist destination and a major economic center in France. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to many famous landmarks and attractions, including the Champs-Élysées, the Eiffel Tower, and the Louvre Museum. Paris is a city of contrasts, with its modern and historic elements blending together to create a unique
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation: AI is expected to become more and more integrated into our daily lives, from manufacturing to customer service. This could lead to a significant increase in automation, where machines will be able to perform tasks that were previously done by humans.
    
    2. Improved privacy and security: As AI becomes more integrated into our lives, there will be an increased need for privacy and security. This could lead to new regulations and technologies that will help to
    


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
    Generated text:  [name], and I'm a [career or role] expert. I specialize in [specific skill or expertise], and I have [number of years of experience] in [field or industry]. I've always been fascinated by [topic or subject] and I'm always eager to learn and expand my knowledge. Whether you're looking for advice on [topic or subject], or you want to hear my insights on [topic or subject], I'm here to help. How can I assist you today? [Name] is passionate about helping others achieve their goals, and I'm excited to help anyone in need. [Name] is always ready
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, commonly referred to as the City of Light, known for its rich history, stunning architecture, and vibrant culture. The city is a major economic center and plays a significant role in France's political, cultural, and social life. Paris is often considered one of the world's most iconic cities and is home to numerous landmarks, museums, and cultural institutions. The city is also known for its cuisine, fashion, and wine. Paris is a popular tourist destination and is often referred to as "The City of Light" due to its vibrant nightlife and cultural events. Its history, art, and food culture make Paris a fascinating destination for those
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by continued advancements and changes in its applications, technologies, and potential applications. Here are some possible future trends in artificial intelligence:
    
    1. Autonomous vehicles: Self-driving cars may become a reality, leading to a significant reduction in accidents caused by human error. AI will also be used to optimize traffic flow, reduce traffic congestion and improve air quality.
    
    2. Medical diagnosis: AI will enable more accurate and precise diagnosis of diseases and health conditions. AI will be used to predict and prevent illnesses, and to develop personalized treatment plans.
    
    3. Personalized medicine: AI will be used to develop personalized treatment plans for diseases and health conditions


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

     am

     a

     [

    career

    ]

     [

    role

    ],

     [

    field

    ].

     I

     bring

     [

    strength

    /

    ability

    ]

     to

     [

    job

    ]

     at

     [

    company

    ].

     My

     name

     is

     [

    Name

    ]

     and

     I

     am

     a

     [

    career

    ]

     [

    role

    ],

     [

    field

    ].

     I

     bring

     [

    strength

    /

    ability

    ]

     to

     [

    job

    ]

     at

     [

    company

    ].

     I

     am

     a

     [

    career

    ]

     [

    role

    ],

     [

    field

    ],

     [

    person

    ].

     I

     bring

     [

    strength

    /

    ability

    ]

     to

     [

    job

    ]

     at

     [

    company

    ].

     I

     am

     a

     [

    career

    ]

     [

    role

    ],

     [

    field

    ],

     [

    person

    ].

     I

     bring

     [

    strength

    /

    ability

    ]

     to

     [

    job

    ]

     at

     [

    company

    ].

     I

     am

     a

    
    
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

     located

     on

     the

     Se

    ine

     river

    .

     It

     is

     known

     for

     its

     historic

     buildings

    ,

     world

    -ren

    owned

     museums

    ,

     and

     cultural

     scene

    .

     The

     city

     is

     also

     famous

     for

     its

     fashion

     industry

     and

     annual

     fashion

     week

    .

     Paris

     is

     a

     city

     of

     contrasts

     and

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

    ,

     including

     the

     Ancient

     Roman

     and

     French

     cities

    .

     It

     is

     a

     unique

     blend

     of

     traditional

     French

     culture

     and

     modern

    ity

    ,

     making

     it

     a

     popular

     destination

     for

     visitors

     from

     all

     over

     the

     world

    .

     The

     city

     is

     home

     to

     many

     famous

     landmarks

     and

     monuments

    ,

     including

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     promising

    ,

     with

     new

     technologies

     emerging

     at

     a

     rapid

     pace

    .

     Here

     are

     some

     possible

     trends

     in

     the

     field

    :
    


    1

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     already

     becoming

     more

     common

     in

     urban

     areas

    ,

     and

     we

     can

     expect

     them

     to

     become

     more

     widespread

     in

     the

     future

    .

     These

     vehicles

     will

     be

     able

     to

     learn

     from

     the

     environment

    ,

     communicate

     with

     each

     other

    ,

     and

     make

     decisions

     on

     their

     own

     without

     human

     intervention

    .
    


    2

    .

     Robotics

    :

     The

     use

     of

     robots

     in

     manufacturing

    ,

     healthcare

    ,

     and

     other

     industries

     is

     increasing

    ,

     and

     we

     can

     expect

     these

     robots

     to

     become

     more

     advanced

     in

     the

     future

    .

     Robots

     will

     be

     able

     to

     perform

     complex

     tasks

     and

     work

     in

     unpredictable

     environments

    .
    


    3

    .

     Personal

    ized

     healthcare

    



```python
llm.shutdown()
```
