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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.29it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:56,  4.16s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:56,  4.16s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:56,  4.16s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:56,  4.16s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:56,  4.16s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.56it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.56it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.56it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.56it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.56it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.56it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.56it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.56it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.63it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.36it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.36it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.36it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.36it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.36it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.36it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.36it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.36it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.36it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.36it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.64it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.64it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.64it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.64it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.64it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.64it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.64it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.64it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.64it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.64it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.64it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.09it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.09it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.09it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.09it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.09it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.09it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.09it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.09it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.09it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.09it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.09it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.39it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.39it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.39it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.39it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.39it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.39it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.39it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.39it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.25it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.26it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.14 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.26it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.26it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.26it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  21%|██        | 12/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.12 GB):  21%|██        | 12/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  21%|██        | 12/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.95it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.95it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.40it/s] Capturing num tokens (num_tokens=896 avail_mem=74.10 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.19it/s]Capturing num tokens (num_tokens=704 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.19it/s]

    Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.19it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.19it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.19it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 29.71it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 29.71it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 29.71it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=288 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.72it/s]

    Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.58it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.58it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.58it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.58it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.58it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.58it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.29it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.17it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 36.95it/s]


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
    Generated text:  Lily and I'm a college student. I want to live in a new city and I'm looking for a good apartment in a desirable area. 
    
    I'm considering a 1-bedroom apartment in the city center. The area is very clean and safe, and the only trouble I have is that I don't have an elevator. But I don't want to call the elevator every day. I prefer to use the stairs. 
    
    The apartment is located on the 3rd floor, and it has a large windowsill that overlooks the street. The apartment has a nice kitchen and a large living room with a dining area. The bedroom
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. To get more jobs, some young people in America choose to go to school. They want to learn new things. They want to improve their English. They want to find better jobs. After learning the language and working hard, some young people become the leaders of the country. It is said that the president of the United States is very important. However, the president of China is different. He is an important person. The president of China has a lot of power. He is the leader of the country. In the United States, the president is the leader of the country. He is the one who always decides
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]
    A. Paris
    B. Lyon
    C. Nice
    D. Marseille
    Answer:
    A
    
    An atom is composed of a nucleus with a positive charge and electrons orbiting around it. The ratio of the number of electrons to protons in the nucleus is called the atomic number. Which of the following statements is incorrect? A. The number of protons in the nucleus of a certain atom is 8.
    B. The number of neutrons in the nucleus of a certain atom is 8.
    C. The number of protons in the nucleus of a certain atom is 8.
    D. The number of electrons in
    ===============================
    Prompt: The future of AI is
    Generated text:  changing, but it can’t change the fact that AI will be in every industry in the future. It is essential to understand the different AI applications in each sector and how they will be utilized. Here are some of the major AI applications in each industry.
    
      1. Healthcare
      2. Manufacturing
      3. Retail
      4. Finance
      5. Transportation
    
    Healthcare: AI in Healthcare
    
    AI in Healthcare can have a broad impact on healthcare in the following areas.
    
      1. Personalized Healthcare: AI can help healthcare providers to identify patterns in medical records and provide personalized healthcare solutions to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] because [reason for passion]. I'm always looking for ways to [action or goal]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a cultural hub with a diverse population and a vibrant nightlife. It is a popular tourist destination and a major economic center in Europe. The city is home to many famous museums, including the Louvre, the Musée d'Orsay
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy.
    
    2. Integration of AI with other technologies: AI will continue to be integrated with other technologies such as IoT, blockchain, and quantum computing. This will create new opportunities for AI to be used in new ways.
    
    3. Development of new AI technologies: AI will continue to
    


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
    Generated text:  [Your Name]. I am [Your Age] years old. I come from [Your Background], a [Your Profession or Experience] in the [Your Field/Industry] field. I am [Your Motivations]. And I am here to [Your Purpose or Task]. I have [Your Skills or Expertise]. This is my [Your Character Name] and I am here to help you achieve your goals. How can I assist you today? I look forward to meeting you and working with you towards your goals. [Your Name] [Your Character Name] [Your Intentions] [Your Achievements] [Your Future]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city and the seat of government of the country, and serves as the headquarters of the French government, the World Trade Organization, and other international organizations. Paris is considered to be the cultural and artistic capital of the world and is the most visited city in the world. The city is famous for its historical landmarks, such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe, and its cuisine and arts scene. Paris is a cosmopolitan city with a diverse mix of French, international, and local cultures, and it is the place where the French language
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very much dependent on a number of factors, including the technology itself, as well as the way that we as humans use it. Here are some possible future trends in AI:
    
    1. Increased reliance on machine learning: As AI systems get more sophisticated, there's a greater emphasis on using machine learning algorithms to improve their performance and accuracy.
    
    2. Greater use of speech recognition and natural language processing: As more and more people become comfortable with AI-powered assistants, such as Siri, Alexa, and Google Assistant, the technology will continue to improve its ability to recognize and understand spoken language.
    
    3. More integration with human decision-making: As AI becomes


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

    ],

     I

     am

     a

     [

    job

     title

    ]

     with

     [

    number

     of

     years

     of

     experience

    ].

     I

     am

     passionate

     about

     [

    reason

     for

     job

    ]

     and

     enjoy

     helping

     people

     through

     [

    the

     reason

     for

     job

    ].

     I

     enjoy

     working

     with

     [

    specific

     tool

     or

     technology

    ]

     and

     have

     a

     keen

     eye

     for

     detail

    .

     I

     am

     a

     [

    mot

    iv

    ational

     trait

    ]

     and

     have

     a

     great

     sense

     of

     humor

     that

     can

     bring

     people

     together

    .

     I

     am

     always

     looking

     for

     ways

     to

     improve

     [

    specific

     skill

     or

     aspect

     of

     my

     job

    ].

     I

     am

     always

     eager

     to

     learn

     and

     take

     on

     new

     challenges

    ,

     and

     I

     am

     a

     team

     player

     who

     values

     diverse

     perspectives

     and

     experiences

    .

     Thank

     you

     for

     considering

     me

     for

     this

     position

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Why

     is

     Paris

     considered

     one

     of

     the

     most

     beautiful

     cities

     in

     the

     world

    ?

     It

     is

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

     many

     other

     landmarks

     that

     make

     it

     a

     must

    -

    visit

     destination

     for

     people

     from

     all

     over

     the

     world

    .

     Its

     stunning

     architecture

    ,

     rich

     history

    ,

     and

     enchant

    ing

     culture

     make

     Paris

     a

     unique

     and

     unforgettable

     destination

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

     world

    -ren

    owned

     museums

    ,

     and

     lively

     nightlife

    .

     Overall

    ,

     the

     beauty

     and

     charm

     of

     Paris

     make

     it

     a

     city

     that

     is

     worth

     exploring

     and

     experiencing

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     full

     of

     possibilities

    ,

     but

     it

     is

     important

     to

     keep

     in

     mind

     that

     it

     is

     still

     in

     its

     early

     stages

     and

     there

     are

     many

     unknown

    s

     ahead

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

     that

     we

     can

     expect

    :
    


    1

    .

     Increased

     AI

     AI

     will

     continue

     to

     evolve

     and

     improve

    ,

     with

     new

     technologies

     being

     developed

     to

     better

     understand

     and

     interact

     with

     humans

    .
    


    2

    .

     Personal

    ization

     Personal

    ized

     AI

     will

     become

     more

     common

    ,

     as

     algorithms

     learn

     to

     better

     understand

     and

     personalize

     the

     user

     experience

    .
    


    3

    .

     Autonomous

     AI

     Autonomous

     AI

     will

     become

     more

     prevalent

    ,

     with

     robots

     and

     other

     autonomous

     systems

     taking

     on

     many

     of

     the

     tasks

     traditionally

     done

     by

     humans

    .
    


    4

    .

     Ethics

     and

     safety

     AI

     will

     need

    



```python
llm.shutdown()
```
