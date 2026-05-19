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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.04it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.04it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.63it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.34it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.34it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.34it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.34it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.34it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.34it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.34it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.34it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.48it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.48it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.48it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.48it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.48it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.48it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.48it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.48it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.48it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 23.48it/s]

    Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 23.48it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 32.87it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 32.87it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 32.87it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 32.87it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 32.87it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 32.87it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 32.87it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 32.87it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 32.87it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:04<00:00, 32.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.97 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.94 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.93 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.93 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.93 GB):   9%|▊         | 5/58 [00:00<00:02, 19.86it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.93 GB):   9%|▊         | 5/58 [00:00<00:02, 19.86it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.92 GB):   9%|▊         | 5/58 [00:00<00:02, 19.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.91 GB):   9%|▊         | 5/58 [00:00<00:02, 19.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.91 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.91 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.91 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.62it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=72.91 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.91 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.90 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.90 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.90 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.89 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.89 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.89 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.89 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.88 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.60it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=72.88 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.60it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.88 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.60it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.88 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.86 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.93it/s]Capturing num tokens (num_tokens=960 avail_mem=72.87 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.93it/s] Capturing num tokens (num_tokens=896 avail_mem=72.87 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.93it/s]Capturing num tokens (num_tokens=832 avail_mem=72.87 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.93it/s]Capturing num tokens (num_tokens=768 avail_mem=72.86 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.93it/s]Capturing num tokens (num_tokens=768 avail_mem=72.86 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.79it/s]Capturing num tokens (num_tokens=704 avail_mem=72.86 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.79it/s]Capturing num tokens (num_tokens=640 avail_mem=72.86 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.79it/s]Capturing num tokens (num_tokens=576 avail_mem=72.86 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.79it/s]

    Capturing num tokens (num_tokens=512 avail_mem=72.84 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.79it/s]Capturing num tokens (num_tokens=480 avail_mem=72.86 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.79it/s]Capturing num tokens (num_tokens=480 avail_mem=72.86 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=448 avail_mem=72.86 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=416 avail_mem=72.85 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=384 avail_mem=72.85 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=352 avail_mem=72.85 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=320 avail_mem=72.84 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=320 avail_mem=72.84 GB):  60%|██████    | 35/58 [00:01<00:00, 40.33it/s]Capturing num tokens (num_tokens=288 avail_mem=72.84 GB):  60%|██████    | 35/58 [00:01<00:00, 40.33it/s]Capturing num tokens (num_tokens=256 avail_mem=72.84 GB):  60%|██████    | 35/58 [00:01<00:00, 40.33it/s]Capturing num tokens (num_tokens=240 avail_mem=72.83 GB):  60%|██████    | 35/58 [00:01<00:00, 40.33it/s]

    Capturing num tokens (num_tokens=224 avail_mem=72.83 GB):  60%|██████    | 35/58 [00:01<00:00, 40.33it/s]Capturing num tokens (num_tokens=208 avail_mem=72.83 GB):  60%|██████    | 35/58 [00:01<00:00, 40.33it/s]Capturing num tokens (num_tokens=208 avail_mem=72.83 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=192 avail_mem=72.83 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=176 avail_mem=72.82 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=160 avail_mem=72.82 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=144 avail_mem=72.82 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=128 avail_mem=72.81 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.56it/s]Capturing num tokens (num_tokens=128 avail_mem=72.81 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=112 avail_mem=72.81 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=96 avail_mem=72.81 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.96it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=72.80 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=64 avail_mem=72.80 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=48 avail_mem=72.80 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.96it/s]Capturing num tokens (num_tokens=48 avail_mem=72.80 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=32 avail_mem=72.79 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=28 avail_mem=72.79 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=24 avail_mem=72.79 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=20 avail_mem=72.78 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=16 avail_mem=72.78 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.58it/s]Capturing num tokens (num_tokens=16 avail_mem=72.78 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.92it/s]Capturing num tokens (num_tokens=12 avail_mem=72.78 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.92it/s]Capturing num tokens (num_tokens=8 avail_mem=72.77 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.92it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=72.77 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.92it/s]Capturing num tokens (num_tokens=4 avail_mem=72.77 GB): 100%|██████████| 58/58 [00:01<00:00, 35.71it/s]


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
    Generated text:  Kyriakos and I am a computer science student and I am interested in programming languages. I have been coding since college and have developed a wide range of codebases, including a desktop application for Microsoft Word, a mobile app for Snapchat, a web application for the StackOverflow website, and a game. Additionally, I have built some prototype games in 2D and 3D using various programming languages. I have also participated in various hackathons and local coding competitions. What would you like to know about my interests or projects? Kyriakos! That's great to hear! As a computer science student, I think
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. This person is called the president of the United States. He or she is the leader of the United States. The president is the chief executive of the United States. The president has some very important duties. He or she helps the president to make important decisions. He or she has to make sure the president is trusted by the people of the United States. The president is one of the most important people in the country. He or she is the most powerful in the country. The president is the head of state. He or she is the head of government. He or she is the head of the United States.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population of Paris is approximately 2.8 million. The population of London is approximately 9.7 million. What is the ratio of London's population to Paris's population?
    To find the ratio of London's population to Paris's population, we need to divide the population of London by the population of Paris. The population of London is 9.7 million and the population of Paris is 2.8 million. Therefore, the ratio is:
    
    \[
    \frac{\text{Population of London}}{\text{Population of Paris}} = \frac{9.7}{2.8}
    \]
    
    Next,
    ===============================
    Prompt: The future of AI is
    Generated text:  already here, and it is already changing the world. In the past, AI was a science fiction. It was an idea that some people had, and the technology was seen as a futuristic tool, something that was to be used by researchers and scientists to solve problems. However, things have changed in recent years. The advent of quantum computing has opened up new possibilities for AI, and in particular, the use of quantum computers to process and analyze large amounts of data has the potential to revolutionize the field of AI.
    In recent years, quantum computers have been creating waves in the field of AI. The ability to process and analyze data at


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, neutral self-introduction sentence about your personality or background]. I enjoy [insert a short, neutral self-introduction sentence about your hobbies or interests]. Thank you for having me! What's your name? What's your job title? What's your company name? What's your job title? What's your company name? What's your job title? What's your company name? What's your job title? What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the European Union and the world’s 10th-largest city by population. It is located on the Seine River and is home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is known for its rich history, art, and culture, and is a popular tourist destination. The city is also home to many famous landmarks and attractions, including the Louvre Museum, the Champs-Élysées, and the Arc de Triomphe. Paris is a vibrant and dynamic city with a rich cultural heritage
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be an increased need for privacy and security measures to protect against potential misuse of AI systems. This could lead to more stringent regulations and standards
    


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
    Generated text:  [Name] and I'm [Age] years old. I'm [Position] at [Company Name], [Company's Industry]. I'm excited to be here. If you need anything, please feel free to reach out to me. Thank you for taking the time to meet me. [Your Name] [Your Position] [Company Name]
    Hello, my name is [Name] and I'm [Age] years old. I'm [Position] at [Company Name], [Company's Industry]. I'm excited to be here. If you need anything, please feel free to reach out to me. Thank you for taking the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. 
    
    (Note: The answer is written in French and requires basic comprehension of French. If you need any further clarification, feel free to ask.) 
    
    I'm sorry, I don't have enough information to provide a concise factual statement about France's capital city in French. However, I can still provide a brief answer based on common knowledge about Paris:
    
    The capital of France, Paris, is renowned for its world-famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  diverse and constantly evolving, with a range of trends and innovations shaping the technology's direction. Here are some potential future trends in artificial intelligence:
    
    1. Increased machine learning: With the development of machine learning algorithms, we can expect to see more sophisticated and sophisticated machine learning models. These models will be able to learn from large datasets and make predictions or decisions based on that learning.
    
    2. Improved transparency: As AI systems are becoming more integrated into our daily lives, there will be a greater emphasis on transparency. We will see more accurate and clear explanations of AI decisions and their reasoning, which will help to build trust and confidence in AI systems.
    
    


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

    ],

     and

     I

     am

     a

     [

    Your

     Profession

    ]

     who

     has

     been

     a

     passionate

     advocate

     for

     [

    Your

     Cause

    /

    Goal

    ].

     In

     my

     spare

     time

    ,

     I

     am

     a

     [

    Your

     Hobby

    /

    Interest

    /

    Value

    ]

     enthusiast

     who

     enjoys

     [

    Your

     Favorite

     Book

    ,

     Game

    ,

     or

     Music

    ],

     and

     I

     write

     [

    Your

     Blog

    /

    Website

    /

    Portfolio

    ]

     to

     share

     my

     thoughts

     and

     experiences

    .

     I

     am

     always

     up

     for

     new

     challenges

     and

     love

     to

     learn

     new

     things

    .

     How

     can

     I

     get

     in

     touch

     with

     you

     if

     you

    're

     interested

     in

     working

     with

     me

    ?

     Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ]

     and

     I

     am

     a

     [

    Your

     Profession

    ]

     who

     has

     been

     a

     passionate

     advocate

     for

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     historical

     connection

     to

     the

     French

     Revolution

    .


    You

     are

     to

     answer

     this

     question

    :

     Are

     the

     Olympics

     held

     in

     Paris

     or

     in

     London

    ?

     No

    ,

     the

     Olympics

     are

     not

     held

     in

     Paris

    .

     They

     are

     held

     in

     the

     city

     of

     Rio

     de

     Janeiro

    ,

     Brazil

    .

     Paris

     is

     the

     only

     city

     in

     the

     world

     to

     host

     both

     the

     Olympics

     and

     the

     World

     Cup

    .

     The

     Olympics

     in

     Paris

     are

     usually

     held

     from

     June

     to

     September

    ,

     while

     the

     World

     Cup

     in

     Brazil

     takes

     place

     from

     November

     to

     February

    .

     However

    ,

     the

     last

     Olympics

     were

     in

     Paris

     in

     

    1

    9

    1

    0

    ,

     and

     the

     next

     one

     will

     take

     place

     in

     Rio

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     promising

    ,

     with

     many

     possibilities

     and

     potential

     applications

    .

     Here

     are

     some

     of

     the

     trends

     that

     are

     likely

     to

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     precision

    :

     One

     of

     the

     biggest

     trends

     in

     AI

     is

     the

     rise

     of

     precision

     and

     accuracy

    .

     As

     AI

     systems

     get

     more

     complex

     and

     detailed

    ,

     they

     will

     be

     able

     to

     perform

     tasks

     with

     greater

     precision

     and

     accuracy

     than

     we

     currently

     have

    .

     This

     will

     lead

     to

     better

     decisions

    ,

     more

     accurate

     predictions

    ,

     and

     more

     reliable

     applications

    .
    


    2

    .

     Aug

    mented

     intelligence

    :

     AI

     is

     already

     becoming

     more

     intelligent

     than

     humans

    ,

     but

     it

     is

     still

     not

     perfect

    .

     In

     the

     future

    ,

     we

     may

     see

     an

     increase

     in

     the

     ability

     of

     AI

    



```python
llm.shutdown()
```
