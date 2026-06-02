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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.54it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.19it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.19it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.19it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.19it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.19it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.19it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.19it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.19it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.19it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.19it/s]

    Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.19it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.42it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.42it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.42it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.42it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.42it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.42it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.42it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.42it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.42it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.92it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.05it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.05it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.71it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.71it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.89it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.89it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.89it/s] Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.89it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.89it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.49it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.49it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.49it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.49it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.49it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.49it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.46it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.46it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.46it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.46it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.46it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.46it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.29it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.29it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.29it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.29it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.29it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.29it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.29it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.01it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.64it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.64it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.59it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.59it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.59it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.59it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.59it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.59it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.90it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.90it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.90it/s] Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.90it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 42.20it/s]


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
    Generated text:  Ariadne and I'm a 12-year-old girl from Germany. I'm an English teacher. My lessons are very interesting, but I have a lot of homework. When I get home, I play a lot of video games. My best friend is Clara. Clara is 13 years old, and we are both very clever and smart. She always studies hard and always helps us with the homework. Clara and I are very good friends. I often help her with her homework and she helps me with my English. We are both very happy to help each other. Sometimes I feel like crying because I know that I'm
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. At the end of the year, the president reports to the congress. They make decisions to help the country and work together to solve problems. The president is supposed to work hard and be helpful. Can you be president of the United States? Think about who the presidents are. The president of the United States has been in power for over 40 years. That's longer than the time that most people lived. But they don't have to be perfect. They can make mistakes. The president is supposed to be a good leader. The president should help people get along with each other. The president should keep everyone
    ===============================
    Prompt: The capital of France is
    Generated text:  ____. 
    A. Paris
    B. London
    C. Moscow
    D. Shanghai
    Answer:
    Solution: Paris is the capital of France, so the correct answer is: A. Therefore, the answer is: A.
    
    Among the following numbers, which one is not a fraction? 
    A. 1.0
    B. 3
    C. 0.6
    D. 3.14
    Answer:
    Solution: A, 1.0 is a fraction, so this option is incorrect; B, 3 is a whole number, so this option is incorrect; C, 0.6 is
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it's incredibly exciting. With the rapid development of artificial intelligence, many of the most fundamental and complex problems that humanity is facing, such as climate change, pandemics, and economic inequality, can be solved by AI.
    
    But as you might be aware, the pace of AI development has accelerated rapidly in recent years, creating the need for a more aggressive push from governments and the public to make sure we don't lose the benefits of the technology.
    
    In our recent piece, we looked at how to make AI fair and ethical. We discussed what sorts of innovations, research and development, and policies could come together to make AI fair


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


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is home to iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as a vibrant arts scene and a thriving food and wine industry. Paris is known for its fashion, art, and cuisine, and is a major center of politics, culture, and commerce in Europe. The city is also home to numerous museums, theaters, and other cultural institutions, making it a popular destination for tourists and locals alike. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased use of AI in healthcare: AI is already being used to diagnose and treat diseases, and it has the potential to revolutionize the field of medicine. In the future, we may see even more sophisticated AI systems that can analyze medical data and provide personalized treatment plans.
    
    2. AI in manufacturing: AI is already being used to optimize production processes and improve quality control. In the future, we may see even more advanced AI systems that can analyze data from sensors and machines to predict and prevent
    


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
    Generated text:  [Name], and I'm a [short answer to your profession or area of expertise]. I'm here to help you if you need anything. How can I assist you today? Let me know! 
    #The Given Prompt#:
    Write a short, neutral self-introduction for a fictional character. Hello, my name is [Name], and I'm a [short answer to your profession or area of expertise]. I'm here to help you if you need anything. How can I assist you today? Let me know! 
    #The Created Answer#:
    Hello, my name is [Name] and I am a [short answer to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. The city is a popular tourist destination, known for its stunning architecture, rich history, and vibrant culture. The French Republic, which has its capital in Paris, was founded in 843 and has been the seat of government and principal seat of government for nearly 700 years. Paris is home to the Eiffel Tower, the Louvre Museum, and numerous museums and art galleries. The city is also known for its gastronomic offerings, cuisine, and fashion. Paris is a city that has captured the imagination of the world for centuries and continues to be a major cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, with many potential trends shaping its direction. Here are some possible future trends in AI:
    
    1. Increased Personalization: AI will become even more personalized, allowing machines to learn from the data they process and adapt to the user's needs.
    
    2. Autonomous Systems: AI-powered autonomous systems are expected to become more prevalent, with robots and drones that can operate in a wide range of environments without human intervention.
    
    3. Enhanced Human-AI Interaction: AI will become more capable of understanding and adapting to human emotions, making interactions with humans more natural and intuitive.
    
    4. Increased Robustness: AI will become more robust, with systems


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

     Smith

    .

     I

     am

     a

     self

    -employed

     freelance

     writer

    .

     What

    's

     your

     profession

     and

     what

     do

     you

     do

    ?

     Answer

     in

     a

     neutral

    ,

     un

    pret

    entious

     manner

    .
    


    Hello

     John

    !

     My

     name

     is

     John

     Smith

    ,

     a

     freelance

     writer

    .

     What

     can

     you

     tell

     me

     about

     your

     work

    ?

     Answer

     in

     a

     neutral

    ,

     un

    pret

    entious

     manner

    .

     John

    ,

     can

     you

     tell

     us

     about

     your

     writing

     style

     or

     process

    ?

     Answer

     in

     a

     neutral

    ,

     un

    pret

    entious

     manner

    .

     John

    ,

     describe

     your

     writing

     process

    .

     Answer

     in

     a

     neutral

    ,

     un

    pret

    entious

     manner

    .

     John

    ,

     can

     you

     tell

     us

     about

     your

     most

     popular

     or

     notable

     works

    ?

     Answer

     in

     a

     neutral

    ,

     un

    pret

    entious

     manner

    .

     John

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Petite

     Vie

    ",

     which

     means

     "

    The

     Small

     Life

    "

     in

     French

    .

     It

     is

     a

     city

     that

     is

     home

     to

     many

     of

     France

    's

     most

     iconic

     landmarks

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

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     Paris

     is

     also

     a

     hub

     for

     French

     culture

     and

     cuisine

    ,

     with

     many

     famous

     French

     restaurants

     and

     bars

    ,

     and

     is

     known

     for

     its

     elaborate

     architecture

    ,

     including

     the

     Arc

     de

     Tri

    omp

    he

    .

     With

     its

     diverse

     population

     and

     rich

     history

    ,

     Paris

     is

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     The

     French

     people

     have

     a

     long

     history

     of

     artistic

     and

     literary

     influences

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     diverse

    ,

     and

     it

    's

     shaped

     by

     many

     factors

    .

     Here

     are

     some

     possible

     future

     trends

     that

     could

     shape

     the

     way

     AI

     is

     used

     in

     various

     sectors

    :
    


    1

    .

     Increased

     reliance

     on

     AI

     for

     automation

    :

     As

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     that

     AI

     will

     play

     a

     more

     and

     more

     important

     role

     in

     autom

    ating

     tasks

    .

     This

     could

     include

     tasks

     that

     are

     currently

     handled

     by

     humans

    ,

     such

     as

     data

     analysis

     and

     decision

    -making

    .
    


    2

    .

     Greater

     integration

     of

     AI

     with

     other

     technologies

    :

     AI

     is

     becoming

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     the

     Internet

     of

     Things

     (

    Io

    T

    )

     and

     the

     blockchain

    ,

     creating

     new

     opportunities

     for

     AI

    -driven

     applications

    .

     For

     example

    ,

     AI

     could

     be

     used

    



```python
llm.shutdown()
```
