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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.07it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.39it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.87it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.05it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 24.05it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 24.05it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 24.05it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 24.05it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.08 GB):   3%|▎         | 2/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.08 GB):   3%|▎         | 2/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.07 GB):   3%|▎         | 2/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.07 GB):   3%|▎         | 2/58 [00:00<00:02, 19.52it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=53.07 GB):   9%|▊         | 5/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.07 GB):   9%|▊         | 5/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.06 GB):   9%|▊         | 5/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.06 GB):   9%|▊         | 5/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.06 GB):   9%|▊         | 5/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.66it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.66it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.66it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.04 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.04 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.66it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=53.04 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.02 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.02 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.58it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.00 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.58it/s]Capturing num tokens (num_tokens=960 avail_mem=53.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.58it/s] Capturing num tokens (num_tokens=896 avail_mem=53.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.58it/s]

    Capturing num tokens (num_tokens=832 avail_mem=53.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.58it/s]Capturing num tokens (num_tokens=832 avail_mem=53.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=768 avail_mem=53.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=704 avail_mem=53.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=640 avail_mem=53.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=576 avail_mem=53.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=512 avail_mem=52.98 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.08it/s]Capturing num tokens (num_tokens=512 avail_mem=52.98 GB):  50%|█████     | 29/58 [00:00<00:00, 44.45it/s]Capturing num tokens (num_tokens=480 avail_mem=53.00 GB):  50%|█████     | 29/58 [00:00<00:00, 44.45it/s]Capturing num tokens (num_tokens=448 avail_mem=53.00 GB):  50%|█████     | 29/58 [00:00<00:00, 44.45it/s]Capturing num tokens (num_tokens=416 avail_mem=52.99 GB):  50%|█████     | 29/58 [00:00<00:00, 44.45it/s]Capturing num tokens (num_tokens=384 avail_mem=52.99 GB):  50%|█████     | 29/58 [00:00<00:00, 44.45it/s]

    Capturing num tokens (num_tokens=352 avail_mem=52.99 GB):  50%|█████     | 29/58 [00:00<00:00, 44.45it/s]Capturing num tokens (num_tokens=352 avail_mem=52.99 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.05it/s]Capturing num tokens (num_tokens=320 avail_mem=52.98 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.05it/s]Capturing num tokens (num_tokens=288 avail_mem=52.98 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.05it/s]Capturing num tokens (num_tokens=256 avail_mem=52.98 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.05it/s]Capturing num tokens (num_tokens=240 avail_mem=52.97 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.05it/s]Capturing num tokens (num_tokens=224 avail_mem=52.97 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.05it/s]Capturing num tokens (num_tokens=208 avail_mem=52.97 GB):  59%|█████▊    | 34/58 [00:00<00:00, 46.05it/s]Capturing num tokens (num_tokens=208 avail_mem=52.97 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.46it/s]Capturing num tokens (num_tokens=192 avail_mem=52.97 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.46it/s]Capturing num tokens (num_tokens=176 avail_mem=52.96 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=160 avail_mem=52.96 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.46it/s]

    Capturing num tokens (num_tokens=144 avail_mem=52.96 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=128 avail_mem=52.44 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.46it/s]Capturing num tokens (num_tokens=128 avail_mem=52.44 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=112 avail_mem=52.44 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=96 avail_mem=52.44 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.21it/s] Capturing num tokens (num_tokens=80 avail_mem=52.43 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=64 avail_mem=52.43 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=48 avail_mem=52.42 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.21it/s]

    Capturing num tokens (num_tokens=48 avail_mem=52.42 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=32 avail_mem=52.42 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=28 avail_mem=52.42 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=24 avail_mem=52.41 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=20 avail_mem=52.41 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=16 avail_mem=52.41 GB):  86%|████████▌ | 50/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=16 avail_mem=52.41 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.32it/s]Capturing num tokens (num_tokens=12 avail_mem=52.40 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.32it/s]Capturing num tokens (num_tokens=8 avail_mem=52.40 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.32it/s] Capturing num tokens (num_tokens=4 avail_mem=52.40 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.32it/s]Capturing num tokens (num_tokens=4 avail_mem=52.40 GB): 100%|██████████| 58/58 [00:01<00:00, 40.94it/s]


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
    Generated text:  Kana. I'm a kind girl and enjoy playing with my friends. I'm very active and always want to play more. I'm learning English as a second language and have a lot of homework to do. I need a Chinese dictionary to learn English. I want to learn English very fast. Can you please help me? I will be very grateful. Thank you. According to the passage, which of the following statements is NOT correct? A) Kana is a girl. B) Kana has a lot of homework. C) Kana is very active. D) Kana wants to learn English very fast.
    A)
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to celebrate Martin Luther King Jr.'s birthday. 
    
    According to the 2016 US Census, the population of the US is 323,000,000, and the president believes that the population grows at an average rate of 500,000 per year. If the president decides to celebrate King Jr.'s birthday, how many years would it take for the population to reach 323,000,000? 
    
    Remember, the president wants the population to reach 323,000,000 as
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 错误
    B. 正确
    答案:
    
    A
    
    5.关于伤口的处理，下列说法正确的是:
    A. 伤口感染的处理:伤口感染的处理:伤口感染的处理:伤口感染的处理:伤口感染的处理
    B. 伤口感染的处理:伤口感染的处理:伤口感染的处理:伤口感染的处理:伤口感染的处理
    C. 伤口感染的处理:伤口感染的处理:伤口感染的处理:伤口感染的处理:伤口感染的处理
    D. 伤口感染的处理:伤口感染的处理:伤口
    ===============================
    Prompt: The future of AI is
    Generated text:  dark, says Walter Dean Myers
    
    Two weeks ago, Walter Dean Myers made a statement about the future of AI that was met with alarm from most of the AI community. As a member of the new AI group at Stanford, I had the opportunity to have a deep discussion about this with Myers.
    
    Myers wrote: "If you’re like me, you’re smart, you’re capable, but you’re not in control of your decisions. You’re a product of your beliefs, of your values, of your culture, of your education. But you’re not in control of what’s going on. You’re not in control of what’s


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its fashion industry, art, and cuisine. Paris is a vibrant and diverse city with a population of over 2.5 million people. It is a major hub for business, culture, and tourism in Europe. The city is also home to many world-renowned museums, including the Louvre and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient monitoring.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management and fraud detection. As AI technology continues
    


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
    Generated text:  [Name], and I'm here to meet you. I'm [Name]'s friend, and we've been friends for a few years now. Our names sound familiar, right? Who knows, maybe I'll be able to become your best friend or even a partner in crime? Just so you know, I'm a lawyer with a focus on personal injury law. It's really cool to meet you and get to know you better. What can I say, I'm so glad to have you as a friend. Hey! I'm [Name]. How was your day? Are you tired from all the running around? No way!
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the south of the country and is known as the city of love. It is the oldest capital city in Europe and hosts the famous Eiffel Tower. The city is also famous for its art and culture, including the Louvre Museum and the Notre-Dame Cathedral. Paris is a vibrant and dynamic city with a rich cultural and historical heritage. It has a population of over 2 million people and is home to many of the world's most famous landmarks. It is a bustling city that is known for its fashion, food, and entertainment industry. Paris is a popular tourist destination and attracts millions of visitors each year.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly anticipated by experts and investors alike. Here are some potential trends in AI that could shape our world in the coming years:
    
    1. Increased automation: One of the most prominent trends in AI is the increased automation of manual tasks. AI systems are being used to automate repetitive and routine work, and it is expected that this trend will continue in the future.
    
    2. Artificial intelligence in healthcare: AI has already been used to diagnose and treat diseases, and it has the potential to revolutionize the healthcare industry. AI is being used to analyze medical images, predict disease outcomes, and assist in diagnosis and treatment planning.
    
    3. AI in finance:


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

     [

    Position

    ]

     at

     [

    Company

     Name

    ].

     I

     am

     a

     passionate

     advocate

     for

     [

    Company

    's

     values

    ]

     and

     [

    Company

    's

     mission

    ].

     I

     am

     a

     team

     player

     who

     thr

    ives

     on

     collaboration

     and

     always

     put

     others

     before

     myself

    .

     I

     enjoy

     working

     with

     diverse

     teams

     and

     am

     always

     looking

     for

     ways

     to

     improve

     our

     team

    's

     productivity

    .

     I

     am

     a

     proactive

     problem

    -s

    olver

     who

     is

     always

     ready

     to

     take

     action

     to

     address

     any

     issues

     that

     arise

    .

     I

     am

     dedicated

     to

     achieving

     our

     goals

     and

     contributing

     to

     the

     company

    's

     success

    .

     In

     short

    ,

     I

     am

     a

     [

    Company

    's

     name

    ]

     dedicated

     to

     [

    Company

    's

     mission

    ].


    [

    Your

     Name

    ].

     How

     are

    
    
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

     both

     Europe

     and

     the

     world

    ,

     and

     home

     to

     the

     European

     Parliament

    ,

     the

     French

     National

     Library

    ,

     and

     numerous

     historic

     landmarks

    .

     Paris

     is

     renowned

     for

     its

     architecture

    ,

     cuisine

    ,

     and

     fashion

    ,

     and

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

     Its

     unique

     cultural

     and

     historical

     heritage

    ,

     along

     with

     its

     contemporary

     fashion

     scene

    ,

     has

     made

     Paris

     a

     major

     global

     city

    .

     In

     terms

     of

     its

     economy

    ,

     Paris

     is

     a

     major

     hub

     for

     international

     trade

     and

     tourism

    ,

     and

     is

     a

     significant

     player

     in

     the

     European

     Union

     and

     beyond

    .

     Overall

    ,

     Paris

     is

     a

     fascinating

     city

     with

     a

     rich

     history

    ,

     diverse

     culture

    ,

     and

     a

     vibrant

     economy

    .

     Its

     status

     as

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     and

     there

     are

     a

     number

     of

     potential

     trends

     that

     could

     shape

     its

     direction

     in

     the

     years

     to

     come

    .

     Here

     are

     some

     potential

     trends

     in

     AI

     that

     are

     currently

     being

     considered

     and

     that

     could

     play

     a

     role

     in

     shaping

     the

     future

    :
    


    1

    .

     Personal

    ization

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     lives

    ,

     there

     will

     be

     an

     increasing

     emphasis

     on

     personal

    ization

    .

     AI

     will

     be

     able

     to

     understand

     and

     tailor

     our

     experiences

     to

     meet

     our

     individual

     needs

     and

     preferences

    ,

     and

     this

     will

     be

     a

     key

     driver

     of

     future

     AI

     development

    .
    


    2

    .

     Autonomous

     vehicles

    :

     AI

     is

     being

     developed

     to

     help

     drive

     autonomous

     vehicles

     that

     can

     navigate

     roads

     and

     navigate

     traffic

    .

     These

     vehicles

     will

     be

     able

     to

     communicate

    



```python
llm.shutdown()
```
