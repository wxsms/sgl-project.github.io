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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.01it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.01it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:07,  4.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:07,  4.33s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:07,  4.33s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:07,  4.33s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:07,  4.33s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.47it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.01it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.05it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.05it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.05it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.05it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.05it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.05it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.05it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.05it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.05it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.05it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.05it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.27it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.27it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.27it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.27it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.27it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.27it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.27it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.26 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.26 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.26 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.26 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.26 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.25 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.24 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.24 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.24 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.24 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.22 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.19it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=71.22 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.19it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.20 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.18 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.39it/s]Capturing num tokens (num_tokens=960 avail_mem=71.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.39it/s] Capturing num tokens (num_tokens=896 avail_mem=71.19 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.39it/s]

    Capturing num tokens (num_tokens=832 avail_mem=71.19 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.39it/s]Capturing num tokens (num_tokens=832 avail_mem=71.19 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=768 avail_mem=71.19 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=704 avail_mem=71.18 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=640 avail_mem=71.18 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=576 avail_mem=71.18 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=512 avail_mem=71.16 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=512 avail_mem=71.16 GB):  50%|█████     | 29/58 [00:00<00:00, 42.95it/s]Capturing num tokens (num_tokens=480 avail_mem=71.18 GB):  50%|█████     | 29/58 [00:00<00:00, 42.95it/s]Capturing num tokens (num_tokens=448 avail_mem=71.18 GB):  50%|█████     | 29/58 [00:00<00:00, 42.95it/s]Capturing num tokens (num_tokens=416 avail_mem=71.18 GB):  50%|█████     | 29/58 [00:00<00:00, 42.95it/s]Capturing num tokens (num_tokens=384 avail_mem=71.17 GB):  50%|█████     | 29/58 [00:00<00:00, 42.95it/s]

    Capturing num tokens (num_tokens=352 avail_mem=71.17 GB):  50%|█████     | 29/58 [00:00<00:00, 42.95it/s]Capturing num tokens (num_tokens=352 avail_mem=71.17 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.81it/s]Capturing num tokens (num_tokens=320 avail_mem=71.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.81it/s]Capturing num tokens (num_tokens=288 avail_mem=71.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.81it/s]Capturing num tokens (num_tokens=256 avail_mem=71.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.81it/s]Capturing num tokens (num_tokens=240 avail_mem=71.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.81it/s]Capturing num tokens (num_tokens=224 avail_mem=71.15 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.81it/s]Capturing num tokens (num_tokens=208 avail_mem=71.15 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.81it/s]Capturing num tokens (num_tokens=208 avail_mem=71.15 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=192 avail_mem=71.15 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=176 avail_mem=71.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=160 avail_mem=71.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.63it/s]

    Capturing num tokens (num_tokens=144 avail_mem=71.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=128 avail_mem=71.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=128 avail_mem=71.14 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=112 avail_mem=71.13 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=96 avail_mem=71.13 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.29it/s] Capturing num tokens (num_tokens=80 avail_mem=71.13 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=64 avail_mem=71.12 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=48 avail_mem=71.12 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=48 avail_mem=71.12 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=32 avail_mem=71.12 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.72it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.11 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=24 avail_mem=71.11 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=20 avail_mem=71.10 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.72it/s]

    Capturing num tokens (num_tokens=16 avail_mem=71.10 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=16 avail_mem=71.10 GB):  95%|█████████▍| 55/58 [00:01<00:00, 26.33it/s]Capturing num tokens (num_tokens=12 avail_mem=71.10 GB):  95%|█████████▍| 55/58 [00:01<00:00, 26.33it/s]Capturing num tokens (num_tokens=8 avail_mem=71.10 GB):  95%|█████████▍| 55/58 [00:01<00:00, 26.33it/s] Capturing num tokens (num_tokens=4 avail_mem=71.09 GB):  95%|█████████▍| 55/58 [00:01<00:00, 26.33it/s]

    Capturing num tokens (num_tokens=4 avail_mem=71.09 GB): 100%|██████████| 58/58 [00:01<00:00, 32.19it/s]


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
    Generated text:  Tim. I'm a student at Columbia University, majoring in computer science and I have a dream to work in a tech company like Apple. I have some friends who are also planning to join the tech industry in the future. I was wondering if you could recommend some books or articles that have helped you with your career goals and how they have helped you grow and improve your skills? Additionally, I would like to know if there are any specific conferences or events that you have attended that have been helpful in your career development? 
    
    Thank you! [Your Name]
    
    Hi, my name is Sarah. I'm a software engineer at a startup
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. The president is the highest-ranking military officer in the United States. His or her job is to represent the country in the United Nations.
    Does this next sentence follow, given the preceding text?
    The president of the United States is the highest-ranking military officer in the United States. 
    Options are:
     a). yes
     b). it is not possible to tell
     c). no
    a). yes
    
    The sentence "The president of the United States is the highest-ranking military officer in the United States" is a direct translation and follows directly from the given information in the text. The text states that the president is described as being
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the capital of the country, the second largest city in the country and the 14th largest city in the world. It has a population of 2.3 million and is the most populous city in the country. The population of the French capital Paris is 2, 3 million people. The population density of the capital Paris is 2550 people per square kilometer. According to the French constitution, the capital of France has the power to govern the country. The French capital Paris, which is the largest city of the country and the largest city in Europe, is the capital of France.
    
    ===============================
    Prompt: The future of AI is
    Generated text:  now, and it will be digital and mobile-first
    
    In the digital age, the adoption of AI is a digital and mobile-first trend. A more detailed analysis of the future of AI will be presented.
    
    There is an ever-growing and growing need for more AI in the digital and mobile market, and so it is essential to understand the future of AI and how it will shape the digital landscape.
    
    A more detailed analysis of the future of AI will be presented.
    
    The age of AI is here, but the technology is still a work in progress. We are already seeing a lot of AI being implemented into the digital and mobile market, and the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have a [job title] at [company name]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [mention a hobby or activity]. I'm always looking for new experiences and adventures. What's your favorite book or movie? I love [mention a book or movie]. I'm always looking for new ideas and inspiration
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" and "La Ville de la Rose" (the Rose City). It is the largest city in France and the second-largest city in the European Union, with a population of over 10 million people. Paris is known for its rich history, art, and culture, and is a major tourist destination. It is also home to many famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also home to many important institutions, including the French Academy of Sciences and the French National Library. Paris is a vibrant and dynamic city,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some possible future trends in AI include:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve the quality of care. As AI technology continues to advance, we can expect to see even more sophisticated and accurate AI systems being used in healthcare, such as in diagnosing diseases, predicting patient outcomes
    


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
    Generated text:  [Your Name], and I am a/an [character's name] who has been in the [character's occupation] industry for [number] years. I have a passion for [character's profession], and I love working hard and being dedicated to my craft. I am always looking for new challenges and learning new skills, and I am always eager to share my knowledge with others. I enjoy helping people and making a positive impact on the world, and I am excited to continue this journey with you. Please let me know if you would like to chat more about this character or if you have any questions on how to get started. [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Is the following statement true or false: "Paris is the capital of the United States." The statement is false because Paris is a city in France, not the United States. Paris is the capital city of France, but the statement about it is incorrect. 
    
    To properly answer the question, the correct statement would be: "Paris is the capital of France." 
    
    It is important to note that while Paris is indeed the largest city in France and one of its most famous landmarks, it is not the capital of the United States. The capital of the United States is Washington, D.C., located in the state of Maryland. 
    
    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting, with many exciting developments and possible future trends. Here are some of the most likely areas of growth:
    
    1. Faster Learning and Adaptation: One of the biggest challenges facing AI is adapting to new and changing data and environments. This is where faster learning and adaptation are likely to play a key role. As AI systems become more complex, they will require more data to learn and develop their skills, which can help them adapt to changing situations.
    
    2. Personalization: With AI, businesses are likely to find ways to personalize their experiences. For example, chatbots and virtual assistants can use AI to understand and respond to customer queries in


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

    insert

     character

     name

    ].

     I

    'm

     from

     [

    insert

     hometown

     or

     location

    ]

     and

     I

     have

     lived

     here

     for

     [

    insert

     number

     of

     years

    ]

     years

     now

    .

     I

    'm

     a

     [

    insert

     occupation

     or

     profession

    ]

     who

     has

     always

     been

     [

    insert

     personality

     trait

     or

     quality

    ],

     but

     I

     recently

     [

    insert

     accomplishment

     or

     achievement

    ].

     I

     enjoy

     [

    insert

     hobby

     or

     activity

    ]

     with

     my

     family

     and

     friends

    ,

     and

     I

     always

     strive

     to

     [

    insert

     goal

     or

     aspiration

    ].

     I

     believe

     in

     [

    insert

     value

     or

     belief

    ],

     and

     I

    'm

     a

     [

    insert

     education

     level

    ]

     with

     [

    insert

     major

     or

     major

     in

     college

    ].

     I

     have

     a

     degree

     in

     [

    insert

     field

     of

     study

    ],

     and

     I

     believe

     in

     [

    insert

     personal

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

     concise

     factual

     statement

     about

     the

     famous

     work

     of

     art

     Mona

     Lisa

     by

     Leonardo

     da

     Vinci

    ,

     including

     the

     title

     and

     the

     work

     of

     art

     itself

    .

     The

     work

     of

     art

     Mona

     Lisa

     by

     Leonardo

     da

     Vinci

     is

     titled

     "

    The

     Last

     Sup

    per

    ,"

     and

     it

     is

     one

     of

     the

     most

     famous

     paintings

     in

     the

     world

    .

     
    


    A

     concise

     factual

     statement

     about

     the

     landscape

     of

     the

     Czech

     Republic

    ,

     including

     the

     capital

     city

     Prague

     and

     the

     country

    's

     three

     UNESCO

     World

     Heritage

     Sites

    .

     Prague

     is

     the

     capital

     of

     the

     Czech

     Republic

     and

     is

     known

     for

     its

     beautiful

     architecture

    ,

     including

     the

     Prague

     Castle

    ,

     Old

     Town

     Square

    ,

     and

     historic

     Old

     Town

    .

     Prague

     is

     also

     home

     to

     UNESCO

     World

     Heritage

     Sites

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     several

     trends

     that

     are

     likely

     to

     shape

     the

     development

     and

     evolution

     of

     this

     technology

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

     Increasing

    ly

     Natural

     Language

     Processing

    :

     AI

     systems

     are

     expected

     to

     become

     more

     natural

     and

     human

    -like

    ,

     with

     the

     ability

     to

     understand

     and

     interpret

     human

     language

     in

     ways

     that

     are

     more

     nuanced

     and

     context

    -dependent

    .

     This

     will

     require

     the

     development

     of

     more

     sophisticated

     language

     models

     that

     can

     understand

     and

     respond

     to

     a

     wider

     range

     of

     natural

     language

     inputs

    .
    


    2

    .

     Enhanced

     Emotional

     AI

    :

     AI

     systems

     are

     expected

     to

     become

     even

     more

     capable

     of

     detecting

     and

     understanding

     human

     emotions

    ,

     including

     both

     positive

     and

     negative

     emotions

    .

     This

     will

     require

     the

     development

     of

     more

    



```python
llm.shutdown()
```
