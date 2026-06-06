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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.47it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.47it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.22it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.22it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.22it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.22it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.22it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.22it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.22it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.22it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.22it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.22it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.22it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.21it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.21it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.21it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.21it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.21it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.21it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.21it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.08 GB):   3%|▎         | 2/58 [00:00<00:03, 18.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=47.09 GB):   3%|▎         | 2/58 [00:00<00:03, 18.07it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=47.09 GB):   3%|▎         | 2/58 [00:00<00:03, 18.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=47.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.01it/s]Capturing num tokens (num_tokens=5632 avail_mem=47.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.01it/s]Capturing num tokens (num_tokens=5120 avail_mem=47.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=47.07 GB):   9%|▊         | 5/58 [00:00<00:02, 21.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=47.07 GB):   9%|▊         | 5/58 [00:00<00:02, 21.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=47.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=3840 avail_mem=47.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=3584 avail_mem=47.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=47.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=3072 avail_mem=47.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=47.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=47.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=47.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=47.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=47.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=47.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=47.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=47.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=47.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=47.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.57it/s]

    Capturing num tokens (num_tokens=960 avail_mem=47.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.57it/s] Capturing num tokens (num_tokens=896 avail_mem=47.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.57it/s]Capturing num tokens (num_tokens=832 avail_mem=47.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.57it/s]Capturing num tokens (num_tokens=832 avail_mem=47.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=768 avail_mem=47.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=704 avail_mem=47.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=640 avail_mem=47.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=576 avail_mem=47.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=512 avail_mem=47.00 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=512 avail_mem=47.00 GB):  50%|█████     | 29/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=480 avail_mem=47.02 GB):  50%|█████     | 29/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=448 avail_mem=47.01 GB):  50%|█████     | 29/58 [00:00<00:00, 42.54it/s]

    Capturing num tokens (num_tokens=416 avail_mem=47.01 GB):  50%|█████     | 29/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=384 avail_mem=47.01 GB):  50%|█████     | 29/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=352 avail_mem=47.00 GB):  50%|█████     | 29/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=352 avail_mem=47.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=320 avail_mem=47.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=288 avail_mem=47.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=256 avail_mem=46.99 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=240 avail_mem=46.99 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=224 avail_mem=46.99 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=224 avail_mem=46.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=208 avail_mem=46.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=192 avail_mem=46.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.51it/s]

    Capturing num tokens (num_tokens=176 avail_mem=46.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=160 avail_mem=46.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=144 avail_mem=46.97 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=144 avail_mem=46.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=128 avail_mem=46.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=112 avail_mem=46.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=96 avail_mem=46.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.41it/s] Capturing num tokens (num_tokens=80 avail_mem=46.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=64 avail_mem=46.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.41it/s]Capturing num tokens (num_tokens=64 avail_mem=46.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.33it/s]Capturing num tokens (num_tokens=48 avail_mem=46.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.33it/s]Capturing num tokens (num_tokens=32 avail_mem=46.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.33it/s]

    Capturing num tokens (num_tokens=28 avail_mem=46.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.33it/s]Capturing num tokens (num_tokens=24 avail_mem=46.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.33it/s]Capturing num tokens (num_tokens=20 avail_mem=46.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.33it/s]Capturing num tokens (num_tokens=20 avail_mem=46.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=16 avail_mem=46.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=12 avail_mem=46.93 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=8 avail_mem=46.93 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.67it/s] Capturing num tokens (num_tokens=4 avail_mem=46.93 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=4 avail_mem=46.93 GB): 100%|██████████| 58/58 [00:01<00:00, 40.95it/s]


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
    Generated text:  Yuki. I'm a Grade 8 student at Nankai University. At the age of 20, I have a dream to be a teacher, not only because I love teaching but also because I want to make a difference in the lives of the students who need help. When I was young, I spent a lot of time outdoors, reading stories and writing poems, which gave me an advantage in my studies. I used to be the best student in my class. I also loved spending my free time with my family. 
    
    Now, I am a full-time teacher. I believe that by teaching, I can make a
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking official of the government of the United States who represents the country's interests in the international community and exercises authority in the United States, as well as the functions of a state and its government. The president serves a term of four years, and the United States president is elected annually by the citizens of the United States. The United States president is the head of state and the head of government of the United States, and is the Commander-in-Chief of the United States Armed Forces. The president, together with other members of Congress, form the United States Congress, which is the most powerful legislative body in the United States. The
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the first city of what country?  A. germany  B. india  C. canada  D. mexico  E. america  The thought process to arrive at the answer: To answer this question, we need to determine the capital of France and then find the country it belongs to. The capital of France is Paris, and the capital of France is located in the country of France. Therefore, the answer is E. America. The other options (Germany, India, Canada, and Mexico) are not capitals of France. Germany is the capital of Germany, India is the capital of India, Canada is the capital of
    ===============================
    Prompt: The future of AI is
    Generated text:  incredibly exciting. It is essential for every industry to be able to harness the power of AI in order to compete in the market and succeed in the future.
    One of the most promising areas of AI is robotics. Robotics is a field that combines the principles of engineering, computer science, and artificial intelligence to create intelligent machines that can perform tasks that require human-like capabilities.
    Robots are currently used in a wide range of industries, from manufacturing and healthcare to transportation and entertainment. They can perform complex tasks, such as assembling components, cleaning, and transporting goods, all while minimizing the risk of accidents and human error.
    Robots are also used in


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your profession or experience here]. I enjoy [insert a short description of your hobbies or interests here]. I'm always looking for new challenges and opportunities to grow and learn. What do you like to do in your free time? I enjoy [insert a short description of your hobbies or interests here]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the annual Eiffel Tower Festival. It is also the birthplace of French writer Victor Hugo and the home of the Louvre Museum. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also home to the French Parliament and the French National Library. It is the capital of France and is the largest city in the European Union. Paris is known for its vibrant nightlife, fashion, and art scene. It is also a major center for business and finance in Europe. The city is home to many international organizations and is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could emerge in the coming years:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI systems, there will be a greater emphasis on ethical considerations and the development of AI that is designed to be fair, transparent, and accountable.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI becomes more advanced, it is likely to be used
    


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
    Generated text:  [Your Name] and I am a [Your Profession/Position] with [Your Education] in [Your Relevant Subject] and [Your Previous Experience]. I'm [Your Age] years old and I'm currently [Your Job Title] for [Your Company]. My strongest skills are [List your most notable skills here, such as [List one or more of your skills and their application in your current role]].
    
    I enjoy [List one or more of your hobbies and interests, such as [List one or more of your hobbies and interests and how they relate to your character]]. I believe in [List one or more of your core
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the south of the country and is known for its beautiful landmarks, rich history, and world-renowned cuisine. It is the largest and most populous city in the European Union and is also the birthplace of the French language and the symbol of the French national identity. Paris has a rich cultural history, with many famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum, and is known for its lively nightlife, fashion industry, and annual cultural events. Its climate is temperate, with mild winters and hot summers, and it is home to many museums, art galleries, and theaters
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  unpredictable and depends on a number of factors, but some possible trends to expect include:
    
    1. Increasing specialization: AI will continue to become more specialized, with companies and individuals developing their own AI technologies and developing their own AI applications. This will create a more tailored and efficient approach to problem-solving, which could reduce costs and increase productivity.
    
    2. AI will become more ethical: As AI becomes more prevalent in our lives, there will be a growing emphasis on ethical considerations and regulations. This could lead to the development of new ethical AI technologies, such as AI that is designed to minimize harm and promote positive outcomes.
    
    3. AI will become more


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

     am

     a

     [

    Job

     Title

    ]

     in

     a

     [

    Company

     Name

    ].

     I

     have

     always

     been

     a

     [

    Occup

    ation

    ]

     who

     has

     been

     dedicated

     to

     [

    Career

     Goal

    ]

     since

     childhood

    .

     Over

     the

     years

    ,

     I

     have

     hon

    ed

     my

     skills

     and

     developed

     a

     reputation

     as

     a

     [

    Professional

     Trait

     or

     Quality

    ]

     with

     my

     colleagues

    .

     I

     am

     always

     [

    Positive

     Qual

    ities

    ],

     always

     ready

     to

     learn

     and

     grow

    .

     I

     am

     a

     [

    C

    ultural

     Fit

    ]

     and

     I

     enjoy

     [

    Professional

     Experience

    ].

     How

     would

     you

     describe

     your

     personality

     and

     background

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

    Job

     Title

    ]

     in

     a

     [

    Company

     Name

    ].

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    .

     It

     is

     a

     bustling

     met

    ropolis

     with

     a

     rich

     cultural

     heritage

    ,

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

     and

     the

     Lou

    vre

     Museum

    ,

     and

     a

     vibrant

     nightlife

    .

     Paris

     is

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

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     Mus

    ée

     Rod

    in

    ,

     as

     well

     as

     a

     number

     of

     historical

     and

     cultural

     venues

     such

     as

     the

     Pal

    ais

     de

     Justice

     and

     the

     Opera

     Garn

    ier

    .

     Paris

     is

     also

     a

     major

     financial

     center

     and

     home

     to

     a

     number

     of

     famous

     organizations

     and

     institutions

    ,

     including

     the

     French

     National

     Library

     and

     the

     E

    NS

     (

    É

    cole

     nation

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     a

     mix

     of

     progress

     and

     setbacks

    ,

     as

     it

     continues

     to

     develop

     at

     an

     unprecedented

     pace

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

     Improved

     accuracy

     and

     reliability

    :

     As

     AI

     algorithms

     become

     more

     sophisticated

    ,

     they

     are

     likely

     to

     become

     more

     accurate

     and

     reliable

     in

     their

     predictions

     and

     decisions

    .

     This

     could

     lead

     to

     new

     applications

     such

     as

     self

    -driving

     cars

    ,

     medical

     diagnosis

    ,

     and

     personalized

     medicine

    .
    


    2

    .

     Integration

     with

     human

     cognitive

     abilities

    :

     AI

     is

     likely

     to

     become

     more

     integrated

     with

     human

     cognitive

     abilities

     in

     the

     future

    .

     This

     could

     lead

     to

     new

     ways

     of

     interacting

     with

     technology

     and

     the

     ability

     to

     understand

     and

     respond

     to

     human

     emotions

     and

     needs

    .
    


    3

    .

     Increased

    



```python
llm.shutdown()
```
