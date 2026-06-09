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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.89it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.29it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.62it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.47it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.47it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.47it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.47it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.47it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.47it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.47it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.47it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.47it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.47it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.47it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.40it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.40it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.40it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.40it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.40it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.40it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.40it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.40it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.40it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.40it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.40it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.23it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.23it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.23it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.23it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.23it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.23it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.23it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.23it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.28 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.22 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.22 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.21 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.21 GB):   3%|▎         | 2/58 [00:00<00:02, 18.81it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.21 GB):   9%|▊         | 5/58 [00:00<00:02, 21.93it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.21 GB):   9%|▊         | 5/58 [00:00<00:02, 21.93it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.19 GB):   9%|▊         | 5/58 [00:00<00:02, 21.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.19 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.19 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.19 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.18 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.18 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.85it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.18 GB):  21%|██        | 12/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.18 GB):  21%|██        | 12/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.18 GB):  21%|██        | 12/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.17 GB):  21%|██        | 12/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.17 GB):  21%|██        | 12/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.17 GB):  21%|██        | 12/58 [00:00<00:01, 30.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.17 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.14 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=960 avail_mem=72.15 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=72.15 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.68it/s]Capturing num tokens (num_tokens=896 avail_mem=72.15 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.68it/s]Capturing num tokens (num_tokens=832 avail_mem=72.15 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.68it/s]Capturing num tokens (num_tokens=768 avail_mem=72.14 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.68it/s]Capturing num tokens (num_tokens=704 avail_mem=72.14 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.68it/s]Capturing num tokens (num_tokens=640 avail_mem=72.14 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.68it/s]Capturing num tokens (num_tokens=640 avail_mem=72.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=576 avail_mem=72.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=512 avail_mem=72.12 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=480 avail_mem=72.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=448 avail_mem=72.14 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=416 avail_mem=72.13 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.94it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.13 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.50it/s]Capturing num tokens (num_tokens=384 avail_mem=72.13 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.50it/s]Capturing num tokens (num_tokens=352 avail_mem=72.13 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.50it/s]Capturing num tokens (num_tokens=320 avail_mem=72.12 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.50it/s]Capturing num tokens (num_tokens=288 avail_mem=72.12 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.50it/s]Capturing num tokens (num_tokens=256 avail_mem=72.12 GB):  55%|█████▌    | 32/58 [00:00<00:00, 43.50it/s]Capturing num tokens (num_tokens=256 avail_mem=72.12 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.79it/s]Capturing num tokens (num_tokens=240 avail_mem=72.11 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.79it/s]Capturing num tokens (num_tokens=224 avail_mem=72.11 GB):  64%|██████▍   | 37/58 [00:00<00:00, 44.79it/s]Capturing num tokens (num_tokens=208 avail_mem=72.10 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.79it/s]Capturing num tokens (num_tokens=192 avail_mem=72.10 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.79it/s]Capturing num tokens (num_tokens=176 avail_mem=72.10 GB):  64%|██████▍   | 37/58 [00:01<00:00, 44.79it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.10 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=160 avail_mem=72.10 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=144 avail_mem=72.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=128 avail_mem=72.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=112 avail_mem=72.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.59it/s]Capturing num tokens (num_tokens=96 avail_mem=72.09 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.59it/s] Capturing num tokens (num_tokens=96 avail_mem=72.09 GB):  81%|████████  | 47/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=80 avail_mem=72.08 GB):  81%|████████  | 47/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=64 avail_mem=72.08 GB):  81%|████████  | 47/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=48 avail_mem=72.08 GB):  81%|████████  | 47/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=32 avail_mem=72.07 GB):  81%|████████  | 47/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=28 avail_mem=72.07 GB):  81%|████████  | 47/58 [00:01<00:00, 45.69it/s]

    Capturing num tokens (num_tokens=28 avail_mem=72.07 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=24 avail_mem=72.07 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=20 avail_mem=72.06 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=16 avail_mem=72.06 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=12 avail_mem=72.06 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.73it/s]Capturing num tokens (num_tokens=8 avail_mem=72.05 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.73it/s] Capturing num tokens (num_tokens=8 avail_mem=72.05 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.24it/s]Capturing num tokens (num_tokens=4 avail_mem=72.05 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.24it/s]Capturing num tokens (num_tokens=4 avail_mem=72.05 GB): 100%|██████████| 58/58 [00:01<00:00, 40.75it/s]


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
    Generated text:  Lacie and I'm a tutor at a public school. I'm from England and I'm a math tutor. I'm really good at math and I can help you learn math too. Come see me at my home or at my school. -This speech is suitable for the purpose of _____. A. advertising B. introducing C. explaining D. selling
    Answer:
    
    A
    
    After the outbreak of the SARS epidemic, the Chinese government took an extremely strong and decisive response, successfully controlling the epidemic. Which of the following measures did the Chinese government take to control the epidemic?
    A. Conducting a census
    B. Directly
    ===============================
    Prompt: The president of the United States is
    Generated text:  a powerful leader who is willing to make difficult decisions. This is because the U.S. government is a government of the people. The president is directly elected by the people who vote for him, and he is accountable to them. The president’s decisions are made based on what is best for the country. The president must be aware of the needs and desires of the American people, and he must make decisions that will be in the best interest of the country as a whole.
    As a leader, the president is responsible for the overall direction of the country. He is responsible for making decisions that will help the country be successful in its goals.
    ===============================
    Prompt: The capital of France is
    Generated text:  (  )
    A: Lyon
    B: Paris
    C: Paris, Lyon
    D: Paris, Marseille
    
    To determine the capital of France, we need to consider the historical and political context of the country. The French capital is typically Paris, as it is the largest city in France and one of the most populous cities in Europe. Let's go through the options one by one:
    
    A: Lyon
    - Lyon is a city in France, but it is not the capital. Lyon is the capital of the French department of North-Western regions.
    
    B: Paris
    - Paris is the capital of France. It is located in the
    ===============================
    Prompt: The future of AI is
    Generated text:  now, and it is fast. The major trends in AI are coming from quantum computing, blockchain, the Internet of Things, and more.
    There’s a lot of hype about how AI could change jobs in the future. But just how far will AI be in the way of job displacement? In this blog post, I want to focus on how blockchain and quantum computing could impact jobs in the future.
    Quantum computing is a technology that uses quantum-mechanical phenomena to perform operations on data. It is a major force driving the development of new technologies in areas such as computing, telecommunications, and medicine. It could revolutionize the way we


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character or profession]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new experiences and learning opportunities. What are some of your favorite things to do? I love [insert a short description of your favorite activities or hobbies]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite thing to do in your free time? I love [insert
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the Middle Ages. It is a major transportation hub and a major tourist destination. The city is known for its cuisine, including French cuisine, and is home to many famous French restaurants and cafes. Paris is also known for its fashion industry, with many famous designers and boutiques. The city is a major economic center and a major player
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence. This could lead to more efficient and effective use of AI, as well as more accurate and nuanced decision-making.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations. This could lead to more rigorous testing and evaluation of AI systems, as well as more transparent and accountable use of AI.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs.
    


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
    Generated text:  [insert name], and I'm a [insert occupation or profession] with [insert relevant background, such as education, experience, or leadership role]. I've been [insert relevant experience or achievements] and I'm always seeking to learn and grow in my field. I enjoy [insert hobbies or interests], and I'm always looking to create new experiences and experiences to improve my skills. I'm passionate about [insert a personal interest or passion], and I'm committed to doing my best to make a positive impact in my community. Whether it's through helping someone in need or participating in volunteer work, I strive to make a difference in the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    To expand on this point, provide the population of Paris. Approximately 2 million people live there. 
    
    Add a sentence explaining that Paris is one of the most important cultural and political centers of the world. 
    
    Please include the current status of Paris in its population of 2 million. Paris is home to the European Parliament, a huge government body. 
    
    Lastly, please provide a sentence about Paris's famous landmark, the Eiffel Tower. It is a famous landmark in Paris. The Eiffel Tower is a prominent structure in the city and is a symbol of Paris.
    
    Use only the information given and organize your response
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and constantly evolving. Here are some possible trends that could shape the field in the coming years:
    
    1. Deep learning: As the number of training examples grows, so does the effectiveness of deep learning algorithms. Researchers are developing more sophisticated models that can recognize patterns in images and videos, as well as in natural language.
    
    2. Computer vision: AI systems that can interpret and understand visual information are already being used in a variety of applications, from autonomous vehicles to security cameras to medical diagnosis.
    
    3. Natural language processing: As language models become more sophisticated, natural language processing techniques will become more common in AI applications. This includes things like chat


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

    ,

     and

     I

     love

     reading

     and

     writing

    .

     I

     have

     a

     passion

     for

     exploring

     the

     mysteries

     of

     the

     universe

     and

     trying

     to

     make

     sense

     of

     the

     seemingly

     impossible

    .

     I

    'm

     a

     curious

     and

     adventurous

     person

     who

     enjoys

     trying

     new

     experiences

     and

     always

     looking

     for

     new

     and

     exciting

     ways

     to

     learn

    .

     I

    'm

     a

     strong

     believer

     in

     the

     power

     of

     the

     human

     spirit

     to

     overcome

     even

     the

     most

     daunting

     challenges

     and

     have

     always

     been

     fascinated

     by

     the

     idea

     of

     living

     in

     harmony

     with

     the

     natural

     world

    .

     I

     enjoy

     spending

     time

     outdoors

    ,

     exploring

     new

     places

    ,

     and

     trying

     new

     foods

    .

     I

    'm

     always

     looking

     for

     new

     and

     exciting

     ways

     to

     learn

     and

     grow

     as

     a

     person

    .

     I

    'm

     also

     a

     strong

     advocate

     for

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Its

     name

     in

     French

     is

     "

    Paris

    "

     and

     it

     is

     located

     in

     the

     southern

     part

     of

     France

    .

     The

     city

     was

     founded

     in

     the

     

    6

    th

     century

    ,

     and

     it

     is

     home

     to

     numerous

     historical

     landmarks

    ,

     including

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     a

     major

     international

     hub

     of

     culture

    ,

     commerce

    ,

     and

     fashion

    ,

     and

     is

     a

     UNESCO

     World

     Heritage

     site

    .

     The

     city

     is

     also

     known

     for

     its

     vibrant

     nightlife

    ,

     fashion

    ,

     and

     food

     scene

    .

     According

     to

     a

     

    2

    0

    2

    3

     census

    ,

     Paris

     has

     a

     population

     of

     approximately

     

    1

    0

    .

    7

     million

     people

    .

     With

     its

     rich

     history

    ,

     diverse

     culture

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     dynamic

     and

     depends

     on

     a

     number

     of

     factors

     such

     as

     technological

     advancements

    ,

     changing

     regulatory

     frameworks

    ,

     and

     evolving

     societal

     needs

    .

     Here

     are

     some

     of

     the

     possible

     trends

     in

     AI

     that

     could

     shape

     the

     future

    :
    


    1

    .

     Increased

     automation

    :

     As

     AI

     continues

     to

     become

     more

     sophisticated

    ,

     the

     possibility

     of

     automation

     in

     various

     industries

     is

     increasing

    .

     We

     may

     see

     more

     people

     working

     less

     and

     less

     as

     AI

     takes

     over

     routine

     tasks

     and

     adds

     value

     to

     the

     workforce

    .
    


    2

    .

     Improved

     privacy

    :

     With

     the

     rise

     of

     AI

     and

     big

     data

    ,

     there

     will

     be

     a

     greater

     emphasis

     on

     protecting

     user

     privacy

    .

     We

     will

     see

     more

     stringent

     data

     protection

     regulations

    ,

     more

     advanced

     privacy

    -pres

    erving

     techniques

    ,

     and

     more

     personalized

     data

    



```python
llm.shutdown()
```
