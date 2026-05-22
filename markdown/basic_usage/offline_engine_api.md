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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.15it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.14it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.45it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.99it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.30it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.47it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.47it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.47it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.47it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.47it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.47it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.47it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.47it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.29it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.29it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.29it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.53it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.53it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.53it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.53it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.53it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.53it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.38it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.37it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.37it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.37it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.37it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.37it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.65it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 43.29it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 43.29it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 43.29it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 43.29it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 43.29it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 43.29it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.48it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.48it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.48it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.48it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.48it/s]

    Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.48it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.48it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.63it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.63it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.82it/s]

    Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.82it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.19it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.19it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.19it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 48.19it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 42.45it/s]


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
    Generated text:  Alexandra and I am currently an undergraduate student at California State University, Los Angeles. I have always been a creative person with a passion for music, so I love to explore different genres of music and create my own musical compositions. I have taken many music classes and workshops to learn various instruments and compose music. I have a strong commitment to creative writing and have written several short stories and poetry. Currently, I am a member of the school's Science Fiction Society and have volunteered at a local museum.
    I am also an active member of the campus band, the Cal Ripley Wind Band, and have participated in various other music-related activities on campus
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to fire a certain executive. The executive has been paid a salary of $150,000 per year for 10 years. The executive has also been offered a job at a company that is currently undergoing a merger, and will make an additional $500,000 per year. The president has also made other proposals to the executive to increase the salary, but the executive refuses to accept them. 
    
    Assuming that the executive is a career professional who believes in self-employment and has no other job, what would be the net effect on the president's salary if he decides to
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Nantes
    C. Lyon
    D. Marseille
    Answer:
    A
    
    After the Third Plenary Session of the Eleventh Central Committee of the Communist Party of China, what is the most distinctive feature of our country's economic system reform?
    A. Accelerating economic restructuring
    B. Improving enterprise management systems
    C. Reforming state-owned enterprises
    D. Developing foreign trade
    Answer:
    C
    
    The construction of the network economy is driven by the significant changes in human life. The development of information technology, the increase in the number of users, and the emergence of new platforms have propelled the
    ===============================
    Prompt: The future of AI is
    Generated text:  not just about building better and more powerful computers. It’s about building better and more accurate robots that can work alongside humans in various roles. In this blog post, we will explore the key areas of focus and emerging technologies in the field of robotics and AI. We will also examine the different ways that robots are being used today and how they will continue to evolve in the future. By the end of this post, you will have a better understanding of the role that robots and AI will play in society in the coming years. 
    The first thing to understand about robotics and AI is that they are two distinct but related fields. Robotics is the


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the seat of the French government and home to many of the country's cultural institutions. Paris is a bustling metropolis with a rich history and a diverse population, making it a popular tourist destination. The city is also known for its cuisine, including French cuisine, and its fashion industry. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has been a hub of culture and innovation for centuries, and continues to be a major center
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy. AI developers will need to be more mindful of the potential impact of their technology on society and work to ensure that it is used in a responsible and ethical manner.
    
    2. Greater integration with other technologies: AI is likely to become more integrated with other technologies, such
    


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
    Generated text: ... (insert character's name) and I am a/an (insert fictional character's profession or background) who is passionate about (insert why you love what you do).
    
    My experience has taught me that (insert why you're the best at what you do), and I'm dedicated to (insert why you're doing this). Let me know if you'd like to discuss anything in more detail or if you'd like me to give you a taste of my work. #YourName
    
    Sure, I'd love to give you a taste of my work! Can you tell me a bit more about your background and how you ended up at this
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a major city and the largest city in Europe by population, with an estimated population of over 2.7 million. Paris is also known for its rich history, cultural diversity, and iconic landmarks, such as the Eiffel Tower and Notre-Dame Cathedral. The city's status as a global hub of art, culture, and politics has made it a UNESCO World Heritage site. Despite its size, Paris is still a lively and vibrant city with a rich history and diverse culture that has inspired countless artists, writers, and musicians. It is considered one of the most beautiful cities in the world. The city is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid progress and change. Some possible trends in AI include:
    
    1. Increased AI integration with other technologies: AI is already being integrated into many other technologies, such as smart cities, autonomous vehicles, and health monitoring devices. In the future, it is likely that AI will be integrated into more of these technologies, creating even more efficient and adaptive systems.
    
    2. Increased focus on ethical considerations: As AI is increasingly used in human contexts, there will be increased focus on ethical considerations and regulations to ensure that AI is used in a way that is fair and responsible.
    
    3. AI-powered healthcare advancements: AI is already being


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

    'm

     a

    /an

     [

    职业

    ]

     [

    职业

    描述

    ]

     [

    职业

    描述

    ]

     [

    职业

    描述

    ]

    .
    


    I

    ’m

     currently

     [

    当前

    职位

    ]

     at

     [

    公司

    名

    ],

     working

     towards

     [

    职业

    目标

    ]

     in

     [

    行业

    或

    领域

    ]

     (

    e

    .g

    .,

     Marketing

    ,

     Finance

    ,

     IT

    ).

     Throughout

     my

     career

    ,

     I

     have

     always

     valued

     [

    why

     I

     value

     this

    ],

     and

     I

     am

     passionate

     about

     [

    what

     I

     do

     for

     fun

    ]

     and

     [

    my

     personal

     interest

     in

    /

    need

     for

    ]

     [

    interest

     or

     hobby

    ].
    


    I

     believe

     in

     the

     importance

     of

     [

    why

    ]

     and

     my

     commitment

     to

     [

    why

     I

     believe

     this

    ],

     and

     I

     strive

     to

     be

     [

    a

    /an

    ]

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     capital

     of

     France

    .

     Its

     population

     is

     approximately

     

    2

    .

    2

     million

     as

     of

     

    2

    0

    2

    1

    .

     It

     is

     a

     cosm

    opolitan

     city

    ,

     famous

     for

     its

     historical

     landmarks

    ,

     such

     as

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

     also

     known

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     art

     scene

    .

     The

     city

     has

     a

     rich

     cultural

     heritage

     and

     is

     a

     major

     economic

     and

     tourist

     center

     in

     Europe

    .

     It

     is

     also

     home

     to

     many

     notable

     universities

    ,

     including

     the

     É

    cole

     poly

    techn

    ique

     and

     the

     Sor

    bon

    ne

    .

     Paris

     is

     a

     popular

     destination

     for

     business

    ,

     leisure

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

    ,

     and

     there

     is

     no

     doubt

     that

     it

     will

     continue

     to

     evolve

     and

     develop

     in

     the

     coming

     years

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     Integration

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     become

     more

     integrated

     with

     other

     technologies

     such

     as

     IoT

    ,

     blockchain

    ,

     and

     blockchain

    -based

     systems

    .

     This

     integration

     will

     enable

     AI

     to

     be

     more

     efficient

    ,

     adaptable

    ,

     and

     effective

     in

     managing

     and

     optimizing

     the

     performance

     of

     other

     systems

    .
    


    2

    .

     Personal

    ization

     and

     Ad

    ap

    ting

     to

     User

     Needs

    :

     As

     AI

     continues

     to

     improve

    ,

     it

     will

     become

     more

     personalized

     and

     adaptable

     to

     user

     needs

    .

     AI

     systems

     will

     be

     able

     to

     learn

     from

     user

     data

     and

     feedback

    ,

     and

    



```python
llm.shutdown()
```
