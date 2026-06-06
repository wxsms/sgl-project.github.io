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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.23it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.37it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.78it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.78it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.78it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.78it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.78it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.78it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.78it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.78it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.78it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.78it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.81it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.81it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.81it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.81it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.81it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.81it/s]

    Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.81it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.81it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.81it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 28.50it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 28.50it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 28.50it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 28.50it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 28.50it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 28.50it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 28.50it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 28.50it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 31.29it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 31.29it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 31.29it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.28 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.28 GB):   3%|▎         | 2/58 [00:00<00:02, 19.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.27 GB):   3%|▎         | 2/58 [00:00<00:02, 19.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.27 GB):   3%|▎         | 2/58 [00:00<00:02, 19.00it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=59.27 GB):   3%|▎         | 2/58 [00:00<00:02, 19.00it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.27 GB):   9%|▊         | 5/58 [00:00<00:02, 20.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.27 GB):   9%|▊         | 5/58 [00:00<00:02, 20.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.26 GB):   9%|▊         | 5/58 [00:00<00:02, 20.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.25 GB):   9%|▊         | 5/58 [00:00<00:02, 20.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.25 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.25 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.25 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.24 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.97it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=59.24 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.24 GB):  21%|██        | 12/58 [00:00<00:01, 29.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.24 GB):  21%|██        | 12/58 [00:00<00:01, 29.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.24 GB):  21%|██        | 12/58 [00:00<00:01, 29.58it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.23 GB):  21%|██        | 12/58 [00:00<00:01, 29.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.23 GB):  21%|██        | 12/58 [00:00<00:01, 29.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.23 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.23 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.22 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.22 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.96it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=59.22 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.22 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.20 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.69it/s]Capturing num tokens (num_tokens=960 avail_mem=59.21 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.69it/s] Capturing num tokens (num_tokens=896 avail_mem=59.21 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.69it/s]Capturing num tokens (num_tokens=832 avail_mem=59.21 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.69it/s]Capturing num tokens (num_tokens=768 avail_mem=59.20 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.69it/s]Capturing num tokens (num_tokens=768 avail_mem=59.20 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=704 avail_mem=59.20 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=640 avail_mem=59.20 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=576 avail_mem=59.20 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=512 avail_mem=59.18 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.28it/s]

    Capturing num tokens (num_tokens=480 avail_mem=59.20 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=480 avail_mem=59.20 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.45it/s]Capturing num tokens (num_tokens=448 avail_mem=59.19 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.45it/s]Capturing num tokens (num_tokens=416 avail_mem=59.19 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.45it/s]Capturing num tokens (num_tokens=384 avail_mem=59.19 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.45it/s]Capturing num tokens (num_tokens=352 avail_mem=59.18 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.45it/s]Capturing num tokens (num_tokens=320 avail_mem=59.18 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.45it/s]

    Capturing num tokens (num_tokens=320 avail_mem=59.18 GB):  60%|██████    | 35/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=288 avail_mem=59.18 GB):  60%|██████    | 35/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=256 avail_mem=59.17 GB):  60%|██████    | 35/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=240 avail_mem=59.17 GB):  60%|██████    | 35/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=224 avail_mem=59.17 GB):  60%|██████    | 35/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=208 avail_mem=59.16 GB):  60%|██████    | 35/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=208 avail_mem=59.16 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=192 avail_mem=59.16 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=176 avail_mem=59.16 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=160 avail_mem=59.16 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=144 avail_mem=59.15 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=128 avail_mem=59.15 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.47it/s]

    Capturing num tokens (num_tokens=128 avail_mem=59.15 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=112 avail_mem=59.15 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=96 avail_mem=59.15 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.26it/s] Capturing num tokens (num_tokens=80 avail_mem=59.14 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=64 avail_mem=59.14 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=48 avail_mem=59.13 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=48 avail_mem=59.13 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=32 avail_mem=59.13 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=28 avail_mem=59.13 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=24 avail_mem=59.12 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=20 avail_mem=59.12 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=16 avail_mem=59.12 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.84it/s]

    Capturing num tokens (num_tokens=16 avail_mem=59.12 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.51it/s]Capturing num tokens (num_tokens=12 avail_mem=59.11 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.51it/s]Capturing num tokens (num_tokens=8 avail_mem=59.11 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.51it/s] Capturing num tokens (num_tokens=4 avail_mem=59.11 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.51it/s]Capturing num tokens (num_tokens=4 avail_mem=59.11 GB): 100%|██████████| 58/58 [00:01<00:00, 37.49it/s]


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
    Generated text:  Mark and I am a software developer. I have been working in the industry for over a decade and have spent my time building exciting projects in a wide range of technologies, from JavaScript and Node.js to Ruby and PHP. My passion for coding has always been in the way that I can make things work. 
    My knowledge has also helped me to expand my career, as I have worked with a variety of different clients and clients have recommended me and my skills to the industry. I have experience with several different web technologies, including Ruby on Rails, Node.js, Java, and Python.
    My current projects include a mobile application that I created,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He or she is like a king or a queen, so many people like to think that he or she is a god. But in reality, the president is only like a leader. He or she has a lot of powers, and the people can have any power. The president does not always make the laws. Law makers make the laws, and they decide what rules and punishments to use in the country. They also decide what kind of people they want to work with in the country.
    
    Who is the president of the United States? 
    A) A king or a queen
    B) The president
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A) Paris B) Nice C) Marseille D) Lyon
    Answer:
    
    A) Paris
    
    Paris is the capital of France, which is a country in Western Europe. It is known for its historical landmarks such as Notre-Dame Cathedral, Eiffel Tower, and the Louvre Museum. Paris is a modern city with a rich cultural heritage, and it is a popular tourist destination for tourists and locals alike. It is also a financial center, with many famous financial institutions and companies headquartered there. Despite being a historical city, Paris is a modern and vibrant city with a high standard of living. Its famous landmarks and cultural attractions
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but the acceptance of new technologies is undeniable. The rapid increase of AI-powered devices and software has created a thriving market for AI professionals, which has led to a growing need for qualified AI professionals.
    
    AI professionals can be found in a variety of roles, including but not limited to, data scientists, machine learning engineers, and computer vision researchers. They are responsible for developing and implementing AI systems, creating algorithms, and improving the performance of AI models.
    
    The rise of AI has also led to a growing need for AI designers, which includes areas such as data science, machine learning, computer vision, and natural language processing. These roles require


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


    Generated text:  [Name] and I'm a [occupation] who has been working in the [industry] for [number] years. I'm passionate about [reason for interest] and have always been [character trait]. I'm always looking for opportunities to [action or goal]. I'm [character trait] and I'm excited to [reason for excitement]. What's your name, and what do you do for a living? [Name] [Name] [Name] [Name] [Name] [Name] [Name] [Name] [Name] [Name] [Name] [Name] [Name] [Name] [Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. The city is home to many famous French artists, writers, and musicians, and is a major hub for international business and diplomacy. Paris is a vibrant and dynamic city, with a diverse population and a rich cultural heritage. Its status as the capital of France has made it a major political and economic center
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Greater emphasis on ethical considerations: As AI becomes more prevalent in various industries, there will be a greater emphasis on ethical considerations and regulations. This could lead to more stringent standards and guidelines for AI development and deployment, as well as increased scrutiny of AI systems in the event of unintended consequences.
    
    3.
    


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
    Generated text:  [Your Name]. I have been working for [Company Name] for [Number] years, and I am passionate about [Your Profession/Interest]. I enjoy [Your Profession/Interest] because [Your Perspective]. I strive to be a [Your Preferred Role] to [Your Position or Job Title], and I am always looking for ways to [Your Preferred Activity or Hobby]. What makes you unique and who are some of your favorite experiences? [Your Name] wants to start a new chapter in their career and is looking for someone who is passionate about [Your Profession/Interest] and is always seeking ways to grow and learn. Can
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city that is known as "The City of Light" for its vibrant culture and beautiful architecture.
    
    Paris is the capital of France and is also known as "The City of Light" for its distinctive culture, beautiful architecture, and historic sites such as the Eiffel Tower and the Louvre Museum. It is also the center of politics, culture, and industry in France. 
    
    Paris is often referred to as the "City of Love" due to its numerous romantic attractions such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. It is also home to some of the world's most
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, with many potential developments and innovations on the horizon. Here are some possible future trends in AI:
    
    1. Enhanced cognitive functions: AI is becoming more capable of processing and understanding human language, emotions, and social interactions. This means that future AI will be able to learn and adapt to a variety of complex situations, improving decision-making and problem-solving capabilities.
    
    2. Universal comprehension: AI will continue to learn and improve its ability to comprehend human language, which will allow it to communicate effectively with people from all walks of life.
    
    3. Autonomous vehicles: With the increasing use of AI in transportation, autonomous vehicles will become more common.


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

    ].

     I

     am

     a

     [

    Gender

    ]

     who

     lives

     in

     [

    City

    ].

     I

     am

     passionate

     about

     [

    Field

     of

     Interest

    ].

     I

     am

     determined

     to

     [

    Achie

    vement

    ].

     I

     am

     [

    Age

    ].

     I

     hope

     to

     have

     the

     [

    Number

    ]

     of

     friends

     on

     my

     list

    .

     I

     am

     [

    Company

     Name

    ]'

    s

     [

    Role

    ].

     Welcome

     to

     my

     world

    .

     Let

    's

     get

     to

     know

     each

     other

     better

    .


    [

    Name

    ]:

     Hello

    ,

     my

     name

     is

     [

    Name

    ].

     I

     am

     a

     [

    Gender

    ]

     who

     lives

     in

     [

    City

    ].

     I

     am

     passionate

     about

     [

    Field

     of

     Interest

    ].

     I

     am

     determined

     to

     [

    Achie

    vement

    ].

     I

     am

     [

    Age

    ].

     I

     hope

     to

     have

     the

     [

    Number

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     south

     of

     the

     country

    .

     It

     is

     the

     largest

     city

     in

     Europe

     and

     is

     known

     as

     the

     "

    City

     of

     Light

    "

     due

     to

     its

     rich

     history

    ,

     culture

    ,

     and

     artistic

     scene

    .

     Paris

     is

     famous

     for

     its

     iconic

     landmarks

     such

     as

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

    ,

     and

     is

     also

     home

     to

     many

     famous

     museums

    ,

     including

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     The

     city

     is

     also

     known

     for

     its

     cuisine

    ,

     especially

     its

     French

     cooking

    ,

     and

     is

     home

     to

     numerous

     festivals

     and

     events

     throughout

     the

     year

    .

     Overall

    ,

     Paris

     is

     a

     city

     that

     is

     steep

    ed

     in

     history

    ,

     culture

    ,

     and

     art

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

     and

     uncertain

    .

     Some

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

     include

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

     and

     deep

     learning

    :

     Advances

     in

     AI

     technology

    ,

     such

     as

     machine

     learning

     and

     deep

     learning

    ,

     will

     likely

     lead

     to

     more

     sophisticated

     and

     powerful

     AI

     systems

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     expected

     to

     play

     an

     increasingly

     important

     role

     in

     healthcare

    ,

     from

     diagnosis

     and

     treatment

     to

     patient

     care

     and

     monitoring

    .
    


    3

    .

     Integration

     with

     human

     experts

    :

     AI

     systems

     are

     likely

     to

     be

     used

     in

     conjunction

     with

     human

     experts

     to

     improve

     decision

    -making

     and

     decision

    -making

     accuracy

    .
    


    4

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

    



```python
llm.shutdown()
```
