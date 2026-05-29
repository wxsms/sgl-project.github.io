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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.16it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:46,  3.97s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:46,  3.97s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:46,  3.97s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:46,  3.97s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:46,  3.97s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  9.10it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:04<00:04,  9.10it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 23.47it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:04<00:00, 23.47it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 33.13it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 33.13it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 33.13it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 33.13it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 33.13it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 33.13it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 33.13it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:04<00:00, 33.13it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:04<00:00, 33.13it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:04<00:00, 33.13it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:04<00:00, 33.13it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:04<00:00, 33.13it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.33it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.17it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.17it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.17it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.88it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.88it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.88it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.88it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 36.25it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.51it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.51it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.51it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.51it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.51it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.51it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.85it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.70it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.16it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.10it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.10it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.18it/s]

    Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.18it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.35it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.35it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 45.71it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 40.40it/s]


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
    Generated text:  Sam and I am a third year student at the University of Toronto. I came to the United States to pursue a Master's in Chemistry, but before that I did a Bachelor's in English. I have experience teaching elementary school math, and I am very passionate about helping students succeed. I have an interest in promoting international understanding and my background in the humanities means I can also engage with students who are learning languages. Currently I am working on a language course for international students. I am interested in learning more about the diversity of the Canadian language scene and would like to explore further opportunities to study with students from various backgrounds.
    What is your background
    ===============================
    Prompt: The president of the United States is
    Generated text:  from which country?
    What is the answer? (A). England (B). France (C). Italy (D). United States
    The answer is D. United States. The president of the United States is the President of the United States, and he or she is from the United States. The other options are incorrect because France and Italy are not countries in the same way the United States is. England, on the other hand, is a country and not a person in this scenario.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The abbreviation of Paris is P.
    A. Correct
    B. Incorrect
    Answer:
    
    A
    
    According to the "People's Republic of China Production Safety Law", production and operation entities should establish and improve systems for the investigation and management of accident隐患and take corresponding ___ measures based on the investigation and management of accident隐患.
    A. Safety Management
    B. Safety Inspection
    C. Safety Protection
    D. Hidden Hazards Governance
    Answer:
    
    D
    
    The Municipal Bureau of Industry and Commerce has the authority to formulate the rules for the establishment and operation of industry and commerce enterprises in its region, and it shall be announced to the public
    ===============================
    Prompt: The future of AI is
    Generated text:  fast becoming the future of AI. The "future of AI" is the direction that is advancing the field. It is the future of AI in the future. We will see many new advancements in AI in the years to come. We will see more and more smart machines taking over the world. If you have any questions about the future of AI, you are welcome to ask them.
    
    What are the possible uses of artificial intelligence in the future?
    
    AI is changing the way we live, work and think. It is changing the way we use technology and it is changing the way we interact with technology. It is changing the way we approach problems


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


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art, and cuisine. Paris is a cultural and economic center of France and a major tourist destination. It is home to many famous landmarks and attractions, including the Louvre, the Eiffel Tower, and the Champs-Élysées. Paris is a city that has a rich history and continues to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into various industries, from manufacturing to healthcare to transportation. This automation will likely lead to the creation of new jobs and the displacement of existing ones, but it will also create new opportunities for people to work in areas such as data analysis, software development, and machine learning.
    
    2. Improved privacy and security: As AI becomes more integrated into our daily lives, there will be a growing need for measures to protect the
    


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
    Generated text:  [Name], and I'm a [job title] at [company]. I'm an [age] year old and I enjoy [interests/experiences] with a strong sense of [sense of self]. How can I help you today? Start with a simple yet effective introduction that captures the essence of your character and leaves a lasting impression. Consider using a conversational tone to make your self-introduction feel more approachable and genuine.
    
    [Name] is a [job title] at [company], with a passion for [personal interest/experience], which I've found to be as stimulating and rewarding as they are unique to me.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the region of the same name, on the western coast of the Mediterranean Sea, and is the largest city in Europe and the second largest city in the world by population. It is known for its historical and cultural importance, and its world-renowned museums, art galleries, and fashion brands. Paris is also a hub for entertainment, music, and film, and is famous for its annual festivals, including the Eiffel Tower, La Fête de la Feuille, and the Musée de la République. The city's skyline is dotted with tall buildings and iconic landmarks, including the Eiffel Tower
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to involve several key trends, including:
    
    1. Increased use of AI in healthcare: With the prevalence of chronic diseases, AI can be used to analyze medical data, predict disease progression, and develop personalized treatment plans.
    
    2. Increased use of AI in finance: AI can be used to automate trading, identify fraud, and optimize investment portfolios.
    
    3. Increased use of AI in transportation: AI can be used to improve traffic management, predict traffic congestion, and optimize travel routes.
    
    4. Increased use of AI in education: AI can be used to personalize learning experiences, analyze student performance, and identify learning gaps.
    
    5. Increased use of


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

    character

     type

    ]

     with

     [

    personal

     information

    ].

     My

     [

    character

     type

    ]

     is

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

     I

     am

     [

    character

     type

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

     with

     a

     rich

     history

     and

     diverse

     culture

    .

     It

     is

     located

     on

     the

     Se

    ine

     River

     and

     is

     known

     for

     its

     romantic

     architecture

    ,

     charming

     streets

    ,

     and

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

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

     cosm

    opolitan

     city

     with

     a

     vibrant

     cultural

     scene

     and

     is

     considered

     one

     of

     the

     most

     important

     cities

     in

     the

     world

    .

     Its

     significance

     has

     made

     it

     the

     heart

     of

     French

     culture

     and

     influence

    ,

     and

     it

     remains

     a

     major

     global

     hub

     for

     trade

    ,

     entertainment

    ,

     and

     education

    .

     Its

     status

     as

     the

     capital

     of

     France

     is

     reflected

     in

     its

     elegant

     and

     influential

     political

    ,

     economic

    ,

     and

     cultural

     institutions

    .

     Paris

     is

     also

     known

     for

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     key

     trends

    ,

     including

    :
    


    1

    .

     Increased

     Use

     of

     AI

     for

     Automation

    :

     AI

     is

     expected

     to

     become

     even

     more

     prevalent

     in

     everyday

     life

    ,

     with

     a

     wide

     range

     of

     applications

     that

     can

     automate

     routine

     tasks

     and

     save

     time

    .

     This

     will

     likely

     lead

     to

     job

     displacement

    ,

     but

     also

     create

     new

     opportunities

     for

     workers

     who

     can

     use

     AI

     to

     perform

     tasks

     that

     are

     repetitive

     or

     mundane

    .
    


    2

    .

     Greater

     Integration

     of

     AI

     into

     Everyday

     Life

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     become

     more

     integrated

     into

     everyday

     life

    ,

     from

     voice

     assistants

     like

     Siri

     and

     Alexa

     to

     virtual

     assistants

     that

     can

     understand

     complex

     natural

     language

     queries

    .

     This

     will

     likely

     lead

     to

     a

     more

     seamless

     and

     intuitive

    



```python
llm.shutdown()
```
