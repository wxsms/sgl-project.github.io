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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.33it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.87it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.87it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.87it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.87it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.87it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.87it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.87it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.87it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.87it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.87it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.63it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.63it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.63it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.63it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.63it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.63it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.63it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.63it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.63it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.63it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:01, 14.52it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.47it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 29.56it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 29.56it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 40.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.89it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.45it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.45it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.45it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.45it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.45it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.45it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s] Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.25it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.25it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.25it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.25it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.25it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.25it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.48it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.48it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.48it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.48it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.48it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.48it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.63it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.63it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.63it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.63it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.63it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.63it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.64it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.64it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.19it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.11it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=16 avail_mem=72.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.17it/s] Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 44.86it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 39.63it/s]


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
    Generated text:  Luke and I am 23 years old.
    I grew up in the small town of Dundrum, Ireland, and attended Trinity College, Dublin. I graduated with a Bachelor of Arts in Music Composition in 2008. While I was at Trinity, I worked at the film production company, 2020 Films, and I am the Musical Director and Vocalist for the Celtic community in Dundrum. I also worked as a teacher and have a passion for music in my free time. I am currently pursuing a Master of Music in Composition with a specialization in Counterpoint, and my performance focus is on vocal music.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a young man named Joe Biden. He is a natural born American. He was born in 1942. He is a member of the Democratic party. He was voted for the presidency by the voters in the 2016 election. He is a member of the U.S. Army. He was the commander of the U.S. Navy's combatant ships, the USS Vincennes and the USS S. Arizona, during the attack on Pearl Harbor. He served in the 77th Infantry Regiment, 2nd Armored Division, and the 10th Armored Division. In 
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Washington
    D. Madrid
    
    Answer the question: The capital of France is:
    
    A. Paris
    
    The capital of France is Paris. Paris is the largest city in France and the capital of the country. It is located on the banks of the River Seine, on the Left Bank of the river. Paris is known for its rich history, art, culture, and cuisine. The city is also famous for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and the Louvre Palace. Paris is home to many renowned museums, theaters
    ===============================
    Prompt: The future of AI is
    Generated text:  looking bright. But for many people, it’s not clear what it is. What does it do? What does it mean?
    
    In this short article, we’ll look at the basics of AI and its potential to change the world.
    
    AI is the study of computer systems that learn, reason and make decisions by using algorithms.
    
    Here’s what you need to know about AI:
    
      • AI is a subset of computer science. It’s a tool that helps computers do work that would otherwise require a human.
      • AI is the creation of software that mimics the human brain. That’s why it’s also called artificial intelligence.
     


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for business, finance, and tourism. The city is home to many museums, theaters, and other cultural institutions. It is a popular tourist destination and a major economic hub in Europe. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is known for its annual Eiff
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This could lead to the creation of more efficient and cost-effective solutions, but it could also lead to job displacement for some workers.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we are likely to see even
    


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
    Generated text:  [Name] and I'm a [age] year old girl who grew up in [city/county]. I have a deep love for [activity/interest/subject] and have always been passionate about [job/occupation/role] because I've always wanted to do it. I'm always learning new things, and I'm always looking for ways to improve myself. I'm a strong workaholic who values integrity and honesty in everything I do. I'm a perfectionist and I'm always striving to do my best. I love being around people, and I'm always looking for new experiences and opportunities. I'm someone who
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and the center of its administrative, economic, cultural and political life. The city is known for its iconic landmarks such as Notre-Dame Cathedral, Louvre Museum, Eiffel Tower, and Champs-Élysées. Paris is also known for its cuisine, fashion, and its role as the home of the French Riviera. It is a major tourist destination, attracting millions of visitors each year. The city has a rich and diverse cultural heritage that includes art, music, and literature, and is also known for its artistic expression. The city's vibrant nightlife and its use of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and will likely involve a diverse range of emerging trends. Here are some potential future trends in AI:
    
    1. Integration of AI with other technologies: As more and more industries move towards automation and digitalization, AI will become more integrated with other technologies such as cloud computing, IoT, and machine learning. This will allow AI to collaborate with other systems to perform tasks more efficiently and effectively.
    
    2. Development of ethical AI: AI will continue to evolve and improve, but it will also need to be developed and used in a way that promotes ethical behavior. As the scope of AI continues to expand, ethical concerns will become more prominent, and


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

     am

     a

     [

    Occup

    ation

    ]

    !

     My

     journey

     started

     in

     the

     [

    previous

     place

    ]

     and

     my

     [

    current

     occupation

    ]

     is

     [

    current

     occupation

    ].

     I

     have

     always

     been

     [

    int

    angible

     trait

    ],

     but

     [

    what

     is

     your

     special

     skill

     or

     interest

     that

     makes

     you

     stand

     out

     from

     the

     crowd

    ?]

     [

    Include

     an

     interesting

     fact

     or

     a

     relevant

     fact

     that

     explains

     your

     unique

     nature

     or

     something

     you

     are

     proud

     of

    ].

     My

     goal

     is

     to

     [

    what

     motiv

    ates

     you

     to

     be

     a

     [

    Occup

    ation

    ]

     and

     achieve

     [

    specific

     goals

     or

     milestones

    ?

    ]

     ]

     I

     am

     always

     looking

     for

     opportunities

     to

     [

    what

     are

     you

     interested

     in

     or

     passionate

     about

    ?

     ]

     and

     I

     am

     always

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     landmarks

    ,

     vibrant

     culture

    ,

     and

     rich

     history

    .

     It

     was

     founded

     in

     

    8

    6

    8

     AD

     as

     the

     capital

     of

     the

     Carol

    ing

    ian

     Empire

     and

     continues

     to

     be

     the

     capital

     of

     France

    .

     Paris

     is

     home

     to

     museums

    ,

     art

     galleries

    ,

     and

     various

     cultural

     institutions

    ,

     and

     is

     a

     world

    -ren

    owned

     culinary

     destination

    .

     Its

     annual

     E

    iff

    el

     Tower

     exhibition

    ,

     held

     every

     March

    ,

     attracts

     millions

     of

     visitors

     from

     around

     the

     world

    .

     Paris

     is

     also

     known

     for

     its

     fashion

     scene

    ,

     including

     cout

    ure

     bout

    iques

    ,

     haute

     cout

    ure

    ,

     and

     Paris

     fashion

     week

    .

     The

     city

     is

     home

     to

     the

     Lou

    vre

     Museum

    ,

     the

     Arc

     de

     Tri

    omp

    he

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     vast

     and

     rapidly

     evolving

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     could

     shape

     the

     next

     

    1

    0

    -

    2

    0

     years

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     more

     individuals

     and

     organizations

     become

     aware

     of

     the

     negative

     impacts

     of

     AI

     on

     society

    ,

     there

     will

     be

     an

     increased

     focus

     on

     ethical

     AI

    .

     This

     could

     include

     regulations

     that

     govern

     AI

     development

    ,

     ensuring

     that

     AI

     is

     not

     used

     to

     harm

     individuals

     or

     the

     environment

    .
    


    2

    .

     AI

     will

     become

     more

     integrated

     into

     daily

     life

    :

     AI

     will

     continue

     to

     evolve

     and

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     from

     self

    -driving

     cars

     and

     drones

     to

     virtual

     assistants

     and

     chat

    bots

    .

     This

     could

     lead

     to

     a

     more

     seamless

     and

     efficient

    



```python
llm.shutdown()
```
