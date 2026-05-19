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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.57it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 11.98it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 11.98it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 11.98it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 11.98it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:02, 11.98it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:02, 11.98it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:02, 11.98it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:01, 16.14it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:01, 16.14it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:01, 16.14it/s]

    Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:01, 16.14it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:01, 16.14it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 16.14it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 16.14it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:05<00:01, 16.14it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 21.96it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 21.96it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 27.65it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 27.65it/s]

    Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 27.65it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 27.65it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 27.65it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 27.65it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 27.65it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 31.20it/s] 

    Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 31.20it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.03it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.00 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.94 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.94 GB):   3%|▎         | 2/58 [00:00<00:04, 13.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.95 GB):   3%|▎         | 2/58 [00:00<00:04, 13.26it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.95 GB):   3%|▎         | 2/58 [00:00<00:04, 13.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.95 GB):   7%|▋         | 4/58 [00:00<00:03, 15.03it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.94 GB):   7%|▋         | 4/58 [00:00<00:03, 15.03it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.93 GB):   7%|▋         | 4/58 [00:00<00:03, 15.03it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.93 GB):  10%|█         | 6/58 [00:00<00:03, 16.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.92 GB):  10%|█         | 6/58 [00:00<00:03, 16.53it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=71.90 GB):  10%|█         | 6/58 [00:00<00:03, 16.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.92 GB):  10%|█         | 6/58 [00:00<00:03, 16.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.92 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.91 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.90 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.90 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.38it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.89 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.38it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=71.89 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.94it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.87 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.86 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.87 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.87 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.87 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.86 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.85 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.85 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.83 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.72it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=71.83 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.00it/s]Capturing num tokens (num_tokens=960 avail_mem=71.84 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.00it/s] Capturing num tokens (num_tokens=896 avail_mem=71.81 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.00it/s]Capturing num tokens (num_tokens=832 avail_mem=71.51 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.00it/s]Capturing num tokens (num_tokens=768 avail_mem=71.79 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.00it/s]Capturing num tokens (num_tokens=768 avail_mem=71.79 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.12it/s]Capturing num tokens (num_tokens=704 avail_mem=71.78 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.12it/s]

    Capturing num tokens (num_tokens=640 avail_mem=71.77 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.12it/s]Capturing num tokens (num_tokens=576 avail_mem=71.55 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.12it/s]Capturing num tokens (num_tokens=576 avail_mem=71.55 GB):  48%|████▊     | 28/58 [00:01<00:01, 25.40it/s]Capturing num tokens (num_tokens=512 avail_mem=71.54 GB):  48%|████▊     | 28/58 [00:01<00:01, 25.40it/s]Capturing num tokens (num_tokens=480 avail_mem=71.75 GB):  48%|████▊     | 28/58 [00:01<00:01, 25.40it/s]Capturing num tokens (num_tokens=448 avail_mem=71.74 GB):  48%|████▊     | 28/58 [00:01<00:01, 25.40it/s]

    Capturing num tokens (num_tokens=448 avail_mem=71.74 GB):  53%|█████▎    | 31/58 [00:01<00:01, 24.96it/s]Capturing num tokens (num_tokens=416 avail_mem=71.73 GB):  53%|█████▎    | 31/58 [00:01<00:01, 24.96it/s]Capturing num tokens (num_tokens=384 avail_mem=71.72 GB):  53%|█████▎    | 31/58 [00:01<00:01, 24.96it/s]Capturing num tokens (num_tokens=352 avail_mem=71.71 GB):  53%|█████▎    | 31/58 [00:01<00:01, 24.96it/s]Capturing num tokens (num_tokens=352 avail_mem=71.71 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.15it/s]Capturing num tokens (num_tokens=320 avail_mem=71.70 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.15it/s]Capturing num tokens (num_tokens=288 avail_mem=71.69 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.15it/s]Capturing num tokens (num_tokens=256 avail_mem=71.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.15it/s]

    Capturing num tokens (num_tokens=256 avail_mem=71.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.10it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.10it/s]Capturing num tokens (num_tokens=224 avail_mem=71.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.10it/s]Capturing num tokens (num_tokens=208 avail_mem=71.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.10it/s]Capturing num tokens (num_tokens=192 avail_mem=71.65 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.10it/s]Capturing num tokens (num_tokens=192 avail_mem=71.65 GB):  71%|███████   | 41/58 [00:01<00:00, 28.47it/s]Capturing num tokens (num_tokens=176 avail_mem=71.64 GB):  71%|███████   | 41/58 [00:01<00:00, 28.47it/s]Capturing num tokens (num_tokens=160 avail_mem=71.64 GB):  71%|███████   | 41/58 [00:01<00:00, 28.47it/s]Capturing num tokens (num_tokens=144 avail_mem=71.63 GB):  71%|███████   | 41/58 [00:01<00:00, 28.47it/s]Capturing num tokens (num_tokens=128 avail_mem=71.61 GB):  71%|███████   | 41/58 [00:01<00:00, 28.47it/s]

    Capturing num tokens (num_tokens=128 avail_mem=71.61 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.29it/s]Capturing num tokens (num_tokens=112 avail_mem=71.62 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.29it/s]Capturing num tokens (num_tokens=96 avail_mem=71.62 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.29it/s] Capturing num tokens (num_tokens=80 avail_mem=71.59 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.29it/s]Capturing num tokens (num_tokens=64 avail_mem=71.58 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.29it/s]Capturing num tokens (num_tokens=64 avail_mem=71.58 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=48 avail_mem=71.58 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=32 avail_mem=71.57 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=28 avail_mem=71.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=24 avail_mem=71.57 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.23it/s]

    Capturing num tokens (num_tokens=24 avail_mem=71.57 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.80it/s]Capturing num tokens (num_tokens=20 avail_mem=71.56 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.80it/s]Capturing num tokens (num_tokens=16 avail_mem=71.55 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.80it/s]Capturing num tokens (num_tokens=12 avail_mem=71.57 GB):  91%|█████████▏| 53/58 [00:01<00:00, 34.80it/s]Capturing num tokens (num_tokens=8 avail_mem=71.56 GB):  91%|█████████▏| 53/58 [00:02<00:00, 34.80it/s] Capturing num tokens (num_tokens=8 avail_mem=71.56 GB):  98%|█████████▊| 57/58 [00:02<00:00, 35.85it/s]Capturing num tokens (num_tokens=4 avail_mem=71.55 GB):  98%|█████████▊| 57/58 [00:02<00:00, 35.85it/s]Capturing num tokens (num_tokens=4 avail_mem=71.55 GB): 100%|██████████| 58/58 [00:02<00:00, 28.00it/s]


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
    Generated text:  Liz and I’m a freelance English teacher in the UK. I have been working in the UK since 2013 and now my school is fully online. I currently teach English as a second language at the school and I am also the lead teacher for the course of the year. I love to travel, play sports and meet new people. Can you tell me more about your teaching style? I would like to know more about your teaching methods. **Q&A session**
    Liz: Hi, thank you for taking the time to talk to me. I am a teacher who has been teaching English for 13 years now,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she is like the boss of the country. This person is usually very kind and he or she always does the right thing. But sometimes they might make some bad decisions.  But many of the presidents who are good at being kind and always do the right thing have a special name. They are called "Teddy Bears Presidents".  What is the main idea of the passage? The main idea of the passage is to tell us that the presidents who are good at being kind and always do the right thing are called "Teddy Bears Presidents". The answer is "tobeforbestandalwaysdo
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The city was founded in the 6th century by King Clovis I. In the 13th century, it was developed by many kings, including Charles I and Henry II. This is the first capital in Europe which is not in the countryside, the first city with a wall and towers. This city is the first European capital to have a castles and towers.
    In 1790, the city was seized by the French Revolution. The new government wanted to name it the capital, but King Louis XV said no. At that time, the French capital was not in the countryside, the French capital was
    ===============================
    Prompt: The future of AI is
    Generated text:  transforming all areas of life, but in the field of education, it is more intricately linked to the future of the human brain.
    By Ashley Fillo, December 16, 2017
    The future of AI is changing every area of life. With every year of progress in technology, a new field emerges – artificial intelligence, or AI. AI, or artificial intelligence, is the technology that enables computers to imitate human intelligence. It includes the development of machine learning algorithms and the creation of artificial intelligence systems that can learn from data, process and understand it, and then generate new outputs. There are many potential uses


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] and I'm always looking for ways to [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is home to many notable French artists, writers, and musicians, and is known for its rich history and cultural heritage. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage, and is a major center of European politics and diplomacy. The city is also known for its fashion industry, with many famous designers
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and preferences.
    
    2. Enhanced machine learning capabilities: AI is likely to become more capable of learning from large amounts of data and making more accurate predictions and decisions. This could lead to more efficient and effective AI systems that can handle a wider range of tasks and scenarios.
    
    3. Greater emphasis on ethical considerations: As AI becomes more integrated
    


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
    Generated text:  [Name], I am 35 years old. I am an entrepreneur, multi-talented, and I thrive on innovation. My passion is to connect people with resources, ideas, and opportunities. I am passionate about sharing knowledge and learning new skills to help others achieve their goals. My goal is to create a better world through my work. What would be your first or best advice for someone who wants to achieve this? To me, it is essential to learn from my mistakes and setbacks, and not to give up on my dreams. I will never be satisfied with my current state of affairs, and I will always strive to improve
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city where the Eiffel Tower, the Louvre Museum, and other iconic landmarks sit tall. It is a cosmopolitan and historic city, home to numerous museums, theaters, and cultural institutions, and a global center of finance and commerce. Visitors to Paris can explore its rich history, vibrant culture, and bustling streets, making it a must-visit destination for both locals and tourists. Paris has a reputation for being a vibrant and exciting city, with its French culture and cuisine being a reflection of its status as a major European city. As of 2021, Paris had an estimated population of approximately 2
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving and there are many possibilities and possibilities ahead for how this technology will continue to change. Some possible future trends in AI include:
    
    1. Increased automation and robotics: The future of AI will see even more automation and automation in industries like manufacturing, transportation, and healthcare. Robots and AI-powered systems will become more integrated into our daily lives, increasing efficiency and reducing the need for human intervention.
    
    2. Enhanced human-machine interaction: AI will continue to improve our ability to communicate and collaborate with machines. This could lead to more efficient and effective ways of doing business, as well as new forms of entertainment and social interaction.
    
    3. Enhanced decision


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

     __

    ________

    .

     I

    'm

     an

     experienced

     __

    ________

    __

     with

     a

     strong

     passion

     for

     __

    ________

    __.

     As

     a

     leader

     in

     the

     __

    ________

    __

     field

    ,

     I

     bring

     a

     wide

     range

     of

     skills

     and

     a

     unique

     approach

     to

     problem

    -solving

    .

     I

     am

     a

     committed

     __

    ________

    _

     to

     __

    ________

    __.

     I

     thrive

     on

     __

    ________

    _,

     and

     I

     am

     always

     looking

     for

     ways

     to

     __

    ________

    _.

     I

     am

     excited

     to

     be

     a

     part

     of

     your

     team

     and

     look

     forward

     to

     working

     with

     you

     to

     achieve

     your

     goals

    .

     [

    Write

     out

     what

     else

     you

     want

     to

     mention

    ]

     [

    Specify

     the

     profession

     or

     field

     you

    're

     in

    ,

     the

     specific

     skills

     or

     qualities

     you

     bring

     to

     the

     table

    ,

     or

     your

     passion

     or

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     cosm

    opolitan

     city

     with

     many

     historical

     landmarks

    ,

     such

     as

     the

     Lou

    vre

     Museum

     and

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     known

     for

     its

     vibrant

     cultural

     scene

    ,

     including

     the

     annual

     E

    iff

    el

     Tower

     Festival

    .

     Paris

     is

     a

     city

     of

     contrasts

    ,

     with

     its

     modern

     skys

    crap

    ers

     and

     historic

     districts

    ,

     making

     it

     a

     UNESCO

     World

     Heritage

     site

    .

     It

     is

     a

     popular

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

     Overall

    ,

     Paris

     is

     a

     city

     that

     reflects

     the

     rich

     history

    ,

     culture

    ,

     and

     art

    istry

     of

     France

    .

     Its

     annual

     E

    iff

    el

     Tower

     Festival

     is

     a

     major

     event

     that

     brings

     in

     visitors

     from

     around

     the

     world

    .

     The

     French

     language

     is

     widely

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

    ,

     and

     it

     is

     likely

     that

     it

     will

     continue

     to

     evolve

     in

     ways

     that

     are

     both

     transformative

     and

     challenging

    .

     Here

     are

     some

     potential

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     AI

     will

     become

     more

     diverse

     and

     inclusive

    :

     With

     the

     increasing

     number

     of

     individuals

     with

     disabilities

    ,

     AI

     is

     likely

     to

     become

     even

     more

     accessible

     to

     those

     who

     need

     it

    .

     This

     could

     mean

     increased

     training

     and

     development

     for

     algorithms

     to

     better

     accommodate

     the

     diverse

     needs

     of

     people

     with

     disabilities

    .
    


    2

    .

     AI

     will

     be

     more

     autonomous

    :

     As

     autonomous

     vehicles

     become

     more

     advanced

    ,

     they

     may

     become

     more

     autonomous

    ,

     freeing

     up

     humans

     to

     focus

     on

     more

     complex

     tasks

    .

     This

     could

     also

     mean

     that

     AI

    -powered

     robots

     and

     other

     autonomous

     systems

     will

    



```python
llm.shutdown()
```
