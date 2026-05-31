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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.13it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:48,  5.05s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:48,  5.05s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:48,  5.05s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:48,  5.05s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:48,  5.05s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:41,  1.29it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.86it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.20it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.20it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.20it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.20it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.20it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.20it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.20it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.20it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 12.26it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 12.26it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 12.26it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 12.26it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 12.26it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 12.26it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 12.26it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 12.26it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 12.26it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:05<00:02, 12.26it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 18.83it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 26.56it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 26.56it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 26.56it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 26.56it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 26.56it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 26.56it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 26.56it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 26.56it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 26.56it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 26.56it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 26.56it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 36.32it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 36.32it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 36.32it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:06,  7.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:06,  7.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:06,  7.88it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:06,  7.88it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.06it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.06it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.06it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.43it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.43it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.43it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.43it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.43it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.41it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.82it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.42it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.42it/s]

    Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.42it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.42it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.42it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 38.79it/s]

    Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  60%|██████    | 35/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.90it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.90it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.90it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.90it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.90it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 40.90it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.85it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.85it/s]

    Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.85it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.85it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.85it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.85it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.40it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.40it/s]

    Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.40it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.40it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 33.38it/s]


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
    Generated text:  Hannah and I am going to do a science fair project about recycling. I have some plastic bottles that I want to recycle. Can you give me some ideas for recycling them?
    
    Certainly! Recalling your project, I understand that you want to recycle plastic bottles. Here are a few ideas:
    
    1. **Collection at Home**: Start by taking the plastic bottles home and sorting them into different bins. Plastic bottles can be categorized into categories like recyclable, hazardous, and other.
    
    2. **Recycling Centers**: Consider visiting a local recycling center or taking your bottles directly to a recycling facility. Many communities offer recycling programs that include plastic bottles.
    
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy all the time, but he has only one hour to see a lot of people. The president has a long meeting at the White House on January 3rd, and the meeting starts at 12:00 PM. If the meeting lasts for 2 hours, how much time will be left in the hour for the president to see other people? To determine how much time will be left in the hour for the president to see other people, we need to follow these steps:
    
    1. Identify the start time of the meeting.
    2. Identify the end time of the meeting.
    3. Calculate the total duration of the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the seat of government, political and administrative authority, and cultural institutions. It is located in the centre of the region of France, on the Île de France, in the south of the country. It is the largest city in Europe by population, and the 10th largest city in the world. The capital is located on the left bank of the Seine river, facing the city of Paris in the west. The northern limit of the city is the Loire river, located between the Île de France and the city of Paris. The city is situated at 48°51′47
    ===============================
    Prompt: The future of AI is
    Generated text:  no longer in the office.
    Today, a recent study from the University of Pennsylvania showed that when trained with the right algorithms, neural networks are capable of generating art on their own. The study is the first to show that neural networks can produce such art without human intervention. In an interview, a professor of computer science, Albert-Ludwig Scherzer, explained that this is the future of AI and its potential to create breathtaking art. According to Scherzer, "the future of AI is about to come to life."
    This study is particularly significant as the vision of the AI generation is the creation of art. Scherzer believes


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a brief description of your character or personality]. What do you enjoy doing in your free time? I enjoy [insert a hobby or activity]. What's your favorite book or movie? I love [insert a favorite book or movie]. What's your dream job? I dream of [insert a dream job]. What's your favorite hobby? I love [insert a favorite hobby]. What's your favorite place to relax? I love [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of French Revolution and Romanticism, and its influence on modern French culture and politics. It is also home to many famous French artists, writers, and musicians. Paris is a popular tourist destination, with millions of visitors annually. The city is also known for its cuisine, with dishes such as croissants, escargot, and charcut
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy. AI developers will need to be more mindful of the potential consequences of their work and work to ensure that it is used in a responsible and ethical manner.
    
    2. Integration with human decision-making: AI is likely to become more integrated with human decision-making in the future.
    


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
    Generated text:  [Name], and I am a [Background] with a passion for [Favorite Subject/Activity]. I love [Interesting fact/Quote] about [Subject]. I have always been passionate about [Subject], and my love for it has driven me to [Action]. I am confident that I can achieve my goals, and I am ready to take on anything that comes my way. I am a [Attraction/Advantage/Comfort] person, and I am eager to make new friends and make memories with them. I am a [Positive/Attitude] person, and I love spending time with my friends, family, and my
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement encapsulates the factual information provided in the prompt, stating that Paris is the capital city of France. No additional context or elaboration was requested, as the given information is sufficient to fulfill the requested information. The statement is concise, accurate, and relevant to the prompt. 
    
    The statement "The capital of France is Paris" is a direct and clear answer to the question "What is the capital of France?" It provides a specific and verifiable fact that answers the given query. The statement is factual, accurate, and relevant to the prompt, meeting the criteria of clarity, accuracy, and relevance. 
    
    Therefore,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  predicted to be a highly dynamic and unpredictable one, shaped by many factors, including technological advancements, societal shifts, and ethical considerations. Here are some potential trends that could shape the future of AI:
    
    1. Increased AI integration with human communication: As AI becomes more integrated into our daily lives, we are likely to see a greater emphasis on how AI is used in human communication, particularly in areas such as customer service, healthcare, and education.
    
    2. AI becoming more dependent on data: With the increasing amount of data being generated and used by AI, it is likely that AI will become more dependent on data in the future. This will require


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

     [

    Skill

     Level

    ]

     player

     in

     a

     [

    Game

    /

    Character

    ]

     with

     [

    Number

     of

     Episodes

    ]

     episodes

     total

    .

     I

    'm

     originally

     from

     [

    Location

    ],

     but

     now

     I

     live

     in

     [

    New

     Location

    ].

     I

    'm

     [

    Age

    ]

     years

     old

    ,

     and

     I

    'm

     [

    Gender

    ]

     [

    Race

    ].

     I

    'm

     a

     [

    Character

     Trait

    ]

     player

    .

     I

     like

     to

     [

    Reason

     for

     Interest

    ]

     in

     [

    Technology

    /

    Anime

    /

    Video

     Game

    ].

     I

    'm

     also

     [

    H

    obby

    /

    Att

    itude

    ]

     in

     [

    Activity

    ].

     I

     like

     to

     [

    Express

     Passion

     for

    )

     and

     I

     enjoy

     [

    What

     I

     Enjoy

     Most

     About

     My

     Job

    ].

     I

    'm

     a

     [

    Positive

     Traits

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

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

     annual

     cultural

     festivals

    .

     
    


    To

     expand

     this

     statement

    ,

     please

     provide

     the

     meaning

     of

     "

    annual

     cultural

     festivals

    ".

     Include

     at

     least

     two

     additional

     details

     about

     Paris

    ian

     cuisine

    ,

     including

     a

     dish

     that

     is

     popular

     and

     unique

     to

     the

     city

    .

     Additionally

    ,

     provide

     the

     year

     of

     its

     founding

     and

     the

     name

     of

     its

     current

     president

    .

     The

     statement

     should

     also

     highlight

     the

     importance

     of

     Paris

     as

     a

     global

     city

     and

     its

     role

     in

     European

     culture

    .

     Finally

    ,

     please

     provide

     a

     brief

     analysis

     of

     Paris

    's

     current

     economic

     challenges

     and

     its

     potential

     for

     future

     growth

    .

     
    


    Note

    :

     While

     you

     should

     not

     include

     any

     copyrighted

     or

     protected

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     a

     highly

     dynamic

     and

     rapidly

     evolving

     field

     with

     endless

     possibilities

     for

     innovation

     and

     advancement

    .

     Here

     are

     some

     possible

     trends

     that

     we

     can

     expect

     to

     see

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     integration

     with

     human

     capabilities

    :

     One

     of

     the

     key

     trends

     we

     can

     expect

     to

     see

     is

     an

     increased

     integration

     of

     AI

     with

     human

     capabilities

    .

     AI

    -powered

     systems

     will

     be

     able

     to

     understand

     and

     adapt

     to

     human

     behavior

    ,

     emotions

    ,

     and

     preferences

    ,

     allowing

     for

     more

     natural

     and

     effective

     interactions

     between

     humans

     and

     machines

    .
    


    2

    .

     Enhanced

     transparency

     and

     accountability

    :

     AI

     systems

     will

     be

     required

     to

     be

     more

     transparent

     and

     accountable

    ,

     as

     they

     will

     be

     subject

     to

     greater

     scrutiny

     and

     oversight

    .

     This

     will

     lead

     to

     a

    



```python
llm.shutdown()
```
