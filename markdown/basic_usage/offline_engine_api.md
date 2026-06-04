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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.87it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:17,  5.57s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:17,  5.57s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:17,  5.57s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:17,  5.57s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:17,  5.57s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:45,  1.17it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:45,  1.17it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:45,  1.17it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:45,  1.17it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:45,  1.17it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:45,  1.17it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:45,  1.17it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:45,  1.17it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:13,  3.53it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  7.51it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  7.51it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  7.51it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  7.51it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  7.51it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  7.51it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  7.51it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  7.51it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  7.51it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:06<00:02, 11.99it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:06<00:02, 11.99it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:06<00:02, 11.99it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:06<00:02, 11.99it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:06<00:02, 11.99it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:06<00:02, 11.99it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:06<00:02, 11.99it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:06<00:02, 11.99it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:06<00:02, 11.99it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:01, 17.39it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:01, 17.39it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:01, 17.39it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:01, 17.39it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:06<00:01, 17.39it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:06<00:01, 17.39it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:06<00:01, 17.39it/s]

    Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:06<00:01, 17.39it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:06<00:01, 17.39it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:06<00:01, 17.39it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:06<00:00, 24.62it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:06<00:00, 24.62it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 32.79it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 32.79it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 32.79it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 32.79it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:06,  8.16it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:06,  8.16it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:06,  8.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:06,  8.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:03, 14.13it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 23.60it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 23.60it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 23.60it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 23.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 23.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 23.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.34it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.24it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 32.87it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 32.87it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.87it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.87it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.87it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 33.72it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 36.40it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.59it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.59it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.59it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.59it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.59it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.39it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.39it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.39it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.39it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.39it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.96it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.96it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.96it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 31.78it/s]


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
    Generated text:  Alice and I'm an educator, marketing director, and educator from the United States. I currently teach at the University of San Francisco and I also teach at the University of Missouri. My research focuses on educational innovation, particularly in the area of family learning and home education. The common thread in my work is a love for learning. My love for learning can be found throughout my life. My parents instilled in me the desire to know more about the world and to find answers to the questions that life has to offer. When I was younger, I thought that learning was about memorizing facts and figures. I also knew that I wanted to
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to improve the quality of life for the people in his state. He has identified a number of potential areas for improvement, including reducing traffic congestion and increasing access to public transportation. He has also suggested that he work with other states to create a new state highway system that would improve the connectivity of the state.
    
    He has received several reports from citizens on his efforts, which are positive in nature, but he has also received many negative reports. He has been criticized for not doing enough to address traffic congestion and for his actions in the states to work with other states.
    
    Write an editorial in the style of the New York Times that defends the president
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris  
    B. Brussels  
    C. Nantes  
    D. Seine  
    E. Rome
    
    To determine the capital of France, let's recall the official capital cities of France. The official capital cities of France are:
    
    A. Paris
    B. Brussels
    C. Nantes
    D. Seine
    E. Rome
    
    Based on this information, the correct answer is:
    
    A. Paris
    
    Therefore, the capital of France is **Paris**. 
    
    To double-check, I will use a simple Python script to verify this information.
    
    ```python
    # List of official capital cities of France
    capital_cities =
    ===============================
    Prompt: The future of AI is
    Generated text:  about more than software and data. The technologies used to generate, analyze, and understand large amounts of information will be essential for how AI is used in the future.
    In order to drive innovation and improve the impact of AI, we need to start thinking outside the box, not just in the realm of software and data. In today’s world, there are a lot of different ways that AI can be applied to different areas of work. It is important to think about all of the different technologies that are available, and to consider how they can be used to generate new ideas and improve existing ones.
    In addition to software and data, there are


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Square. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. It is a popular tourist destination and a major economic center in Europe. The city is known for its cuisine, fashion, and art, and is home to many famous museums and galleries. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of people, culture, and history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive systems, as well as more efficient and effective ways of interacting with humans.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for the development and use of
    


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
    Generated text:  [Name], and I'm a [job title] at [Company]. I've been working for [duration] years now, and I've always had a strong passion for [related area of interest]. What brings you to [Company] today?
    [Company] has been my [job title] for [duration], and I'm excited to join the team at [Company]. I'm confident that my skills and experiences will be invaluable to the company, and I'm excited to contribute to a dynamic and innovative team. What do you think, [Person], is it? [Company] has a great culture, and I'm honored to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Its official language is French and it is the largest city in the European Union. The city is known for its landmarks such as the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. It is also known for its rich history and culture. In addition, Paris is also one of the most touristy cities in the world and is a major hub for international business and diplomacy. It is also home to the Statue of Liberty, the Louvre, and the Champ de Mars. Overall, Paris is a city of contrasts and beauty that has been a global cultural and political center for centuries. The Eiffel Tower
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities, and it's difficult to predict exactly where it will take us, but here are some of the key trends that we can anticipate:
    
    1. Personalization: With the help of machine learning algorithms, AI will be able to learn more about individuals and tailor their experiences to their needs. This will allow for more personalized interactions and experiences, from product recommendations to healthcare appointments.
    
    2. Autonomous vehicles: AI is already playing an important role in the development of self-driving cars, which will help to reduce traffic accidents and improve the overall safety of the public. In the future, we may see more advanced self-driving cars that can


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

     an

     [

    insert

     age

     range

    ]

     year

     old

     male

    .

     I

     live

     in

     [

    insert

     current

     location

    ].

     I

     enjoy

     [

    insert

     one

     or

     two

     hobbies

    ,

     interests

    ,

     or

     passions

    ].

     I

    'm

     [

    insert

     your

     age

    ],

     and

     I

     have

     an

     [

    insert

     occupation

    ]

     job

    .

     I

     like

     to

     [

    insert

     what

     they

     like

     to

     do

     on

     a

     regular

     basis

    ].

     I

     have

     always

     been

     [

    insert

     one

     or

     two

     qualities

     or

     traits

     that

     define

     you

    ].

     I

    'm

     [

    insert

     your

     most

     significant

     accomplishment

    ].

     I

     enjoy

     [

    insert one

     or

     two

     things

     that

     bring

     joy

     to

     your

     life

    ].


    As

     you

     can

     see

    ,

     I

    'm

     a

     man

     who

     is

     passionate

     about

     work

     and

     life

    ,

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     medieval

     architecture

    ,

     and

     lively

     cultural

     scene

    .

     It

     is

     home

     to

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    ,

     as

     well

     as

     a

     diverse

     population

     of

     about

     

    2

    .

    7

     million

     people

    .

     Paris

     is

     also

     the

     financial

     hub

     of

     the

     country

     and

     is

     a

     popular

     tourist

     destination

     for

     tourists

     from

     all

     over

     the

     world

    .

     It

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     romantic

     atmosphere

     and

     romantic

     architecture

    .

     The

     city

     has

     a

     strong

     sense

     of

     French

     culture

     and

     is

     known

     for

     its

     traditional

     French

     cuisine

    ,

     including

     b

    oul

    anger

    ies

    ,

     which

     serve

     a

     wide

     variety

     of

     past

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     wide

     range

     of

     trends

     and

     developments

    ,

     including

    :
    


    1

    .

     Increased

     automation

    :

     AI

     is

     expected

     to

     automate

     a

     growing

     number

     of

     tasks

    ,

     from

     manufacturing

     to

     customer

     service

     to

     healthcare

    .

     This

     could

     lead

     to

     job

     losses

     in

     certain

     industries

    ,

     but

     also

     create

     new

     opportunities

     for

     those

     who

     are

     skilled

     in

     AI

    -related

     skills

    .
    


    2

    .

     Enhanced

     personal

    ization

    :

     AI

     will

     enable

     more

     personalized

     experiences

    ,

     from

     recommending

     products

     and

     services

     to

     providing

     recommendations

     on

     social

     media

    .

     This

     could

     lead

     to

     increased

     convenience

     and

     satisfaction

     for

     users

    .
    
    3

    .

     Improved

     ethics

     and

     transparency

    :

     As

     AI

     becomes

     more

     advanced

     and

     complex

    ,

     it

     will

     become

     increasingly

     important

     to

     ensure

     that

     AI

     systems

     are

     transparent

     and

     ethical

    .

    



```python
llm.shutdown()
```
