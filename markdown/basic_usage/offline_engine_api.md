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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.63it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.63it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:03,  1.16s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:03,  1.16s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:03,  1.16s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.74it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.74it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.74it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:18,  2.74it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.78it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.78it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.78it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.78it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.23it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.23it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  7.23it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  7.23it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  7.23it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.08it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.08it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.08it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.08it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 13.55it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:02, 13.55it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 26.31it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 26.31it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 26.31it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 26.31it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 26.31it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 26.31it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 26.31it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:01, 26.31it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:01, 26.31it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:01, 26.31it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:05<00:01, 26.31it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 40.37it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 40.37it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 40.37it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 40.37it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 40.37it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 40.37it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 40.37it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 40.37it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 40.37it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 40.37it/s]

    Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 40.37it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 52.96it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 52.96it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 52.96it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 52.96it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 52.96it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 52.96it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 52.96it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 52.96it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 52.96it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 52.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.52it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.52it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.54it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.54it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.54it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.54it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.03it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 44.23it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.59it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.59it/s]Capturing num tokens (num_tokens=288 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.59it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.59it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.59it/s]Capturing num tokens (num_tokens=224 avail_mem=76.66 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.59it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.59it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=160 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.95it/s]

    Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.95it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.99it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.99it/s]Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.99it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.99it/s]Capturing num tokens (num_tokens=64 avail_mem=76.63 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.99it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.99it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.40it/s]

    Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.40it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.68it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.68it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.68it/s] Capturing num tokens (num_tokens=4 avail_mem=76.60 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.68it/s]Capturing num tokens (num_tokens=4 avail_mem=76.60 GB): 100%|██████████| 58/58 [00:01<00:00, 42.20it/s]


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
    Generated text:  George, I’m a computer programmer. I specialize in web development, but I’m also well-versed in mobile app development, frontend development, and back-end development.
    I’m a self-taught programmer with a passion for creating amazing and secure software. My coding style is influenced by a mixture of modern programming paradigms and a love for originality. I’m passionate about learning and teaching, and I enjoy mentoring and supporting my students.
    My skills include JavaScript, CSS, HTML, and back-end technologies such as Node.js, Python, and Ruby. I also have extensive experience in working with frameworks like Angular and React, and
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting a third country. This country uses a different currency than the United States. The president arrives in the country and begins to count the number of people, but something goes wrong. He counts 400 people and then realizes that each person in front of him has 3 bills instead of 2. If he now counts the total number of people again, how many people are in the country? To determine the number of people in the country, we need to follow these steps:
    
    1. **Identify the number of people counted initially:**
       - The president counts 400 people.
    
    2. **Determine the
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Nice
    C. London
    D. Rome
    Answer:
    
    A
    
    Which of the following statements about the cell membrane is true?
    A. The cell membrane is the boundary between the cell and its internal environment.
    B. The cell membrane is a double-layered structure composed of phospholipid molecules and proteins.
    C. All the transport substances within cells require the assistance of the cell membrane.
    D. The function of the cell membrane is to regulate the entry and exit of substances.
    Answer:
    
    D
    
    What is the primary goal of the investment banking industry?
    A. To raise funds and finance corporate
    ===============================
    Prompt: The future of AI is
    Generated text:  very interesting. With the big changes occurring in the world today, the future is already here. When I first started studying AI, the biggest fear I had was the possibility of it not being used for good. I saw it as a threat to the economy and to privacy. Today, the future of AI is looking more promising. However, it is also not without some problems. For example, it can be a very dangerous tool for people who misuse it. The world has seen a lot of incidents of hacking and identity theft. Also, the global war on terror is such a devastating thing that AI could become a great tool for terrorism.
    


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


    Generated text:  [Name] and I am a [occupation] with [number of years] years of experience in [field]. I am a [character trait] and [character's name] is my [character trait]. I am passionate about [reason for passion], and I am always looking for ways to [action or goal]. I am [character's personality] and [character's name] is my [character's personality]. I am [character's personality] and [character's name] is my [character's personality]. I am [character's personality] and [character's name] is my [character's personality]. I am [character
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Middle Ages, and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major cultural and economic center, with a diverse population and a thriving arts scene. The city is home to many famous museums, including the Louvre and the Musée d'Orsay, as well as the iconic Eiffel Tower. Paris is a popular tourist destination and is known for its romantic atmosphere and vibrant nightlife. The city is also home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability. AI developers will need to be more mindful of the potential impact of their work on society and the environment.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As
    


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
    Generated text:  [insert character's name] and I am a [insert relevant profession or age] who recently graduated from [insert university or college]. I have been working as a [insert relevant job title] for [insert number of years] and I have always been fascinated by the world of [insert relevant topic or interest]. What inspired you to pursue a career in this field? Additionally, I am excited to share my hopes and dreams for the future of this industry. Lastly, what's something you are particularly proud of? [insert something specific to share, e.g. a winning lottery ticket, achieving a personal goal, etc.]. Lastly,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. It is also home to the French Parliament, the Courthouse, and the Louvre Museum, among other important institutions. France's economy is also thriving, with a reputation for being one of the world's most cost-effective places to live and work. The city is also home to a diverse and culturally rich population, known for its art, cuisine, and music. As the capital, Paris plays a key role in France's political, economic, and cultural life. It is also a UNESCO World Heritage site and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast, with new developments and trends expected to shape its development in the years to come. Some possible trends include:
    
    1. Increased use of AI in healthcare: As the demand for personalized medicine and disease treatment grows, AI will play a more significant role in healthcare. With AI, medical professionals can analyze large amounts of patient data and develop more accurate diagnoses and treatment plans.
    
    2. AI in education: With the increasing focus on education, AI is set to play an even greater role in learning. AI-powered educational tools can provide personalized learning experiences, adaptive learning algorithms, and more, to help students learn in a more effective and efficient way.
    
    


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

    ],

     and

     I

    'm

     a

     [

    职业

    ]

     with

     [

    previous

     experience

    ].

     I

     have

     a

     passion

     for

     [

    职业

    ]

     and

     enjoy

     helping

     others

     achieve

     their

     goals

    .

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     learn

     and

     grow

    .

     What

     do

     you

     do

     for

     a

     living

    ?

     What

     are

     some

     of

     the

     things

     that

     make

     you

     an

     exceptional

     leader

    ?

     I

    'm

     always

     looking

     for

     new

     experiences

     and

     opportunities

     to

     learn

     and

     grow

    .

     How

     do

     you

     stay

     motivated

    ?

     I

     believe

     that

     the

     key

     to

     staying

     motivated

     is

     to

     set

     realistic

     goals

     and

     celebrate

     small

     wins

     along

     the

     way

    .

     What

     are

     your

     strengths

     and

     what

     do

     you

     enjoy

     doing

    ?

     I

     love

     working

     on

     projects

     and

     taking

     on

     new

     challenges

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     known

     as

     the

     "

    City

     of

     Love

    ,"

     is

     the

     second

    -most

     populous

     city

     in

     the

     European

     Union

     and

     the

     largest

     city

     in

     France

    .

     It

     is

     the

     seat

     of

     the

     French

     government

    ,

     the

     country

    's

     capital

    ,

     and

     a

     major

     commercial

     and

     cultural

     center

    ,

     with

     a

     rich

     historical

     and

     artistic

     heritage

    .

     Paris

     is

     famous

     for

     its

     romantic

     architecture

    ,

     museums

    ,

     and

     world

    -ren

    owned

     art

     scene

    .

     The

     city

     also

     hosts

     numerous

     festivals

     and

     events

     throughout

     the

     year

    .

     Paris

     is

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

     and

     is

     home

     to

     many

     notable

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Palace

     of

     Vers

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     undoubtedly

     one

     of

     rapid

     and

     exciting

     growth

    .

     Here

     are

     some

     potential

     trends

     in

     AI

     that

     could

     shape

     the

     industry

     in

     the

     years

     to

     come

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     With

     the

     rise

     of

     ethical

     considerations

     around

     AI

    ,

     we

     may

     see

     more

     focus

     on

     AI

     ethics

    ,

     privacy

    ,

     and

     accountability

    .

     This

     could

     lead

     to

     the

     development

     of

     more

     robust

     AI

     systems

     that

     are

     designed

     to

     be

     transparent

     and

     accountable

    .
    


    2

    .

     Personal

    ized

     AI

    :

     As

     we

     get

     more

     data

     on

     individuals

    ,

     we

     may

     see

     a

     shift

     towards

     more

     personalized

     AI

     systems

    .

     AI

     systems

     could

     learn

     from

     each

     individual

    's

     data

     and

     provide

     tailored

     recommendations

     or

     responses

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     could

     revolution

    ize

    



```python
llm.shutdown()
```
