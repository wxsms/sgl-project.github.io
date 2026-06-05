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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.96it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.76it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.76it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.76it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.76it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.76it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.76it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.76it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.76it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.76it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.76it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.45it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.45it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.45it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.22it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.22it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 29.25it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 29.25it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 29.25it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 29.25it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 29.25it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 29.25it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 29.25it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 29.25it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 29.25it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 29.25it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 29.25it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 29.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 40.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 18.79it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.66 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.80it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.80it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 27.11it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 27.11it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 27.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 27.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 27.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 27.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.27it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.27it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.27it/s] Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.65it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.77it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.77it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.77it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.77it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.77it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.77it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=208 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.75it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=128 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.49it/s] Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 44.46it/s]Capturing num tokens (num_tokens=80 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 44.46it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 44.46it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 44.46it/s]Capturing num tokens (num_tokens=32 avail_mem=74.55 GB):  81%|████████  | 47/58 [00:01<00:00, 44.46it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 44.46it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.52it/s]

    Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.52it/s] Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.43it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 40.43it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 37.71it/s]


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
    Generated text:  Riley and I am a 17 year old girl. I am at a critical time in my life and need some advice. I am very sick, it feels like I am going to die. My parents are very upset about the way I look and they want me to stop going to school. They think I am going to die and that I should not be allowed to go to school. I am not sure how I will deal with this situation and I am scared. I want to tell my parents and have them help me but I don't want to upset them. I have very strong feelings and I feel that I should talk to
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. To which category does this sentence belong? ____
    A. Time, Place
    B. Cause, Effect
    C. Definition
    D. Fact, Principle
    Answer:
    C
    
    The evaluation and selection of primary schools and secondary schools in our country are conducted through the following methods: ① recommendation by provincial, municipal, and county education administrative departments; ② solicitation of opinions from relevant personnel of the education administrative department and relevant schools; ③ the evaluation of the education administrative departments and education administrative departments at all levels; ④ application and assessment by the education administrative department. Which of the following sequences
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and its area is 230,874 hectares. It is divided into two departments: the Centre-Val de Loire and the Aude. It has 70.177 inhabitants, of which 54.497 live in the Centre-Val de Loire and the remainder in the Aude department.
    To find the area of the Aude department, we can start by noting that the total area of France is 230,874 hectares. The Centre-Val de Loire is 70.177 hectares and the Aude department is 
    ===============================
    Prompt: The future of AI is
    Generated text:  about to change significantly, and that’s due to the advancements of quantum computing. The technology is evolving, and it’s becoming increasingly important in the world of AI.
    One of the most exciting developments in quantum computing is the ability to perform certain calculations much faster than traditional computers. This is due to the phenomenon of superposition, which allows a quantum particle to exist in two places at the same time.
    This means that quantum computers can perform calculations at a much faster rate than traditional computers. This is because quantum computers can use a superposition of states to perform calculations simultaneously, which is unlike anything that can be done with classical computers.
    The potential


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, interesting fact about yourself]. And what's your favorite hobby? I love [insert a short, interesting fact about your hobby]. And what's your favorite movie? I love [insert a short, interesting fact about your favorite movie]. And what's your favorite book? I love [insert a short, interesting fact about your favorite book]. And what's your favorite food? I love [insert a short, interesting fact about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for fashion, art, and music. Paris is a popular tourist destination and a major economic hub for France. The city is home to many international organizations and institutions, including the French Academy of Sciences and the European Parliament. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends include:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, we may see even more sophisticated applications in this field.
    
    2. AI in manufacturing: AI is already being used in manufacturing to optimize production processes and reduce costs. As AI technology continues to improve, we may see even more widespread adoption in this area.
    
    3. AI in finance: AI is already being used in finance to improve risk management and fraud detection
    


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
    Generated text:  [Your Name], and I'm a [Type of person] who has been learning programming for [number of years]. I recently completed a degree in [subject] and have been working as a [job title] for [length of time in years]. I enjoy [thing I enjoy doing], and I'm always looking to expand my skills and knowledge. What excites you the most about programming? What's your long-term goal? And what are your hobbies and interests outside of work? Let's chat! [Your Name]. (Maybe include your favorite programming language, some programming projects, or any other interesting facts about you!) I’m
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historical and cultural center with a rich history, including a famous Gare du Nord railway station. The city is known for its iconic landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. Paris is also famous for its cuisine, including the famous Parisian ham fish soup known as "Boeuf de Vitrine." French cuisine is also known for its light and refreshing dishes, which are often served during the summer months when it is particularly hot outside. The city also has a vibrant nightlife scene, with various clubs and bars in the city center, as well as many more on
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of potential, exciting developments that promise to revolutionize the world in ways we've never imagined. Here are some possible trends in AI:
    
    1. Deep Learning: Deep learning is a subfield of AI that focuses on building models that can learn and improve on tasks like image and speech recognition. As more data and algorithms are processed, the accuracy of deep learning models is expected to improve, making them increasingly capable of performing complex tasks.
    
    2. Machine Learning: Machine learning is a subset of AI that involves developing algorithms that can learn from data without being explicitly programmed. It has been used to automate tasks such as voice assistants, self-driving cars


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

    Occup

    ation

    ]

     with

     [

    Number

     of

     Years

     in

     the

     Industry

    ].

     I

    've

     always

     been

     passionate

     about

     [

    Reason

     for

     Passion

    ],

     and

     my

     career

     has

     helped

     me

     achieve

     my

     [

    Achie

    vement

     or

     Goal

    ].

     Whether

     it

    's

     [

    Business

     Strategy

    ],

     [

    Marketing

     Tactics

    ],

     [

    Customer

     Service

    ],

     or

     [

    E

    co

    -F

    riendly

     Practices

    ],

     I

    'm

     confident

     in

     my

     ability

     to

     bring

     my

     unique

     skills

     to

     any

     task

    .

     I

    'm

     always

     eager

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     always

     looking

     for

     new

     opportunities

     to

     bring

     my

     expertise

     to

     new

     challenges

    .

     My

     name

     is

     [

    Name

    ],

     and

     I

    'm

     a

     [

    Occup

    ation

    ]

     with

     [

    Number

     of

     Years

     in

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     culture

    .

     It

     is

     also

     one

     of

     the

     world

    's

     most

     popular

     tourist

     destinations

     and

     hosts

     numerous

     attractions

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

    's

     reputation

     as

     a

     cultural

     center

     and

     entertainment

     hub

     is

     also

     well

    -d

    ocumented

    .

     It

     is

     the

     cultural

     and

     economic

     center

     of

     France

    ,

     hosting

     major

     events

     such

     as

     the

     Summer

     Olympics

    .

     The

     French

     capital

     is

     often

     described

     as

     the

     "

    City

     of

     Light

    "

     and

     the

     "

    C

    rad

    le

     of

     Civilization

    ."

     Despite

     its

     size

     and

     population

    ,

     Paris

     is

     a

     thriving

     city

     that

     continues

     to

     evolve

     and

     attract

     new

     visitors

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     filled

     with

     exciting

     possibilities

     and

     potential

     challenges

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Autonomous

     vehicles

    :

     Self

    -driving

     cars

     and

     trucks

     are

     already

     on

     the

     road

    ,

     and

     it

     is

     expected

     that

     more

     autonomous

     vehicles

     will

     be

     developed

     in

     the

     future

    .

     This

     could

     lead

     to

     a

     decrease

     in

     accidents

     and

     an

     increase

     in

     efficiency

     in

     transportation

    .
    


    2

    .

     Artificial

     intelligence

     for

     healthcare

    :

     AI

     is

     being

     used

     to

     develop

     new

     treatments

     and

     diagnoses

     for

     a

     wide

     range

     of

     diseases

    ,

     from

     cancer

     to

     Alzheimer

    's

    .

     This

     could

     lead

     to

     more

     accurate

     and

     effective

     treatments

     and

     c

    ures

    .
    


    3

    .

     Personal

    ized

     medicine

    :

     With

     the

     help

     of

     AI

    ,

     doctors

     can

     analyze

     a

     patient

    



```python
llm.shutdown()
```
