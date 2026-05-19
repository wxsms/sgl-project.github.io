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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.51it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.01it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.52it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.51it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.51it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.51it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.51it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.51it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.51it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.51it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.51it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.51it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.51it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 28.00it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 28.00it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 28.00it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 28.00it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 28.00it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 28.00it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 28.00it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 28.00it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 28.00it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 35.21it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 35.21it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 35.21it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 35.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   2%|▏         | 1/58 [00:00<00:05,  9.84it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   2%|▏         | 1/58 [00:00<00:05,  9.84it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   2%|▏         | 1/58 [00:00<00:05,  9.84it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   5%|▌         | 3/58 [00:00<00:03, 13.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   5%|▌         | 3/58 [00:00<00:03, 13.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   5%|▌         | 3/58 [00:00<00:03, 13.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:03, 15.75it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:03, 15.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:03, 15.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:03, 15.75it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 24.44it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 24.44it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 24.44it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 24.44it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 24.44it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  21%|██        | 12/58 [00:00<00:01, 24.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.03it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.03it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.03it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.03it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.83it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.83it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.83it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.83it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.83it/s]Capturing num tokens (num_tokens=768 avail_mem=74.03 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.13it/s]Capturing num tokens (num_tokens=704 avail_mem=74.28 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.13it/s]Capturing num tokens (num_tokens=640 avail_mem=74.30 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.13it/s]Capturing num tokens (num_tokens=576 avail_mem=74.30 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.13it/s]Capturing num tokens (num_tokens=512 avail_mem=74.28 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.13it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.28 GB):  50%|█████     | 29/58 [00:01<00:01, 27.42it/s]Capturing num tokens (num_tokens=480 avail_mem=74.30 GB):  50%|█████     | 29/58 [00:01<00:01, 27.42it/s]Capturing num tokens (num_tokens=448 avail_mem=74.10 GB):  50%|█████     | 29/58 [00:01<00:01, 27.42it/s]Capturing num tokens (num_tokens=416 avail_mem=74.11 GB):  50%|█████     | 29/58 [00:01<00:01, 27.42it/s]Capturing num tokens (num_tokens=416 avail_mem=74.11 GB):  55%|█████▌    | 32/58 [00:01<00:00, 27.27it/s]Capturing num tokens (num_tokens=384 avail_mem=74.11 GB):  55%|█████▌    | 32/58 [00:01<00:00, 27.27it/s]Capturing num tokens (num_tokens=352 avail_mem=74.14 GB):  55%|█████▌    | 32/58 [00:01<00:00, 27.27it/s]Capturing num tokens (num_tokens=320 avail_mem=74.26 GB):  55%|█████▌    | 32/58 [00:01<00:00, 27.27it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.26 GB):  60%|██████    | 35/58 [00:01<00:00, 26.86it/s]Capturing num tokens (num_tokens=288 avail_mem=74.25 GB):  60%|██████    | 35/58 [00:01<00:00, 26.86it/s]Capturing num tokens (num_tokens=256 avail_mem=74.25 GB):  60%|██████    | 35/58 [00:01<00:00, 26.86it/s]Capturing num tokens (num_tokens=240 avail_mem=74.24 GB):  60%|██████    | 35/58 [00:01<00:00, 26.86it/s]Capturing num tokens (num_tokens=224 avail_mem=74.23 GB):  60%|██████    | 35/58 [00:01<00:00, 26.86it/s]Capturing num tokens (num_tokens=224 avail_mem=74.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.04it/s]Capturing num tokens (num_tokens=208 avail_mem=74.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.04it/s]Capturing num tokens (num_tokens=192 avail_mem=74.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.04it/s]Capturing num tokens (num_tokens=176 avail_mem=74.21 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.04it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.21 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.04it/s]Capturing num tokens (num_tokens=160 avail_mem=74.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.03it/s]Capturing num tokens (num_tokens=144 avail_mem=74.20 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.03it/s]Capturing num tokens (num_tokens=128 avail_mem=74.18 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.03it/s]Capturing num tokens (num_tokens=112 avail_mem=74.19 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.03it/s]Capturing num tokens (num_tokens=96 avail_mem=74.18 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.03it/s] Capturing num tokens (num_tokens=96 avail_mem=74.18 GB):  81%|████████  | 47/58 [00:01<00:00, 32.36it/s]Capturing num tokens (num_tokens=80 avail_mem=74.18 GB):  81%|████████  | 47/58 [00:01<00:00, 32.36it/s]Capturing num tokens (num_tokens=64 avail_mem=74.17 GB):  81%|████████  | 47/58 [00:01<00:00, 32.36it/s]Capturing num tokens (num_tokens=48 avail_mem=74.16 GB):  81%|████████  | 47/58 [00:01<00:00, 32.36it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.16 GB):  81%|████████  | 47/58 [00:01<00:00, 32.36it/s]Capturing num tokens (num_tokens=28 avail_mem=74.15 GB):  81%|████████  | 47/58 [00:01<00:00, 32.36it/s]Capturing num tokens (num_tokens=28 avail_mem=74.15 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=24 avail_mem=74.13 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=20 avail_mem=74.15 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=16 avail_mem=74.14 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=12 avail_mem=74.14 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=8 avail_mem=74.13 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.96it/s] Capturing num tokens (num_tokens=8 avail_mem=74.13 GB):  98%|█████████▊| 57/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=4 avail_mem=74.12 GB):  98%|█████████▊| 57/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=4 avail_mem=74.12 GB): 100%|██████████| 58/58 [00:01<00:00, 29.44it/s]


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
    Generated text:  John Smith. I'm 35 years old and I've been working in the field of science and technology for a decade. I have a passion for exploring the universe and have a background in physics and astronomy. I'm very interested in learning about the universe and the cosmos, and I'm eager to learn more about it. I'm also interested in taking on new challenges and working on innovative projects that challenge me to be the best I can be. What are some of the most interesting and groundbreaking projects that you are currently working on, and what is your vision for the future of space exploration? Additionally, I am interested in exploring the
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking advice on the most effective approach to managing the relationship between the federal government and the state governments. One possible approach is to divide the responsibility for education among the state governments, similar to how it is done in many countries. This would require the federal government to provide funding for state education programs, and it would require the state governments to provide their own education programs for their citizens. However, some may argue that this approach is too complicated and may lead to inefficiency and duplication of efforts.
    Another approach could be to provide a unified education system across the country, where all states would have access to the same curriculum and teachers. This would eliminate
    ===============================
    Prompt: The capital of France is
    Generated text:  a beautiful city that has a strong French culture. In this city, you can find many different neighborhoods with their own unique landmarks. The city is known for its beautiful gardens, parks, and museums. In this task, you need to identify the capital city of France based on the information provided. The given answer should also be in French.
    Paris
    Le Parisien
    Libération
    Libération magazine
    L'Essor
    L'Essor magazine
    Le Temps
    Le Temps magazine
    L'Avenir
    L'Avenir magazine
    La Croix
    La Croix magazine
    Le Temps
    Le Temps magazine
    
    ===============================
    Prompt: The future of AI is
    Generated text:  largely defined by the speed at which it evolves and progresses. In the past decade, the field has seen rapid advancements in areas such as natural language processing, computer vision, and robotics. While some experts have predicted that these areas will continue to advance at an exponential rate, others are skeptical of the speed of progress. However, it is clear that AI is transforming the way we live and work, and the future of AI will be closely tied to the pace of technological advancement.
    One of the key areas that is set to continue evolving is AI in healthcare. With the increasing use of AI in healthcare, we can expect to see improvements in the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a popular tourist destination and a major economic center in France. It is also known for its fashion industry, art, and cuisine. The city is home to many international organizations and is a major hub for business and trade. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a unique
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems.
    
    2. Enhanced ethical considerations: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical implications. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Greater reliance on AI for decision-making: AI is likely to become more integrated with human decision-making, allowing machines to make more informed and accurate decisions. This could lead to more
    


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
    Generated text:  Alex. I work in marketing for a marketing agency. I'm always looking for creative solutions to problem.
    Hello, my name is Alex. I work in marketing for a marketing agency. I'm always looking for creative solutions to problem. I don't get lost. I never take shortcuts. I'm a hard worker. I have a great sense of humor. I don't get stressed about meeting deadlines. I love to eat ice cream.
    Apologies for any spelling or grammar mistakes, but I didn't receive a copy of the original story or character information. Let me know if you'd like me to change anything for the sake of authenticity
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Paris is a historic city and the largest city in France. It is also the second-most populous city in the European Union, after London. It is the seat of the French government, the head of state, and the seat of the French Parliament. Paris is also a major financial center. The city features numerous landmarks, including the Eiffel Tower, the Louvre Museum, and Notre Dame Cathedral. The French Riviera is a major tourism destination in the southern part of France. Paris is known for its romantic and picturesque architecture, including its Gothic architecture, the Arc de Triomphe, and the Notre Dame Cathedral.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and there are several key trends that are expected to shape the technology and applications of AI in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with other technologies: AI is likely to become even more integrated with other technologies such as IoT, blockchain, and the Internet of Things (IoT). This integration could lead to more efficient and effective use of these technologies, as well as increased predictability and reliability in AI-based systems.
    
    2. Improved privacy and security: As AI systems become more integrated with other technologies, there is a greater risk of privacy and security breaches. There is a growing emphasis


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

    insert

     name

    ],

     and

     I

    'm

     a

     [

    insert

     profession

     or

     background

    ]

     who

     enjoys

     [

    insert

     what

     they

     enjoy

     doing

     that

     they

    're

     good

     at

    ],

     and

     I

    'm

     a

     [

    insert

     your

     profession

     or

     hobby

    ]

     who

     I

    'm

     really

     good

     at

     [

    insert

     skill

     or

     hobby

    ].

     I

     love

     [

    insert

     hobbies

     or

     activities

     that

     make

     me

     happy

    ],

     and

     I

    'm

     really

     [

    insert

     your

     favorite

     thing

     about

     yourself

    ],

     and

     I

     love

     [

    insert

     any

     activities

     or

     interests

     that

     you

     find

     fulfilling

     and

     interesting

    ].

     I

     strive

     to

     be

     the

     best

     I

     can

     be

    ,

     and

     I

    'm

     always

     eager

     to

     learn

     new

     things

     and

     try

     new

     things

    .

     What

     kind

     of

     character

     are

     you

    ?

     [

    insert

     how

     you

     got

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     “

    La

     Petite

     Vie

    ”

     which

     means

     "

    Little

     Life

    ".


    Paris

     is

     a

     bustling

     met

    ropolis

     of

     

    2

    .

    5

     million

     people

    ,

     famous

     for

     its

     medieval

     architecture

    ,

     renowned

     museums

    ,

     and

     world

    -ren

    owned

     restaurants

    .

     The

     city

     is

     also

     home

     to

     important

     historical

     sites

     and

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

    .

     The

     city

     is

     known

     for

     its

     vibrant

     culture

    ,

     including

     music

    ,

     theater

    ,

     and

     dance

    ,

     as

     well

     as

     its

     annual

     E

    iff

    el

     Tower

     Festival

    .

     With

     its

     rich

     history

     and

     stunning

     views

    ,

     Paris

     is

     a

     city

     that

     is

     both

     old

     and

     new

    ,

     with

     a

     unique

     blend

     of

     cultures

     and

     traditions

    .

     The

     city

     is

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

    ,

     and

     there

     are

     many

     exciting

     developments

     and

     trends

     shaping

     the

     technology

    's

     direction

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     autonomy

    :

     With

     the

     development

     of

     machine

     learning

     and

     natural

     language

     processing

    ,

     AI

     systems

     will

     become

     more

     capable

     of

     performing

     tasks

     that

     were

     previously

     thought

     to

     be

     impossible

    .

     This

     includes

     tasks

     such

     as

     autonomous

     vehicles

    ,

     self

    -driving

     planes

    ,

     and

     even

     human

    -like

     conversations

     with

     AI

     assistants

    .
    


    2

    .

     Improved

     accuracy

    :

     AI

     systems

     will

     continue

     to

     improve

     their

     ability

     to

     understand

     and

     interpret

     complex

     data

    ,

     making

     them

     more

     accurate

     and

     reliable

    .

     This

     will

     lead

     to

     a

     wide

     range

     of

     applications

     in

     fields

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    .
    


    3

    



```python
llm.shutdown()
```
