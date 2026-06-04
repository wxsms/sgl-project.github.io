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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.40it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.08it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.08it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.08it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.08it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.08it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.08it/s]

    Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.08it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.08it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.08it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 12.79it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 12.79it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 12.79it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 12.79it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 12.79it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 12.79it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 12.79it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 12.79it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 12.79it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 12.79it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:01, 19.36it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:01, 19.36it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:01, 19.36it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:01, 19.36it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:01, 19.36it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:01, 19.36it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:01, 19.36it/s]

    Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:01, 19.36it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:01, 19.36it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:01, 19.36it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 27.03it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 36.83it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 36.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 17.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 17.63it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 17.63it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.21it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 14.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):  10%|█         | 6/58 [00:00<00:03, 15.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  10%|█         | 6/58 [00:00<00:03, 15.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  10%|█         | 6/58 [00:00<00:03, 15.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  10%|█         | 6/58 [00:00<00:03, 15.74it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.15it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 32.26it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 32.26it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 32.26it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 32.26it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 32.26it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 32.26it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.11it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.11it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.11it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.11it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.11it/s]

    Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.11it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.37it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.37it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.37it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.37it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.37it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.85it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.78it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.78it/s]

    Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.78it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.78it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.78it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.78it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.75it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.75it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.75it/s]

    Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.36it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.36it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.36it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.36it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.36it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.36it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.89it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.89it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.89it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.89it/s]

    Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 33.24it/s]


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
    Generated text:  Michael Green. I am an artist who makes illustrations and paintings for children and adults alike. I have a background in graphic design and have had work exhibited in many galleries and museums throughout the United States. I have a degree in art from the University of the Arts, Shanghai. I was born in Hong Kong and now live in New York. My work tends to be soft and flowing, with a gentle touch that matches the stories of my subjects. I am a very gentle person who loves to share my paintings with others. How did you get started in the art world and what kind of work do you do?
    
    I started my art career as
    ===============================
    Prompt: The president of the United States is
    Generated text:  3/4 the age of the president of Peru. The president of Peru is 2/3 the age of the president of France. If the president of France will be 2400 years old in 100 years, how old is the president of France now? To determine the current age of the president of France, we need to follow a step-by-step approach using the given information.
    
    1. **Determine the current age of the president of Peru:**
       - We know that the president of Peru is \(\frac{2}{3}\) the age of the president of France.
       -
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Brussels
    C. Lyon
    D. London
    
    The capital of France is Paris. 
    
    Therefore, the correct answer is:
    
    A. Paris
    
    To expand on this:
    
    Paris, the capital of France, is a city located in the south of France, on the French Riviera. It is the most populous city in France, with an estimated population of around 2.5 million people as of 2023. Paris is famous for its architecture, art, and culture, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Moulin Rouge. The
    ===============================
    Prompt: The future of AI is
    Generated text:  about to unfold in the making, with new advancements and breakthroughs on the horizon. AI is being used to analyze and understand the world around us, from the patterns in weather to the patterns in our own emotions. It is also being used to create new forms of technology and services that are revolutionizing industries and communities. AI is also being used to solve complex problems in healthcare, where it has the potential to save lives and improve patient outcomes. As we continue to advance and develop AI, it is important to keep in mind that it is not a panacea, but rather a tool that can be used to solve complex problems and create new


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I do for you today? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I do for you today? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I do for you today? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a cultural and economic center, hosting numerous museums, theaters, and restaurants. Paris is a popular tourist destination, known for its rich history, art, and cuisine. It is also a major hub for international business and diplomacy. The city is home to many famous landmarks and attractions, including the Louvre, the Arc de Triomphe, and the Champs-Élysées. Paris is a vibrant and dynamic city, with a rich cultural and artistic heritage. It is a major center of politics
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI systems, there will be a greater emphasis on ethical considerations. This will likely lead to the development of more transparent and accountable AI systems that are designed to minimize bias and ensure fairness.
    
    2. AI will become more integrated into everyday life: As AI becomes more integrated into our daily lives, we may see more widespread adoption of AI technologies. This could lead to a more personalized
    


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
    Generated text:  [Name], and I’m a [job title] with [company name]. I’m here because [reason for being], [accomplishment]. I have [number] years of experience in [industry/field], and I love [thing]. What’s your name? What’s your job? What’s your company? What are your achievements? What’s your experience? What do you do? What’s your background? And what do you do now? I’m glad to meet you. That’s all for my introduction. Have a nice day! 😊
    
    How did you get into this industry? What inspired you to do this
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A. 885 years old
    B. 515 years old
    C. 265 years old
    D. 150 years old
    
    C. 265 years old. 
    
    The city of Paris has a fascinating history that dates back to the 8th century and has evolved into one of the world's most significant cities. The city's name, "Paris," comes from the Latin word "parvis," meaning "part of the door," referring to the ancient wooden doorways that led to the city's large port. The city's architecture, including the iconic E
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by continued innovation, growth, and integration into various industries. Here are some potential future trends in artificial intelligence:
    
    1. Increased automation: With the rise of automation in industries such as manufacturing and transportation, AI is likely to become even more prevalent in the future. Robots and automation will become more advanced, and AI systems will be able to perform tasks that were previously done by humans.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve diagnosis, treatment, and patient outcomes. The future of AI in healthcare is likely to focus on developing more precise and personalized treatments, as well as developing AI-powered tools


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

     a

    /an

     [

    insert

     profession

     or

     role

    ]

     who

     has

     been

     working

     in

     [

    insert

     field

     or

     industry

    ]

     for

     [

    insert

     number

     of

     years

    ]

     years

    .

     I

    've

     [

    insert

     some

     relevant

     experience

     or

     accomplishments

    ]

     and

     have

     been

     [

    insert

     any

     personal

     qualities

     or

     skills

     that

     make

     you

     stand

     out

     to

     me

    ].

     I

    'm

     always

     here

     for

     help

    ,

     and

     if

     you

     have

     any

     questions

     or

     need

     guidance

    ,

     feel

     free

     to

     ask

     me

     anything

    .

     Thank

     you

    !

     

    😊

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    ✨

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     Î

    le

     de

     la

     C

    ité

     in

     the

     Se

    ine

     River

    ,

     and

     has

     a

     population

     of

     over

     

    2

    .

     

    5

     million

     people

    .

     It

     is

     known

     for

     its

     rich

     history

     and

     culture

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

     Dame

     Cathedral

    .

     The

     city

     is

     also

     known

     for

     its

     food

    ,

     fashion

    ,

     and

     arts

     scene

    .

     France

    ’s

     capital

     is

     Paris

    ,

     an

     historic

     city

     with

     a

     rich

     history

     and

     culture

    ,

     known

     for

     its

     rich

     history

     and

     culture

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

     Dame

     Cathedral

    .

     Additionally

    ,

     Paris

     is

     known

     for

     its

     food

    ,

     fashion

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     evolve

     and

     develop

     in

     many

     exciting

     ways

    .

     Here

     are

     some

     of

     the

     potential

     trends

     that

     are

     predicted

     to

     shape

     the

     way

     we

     interact

     with

     technology

     and

     develop

     new

     applications

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     everyday

     life

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     may

     see

     more

     automation

     and

     personal

     assistants

     that

     are

     designed

     to

     interact

     with

     us

     in

     ways

     that

     are

     increasingly

     natural

     and

     intuitive

    .

     For

     example

    ,

     smart

     home

     devices

     that

     can

     sense

     and

     respond

     to

     our

     emotions

    ,

     or

     virtual

     assistants

     that

     can

     provide

     entertainment

     and

     information

     in

     real

    -time

    .
    


    2

    .

     AI

    -powered

     medical

     advancements

    :

     AI

     is

     already

     being

     used

     to

     analyze

     medical

     images

    ,

     diagnose

     diseases

    ,

     and

    



```python
llm.shutdown()
```
