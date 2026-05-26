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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.47it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.46it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.62it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.62it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.62it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.62it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.62it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.62it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.62it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.62it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.62it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 20.28it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 20.28it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 20.28it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 20.28it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:01, 20.28it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 20.28it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 28.61it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 28.61it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 38.72it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 38.72it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 38.72it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 38.72it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.05 GB):   3%|▎         | 2/58 [00:00<00:03, 15.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.05 GB):   3%|▎         | 2/58 [00:00<00:03, 15.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.05 GB):   3%|▎         | 2/58 [00:00<00:03, 15.06it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.05 GB):   3%|▎         | 2/58 [00:00<00:03, 15.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.05 GB):   9%|▊         | 5/58 [00:00<00:02, 19.51it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.04 GB):   9%|▊         | 5/58 [00:00<00:02, 19.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.03 GB):   9%|▊         | 5/58 [00:00<00:02, 19.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.03 GB):   9%|▊         | 5/58 [00:00<00:02, 19.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.03 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.02 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.47it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.02 GB):  21%|██        | 12/58 [00:00<00:01, 28.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.01 GB):  21%|██        | 12/58 [00:00<00:01, 28.57it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.01 GB):  21%|██        | 12/58 [00:00<00:01, 28.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.01 GB):  21%|██        | 12/58 [00:00<00:01, 28.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.00 GB):  21%|██        | 12/58 [00:00<00:01, 28.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.00 GB):  21%|██        | 12/58 [00:00<00:01, 28.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.00 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.99 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.99 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.96it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=57.97 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.96it/s]Capturing num tokens (num_tokens=960 avail_mem=57.99 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.96it/s] Capturing num tokens (num_tokens=960 avail_mem=57.99 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.01it/s]Capturing num tokens (num_tokens=896 avail_mem=57.98 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.01it/s]Capturing num tokens (num_tokens=832 avail_mem=57.98 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.01it/s]Capturing num tokens (num_tokens=768 avail_mem=57.98 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.01it/s]Capturing num tokens (num_tokens=704 avail_mem=57.98 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.01it/s]Capturing num tokens (num_tokens=640 avail_mem=57.97 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.01it/s]Capturing num tokens (num_tokens=640 avail_mem=57.97 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.45it/s]Capturing num tokens (num_tokens=576 avail_mem=57.97 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.45it/s]Capturing num tokens (num_tokens=512 avail_mem=57.96 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.45it/s]

    Capturing num tokens (num_tokens=480 avail_mem=57.97 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.45it/s]Capturing num tokens (num_tokens=448 avail_mem=57.97 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.45it/s]Capturing num tokens (num_tokens=416 avail_mem=57.97 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.45it/s]Capturing num tokens (num_tokens=416 avail_mem=57.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.59it/s]Capturing num tokens (num_tokens=384 avail_mem=57.97 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.59it/s]Capturing num tokens (num_tokens=352 avail_mem=57.96 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.59it/s]Capturing num tokens (num_tokens=320 avail_mem=57.95 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.59it/s]Capturing num tokens (num_tokens=288 avail_mem=57.95 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.59it/s]Capturing num tokens (num_tokens=256 avail_mem=57.95 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.59it/s]Capturing num tokens (num_tokens=256 avail_mem=57.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.32it/s]Capturing num tokens (num_tokens=240 avail_mem=57.95 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.32it/s]Capturing num tokens (num_tokens=224 avail_mem=57.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.32it/s]

    Capturing num tokens (num_tokens=208 avail_mem=57.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.32it/s]Capturing num tokens (num_tokens=192 avail_mem=57.94 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.32it/s]Capturing num tokens (num_tokens=176 avail_mem=56.57 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.32it/s]Capturing num tokens (num_tokens=176 avail_mem=56.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=160 avail_mem=56.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=144 avail_mem=56.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.63it/s]

    Capturing num tokens (num_tokens=128 avail_mem=56.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=112 avail_mem=56.56 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=112 avail_mem=56.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 30.21it/s]Capturing num tokens (num_tokens=96 avail_mem=56.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 30.21it/s] Capturing num tokens (num_tokens=80 avail_mem=56.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 30.21it/s]Capturing num tokens (num_tokens=64 avail_mem=56.55 GB):  79%|███████▉  | 46/58 [00:01<00:00, 30.21it/s]Capturing num tokens (num_tokens=48 avail_mem=56.55 GB):  79%|███████▉  | 46/58 [00:01<00:00, 30.21it/s]

    Capturing num tokens (num_tokens=48 avail_mem=56.55 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.21it/s]Capturing num tokens (num_tokens=32 avail_mem=56.55 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.21it/s]Capturing num tokens (num_tokens=28 avail_mem=56.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.21it/s]Capturing num tokens (num_tokens=24 avail_mem=56.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.21it/s]Capturing num tokens (num_tokens=20 avail_mem=56.53 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.21it/s]Capturing num tokens (num_tokens=20 avail_mem=56.53 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.10it/s]Capturing num tokens (num_tokens=16 avail_mem=56.53 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.10it/s]Capturing num tokens (num_tokens=12 avail_mem=56.53 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.10it/s]Capturing num tokens (num_tokens=8 avail_mem=56.52 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.10it/s] Capturing num tokens (num_tokens=4 avail_mem=56.52 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.10it/s]Capturing num tokens (num_tokens=4 avail_mem=56.52 GB): 100%|██████████| 58/58 [00:01<00:00, 33.54it/s]


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
    Generated text:  Nita. I am a college student at San Jose State University. I am a biology major. I love plants because they are very fascinating and unique. I also love animals because they are amazing. One of my favorite animals is a cat. I have collected a lot of information about cats, but I am still a bit confused about the different types of cats. Can you help me? Let's start by asking you some questions about cats.
    1. What is the scientific name for a cat?
    2. What is the scientific name for dogs?
    3. What is the scientific name for the cat that has stripes?
    4. What is
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to issue a travel ban on a notorious criminal who has been convicted of several crimes, including murder and drug trafficking. The president also has to consider the costs and benefits of issuing the ban. The president has been told that a travel ban on the criminal would cost $100 million and prevent 1000 people from leaving the country. The cost of educating the public about the criminal's crimes and the potential harm they may cause would cost $50 million. The president must also consider the political ramifications of issuing the ban, which could lead to negative consequences such as increased military spending and a loss of President
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Brussels
    C. London
    D. Rome
    
    The capital of France is Paris. Paris is the largest city in France and is known for its historical and cultural significance. It is located on the North Bank of the Seine River and is home to many famous landmarks such as the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The capital of France is chosen to be the largest city within the country. 
    
    Let me know if you need any clarification or have additional questions in mind. 
    
    I apologize, but there are some inconsistencies in your question. The capital of France
    ===============================
    Prompt: The future of AI is
    Generated text:  here
    
    NVIDIA’s Autopilot and Driverless Car
    
    Engineer engineer Jared Fogle
    
    With Elon Musk’s latest Tesla Model S, the world’s first fully autonomous vehicle, the CEO of the company shares his thoughts on the future of AI and the transition to a fully autonomous world.
    
    The idea of a fully autonomous car is not new. It has been around since the 1970s and was even featured in the movie “The Terminator” back in 1984. We’ve seen it in movies like “The Matrix” and “The Terminator,” and now it’s finally here. In fact, it


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


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is also the birthplace of the French Revolution and the birthplace of the French language. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. The city is known for its art, architecture, and cuisine, and is a popular tourist destination. It is also home to many famous landmarks and museums. Paris is a city of contrasts, with its modern skyscrapers and historic neighborhoods, and its vibrant nightlife. The city is a major hub for business and commerce, and is known for its fashion
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to diagnose and treat diseases, and it has the potential to revolutionize the field. AI-powered diagnostic tools, such as AI-powered X-rays and AI-powered pathology software, are already being used to improve patient outcomes.
    
    2. AI in manufacturing: AI is already being used in manufacturing to optimize production processes, reduce costs, and improve quality. AI-powered robots and AI-powered predictive analytics are being used to optimize production
    


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
    Generated text:  [Name] and I am a [Role]. My [Role] is [job or position]. I am a [type] and have [number] years of experience in [field of work]. I currently reside in [Location]. My [Type] is [Your Type]. I am a [yourself] and I am passionate about [what you do for a living]. I believe that I am [something] due to my [what you do for a living]. I am always [what you do for a living] and my [what you do for a living] is always [what you do for a living]. I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. It is also known as the "City of Lights" due to its numerous landmarks, including Notre-Dame Cathedral and the Louvre Museum. Paris is a hub for global culture, fashion, and cuisine, and has been a center of politics, philosophy, science, and literature for centuries. It is also a world-renowned art center, with iconic landmarks such as the Eiffel Tower and the Notre-Dame Cathedral. The city is home to several famous landmarks, including the Champs-Élysées, the Louvre Museum,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and promising, and there are several trends that could shape the field in the coming years. Here are some possible future trends in AI:
    
    1. Increased Human Interaction with AI: As AI becomes more capable, there may be a growing demand for interactions between humans and machines that involve more complex decision-making and problem-solving.
    
    2. Better Understanding of Human Behavior: AI algorithms can learn from large amounts of data and become more sophisticated over time, allowing them to better understand human behavior and emotions. This could lead to more accurate and empathetic AI, which could improve the way we interact with one another.
    
    3. Personalized and Adaptive AI:


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

    job

     title

    ].

     I

     have

     been

     working

     in

     this

     field

     for

     [

    number

     of

     years

    ]

     years

    .

     I

     have

     [

    number

     of

     years

    ]

     years

     of

     experience

    .

     I

     have

     a

     great

     passion

     for

     [

    job

     title

    ],

     and

     I

     am

     a

     dedicated

     and

     dedicated

     employee

    .

     I

     am

     always

     looking

     for

     ways

     to

     improve

     my

     skills

     and

     stay

     up

    -to

    -date

     with

     the

     latest

     trends

     in

     the

     industry

    .

     I

     am

     a

     hard

    working

     and

     reliable

     worker

    ,

     and

     I

     pride

     myself

     on

     being

     a

     good

     team

     player

    .

     I

     am

     always

     willing

     to

     go

     above

     and

     beyond

     to

     meet

     the

     needs

     of

     my

     customers

     and

     the

     company

    .

     I

     am

     confident

     in

     my

     abilities

     and

     I

     am

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Love

    .

     It

     is

     the

     most

     populous

     city

     in

     Europe

    ,

     with

     an

     estimated

     population

     of

     around

     

    1

    0

     million

     people

    .

     Paris

     is

     known

     for

     its

     history

    ,

     art

    ,

     architecture

    ,

     and

     cuisine

    ,

     and

     is

     considered

     the

     heart

     of

     France

     and

     the

     world

    's

     third

    -largest

     city

     by

     population

    .

     The

     city

     is

     home

     to

     many

     of

     the

     world

    's

     most

     famous

     landmarks

    ,

     including

     the

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

     Mont

    mart

    re

    ,

     and

     has

     been

     a

     center

     of

     French

     culture

     and

     politics

     for

     over

     a

     millennium

    .

     Paris

     is

     also

     known

     for

     its

     diverse

     population

    ,

     with

     many

     different

     ethnic

    ities

     and

     religions

     living

     there

    ,

     including

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     rapid

     development

     and

     widespread

     adoption

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     integration

     with

     human

     consciousness

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     they

     may

     begin

     to

     integrate

     more

     deeply

     with

     our

     minds

     and

     consciousness

    .

     This

     could

     lead

     to

     a

     greater

     understanding

     of

     how

     we

     perceive

     and

     experience

     the

     world

     around

     us

    .
    


    2

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

     complex

     and

     powerful

    ,

     there

     will

     be

     increasing

     focus

     on

     addressing

     ethical

     concerns

    .

     This

     could

     include

     issues

     such

     as

     bias

    ,

     transparency

    ,

     and

     accountability

    .
    


    3

    .

     Increased

     focus

     on

     natural

     language

     processing

    :

     With

     the

     rise

     of

     AI

    ,

     there

     is

     likely

     to

     be

     increased

    



```python
llm.shutdown()
```
