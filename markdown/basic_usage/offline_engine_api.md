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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:03,  9.70it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 16.19it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 16.19it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 16.19it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 16.19it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 16.19it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 16.19it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 16.19it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 16.19it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 21.72it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 21.72it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 21.72it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 21.72it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:01, 21.72it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:01, 21.72it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:01, 21.72it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:01, 21.72it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:01, 21.72it/s]

    Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:04<00:01, 21.72it/s]Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:04<00:01, 21.72it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 31.62it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:04<00:00, 31.62it/s]Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:04<00:00, 31.62it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:04<00:00, 43.48it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:04<00:00, 43.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.00it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.00it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.23it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.23it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 36.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 36.52it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 36.52it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 36.52it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.48it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.48it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.48it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.48it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.48it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.48it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.59it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.59it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.59it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.59it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.59it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.59it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 44.29it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 44.29it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.29it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.33it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.33it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.72it/s]

    Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.76it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.76it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 46.57it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 40.82it/s]


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
    Generated text:  Lorianne. I'm a three-year-old girl, and I'm a big fan of all things science. I love to play and discover things. What interests me the most is learning about the universe, galaxies, stars and planets. I'm also great at solving puzzles and making models.
    My favorite hobby is science experiments. I really enjoy the discovery process and the excitement of seeing what happens when I turn materials and substances in a container into something new. I'm also great at using my imagination to create wonderful worlds and worlds beyond our own.
    I have a love for dinosaurs. I love learning about dinosaurs, and I like to explore
    ===============================
    Prompt: The president of the United States is
    Generated text:  a citizen of which country?
    
    The president of the United States is a citizen of the United States.
    
    To elaborate, the United States, which is a democratic republic, is the only nation that actually has an elected president. As such, the president is elected by the citizens of the United States, just like the other elected officials in the government. The president, including the current president, is a representative of the American people, representing them in the legislative branch of government. As such, the president is directly elected by the American people and is a citizen of the United States. The other officials in the government are elected by the people, not
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Rome
    D. Moscow
    E. Tokyo
    
    A. Paris is the capital of France. It is a significant cultural, historical, and economic center in the country. 
    
    Now, let's eliminate the other options:
    
    B. London - While it is the capital of the United Kingdom, it is not the capital of France.
    
    C. Rome - Rome is the capital of Italy, not France.
    
    D. Moscow - Moscow is the capital of Russia, not France.
    
    E. Tokyo - Tokyo is the capital of Japan, not France.
    
    Therefore, the correct answer is:
    A.
    ===============================
    Prompt: The future of AI is
    Generated text:  quickly approaching. The technology has been improved and is becoming more accessible to everyone. It is also more nuanced than ever. AI now uses a greater variety of data and in more varied ways. The technology is more sophisticated. It is faster and more efficient. There are more AI applications and AI projects than ever before. The technology has been explored in more areas and more deeply into the real world. But while this technology is rapidly advancing, it is also open to misuse. As a result, there is a lot of debate about how to control and regulate AI.
    One area of debate is how to regulate AI. There is a growing consensus that


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I have a [number] degree in [field of study]. I'm a [job title] at [company name]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills. I'm always eager to learn and grow, and I'm always willing to take on new challenges. I'm a [job title] at
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is the largest city in France by population. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for art, literature, and music, and is home to many famous museums, theaters, and restaurants. The city is known for its vibrant nightlife and is a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant culture. The city is also home to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve diagnosis, treatment, and patient care. As AI becomes more advanced, it is likely to be used in even more complex and personalized ways.
    
    3. Increased use of AI in manufacturing: AI is already being used in manufacturing to improve efficiency, reduce costs, and increase productivity. As AI becomes more advanced, it is likely to be used
    


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
    Generated text:  [Name], and I am [Age]. I am a [occupation/education] who loves [pastimes/activities] and [goal or dream]. My strength lies in [strength/ability/qualification], and I enjoy [anything I like]. I am [mainly] focused on [how I am currently spending my day]. I am [maybe] looking to [where I am going to go] in the near future. Can you tell me more about yourself? I would love to meet you! Can you tell me more about yourself? Hello, my name is [Name] and I am [Age]. I am a [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the third-largest city in the European Union by population. Paris is known for its historical significance, art and architecture, and vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. Paris is also a significant center for politics, education, and fashion. The city is a global hub for international trade and a cultural center for many countries.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and uncertain, with many potential trends and developments that could shape the future of technology. Here are some possible future trends in artificial intelligence:
    
    1. Autonomous vehicles: Autonomous vehicles are likely to become a major part of daily life in the future, driven by advances in AI, such as cameras and sensors, machine learning algorithms, and neural networks.
    
    2. Medical AI: AI is already being used in medical diagnosis and treatment, but there are many other applications in the future that could revolutionize healthcare. AI could help doctors analyze patient data more efficiently and accurately, identify potential health risks before they develop, and develop personalized treatment plans.
    
    3.


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

    current

     role

     or

     status

    ]

     at

     [

    Company

    ].

     I

    'm

     excited

     to

     meet

     everyone

     and

     learn

     more

     about

     your

     work

     and

     life

    !

     Here

    's

     what

     I

     can

     tell

     you

    :

     [

    list

     your

     relevant

     skills

    ,

     experiences

    ,

     and

     accomplishments

    ].

     Let

    's

     see

     if

     we

     can

     build

     something

     great

     together

    !

     Can

     you

     tell

     me

     a

     bit

     about

     yourself

    ?

     How

     do

     you

     get

     your

     work

     done

    ?

     What

    's

     your

     best

     tip

     for

     staying

     focused

     during

     work

    ?


    [

    Your

     Name

    ],

     please

    .

     I

    'm

     really

     looking

     forward

     to

     meeting

     you

    !

     What

     brings

     you

     here

     today

    ?

     I

    'm

     excited

     to

     meet

     you

     and

     learn

     more

     about

     our

     world

    .

     What

     would

     you

     like

    
    
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

     dating

     back

     to

     the

     Roman

     Empire

     and

     the

     Renaissance

     period

    .

     It

     is

     home

     to

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

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     numerous

     other

     attractions

     and

     landmarks

    .

     Paris

     is

     known

     for

     its

     vibrant

     culture

    ,

     arts

    ,

     and

     fashion

    ,

     as

     well

     as

     its

     iconic

     landmarks

     and

     historical

     sites

    .

     It

     is

     a

     popular

     tourist

     destination

    ,

     with

     millions

     of

     visitors

     annually

     and

     a

     strong

     economy

     driven

     by

     services

    ,

     tourism

    ,

     and

     commerce

    .

     Paris

     has

     been

     a

     cultural

     center

     since

     the

     

    1

    1

    th

     century

     and

     is

     an

     important

     center

     for

     the

     arts

    ,

     literature

    ,

     and

     philosophy

    .

     It

     is

     also

     home

     to

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

    ,

     with

     many

     potential

     applications

     and

     advancements

     that

     could

     revolution

    ize

     our

     lives

     in

     a

     variety

     of

     ways

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

     automation

    :

     AI

     has

     the

     potential

     to

     automate

     a

     wide

     range

     of

     tasks

    ,

     from

     data

     entry

     to

     customer

     service

     to

     manufacturing

    .

     As

     AI

     becomes

     more

     sophisticated

    ,

     we

     may

     see

     a

     significant

     increase

     in

     automation

    ,

     with

     machines

     replacing

     human

     workers

     in

     various

     industries

    .
    


    2

    .

     Eth

    ical

     considerations

    :

     As

     AI

     becomes

     more

     prevalent

    ,

     there

     will

     be

     increased

     scrutiny

     of

     its

     ethical

     implications

    .

     We

     will

     need

     to

     ensure

     that

     AI

     systems

     are

     designed

     and

     used

     in

     a

     way

     that

     respects

     human

     rights

    ,

     privacy

    ,

     and

     security

    .
    


    3

    



```python
llm.shutdown()
```
