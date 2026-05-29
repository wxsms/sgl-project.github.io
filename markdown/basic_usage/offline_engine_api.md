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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.28it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.45it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.45it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.45it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.45it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.45it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.45it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.45it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.45it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.45it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.45it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  7.76it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  7.76it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  7.76it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  7.76it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  7.76it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  7.76it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  7.76it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  7.76it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  7.76it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 18.32it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 18.32it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 18.32it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 18.32it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 18.32it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 18.32it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 18.32it/s]

    Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 18.32it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 18.32it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 18.32it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 25.85it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 25.85it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 25.85it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 25.85it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 25.85it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 25.85it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 25.85it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 25.85it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 25.85it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 32.44it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 32.44it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 17.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 17.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.94it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.94it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.75it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 20.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 20.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.01it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 28.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 28.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 28.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.02it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.02it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.02it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.02it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.02it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.02it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.02it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.28it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.28it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.28it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.28it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.28it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.28it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.38it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.38it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.38it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.38it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.38it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.38it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  71%|███████   | 41/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 41.93it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 41.93it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.51it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.51it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.51it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.51it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.51it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.51it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.45it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.45it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.45it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.45it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.45it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.45it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.54it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 35.97it/s]


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
    Generated text:  Adam and I am an aspiring writer. I'm an engineering major at the University of Texas at Austin and I have a passion for science and technology. I like to think I have a knack for coming up with interesting and original ideas. I always enjoy writing stories and having fun with words.
    
    I read science fiction books and am always looking for new genres to write in. I have some first-hand experience with some of the worlds I've been reading about. I read science fiction to provide inspiration for my stories.
    
    I'd love to get published with you. I can write for free, so feel free to send me a story or your ideas
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the Vice President. The Vice President is elected by the members of both the House and the Senate of the United States. The president, Vice President, and Speaker of the House have the following responsibilities:
    
    * The president is responsible for maintaining the peace and securing the country.
    * The Vice President is responsible for providing advice to the president about foreign affairs and maintaining the country's interests abroad.
    * The Speaker of the House is responsible for the conduct of the legislative process and maintaining the laws.
    
    The table below shows the relative powers and duties of each office:
    
    |Office| Responsibilities|
    |---|---|
    |President| Maintain the peace
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Nice
    C. London
    D. Brussels
    Answer:
    
    A
    
    The total number of seats in a theater is 120. The proportion of the theater's seats occupied by the audience is 0.85. The number of seats that the audience occupies is _____.
    A. 96 seats
    B. 75 seats
    C. 48 seats
    D. 102 seats
    Answer:
    
    C
    
    The basic process of community health education is: A. Information dissemination - Community health education - Community health promotion - Family health education - Community health awareness B
    ===============================
    Prompt: The future of AI is
    Generated text:  bright
    
    If you're reading this, the future is bright for artificial intelligence. One year ago, in a time of great uncertainty and urgency, the world was divided over what will happen to AI. Is it a technology that will do great good and do great harm? The debate is continuing today, and it is clear that this debate is going to continue, but not in a way that has been debated before.
    
    What we see today is the result of a deep divide in our understanding of AI. The difference is the kind of AI that we are developing, and whether we believe it will have good or bad impact on the world. The


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm always eager to learn and grow, and I'm always looking for opportunities to contribute to the company and the industry. What's your name? What's your job title? What's your company name? What's your favorite hobby or activity? What's your favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the second-largest city in the world by population. Paris is known for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for business, finance, and tourism, making it a popular destination for tourists and locals alike. The city is known for its fashion, art, and cuisine, and is a major hub for international trade and diplomacy. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can better understand and respond to human emotions and preferences.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could lead to more transparent and accountable AI
    


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
    Generated text:  [insert first name], and I am a [insert relevant profession or skill] with [insert relevant experience or achievements]. Ever since I was a [insert age] year old, I've always been passionate about [insert a relevant hobby or interest]. I'm a [insert a relevant characteristic or trait] and I love [insert a relevant accomplishment or achievement]. I'm always looking for ways to [insert a relevant skill or hobby] and am always eager to learn and grow. I'm a [insert a relevant personality trait] and I'm always looking for new experiences and opportunities to make the world a better place. Thank you for having
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    You are an AI assistant that helps you understand the responses to questions and answer answers. Here is the general structure of a question and answer:
    
    Question: Where is Paris located?
    
    Answer: Paris is located in the heart of France, on the Mediterranean coast, at the mouth of the Seine River.
    
    What is the answer? (OPTIONS: A) United States B) South Africa C) Australia D) Italy )
    The answer is D) Italy. Paris is located in the heart of France, on the Mediterranean coast, at the mouth of the Seine River. It is one of the most important cities in Europe and a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of potential and exciting possibilities. Here are some possible future trends in artificial intelligence:
    
    1. Personalized AI: As AI continues to improve, we may see more personalization in AI applications. AI will be able to learn and adapt to individual users' preferences and behaviors, leading to more accurate and effective solutions for a wide range of tasks.
    
    2. Automation: AI will become increasingly integrated into our daily lives, from controlling smart home devices to automating transportation and manufacturing processes. Automation will make our jobs easier and more efficient, freeing up more time for us to focus on other important tasks.
    
    3. AI ethics and governance: As AI


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

    ]

     and

     I

    'm

     a

     [

    job

     title

    ]

     at

     [

    company

     name

    ].

     I

    'm

     here

     to

     help

     you

     with

     your

     [

    problem

    ].

     I

    'm

     excited

     to

     help

     you

     and

     provide

     you

     with

     the

     information

     and

     support

     you

     need

    .

     Let

    's

     get

     started

    !

     

    📊

    💼

    💼

    
    


    *

    Note

    :

     Replace

     the

     placeholders

     with

     your

     actual

     information

    .

     The

     goal

     is

     to

     make

     the

     intro

     as

     neutral

     and

     professional

     as

     possible

     while

     still

     being

     engaging

     and

     captivating

    .*

     


    Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ]

     and

     I

    'm

     a

     [

    job

     title

    ]

     at

     [

    company

     name

    ].

     I

    'm

     here

     to

     help

     you

     with

     your

     [

    problem

    ].

     I

    'm

     excited

     to

     help

     you

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     most

     populous

     city

     in

     the

     European

     Union

     and

     is

     home

     to

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

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     diverse

     cultural

     scene

    ,

     and

     beautiful

     architecture

    .

     It

     is

     a

     popular

     tourist

     destination

    ,

     and

     the

     French

     language

     is

     widely

     spoken

    .

     Paris

     is

     also

     known

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     nightlife

    .

     The

     city

     is

     a

     center

     of

     politics

    ,

     business

    ,

     and

     culture

    ,

     and

     has

     been

     the

     seat

     of

     government

     since

     the

     

    1

    2

    th

     century

    .

     Paris

     is

     a

     significant

     cultural

     and

     economic

     hub

     in

     France

    ,

     and

     its

     influence

     extends

     beyond

     the

     city

     limits

     to

     other

     parts

     of

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

     and

     challenges

    .

     Here

     are

     some

     possible

     future

     trends

    :
    


    1

    .

     Increased

     AI

     for

     General

     Intelligence

    :

     As

     AI

     technology

     advances

    ,

     we

     may

     see

     a

     greater

     focus

     on

     making

     AI

     more

     intelligent

     and

     capable

     of

     performing

     a

     wide

     range

     of

     tasks

    .

     This

     could

     include

     tasks

     like

     language

     understanding

    ,

     decision

    -making

    ,

     and

     problem

    -solving

    .
    


    2

    .

     AI

     Integration

     with

     Human

     Intelligence

    :

     The

     integration

     of

     AI

     with

     human

     intelligence

     could

     lead

     to

     more

     accurate

     and

     effective

     decision

    -making

    .

     This

     could

     include

     AI

    -powered

     chat

    bots

     and

     virtual

     assistants

     that

     can

     better

     understand

     and

     respond

     to

     human

     emotions

     and

     nuances

    .
    


    3

    .

     AI

     for

     Personal

    ized

     Learning

    :

     AI

     could

     be

     used

     to

     create

     personalized

     learning

     experiences

     for

    



```python
llm.shutdown()
```
