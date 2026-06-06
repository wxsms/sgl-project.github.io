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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1024):  19%|█▉        | 11/58 [00:04<00:12,  3.91it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]

    Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.32it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:02, 13.08it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:02, 13.08it/s]

    Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:06<00:02, 13.08it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:06<00:02, 13.08it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:06<00:02, 13.08it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:06<00:02, 13.08it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:06<00:02, 13.08it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:06<00:02,  8.57it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:06<00:02,  8.57it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:06<00:02,  8.57it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:06<00:02,  8.57it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:06<00:02,  8.57it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:06<00:02,  8.57it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:06<00:02,  8.57it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:06<00:02,  8.57it/s]Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:06<00:02,  8.57it/s]Compiling num tokens (num_tokens=176):  57%|█████▋    | 33/58 [00:06<00:02,  8.57it/s]Compiling num tokens (num_tokens=160):  57%|█████▋    | 33/58 [00:06<00:02,  8.57it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:06<00:01, 14.29it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:06<00:01, 14.29it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:01, 14.29it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:01, 14.29it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:06<00:01, 14.29it/s] 

    Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:06<00:01, 14.29it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:06<00:01, 14.29it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:06<00:01, 14.29it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:06<00:01, 14.29it/s]Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:06<00:01, 14.29it/s]Compiling num tokens (num_tokens=24):  74%|███████▍  | 43/58 [00:06<00:01, 14.29it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:06<00:00, 21.27it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:06<00:00, 21.27it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:06<00:00, 21.27it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:06<00:00, 21.27it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:06<00:00, 21.27it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:06<00:00, 21.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.16it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=60.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.39 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=7168 avail_mem=60.39 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=6656 avail_mem=60.38 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]Capturing num tokens (num_tokens=6144 avail_mem=60.38 GB):   3%|▎         | 2/58 [00:00<00:02, 19.09it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=60.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=5120 avail_mem=60.37 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.26 GB):   9%|▊         | 5/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.26 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.26 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.18it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=59.26 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.26 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.25 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.87it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.25 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.87it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.25 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.25 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.87it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.24 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.87it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.24 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.83it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.24 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.24 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.83it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=59.23 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.23 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.23 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.83it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.23 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.21 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=960 avail_mem=59.22 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.21it/s] Capturing num tokens (num_tokens=896 avail_mem=59.22 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=832 avail_mem=59.22 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=832 avail_mem=59.22 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.62it/s]Capturing num tokens (num_tokens=768 avail_mem=59.21 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.62it/s]Capturing num tokens (num_tokens=704 avail_mem=59.21 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.62it/s]

    Capturing num tokens (num_tokens=640 avail_mem=59.21 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.62it/s]Capturing num tokens (num_tokens=576 avail_mem=59.21 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.62it/s]Capturing num tokens (num_tokens=512 avail_mem=59.19 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.62it/s]Capturing num tokens (num_tokens=512 avail_mem=59.19 GB):  50%|█████     | 29/58 [00:01<00:00, 37.23it/s]Capturing num tokens (num_tokens=480 avail_mem=59.21 GB):  50%|█████     | 29/58 [00:01<00:00, 37.23it/s]Capturing num tokens (num_tokens=448 avail_mem=59.20 GB):  50%|█████     | 29/58 [00:01<00:00, 37.23it/s]Capturing num tokens (num_tokens=416 avail_mem=59.20 GB):  50%|█████     | 29/58 [00:01<00:00, 37.23it/s]Capturing num tokens (num_tokens=384 avail_mem=59.20 GB):  50%|█████     | 29/58 [00:01<00:00, 37.23it/s]Capturing num tokens (num_tokens=352 avail_mem=59.19 GB):  50%|█████     | 29/58 [00:01<00:00, 37.23it/s]Capturing num tokens (num_tokens=352 avail_mem=59.19 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.68it/s]Capturing num tokens (num_tokens=320 avail_mem=59.19 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.68it/s]Capturing num tokens (num_tokens=288 avail_mem=59.19 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.68it/s]

    Capturing num tokens (num_tokens=256 avail_mem=59.18 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.68it/s]Capturing num tokens (num_tokens=240 avail_mem=59.18 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.68it/s]Capturing num tokens (num_tokens=224 avail_mem=59.18 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.68it/s]Capturing num tokens (num_tokens=224 avail_mem=59.18 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.05it/s]Capturing num tokens (num_tokens=208 avail_mem=59.17 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.05it/s]Capturing num tokens (num_tokens=192 avail_mem=59.17 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.05it/s]Capturing num tokens (num_tokens=176 avail_mem=59.17 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.05it/s]Capturing num tokens (num_tokens=160 avail_mem=59.17 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.05it/s]Capturing num tokens (num_tokens=144 avail_mem=59.16 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.05it/s]Capturing num tokens (num_tokens=144 avail_mem=59.16 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=128 avail_mem=59.16 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=112 avail_mem=59.16 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.83it/s]

    Capturing num tokens (num_tokens=96 avail_mem=59.16 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.83it/s] Capturing num tokens (num_tokens=80 avail_mem=59.15 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=64 avail_mem=59.15 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=64 avail_mem=59.15 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=48 avail_mem=59.14 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=32 avail_mem=59.14 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=28 avail_mem=59.14 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=24 avail_mem=59.13 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=20 avail_mem=59.13 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.69it/s]

    Capturing num tokens (num_tokens=20 avail_mem=59.13 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=16 avail_mem=59.13 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=12 avail_mem=59.12 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=8 avail_mem=59.12 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.71it/s] Capturing num tokens (num_tokens=4 avail_mem=59.12 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.71it/s]Capturing num tokens (num_tokens=4 avail_mem=59.12 GB): 100%|██████████| 58/58 [00:01<00:00, 34.76it/s]


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
    Generated text:  Nishant. I am a fifth year student of Indian Institute of Technology, Roorkee, India. I am a passionate learner and I love to challenge myself by exploring new ways of solving complex problems.
    
    As a frequent user of various online forums, I have observed that many users find it challenging to understand and engage with complex topics. I decided to explore my own learning style and use it to create a platform that is both educational and engaging.
    
    I have a unique educational background, having studied Mathematics at the undergraduate level and a strong passion for coding. I believe that by using a user-friendly interface and a balance of interactive activities, I
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to arrive at a record number of students who will graduate with a bachelor's degree this year. In the past two years, the number of students who graduate with a bachelor's degree has steadily increased. Over the past two years, the number of students who graduated with a bachelor's degree increased by 150%. If the total number of students who graduated with a bachelor's degree over the two years was 7,500, what is the current number of students who graduate with a bachelor's degree? To determine the current number of students who graduate with a bachelor's degree, we need to follow these steps:
    
    1
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the 5th largest city in the world and is also the second largest metropolitan area in the world. It has a population of approximately 2.9 million. It is located in the northwest of the country, between the Mediterranean Sea and the Atlantic Ocean. It is in the region of Île-de-France, which also includes Paris, and the Île of France.
    
    As the capital of France, Paris is known as the "City of Lights" because of its lights. The most famous landmark in Paris is the Eiffel Tower, built in 1889 to commemorate the 100
    ===============================
    Prompt: The future of AI is
    Generated text:  all around us. AI is already used in everything from self-driving cars to virtual assistants to smart homes. But what about the future of AI? Will it continue to dominate the technology industry or will it face similar challenges to previous generations of AI?
    It is a question that has been debated for decades, and the answer is still not clear. However, it is clear that the field of AI will continue to evolve and grow, and that it will face similar challenges to previous generations of AI.
    The challenges that AI is likely to face include issues such as bias, transparency, and accountability. These challenges come from a variety of sources, including the


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a cultural and economic hub, hosting numerous world-renowned museums, theaters, and festivals. Paris is a popular tourist destination and a major economic center, with a rich history and diverse cultural scene. It is often referred to as the "City of Light" and is a major center of French culture and politics. The city is home to many notable French artists, writers, and intellectuals, and is known for its vibrant nightlife and fashion scene. Paris is a city of contrasts, with its modern architecture and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration of AI into everyday life: As AI becomes more integrated into our daily lives, we may see more widespread adoption of AI technologies in areas such as healthcare, transportation, and consumer goods. This could lead to a more efficient and effective use of resources, as well as the development of new products and services that are tailored to individual needs.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more integrated into our daily
    


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
    Generated text:  [Name] and I'm a [character type] who is dedicated to [occupation]. I'm [age] years old, [gender] and [character type]. I'm [career objective]. I've been living in [city, state, or country] for [years] and have always been [what?]. I've always been passionate about [the hobby or passion that drives me] and I strive to [what?]. I'm always looking for [what?]. I'm always up for [what?]. I'm constantly learning and growing as a person. My goals are [what?]. I'm [type]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The other cities of France include:
    
    - Lyon: a major port city and wine region
    - Marseille: a historic port and cultural center
    
    - Nice: a coastal city and seaside resort
    
    - Toulouse: a major city in the northwestern corner of the country
    
    - Strasbourg: a major city in the south of the country
    
    - Nantes: a historic city and port town
    
    - Bordeaux: a major port city and wine region
    
    - Bordeaux: a major port city and wine region
    
    - Lyon: a major port city and wine region
    
    - Marseille: a historic port and cultural center
    
    - Nice: a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a wide range of developments and trends, including:
    
    1. Increased focus on privacy and ethics: As more people become aware of the dangers of AI-powered surveillance and data collection, there will be increased focus on developing ethical guidelines and policies to protect individuals' privacy and privacy rights.
    
    2. AI becoming more integrated into our daily lives: AI is already making significant inroads into our everyday lives, from helping us find directions to helping us make grocery shopping lists. As AI continues to evolve and improve, we can expect to see more seamless integration into our daily routines.
    
    3. AI becoming more autonomous and self-aware: With the development


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

    /an

     [

    age

    ]

     year

     old

     [

    occupation

    ].

     I

     love

     [

    describe

     something

     about

     yourself

    ].

     I

    'm

     confident

    ,

     smart

    ,

     and

     independent

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     I

    'm

     writing

     this

     introduction

     as

     a

     beginner

    ,

     and

     I

    'm

     not

     sure

     where

     to

     begin

    .

     Can

     you

     help

    ?

     I

    'm

     a

     beginner

     writing

     about

     myself

    ,

     and

     I

     just

     need

     some

     help

     getting

     started

    .

     Do

     you

     have

     any

     tips

     on

     how

     to

     start

     writing

     an

     introduction

     for

     a

     character

    ?

     Definitely

    !

     Start

     by

     identifying

     who

     you

     are

     and

     what

     you

     do

    .

     Write

     a

     sentence

     or

     two

     about

     who

     you

     are

     and

     what

     you

     do

    ,

     and

     then

     narrow

     it

     down

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     is

     the

     seat

     of

     government

    ,

     the

     headquarters

     of

     many

     major

     organizations

    ,

     and

     the

     most

     visited

     city

     in

     the

     world

    .

     It

     is

     also

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

     and

     the

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

     vibrant

     culture

    ,

     art

    ,

     and

     cuisine

    ,

     and

     is

     a

     global

     cultural

     and

     economic

     hub

    .

     Its

     annual

     tourism

     industry

     is

     worth

     billions

     of

     euros

    .

     The

     city

     is

     also

     known

     for

     its

     historical

     landmarks

    ,

     including

     the

     Notre

    -D

    ame

     Cathedral

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     home

     to

     many

     famous

     landmarks

     such

     as

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

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     full

     of

     possibilities

    ,

     as

     the

     technology

     continues

     to

     evolve

     at

     an

     unprecedented

     rate

    .

     Some

     of

     the

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     AI

     systems

     will

     likely

     become

     more

     capable

     of

     replic

    ating

     human

     cognitive

     functions

    ,

     including

     learning

    ,

     problem

    -solving

    ,

     and

     decision

    -making

    .
    


    2

    .

     Enhanced

     transparency

    :

     AI

     systems

     will

     become

     more

     transparent

    ,

     allowing

     for

     better

     understanding

     of

     their

     decision

    -making

     processes

     and

     ability

     to

     explain

     their

     actions

    .
    


    3

    .

     Greater

     ethical

     considerations

    :

     AI

     systems

     will

     face

     more

     ethical

     and

     moral

     questions

    ,

     with

     developers

     and

     policymakers

     working

     to

     ensure

     that

     AI

     is

     developed

     and

     used

     in

     a

     way

     that

     promotes

     social

     justice

     and

     equality

    .
    


    4

    .

    



```python
llm.shutdown()
```
