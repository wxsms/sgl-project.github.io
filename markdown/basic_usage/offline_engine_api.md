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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.53it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.53it/s]


    2026-05-15 05:31:11,210 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 05:31:11] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.64it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.64it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.64it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.64it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.64it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.64it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:06,  6.64it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:06,  6.64it/s] Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:06,  6.64it/s]Compiling num tokens (num_tokens=832):  26%|██▌       | 15/58 [00:04<00:06,  6.64it/s]Compiling num tokens (num_tokens=768):  26%|██▌       | 15/58 [00:04<00:06,  6.64it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=352):  43%|████▎     | 25/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=320):  43%|████▎     | 25/58 [00:04<00:02, 13.63it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:01, 21.80it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:01, 21.80it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:01, 21.80it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:01, 21.80it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:04<00:01, 21.80it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:05<00:01, 21.80it/s]

    Compiling num tokens (num_tokens=128):  60%|██████    | 35/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 30.99it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 30.99it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 30.99it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 30.99it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 30.99it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 30.99it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 30.99it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 30.99it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 30.99it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 30.99it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 30.99it/s]Compiling num tokens (num_tokens=12):  78%|███████▊  | 45/58 [00:05<00:00, 30.99it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 42.22it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 42.22it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 42.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.09 GB):   3%|▎         | 2/58 [00:00<00:03, 18.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.08 GB):   3%|▎         | 2/58 [00:00<00:03, 18.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.08 GB):   3%|▎         | 2/58 [00:00<00:03, 18.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.08 GB):   3%|▎         | 2/58 [00:00<00:03, 18.10it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=59.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.07 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.06 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.06 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.48it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=59.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.05 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.03 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.03 GB):  31%|███       | 18/58 [00:00<00:01, 30.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.03 GB):  31%|███       | 18/58 [00:00<00:01, 30.52it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=59.03 GB):  31%|███       | 18/58 [00:00<00:01, 30.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.01 GB):  31%|███       | 18/58 [00:00<00:01, 30.52it/s]Capturing num tokens (num_tokens=960 avail_mem=59.02 GB):  31%|███       | 18/58 [00:00<00:01, 30.52it/s] Capturing num tokens (num_tokens=896 avail_mem=59.02 GB):  31%|███       | 18/58 [00:00<00:01, 30.52it/s]Capturing num tokens (num_tokens=896 avail_mem=59.02 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.71it/s]Capturing num tokens (num_tokens=832 avail_mem=59.01 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.71it/s]Capturing num tokens (num_tokens=768 avail_mem=59.01 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.71it/s]Capturing num tokens (num_tokens=704 avail_mem=59.01 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.71it/s]Capturing num tokens (num_tokens=640 avail_mem=59.00 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.71it/s]Capturing num tokens (num_tokens=576 avail_mem=59.00 GB):  40%|███▉      | 23/58 [00:00<00:01, 34.71it/s]Capturing num tokens (num_tokens=576 avail_mem=59.00 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=512 avail_mem=58.99 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.49it/s]

    Capturing num tokens (num_tokens=480 avail_mem=59.00 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=448 avail_mem=59.00 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=416 avail_mem=59.00 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=384 avail_mem=59.00 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=384 avail_mem=59.00 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=352 avail_mem=58.99 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=320 avail_mem=58.99 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=288 avail_mem=58.99 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.62it/s]Capturing num tokens (num_tokens=256 avail_mem=58.98 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.62it/s]Capturing num tokens (num_tokens=240 avail_mem=58.98 GB):  57%|█████▋    | 33/58 [00:01<00:00, 41.62it/s]Capturing num tokens (num_tokens=240 avail_mem=58.98 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=224 avail_mem=58.98 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.63it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.97 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=192 avail_mem=58.97 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=176 avail_mem=58.97 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=160 avail_mem=58.96 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=160 avail_mem=58.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.65it/s]Capturing num tokens (num_tokens=144 avail_mem=58.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.65it/s]Capturing num tokens (num_tokens=128 avail_mem=58.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.65it/s]Capturing num tokens (num_tokens=112 avail_mem=58.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.65it/s]Capturing num tokens (num_tokens=96 avail_mem=58.95 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.65it/s] Capturing num tokens (num_tokens=80 avail_mem=58.95 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.65it/s]Capturing num tokens (num_tokens=80 avail_mem=58.95 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.22it/s]Capturing num tokens (num_tokens=64 avail_mem=58.94 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.22it/s]

    Capturing num tokens (num_tokens=48 avail_mem=58.94 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.22it/s]Capturing num tokens (num_tokens=32 avail_mem=58.94 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.22it/s]Capturing num tokens (num_tokens=28 avail_mem=58.93 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.22it/s]Capturing num tokens (num_tokens=24 avail_mem=58.93 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.22it/s]Capturing num tokens (num_tokens=24 avail_mem=58.93 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=20 avail_mem=58.93 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=16 avail_mem=58.93 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=12 avail_mem=58.89 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=8 avail_mem=58.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.09it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=58.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.09it/s]Capturing num tokens (num_tokens=4 avail_mem=58.88 GB): 100%|██████████| 58/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=4 avail_mem=58.88 GB): 100%|██████████| 58/58 [00:01<00:00, 37.32it/s]


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
    Generated text:  Andrew Chen and I am currently a senior at the University of Michigan. My background includes a Master of Arts degree from the University of Michigan and a Master of Science degree from the University of Michigan. My areas of research interest include applied mathematics, computational mathematics, and data science.
    In my undergraduate years, I am particularly interested in computational mathematics. My undergraduate thesis was about solving inverse problems of mathematical physics. I was particularly fascinated by the mathematical connections between physics and mathematics and had the opportunity to work with several other students who were interested in the same topic. To earn my Master of Arts degree, I did my research in the area of computational mathematics
    ===============================
    Prompt: The president of the United States is
    Generated text:  a powerful man and many people want to be president. How do they do it? The president chooses someone who knows how to get things done. Some people want to be president to be the leader of the country. For example, if a president is not very good at taking care of the country, the people who voted for him might not like him. There are also many people who want to be president to be leaders in different countries. Sometimes, countries like to have leaders who are brave and help others. These people are called presidents.
    In answer to these questions, what are some of the qualities that the person who wants to be president
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    
    Does it follow that "The capital of France is in Paris."?
    Select from: 1). yes; 2). it is not possible to tell; 3). no;
    
    1). yes
    
    The statement "The capital of France is in Paris" is true. If Paris is the capital of France, then it logically follows that the capital of France is also in Paris. The answer is yes. 
    2). it is not possible to tell is incorrect because the given information is sufficient to determine that the capital of France is in Paris. 
    3). no is incorrect because it contradicts the information provided in the original
    ===============================
    Prompt: The future of AI is
    Generated text:  more about building the right capabilities and addressing the right problems. As the industry moves forward, it’s important to keep an eye on the latest trends and innovations. These innovations, and more, will continue to shape the future of AI.
    
    One of the most exciting trends in AI is the introduction of machine learning models. Machine learning is a type of AI that allows computers to learn from data and make predictions or decisions based on that data. This has allowed us to build more accurate and efficient systems, such as self-driving cars and chatbots.
    
    Another exciting trend is the development of more sophisticated models that can handle complex tasks and provide more personalized and


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination, known for its rich history, beautiful architecture, and vibrant nightlife. The city is also home to many notable French artists, writers, and musicians. Paris is a city of contrasts, with its modern and traditional elements blending together to create a unique and fascinating place to live and visit. The city is also home to many international organizations and institutions, including the French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Enhanced ethical considerations: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical implications. This could lead to more stringent regulations and guidelines to ensure that AI systems are developed and used in a responsible
    


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
    Generated text:  [Name] and I am a [occupation], a [type of knowledge] with a [number] of years of experience in the [relevant field]. I am dedicated to [your professional goal or mission]. 
    
    Please provide the necessary information to get to know you better. What's your name? What are your current career goals? Where do you come from? What do you like to do on a regular basis? What are your strengths and weaknesses? How do you stay updated on the latest news? What's the most exciting thing you've done in your career? How do you balance your work and personal life? What kind of person
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the 12th largest in the world, with an estimated population of around 2. 5 million. Paris is a bustling metropolis with many historic landmarks, art galleries, and museums, as well as a vibrant nightlife and food scene. The city is home to over 500 million people, making it the world's most populous city. Paris is also a major cultural and commercial center, hosting numerous prestigious events and festivals throughout the year. The city has a rich history, including the ancient city of Paris, as well as French colonial and imperial history. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but here are some possible trends that could potentially shape the industry in the coming years:
    
    1. Increased integration of AI into various industries: AI is likely to become more integrated into various industries, such as healthcare, finance, manufacturing, and transportation, leading to more automation and efficiency improvements.
    
    2. AI will become more accessible to all: As AI becomes more integrated into our daily lives, it is likely that we will see more people adopting AI technology. This could lead to increased accessibility and adoption of AI in different sectors, such as education, retail, and entertainment.
    
    3. AI will become more ethical and responsible: There will be


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

     Sarah

    ,

     and

     I

     am

     a

     professional

     writer

     and

     public

     speaker

    .

     My

     love

     for

     creativity

     is

     infectious

    ,

     and

     I

     strive

     to

     bring

     my

     ideas

     to

     life

     through

     my

     writing

     and

     public

     speaking

    .

     Whether

     it

    's

     writing

     a

     novel

     or

     giving

     a

     speech

     at

     a

     conference

    ,

     I

     am

     always

     up

     for

     the

     challenge

    .

     Thanks

     for

     taking

     the

     time

     to

     learn

     more

     about

     me

    .

     What

     is

     your

     favorite

     hobby

     or

     activity

     outside

     of

     work

    ?

     As

     an

     AI

     language

     model

    ,

     I

     do

     not

     have

     hobbies

     or

     activities

     like

     humans

     do

    ,

     but

     I

     can

     suggest

     some

     popular

     activities

     that

     you

     might

     enjoy

    .

     Would

     you

     like

     me

     to

     help

     you

     find

     some

     suggestions

    ?

     Do

     you

     have

     a

     favorite

     hobby

     or

     activity

     that

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     where

     the

     E

    iff

    el

     Tower

     stands

     tall

     and

     iconic

    .

     It

     is

     a

     bustling

     met

    ropolis

     with

     a

     rich

     history

    ,

     diverse

     culture

    ,

     and

     a

     reputation

     for

     its

     op

    ulent

     and

     eclectic

     architecture

    .

     The

     city

     has

     a

     vibrant

     nightlife

     and

     is

     home

     to

     numerous

     museums

    ,

     galleries

    ,

     and

     landmarks

    .

     Paris

     is

     known

     for

     its

     art

    ,

     fashion

    ,

     food

    ,

     and

     music

     scenes

    ,

     as

     well

     as

     its

     annual

     Carn

    aval

     celebrations

    .

     The

     city

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

     Lou

    vre

     Museum

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    ,

     among

     other

     notable

     buildings

     and

     attractions

    .

     Paris

     is

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    ,

     offering

     a

     unique

     blend

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

     trends

    :
    


    1

    .

     Enhanced

     AI

     capabilities

    :

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     learn

     and

     adapt

    ,

     leading

     to

     even

     more

     advanced

     and

     efficient

     systems

    .
    


    2

    .

     AI

    -in

    formed

     decision

    -making

    :

     AI

     will

     be

     integrated

     into

     a

     broader

     range

     of

     decision

    -making

     processes

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    ,

     to

     help

     make

     more

     informed

     and

     effective

     decisions

    .
    


    3

    .

     AI

     for

     sustainability

    :

     AI

     will

     be

     used

     to

     improve

     energy

     efficiency

    ,

     reduce

     waste

    ,

     and

     promote

     sustainable

     practices

    ,

     making

     it

     a

     key

     driver

     of

     the

     transition

     to

     a

     more

     sustainable

     economy

    .
    


    4

    .

     AI

     for

     customer

     service

    :

     AI

     will

     be

     used

     to

     provide

     personalized

     and

     efficient

    



```python
llm.shutdown()
```
