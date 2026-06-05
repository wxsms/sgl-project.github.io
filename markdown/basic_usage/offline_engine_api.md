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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.76it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:29,  5.79s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:29,  5.79s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:29,  5.79s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:29,  5.79s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:29,  5.79s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:06<00:47,  1.13it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:13,  3.40it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:13,  3.40it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:06<00:13,  3.40it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:06<00:13,  3.40it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:06<00:13,  3.40it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:06<00:13,  3.40it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:06<00:13,  3.40it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:06<00:13,  3.40it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:06<00:13,  3.40it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:05,  6.80it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:05,  6.80it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:06<00:05,  6.80it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:06<00:05,  6.80it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:06<00:05,  6.80it/s]

    Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:06<00:05,  6.80it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:06<00:05,  6.80it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:06<00:05,  6.80it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:06<00:02, 10.45it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:06<00:02, 10.45it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:06<00:02, 10.45it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:06<00:02, 10.45it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:06<00:02, 10.45it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:06<00:02, 10.45it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:06<00:02, 10.45it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:06<00:01, 14.21it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:06<00:01, 14.21it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:06<00:01, 14.21it/s]

    Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:06<00:01, 14.21it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:06<00:01, 14.21it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:06<00:01, 14.21it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:06<00:01, 14.21it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:06<00:01, 18.19it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:06<00:01, 18.19it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:06<00:01, 18.19it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:06<00:01, 18.19it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:06<00:01, 18.19it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:06<00:01, 18.19it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:06<00:01, 18.19it/s]

    Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 22.84it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 22.84it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 22.84it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 22.84it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:06<00:00, 22.84it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:06<00:00, 22.84it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:06<00:00, 22.84it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:06<00:00, 22.84it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:06<00:00, 22.84it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:06<00:00, 30.80it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:06<00:00, 30.80it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:06<00:00, 30.80it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:06<00:00, 30.80it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:06<00:00, 30.80it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:06<00:00, 30.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.84 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.81 GB):   3%|▎         | 2/58 [00:00<00:03, 16.17it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.81 GB):   3%|▎         | 2/58 [00:00<00:03, 16.17it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.80 GB):   3%|▎         | 2/58 [00:00<00:03, 16.17it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.80 GB):   7%|▋         | 4/58 [00:00<00:03, 17.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.80 GB):   7%|▋         | 4/58 [00:00<00:03, 17.73it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.80 GB):   7%|▋         | 4/58 [00:00<00:03, 17.73it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.79 GB):   7%|▋         | 4/58 [00:00<00:03, 17.73it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.79 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.78 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.48it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.78 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.78 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.48it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=56.78 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.68it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.77 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.68it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.77 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.77 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.77 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.68it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.77 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.76 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.76 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.76 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.75 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.64it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=56.75 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.75 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.75 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.73 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.00it/s]Capturing num tokens (num_tokens=960 avail_mem=56.74 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.00it/s] Capturing num tokens (num_tokens=896 avail_mem=56.74 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.00it/s]Capturing num tokens (num_tokens=896 avail_mem=56.74 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=832 avail_mem=56.74 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=768 avail_mem=56.73 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.90it/s]

    Capturing num tokens (num_tokens=704 avail_mem=56.73 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=640 avail_mem=56.73 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=640 avail_mem=56.73 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.07it/s]Capturing num tokens (num_tokens=576 avail_mem=56.73 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.07it/s]Capturing num tokens (num_tokens=512 avail_mem=56.71 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.07it/s]Capturing num tokens (num_tokens=480 avail_mem=56.73 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.07it/s]Capturing num tokens (num_tokens=448 avail_mem=56.73 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=416 avail_mem=56.72 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=416 avail_mem=56.72 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=384 avail_mem=56.72 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=352 avail_mem=56.72 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.44it/s]

    Capturing num tokens (num_tokens=320 avail_mem=56.71 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=288 avail_mem=56.71 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=288 avail_mem=56.71 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.73it/s]Capturing num tokens (num_tokens=256 avail_mem=56.71 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.73it/s]Capturing num tokens (num_tokens=240 avail_mem=56.70 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.73it/s]Capturing num tokens (num_tokens=224 avail_mem=56.70 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.73it/s]Capturing num tokens (num_tokens=208 avail_mem=56.69 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.73it/s]Capturing num tokens (num_tokens=208 avail_mem=56.69 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.51it/s]Capturing num tokens (num_tokens=192 avail_mem=56.69 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.51it/s]Capturing num tokens (num_tokens=176 avail_mem=56.69 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.51it/s]

    Capturing num tokens (num_tokens=160 avail_mem=56.69 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.51it/s]Capturing num tokens (num_tokens=144 avail_mem=56.68 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.51it/s]Capturing num tokens (num_tokens=144 avail_mem=56.68 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=128 avail_mem=56.68 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=112 avail_mem=56.68 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=96 avail_mem=56.68 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.24it/s] Capturing num tokens (num_tokens=80 avail_mem=56.67 GB):  76%|███████▌  | 44/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=80 avail_mem=56.67 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.18it/s]Capturing num tokens (num_tokens=64 avail_mem=56.67 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.18it/s]

    Capturing num tokens (num_tokens=48 avail_mem=56.66 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.18it/s]Capturing num tokens (num_tokens=32 avail_mem=56.66 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.18it/s]Capturing num tokens (num_tokens=28 avail_mem=56.66 GB):  83%|████████▎ | 48/58 [00:01<00:00, 35.18it/s]Capturing num tokens (num_tokens=28 avail_mem=56.66 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=24 avail_mem=56.65 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=20 avail_mem=56.65 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=16 avail_mem=56.65 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.93it/s]

    Capturing num tokens (num_tokens=12 avail_mem=56.65 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=12 avail_mem=56.65 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.36it/s]Capturing num tokens (num_tokens=8 avail_mem=56.64 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.36it/s] Capturing num tokens (num_tokens=4 avail_mem=56.64 GB):  97%|█████████▋| 56/58 [00:01<00:00, 32.36it/s]Capturing num tokens (num_tokens=4 avail_mem=56.64 GB): 100%|██████████| 58/58 [00:01<00:00, 31.87it/s]


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
    Generated text:  Steven. I am a student of mathematics. In my spare time, I like playing chess and watching the movies. I am very good at using English. I can speak English well. I will be an English teacher in a middle school in the future. My classmate, Li Lei, is a student of computer science. He has a unique way of thinking and is very smart. He can use his computer skills to play chess and watch the movies. He often plays chess with me and I often help him with his computer skills. I enjoy helping him and he enjoys playing chess with me. My classmate, Li Lei, is a
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is the leader of the country. He helps the people to make the country better. The president is very important. He makes a lot of decisions. He is very popular. Many people like the president. The president is the president of the United States. His job is to make sure that the country is running well. In the past, the president was the leader of the country. Before that, there was a president who was a helper. He helped the country to make decisions. Now, there are two presidents. The first president was a helper. He helped the country to make decisions. But now the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Berlin
    C. Rome
    D. London
    
    A. Paris
    
    The capital of France is Paris, which is located in the south of the country. It is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Notre-Dame de Paris. Paris is also a popular destination for tourists and has a rich history dating back over 2,500 years.
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. By 2025, one in every ten people will use artificial intelligence to make decisions about their lives. That is to say, AI is going to be more widely used and will change all kinds of jobs. This is not only a blessing but also a curse. What does AI mean? AI refers to the technology that allows machines to learn and adapt to new situations. It can be used for a variety of applications, such as language translation, facial recognition, and image recognition. These are just a few examples of how AI can be used. However, it is important to note that AI is not yet perfect. It


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also home to the French Riviera, a popular tourist destination for its beaches and luxury resorts. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is known for its fashion, art, and cuisine, and is a major hub for international
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are already being used in a wide range of applications, from self-driving cars to personalized medicine to fraud detection. As these technologies continue to improve, we can expect to see even more innovative applications emerge, such as robots that can perform tasks that were previously thought to be impossible. Additionally, AI is likely to play an increasingly important role in shaping society, from the way we interact with technology to the way we make decisions about our own lives. As AI continues to evolve, it is likely to have a significant impact on our
    


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
    Generated text:  [Name] and I'm a [occupation] with [number] years of experience. I'm always looking for opportunities to learn and grow. What can you tell me about yourself? [Tell me about yourself, including any relevant experience or skills that will help me engage with the character and make you feel like a mentor.] As an AI language model, I don't have a physical appearance, but I can help you understand and interact with other humans in a neutral and informative manner. How can I assist you today? What do you want to know or discuss? Keep in mind that I'm programmed to be helpful and accommodating, so please
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris is the capital city of France and one of the most famous cities in the world. It's home to many of France's most iconic landmarks and is known for its rich history, beautiful architecture, and vibrant culture. The city is also home to the Eiffel Tower, the Louvre Museum, and many other important tourist attractions. As the capital, Paris plays an important role in the country's political and economic life. Overall, Paris is an important and beautiful city with much to offer visitors from around the world. 
    
    Paris: Capital of France | Citylife | Wikipedia
    
    Note: This fact is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be one of rapid development and innovation, with many potential applications and opportunities for its development. Some possible future trends in AI include:
    
    1. Increased integration with other technologies: As AI becomes more integrated with other technologies, such as machine learning, natural language processing, and computer vision, it will become even more capable of understanding and responding to complex human-like behavior.
    
    2. Advances in privacy and security: As AI becomes more integrated with other technologies, there is a risk that it may be used for unethical or malicious purposes, such as tracking users or harvesting personal data. To address this, there will be an increase in research and development


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

     I

    'm

     a

     [

    insert

     occupation

     or

     title

    ]

     from

     [

    your

     fictional

     country

     or

     region

    ].

     I

    'm

     [

    insert

     your

     age

     or

     height

    ]

     years

     old

     and

     [

    insert

     your

     occupation

     or

     title

    ],

     and

     I

    've

     been

     living

     in

     [

    your

     fictional

     country

     or

     region

    ]

     for

     [

    insert

     number

     of

     years

    ].

     I

     enjoy

     reading

     books

    ,

     playing

     games

    ,

     and

     spending

     time

     outdoors

    .

     What

    's

     your

     favorite

     hobby

    ?

     As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     feelings

     or

     emotions

    ,

     so

     I

     don

    't

     have

     a

     favorite

     hobby

    .

     However

    ,

     I

     can

     suggest

     some

     books

     or

     games

     that

     I

     enjoy

     reading

     or

     playing

    !

     How

     can

     I

     help

     you

     learn

     more

     about

     my

     character

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Here

     is

     a

     concise

     factual

     statement

     about

     France

    ’s

     capital

     city

    :
    


    The

     capital

     of

     France

     is

     Paris

    .

     
    


    This

     statement

     captures

     the

     essential

     fact

     that

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     specifying

     its

     name

     as

     both

     the

     official

     name

     and

     the

     more

     familiar

     nickname

     "

    Paris

    ."

     It

     uses

     a

     straightforward

    ,

     concise

     format

     typical

     of

     such

     statements

    .

     The

     sentence

     is

     accurate

     and

     easy

     to

     understand

    ,

     providing

     the

     reader

     with

     a

     clear

     and

     concise

     overview

     of

     the

     capital

     city

    's

     status

     and

     name

    .

     Here

    's

     a

     slightly

     altered

     version

     for

     emphasis

    :
    


    -

     **

    Paris

    **

     is

     France

    's

     **

    capital

    **

    .
    


    This

     version

     explicitly

     states

     the

     name

     "

    Paris

    "

     as

     the

     capital

     city

    ,

     enhancing

     clarity

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     rapidly

     evolving

    ,

     with

     numerous

     potential

     trends

     shaping

     the

     landscape

    .

     Here

     are

     some

     of

     the

     most

     notable

     ones

    :
    


    1

    .

     Increased

     integration

     with

     everyday

     technology

    :

     AI

     will

     become

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     voice

     assistants

    ,

     self

    -driving

     cars

    ,

     and

     smart

     home

     devices

    .

     This

     will

     lead

     to

     a

     more

     seamless

     and

     convenient

     experience

     for

     users

    .
    


    2

    .

     Deep

     learning

     and

     machine

     learning

    :

     AI

     will

     continue

     to

     advance

     with

     the

     development

     of

     deep

     learning

     and

     machine

     learning

     algorithms

    .

     These

     technologies

     will

     enable

     AI

     to

     learn

     from

     large

     datasets

     and

     make

     more

     accurate

     predictions

     and

     decisions

    .
    


    3

    .

     AI

     ethics

     and

     safety

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

     there

     will

     be

    



```python
llm.shutdown()
```
