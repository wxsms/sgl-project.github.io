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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  9.23it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.63it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 23.20it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 23.20it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 23.20it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 23.20it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 23.20it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 23.20it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 23.20it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 23.20it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 23.20it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 23.20it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:04<00:00, 23.20it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:04<00:00, 32.70it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=41.93 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=41.93 GB):   2%|▏         | 1/58 [00:00<00:08,  6.88it/s]Capturing num tokens (num_tokens=7680 avail_mem=41.90 GB):   2%|▏         | 1/58 [00:00<00:08,  6.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=41.89 GB):   2%|▏         | 1/58 [00:00<00:08,  6.88it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=41.89 GB):   2%|▏         | 1/58 [00:00<00:08,  6.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=41.89 GB):   7%|▋         | 4/58 [00:00<00:03, 15.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=41.89 GB):   7%|▋         | 4/58 [00:00<00:03, 15.65it/s]Capturing num tokens (num_tokens=5632 avail_mem=41.88 GB):   7%|▋         | 4/58 [00:00<00:03, 15.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.87 GB):   7%|▋         | 4/58 [00:00<00:03, 15.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.87 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.87 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.87 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.61it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=41.87 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.86 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.86 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.86 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.85 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.85 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.85 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.85 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.85 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.84 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.84 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.48it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=41.84 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.83 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.82 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.82 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.26it/s]Capturing num tokens (num_tokens=960 avail_mem=41.83 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.26it/s] Capturing num tokens (num_tokens=896 avail_mem=41.83 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.26it/s]Capturing num tokens (num_tokens=832 avail_mem=41.82 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.26it/s]Capturing num tokens (num_tokens=768 avail_mem=41.82 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.26it/s]Capturing num tokens (num_tokens=704 avail_mem=41.82 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.26it/s]

    Capturing num tokens (num_tokens=704 avail_mem=41.82 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.81it/s]Capturing num tokens (num_tokens=640 avail_mem=41.81 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.81it/s]Capturing num tokens (num_tokens=576 avail_mem=41.81 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.81it/s]Capturing num tokens (num_tokens=512 avail_mem=41.80 GB):  45%|████▍     | 26/58 [00:01<00:00, 36.81it/s]

    Capturing num tokens (num_tokens=480 avail_mem=41.81 GB):  45%|████▍     | 26/58 [00:01<00:00, 36.81it/s]Capturing num tokens (num_tokens=480 avail_mem=41.81 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.62it/s]Capturing num tokens (num_tokens=448 avail_mem=41.81 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.62it/s]

    Capturing num tokens (num_tokens=416 avail_mem=41.81 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.62it/s]Capturing num tokens (num_tokens=384 avail_mem=41.81 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.62it/s]Capturing num tokens (num_tokens=384 avail_mem=41.81 GB):  57%|█████▋    | 33/58 [00:01<00:01, 15.69it/s]Capturing num tokens (num_tokens=352 avail_mem=41.80 GB):  57%|█████▋    | 33/58 [00:01<00:01, 15.69it/s]

    Capturing num tokens (num_tokens=320 avail_mem=41.80 GB):  57%|█████▋    | 33/58 [00:01<00:01, 15.69it/s]Capturing num tokens (num_tokens=288 avail_mem=41.79 GB):  57%|█████▋    | 33/58 [00:01<00:01, 15.69it/s]Capturing num tokens (num_tokens=288 avail_mem=41.79 GB):  62%|██████▏   | 36/58 [00:01<00:01, 14.64it/s]Capturing num tokens (num_tokens=256 avail_mem=41.79 GB):  62%|██████▏   | 36/58 [00:01<00:01, 14.64it/s]

    Capturing num tokens (num_tokens=240 avail_mem=41.79 GB):  62%|██████▏   | 36/58 [00:01<00:01, 14.64it/s]Capturing num tokens (num_tokens=240 avail_mem=41.79 GB):  66%|██████▌   | 38/58 [00:01<00:01, 14.36it/s]Capturing num tokens (num_tokens=224 avail_mem=41.78 GB):  66%|██████▌   | 38/58 [00:01<00:01, 14.36it/s]Capturing num tokens (num_tokens=208 avail_mem=41.78 GB):  66%|██████▌   | 38/58 [00:02<00:01, 14.36it/s]

    Capturing num tokens (num_tokens=208 avail_mem=41.78 GB):  69%|██████▉   | 40/58 [00:02<00:01, 14.01it/s]Capturing num tokens (num_tokens=192 avail_mem=41.78 GB):  69%|██████▉   | 40/58 [00:02<00:01, 14.01it/s]Capturing num tokens (num_tokens=176 avail_mem=41.78 GB):  69%|██████▉   | 40/58 [00:02<00:01, 14.01it/s]Capturing num tokens (num_tokens=176 avail_mem=41.78 GB):  72%|███████▏  | 42/58 [00:02<00:01, 13.62it/s]Capturing num tokens (num_tokens=160 avail_mem=41.77 GB):  72%|███████▏  | 42/58 [00:02<00:01, 13.62it/s]

    Capturing num tokens (num_tokens=144 avail_mem=41.77 GB):  72%|███████▏  | 42/58 [00:02<00:01, 13.62it/s]Capturing num tokens (num_tokens=144 avail_mem=41.77 GB):  76%|███████▌  | 44/58 [00:02<00:01, 13.53it/s]Capturing num tokens (num_tokens=128 avail_mem=41.77 GB):  76%|███████▌  | 44/58 [00:02<00:01, 13.53it/s]Capturing num tokens (num_tokens=112 avail_mem=41.77 GB):  76%|███████▌  | 44/58 [00:02<00:01, 13.53it/s]

    Capturing num tokens (num_tokens=112 avail_mem=41.77 GB):  79%|███████▉  | 46/58 [00:02<00:00, 13.48it/s]Capturing num tokens (num_tokens=96 avail_mem=41.76 GB):  79%|███████▉  | 46/58 [00:02<00:00, 13.48it/s] Capturing num tokens (num_tokens=80 avail_mem=41.76 GB):  79%|███████▉  | 46/58 [00:02<00:00, 13.48it/s]

    Capturing num tokens (num_tokens=80 avail_mem=41.76 GB):  83%|████████▎ | 48/58 [00:02<00:00, 11.70it/s]Capturing num tokens (num_tokens=64 avail_mem=41.75 GB):  83%|████████▎ | 48/58 [00:02<00:00, 11.70it/s]Capturing num tokens (num_tokens=48 avail_mem=41.75 GB):  83%|████████▎ | 48/58 [00:02<00:00, 11.70it/s]

    Capturing num tokens (num_tokens=48 avail_mem=41.75 GB):  86%|████████▌ | 50/58 [00:03<00:00,  9.63it/s]Capturing num tokens (num_tokens=32 avail_mem=41.75 GB):  86%|████████▌ | 50/58 [00:03<00:00,  9.63it/s]Capturing num tokens (num_tokens=28 avail_mem=41.74 GB):  86%|████████▌ | 50/58 [00:03<00:00,  9.63it/s]

    Capturing num tokens (num_tokens=28 avail_mem=41.74 GB):  90%|████████▉ | 52/58 [00:03<00:00,  8.84it/s]Capturing num tokens (num_tokens=24 avail_mem=41.74 GB):  90%|████████▉ | 52/58 [00:03<00:00,  8.84it/s]Capturing num tokens (num_tokens=24 avail_mem=41.74 GB):  91%|█████████▏| 53/58 [00:03<00:00,  8.50it/s]Capturing num tokens (num_tokens=20 avail_mem=41.74 GB):  91%|█████████▏| 53/58 [00:03<00:00,  8.50it/s]

    Capturing num tokens (num_tokens=20 avail_mem=41.74 GB):  93%|█████████▎| 54/58 [00:03<00:00,  8.32it/s]Capturing num tokens (num_tokens=16 avail_mem=41.74 GB):  93%|█████████▎| 54/58 [00:03<00:00,  8.32it/s]Capturing num tokens (num_tokens=16 avail_mem=41.74 GB):  95%|█████████▍| 55/58 [00:03<00:00,  8.19it/s]Capturing num tokens (num_tokens=12 avail_mem=41.73 GB):  95%|█████████▍| 55/58 [00:03<00:00,  8.19it/s]

    Capturing num tokens (num_tokens=12 avail_mem=41.73 GB):  97%|█████████▋| 56/58 [00:03<00:00,  8.26it/s]Capturing num tokens (num_tokens=8 avail_mem=41.73 GB):  97%|█████████▋| 56/58 [00:03<00:00,  8.26it/s] Capturing num tokens (num_tokens=8 avail_mem=41.73 GB):  98%|█████████▊| 57/58 [00:04<00:00,  8.19it/s]Capturing num tokens (num_tokens=4 avail_mem=41.73 GB):  98%|█████████▊| 57/58 [00:04<00:00,  8.19it/s]

    Capturing num tokens (num_tokens=4 avail_mem=41.73 GB): 100%|██████████| 58/58 [00:04<00:00,  8.17it/s]Capturing num tokens (num_tokens=4 avail_mem=41.73 GB): 100%|██████████| 58/58 [00:04<00:00, 14.00it/s]


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
    Generated text:  Ian. I am 14 years old and I am a boy. I like reading. I have a lot of books. I can read a book for more than an hour. I love to learn new things and I think I can become a great writer. I like to be with my family and play games. I like to do my homework. I have a good friend named Emily. She is only 12. She likes to go to the zoo. She says it is fun. She likes to play with the animals. We often spend time together. Now I'm getting into trouble. Can you tell me about the problem
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He is in charge of the government of the country. He is also in charge of the people's lives. He is very important. He is the leader of the country. When we hear the word "leader", we think of a man or a woman who is not afraid of the danger. He is the one who leads the country. He is the one who decides what to do and when to do it. Some presidents like to be popular and some like to be rich. But the people believe that presidents are very important. They want to be president. A president is not like other people.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a beautiful city that is famous for its parks and beautiful architecture. When you walk around Paris, you may see many people with glasses. The glasses are made by artisans, and the artisans are people who make things that are made of glass.
    
    If you visit Paris, you will also notice that there are many restaurants and cafes. These are places where people can eat and drink. In Paris, there is no fast food. It is a very traditional way of eating in France. 
    
    When you are not in Paris, you may have the habit of wearing sunglasses or a hat. This is because Paris is quite hot in the
    ===============================
    Prompt: The future of AI is
    Generated text:  increasingly dependent on the availability of AI-powered machines that can communicate with humans. Which of the following is true according to the passage?
    A) The majority of AI-powered machines are currently in use.
    B) The vast majority of AI-powered machines are currently in use.
    C) All the AI-powered machines are currently in use.
    D) The vast majority of AI-powered machines will not be in use.
    Answer: B) The vast majority of AI-powered machines are currently in use. 
    
    The passage states that "The future of AI is increasingly dependent on the availability of AI-powered machines that can communicate with humans." This implies that there are currently


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


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to many famous museums, including the Musée d'Orsay and the Musée Rodin. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also known for its food scene, with many famous restaurants and cafes serving delicious cuisine. The city is also home to many cultural institutions, including the Louvre Museum, the Musée d'Orsay, and the Musée Rodin. Overall, Paris is a vibrant and exciting city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some possible future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we are likely to see more automation and artificial intelligence in our daily lives. This could include things like self-driving cars, robots in manufacturing, and even virtual assistants like Siri and Alexa.
    
    2. Improved privacy and security: As AI technology becomes more advanced, we are likely to see more privacy and security concerns. This could include issues like data breaches, surveillance, and the use of AI to
    


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
    Generated text:  [Name], and I'm a [career objective or hobby]. What are some of your interests and passions in life? I enjoy reading, traveling, and exploring new cultures. And I've always been fascinated by [future goal or opportunity], so I'm always looking for ways to make a positive impact in my community or beyond. How can I best showcase my interest in [future goal or opportunity] to others? I'd like to convey my passion and dedication in a concise and effective manner. Good luck with your future endeavors, and let me know if you need any further assistance. Best regards, [Your name].
    Hello, my name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the capital city of France, located in the south of the country and served as the capital of France from 1804 until 1970. It is known for its rich history, art, cuisine, and fashion, as well as its vibrant nightlife and annual culture festivals such as the Eiffel Tower celebration. Paris is home to the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and many other landmarks and attractions. It is also known for its role in the French Revolution and is the largest city in the European Union and the 29th most populous city in the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be rapidly evolving, driven by a range of technological advancements and changing needs. Some potential trends that could influence AI in the years to come include:
    
    1. Increased reliance on AI in healthcare and medical diagnosis: AI-powered diagnostic tools could help doctors make faster, more accurate diagnoses, leading to better patient outcomes. However, there will also be concerns about privacy and data security.
    
    2. Integration of AI into everyday life: AI is already being integrated into many areas of everyday life, from personal assistants to self-driving cars. As AI continues to improve, it is likely that this trend will continue, leading to a more connected, intelligent world


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

    ],

     and

     I

    'm

     a

     [

    Career

    ]

     who

     has

     always

     been

     passionate

     about

     [

    Your

     Specialty

    /

    Interest

    ],

     despite

     having

     a

     few

     minor

     setbacks

     along

     the

     way

    .

     Whether

     it

    's

     in

     the

     office

    ,

     the

     office

    ,

     or

     in

     the

     office

    ,

     I

    'm

     always

     up

     for

     learning

     and

     expanding

     my

     skills

    .

     I

     enjoy

     being

     proactive

     and

     creative

     in

     my

     approach

     to

     problem

    -solving

    ,

     and

     I

     strive

     to

     be

     a

     role

     model

     for

     others

     who

     face

     challenges

    .

     If

     you

    're

     interested

    ,

     I

    'd

     love

     to

     learn

     more

     about

     you

     and

     how

     I

     can

     potentially

     be

     of

     help

     to

     you

    .

     [

    Add

     a

     brief

     introduction

     or

     quote

     about

     your

     journey

     or

     experiences

     if

     desired

    ]


    Good

     afternoon

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     the

     city

     of

     love

     and

     fashion

    ,

     is

     known

     for

     its

     historic

     landmarks

    ,

     vibrant

     arts

     scene

    ,

     and

     iconic

     E

    iff

    el

     Tower

    .

     Its

     population

     is

     approximately

     

    2

    .

    7

     million

     people

    ,

     making

     it

     the

     most

     populous

     city

     in

     France

    .

     The

     city

    's

     architecture

    ,

     cuisine

    ,

     and

     culture

     are

     deeply

     ingr

    ained

     in

     its

     history

     and

     culture

    ,

     making

     Paris

     a

     globally

     recognized

     city

    .

     The

     capital

     of

     France

     is

     renowned

     for

     its

     rich

     cultural

     and

     historical

     heritage

    ,

     with

     its

     unique

     architectural

     style

     and

     stunning

     sights

    ,

     making

     it

     a

     must

    -

    visit

     destination

     for

     anyone

     interested

     in

     French

     culture

     and

     history

    .

     Paris

     is

     also

     known

     for

     its

     fashion

     industry

     and

     its

     status

     as

     the

     world

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     promises

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

     Advanced

     Machine

     Learning

    :

     With

     the

     development

     of

     machine

     learning

    ,

     AI

     will

     become

     more

     capable

     of

     learning

     and

     understanding

     complex

     patterns

     and

     behaviors

    .

     This

     will

     lead

     to

     more

     accurate

     and

     efficient

     predictions

    ,

     decision

    -making

    ,

     and

     automation

    .
    


    2

    .

     Artificial

     General

     Intelligence

     (

    AG

    I

    ):

     AG

    I

     is

     the

     theory

     that

     machines

     can

     fully

     understand

     and

     think

     like

     humans

    .

     It

     is

     currently

     a

     theoretical

     concept

    ,

     but

     research

     in

     this

     area

     is

     growing

     rapidly

    .

     AG

    I

     could

     lead

     to

     AI

     systems

     that

     can

     perform

     a

     wide

     range

     of

     tasks

    ,

     from

     decision

    -making

     to

     creative

     problem

    -solving

    .
    


    3

    .

     Autonomous

     vehicles

    :

     With

    



```python
llm.shutdown()
```
