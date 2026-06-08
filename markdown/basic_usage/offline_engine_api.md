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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.24it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.24it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:16,  3.06it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:16,  3.06it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:16,  3.06it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:16,  3.06it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:09,  4.94it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:09,  4.94it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:09,  4.94it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:09,  4.94it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:09,  4.94it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:05,  8.14it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:05,  8.14it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:05,  8.14it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:05,  8.14it/s]

    Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:05,  8.14it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:03, 11.80it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:03, 11.80it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:03, 11.80it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:03, 11.80it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:03, 11.80it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:03, 11.80it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:02, 16.93it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:02, 16.93it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:02, 16.93it/s]

    Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:02, 16.93it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:02, 16.93it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 20.28it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 27.06it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 27.06it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 27.06it/s]

    Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 27.06it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 27.06it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:00, 27.06it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 31.68it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 31.68it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 31.68it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 31.68it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 31.68it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 31.68it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 35.57it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 35.57it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 35.57it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 35.57it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 35.57it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 35.57it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:06<00:00, 38.04it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:06<00:00, 38.04it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:06<00:00, 38.04it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:06<00:00, 38.04it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:06<00:00, 38.04it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:06<00:00, 38.04it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:06<00:00, 38.04it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:06<00:00, 41.98it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:06<00:00, 41.98it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 41.98it/s] 

    Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 41.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.12 GB):   3%|▎         | 2/58 [00:00<00:04, 11.42it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.12 GB):   3%|▎         | 2/58 [00:00<00:04, 11.42it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=57.14 GB):   3%|▎         | 2/58 [00:00<00:04, 11.42it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.14 GB):   7%|▋         | 4/58 [00:00<00:04, 13.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.14 GB):   7%|▋         | 4/58 [00:00<00:04, 13.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.14 GB):   7%|▋         | 4/58 [00:00<00:04, 13.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.14 GB):  10%|█         | 6/58 [00:00<00:03, 15.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.13 GB):  10%|█         | 6/58 [00:00<00:03, 15.16it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=57.13 GB):  10%|█         | 6/58 [00:00<00:03, 15.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.13 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.13 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.39it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.12 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.39it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.15 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.39it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.15 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.15 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.70it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=57.15 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.15 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.15 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.14 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.13 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.99it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.12 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.99it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.38it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=57.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.38it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.38it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.08 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=960 avail_mem=57.09 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.79it/s] Capturing num tokens (num_tokens=896 avail_mem=57.09 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.79it/s]Capturing num tokens (num_tokens=832 avail_mem=57.09 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.79it/s]Capturing num tokens (num_tokens=768 avail_mem=57.08 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.79it/s]

    Capturing num tokens (num_tokens=768 avail_mem=57.08 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.79it/s]Capturing num tokens (num_tokens=704 avail_mem=57.07 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.79it/s]Capturing num tokens (num_tokens=640 avail_mem=57.07 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.79it/s]Capturing num tokens (num_tokens=576 avail_mem=57.06 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.79it/s]Capturing num tokens (num_tokens=512 avail_mem=57.04 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.79it/s]Capturing num tokens (num_tokens=512 avail_mem=57.04 GB):  50%|█████     | 29/58 [00:01<00:00, 29.94it/s]Capturing num tokens (num_tokens=480 avail_mem=57.06 GB):  50%|█████     | 29/58 [00:01<00:00, 29.94it/s]Capturing num tokens (num_tokens=448 avail_mem=57.05 GB):  50%|█████     | 29/58 [00:01<00:00, 29.94it/s]Capturing num tokens (num_tokens=416 avail_mem=57.05 GB):  50%|█████     | 29/58 [00:01<00:00, 29.94it/s]

    Capturing num tokens (num_tokens=384 avail_mem=57.04 GB):  50%|█████     | 29/58 [00:01<00:00, 29.94it/s]Capturing num tokens (num_tokens=384 avail_mem=57.04 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.97it/s]Capturing num tokens (num_tokens=352 avail_mem=57.03 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.97it/s]Capturing num tokens (num_tokens=320 avail_mem=57.02 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.97it/s]Capturing num tokens (num_tokens=288 avail_mem=57.02 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.97it/s]Capturing num tokens (num_tokens=256 avail_mem=57.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.97it/s]Capturing num tokens (num_tokens=256 avail_mem=57.01 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=240 avail_mem=57.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=224 avail_mem=57.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=208 avail_mem=56.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.40it/s]

    Capturing num tokens (num_tokens=192 avail_mem=56.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=176 avail_mem=56.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=176 avail_mem=56.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=160 avail_mem=56.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=144 avail_mem=56.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=128 avail_mem=56.94 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=112 avail_mem=56.70 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.17it/s]Capturing num tokens (num_tokens=112 avail_mem=56.70 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.18it/s]Capturing num tokens (num_tokens=96 avail_mem=56.69 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.18it/s] Capturing num tokens (num_tokens=80 avail_mem=56.69 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.18it/s]

    Capturing num tokens (num_tokens=64 avail_mem=56.69 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.18it/s]Capturing num tokens (num_tokens=48 avail_mem=56.68 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.18it/s]Capturing num tokens (num_tokens=32 avail_mem=56.68 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.18it/s]Capturing num tokens (num_tokens=32 avail_mem=56.68 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.82it/s]Capturing num tokens (num_tokens=28 avail_mem=56.67 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.82it/s]Capturing num tokens (num_tokens=24 avail_mem=56.67 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.82it/s]Capturing num tokens (num_tokens=20 avail_mem=56.67 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.82it/s]Capturing num tokens (num_tokens=16 avail_mem=56.67 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.82it/s]Capturing num tokens (num_tokens=12 avail_mem=56.66 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.82it/s]Capturing num tokens (num_tokens=12 avail_mem=56.66 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=8 avail_mem=56.66 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.69it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=56.66 GB):  97%|█████████▋| 56/58 [00:01<00:00, 39.69it/s]Capturing num tokens (num_tokens=4 avail_mem=56.66 GB): 100%|██████████| 58/58 [00:01<00:00, 29.25it/s]


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
    Generated text:  Mary. I'm a kind and outgoing girl. I am in Grade 8. My favorite subject is English. And my favorite food is hamburgers. I like playing the guitar and singing very much. What do you think of Mary? I think Mary is very kind and outgoing. She always tries to help others. She can play the guitar very well and she can sing very well. And she doesn't like hamburgers very much. What's Mary's favorite subject? A) English B) Music C) Science D) Art
    A:
    Mary's favorite subject is English. 
    
    The question does not explicitly state Mary's favorite subject
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. Some people think the president has more power than the president of the United States. Other people think the president of the United States has more power than the president of the United States. 
    
    Write a short summary of the two main points of the two different opinions about who is the most powerful president. The president of the United States is important, and some people think the president of the United States has more power than the president of the United States. The president of the United States is the person in charge of the country and the president of the United States has more power than the president of the United States. The president
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 正确
    B. 错误
    答案:
    A
    
    客户张三和李四将自己所有的房产以200万元的价格转让给赵五。赵五在对房产进行产权登记之前，就将房产折旧登记在自己名下。在张三和李四将房产卖给赵五的情况下，赵五是否可以对张三和李四的房产享有优先购买权？（将房产折旧登记在自己名下不属于对房产的合法处分）
    A. 正确
    B. 错误
    答案:
    B
    
    肝昏迷患者在进食
    ===============================
    Prompt: The future of AI is
    Generated text:  a future of continuous development, which is also a future of continuous testing. The development of AI technology has reached a point where it is currently in the infancy stage, and its development level is still not satisfactory. Its shortcomings include ___.
    A. Incomplete theoretical foundation
    B. Poor hardware environment
    C. Narrow application field
    D. Incomplete industry support
    Answer:
    A
    
    The different combinations of attitudes towards risk among individuals can affect their risk-taking behaviors. This illustrates that risk-taking behavior belongs to ____.
    A. Intrinsic Risk Behavior
    B. Transactional Risk Behavior
    C. Social Risk Behavior
    D. Behavioral Risk Behavior


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I'm [insert a short description of your personality or background]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always looking for new challenges and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of French Revolution and Napoleon Bonaparte, and its influence on modern French culture and politics. The city is also known for its diverse cuisine, including French cuisine, and its role in hosting the 2012 Summer Olympics. Paris is a popular tourist destination, attracting millions of visitors each year. The city is also home to many international
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient monitoring.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve risk management, fraud detection, and portfolio optimization. As AI technology continues to
    


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
    Generated text:  [Name], and I am a [Type] who is [Status]. [Name] is a [job title] with over [number] of [job role]. I've always been passionate about [why you're passionate], and I'm always up for [what you're up for]. I enjoy [why you like [the subject]], and I strive to [what you do to make people happy]. I'm [how you would describe yourself] and I love [why you love what you do]. I have a [number] of [experience] of [what you've done]. I'm always looking for new challenges and opportunities
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement succinctly captures the fact that Paris serves as the primary city in the French administrative system and is considered to be the most important cultural, political, and economic center of France. While other cities in France are significant, Paris remains the capital due to its historical, cultural, and architectural significance, as well as its status as the national capital of France. The statement is accurate and provides a clear understanding of the importance of Paris as the capital of France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly unpredictable and will likely evolve in ways we can never fully predict. Here are some potential trends that could shape the future of AI:
    
    1. Artificial Intelligence will become more pervasive in our daily lives, from autonomous vehicles to chatbots and virtual assistants.
    
    2. AI will become more personalized and effective at predicting and adapting to human behavior.
    
    3. AI will become more sophisticated at simulating human decision-making and problem-solving.
    
    4. AI will become more integrated into human work processes, such as in manufacturing and healthcare.
    
    5. AI will become more ethical and considerate, with efforts to ensure that AI systems are developed and used in a way


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

    Age

    ]

     year

     old

    ,

     [

    Prof

    ession

    ]

     person

    ,

     and

     I

     recently

     moved

     to

     [

    City

    ]

     to

     pursue

     my

     [

    Professional

     Goal

    ].

     I

    'm

     an

     [

    Ext

    ro

    verted

     or

     Intro

    verted

    ]

     personality

    ,

     with

     a

     [

    特长

    ]

     that

     sets

     me

     apart

     from

     others

    ,

     and

     I

    'm

     [

    Positive

     or

     Negative

    ]

     person

    .

     I

    'm

     a

     [

    loy

    al

     or

     Independent

    ]

     person

    ,

     with

     a

     [

    Interest

    /

    Att

    itude

    ]

     that

     I

    'm

     passionate

     about

    ,

     and

     I

    'm

     [

    Positive

     or

     Negative

    ]

     about

     my

     current

     situation

     and

     my

     future

    .

     I

    'm

     [

    Friendly

     or

     Int

    imid

    ating

    ]

     and

     am

     known

     for

     my

     [

    Strong

     or

     Weak

    ]

    
    
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

     Europe

     by

     area

     and

     population

    ,

     with

     an

     estimated

     population

     of

     over

     

    1

    .

    7

     million

     people

    .

     Paris

     is

     known

     for

     its

     artistic

    ,

     architectural

    ,

     and

     cultural

     influences

    ,

     including

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

     The

     city

     has

     a

     rich

     history

     dating

     back

     to

     the

     ancient

     Roman

     Empire

     and

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     landmarks

     that

     highlight

     its

     cultural

     and

     artistic

     heritage

    .

     The

     city

     is

     also

     known

     for

     its

     fashion

    ,

     nightlife

    ,

     and

     food

     scene

    ,

     and

     is

     a

     major

     hub

     of

     business

     and

     commerce

     in

     the

     country

    .

     Paris

     is

     a

     bustling

     city

     with

     a

     rich

     cultural

     and

     historical

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     anticipated

    ,

     and

     there

     are

     several

     potential

     trends

     that

     are

     likely

     to

     shape

     how

     it

     will

     evolve

    .

     Some

     of

     the

     most

     notable

     trends

     include

    :
    


    1

    .

     Personal

    ization

    :

     AI

     is

     becoming

     more

     personal

     and

     tailored

     to

     the

     individual

     user

    ,

     leading

     to

     the

     development

     of

     more

     sophisticated

     algorithms

     that

     can

     understand

     and

     process

     complex

     human

     experiences

    .

     This

     trend

     will

     continue

     to

     drive

     the

     development

     of

     more

     personalized

     and

     context

    -aware

     AI

     systems

    .
    


    2

    .

     Rob

    otic

     augmentation

    :

     AI

     is

     already

     being

     used

     to

     augment

     humans

     in

     a

     variety

     of

     ways

    ,

     such

     as

     through

     machine

     learning

     algorithms

     that

     can

     perform

     tasks

     that

     are

     difficult

     or

     impossible

     for

     humans

     to

     do

    .

     In

     the

     future

    ,

     we

     may

     see

     more

     widespread

     use

    



```python
llm.shutdown()
```
