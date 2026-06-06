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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.93it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:08,  1.24s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:08,  1.24s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:08,  1.24s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:16,  3.11it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:16,  3.11it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:16,  3.11it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:16,  3.11it/s]

    Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:16,  3.11it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:08,  5.75it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:08,  5.75it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:08,  5.75it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:08,  5.75it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:08,  5.75it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:08,  5.75it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:04,  9.83it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:04,  9.83it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:04,  9.83it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:04,  9.83it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:04,  9.83it/s]

    Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:05<00:04,  9.83it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 14.52it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 14.52it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 14.52it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 14.52it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 14.52it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:02, 14.52it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:02, 14.52it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:09<00:09,  3.04it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:09<00:09,  3.04it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:09<00:09,  3.04it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:09<00:09,  3.04it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:09<00:09,  3.04it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:09<00:09,  3.04it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:09<00:09,  3.04it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:09<00:09,  3.04it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:09<00:04,  4.94it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:09<00:04,  4.94it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:09<00:04,  4.94it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:09<00:04,  4.94it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:09<00:04,  4.94it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:09<00:04,  4.94it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:09<00:04,  4.94it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:09<00:02,  7.09it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:09<00:02,  7.09it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:09<00:02,  7.09it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:09<00:02,  7.09it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:09<00:02,  7.09it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:09<00:02,  7.09it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:09<00:02,  7.09it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:09<00:02,  7.09it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:09<00:00, 10.39it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:09<00:00, 10.39it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:09<00:00, 10.39it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:09<00:00, 10.39it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:09<00:00, 10.39it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:09<00:00, 10.39it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:09<00:00, 10.39it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:09<00:00, 10.39it/s]

    Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:09<00:00, 14.56it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:09<00:00, 14.56it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:09<00:00, 14.56it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:09<00:00, 14.56it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  5.82it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.68 GB):   2%|▏         | 1/58 [00:00<00:06,  9.08it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.65 GB):   2%|▏         | 1/58 [00:00<00:06,  9.08it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.65 GB):   3%|▎         | 2/58 [00:00<00:06,  8.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.65 GB):   3%|▎         | 2/58 [00:00<00:06,  8.99it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.65 GB):   3%|▎         | 2/58 [00:00<00:06,  8.99it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.65 GB):   7%|▋         | 4/58 [00:00<00:04, 11.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.65 GB):   7%|▋         | 4/58 [00:00<00:04, 11.07it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.64 GB):   7%|▋         | 4/58 [00:00<00:04, 11.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.64 GB):  10%|█         | 6/58 [00:00<00:03, 13.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.63 GB):  10%|█         | 6/58 [00:00<00:03, 13.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.63 GB):  10%|█         | 6/58 [00:00<00:03, 13.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.63 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.96it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.63 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.96it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=55.62 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.62 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.62 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.61 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.61 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.61 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.76it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.61 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.76it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=55.60 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.76it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.60 GB):  22%|██▏       | 13/58 [00:00<00:02, 18.76it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.60 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.60 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.59 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.59 GB):  28%|██▊       | 16/58 [00:01<00:02, 20.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.59 GB):  33%|███▎      | 19/58 [00:01<00:01, 23.13it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.59 GB):  33%|███▎      | 19/58 [00:01<00:01, 23.13it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=55.57 GB):  33%|███▎      | 19/58 [00:01<00:01, 23.13it/s]Capturing num tokens (num_tokens=960 avail_mem=55.58 GB):  33%|███▎      | 19/58 [00:01<00:01, 23.13it/s] Capturing num tokens (num_tokens=896 avail_mem=55.58 GB):  33%|███▎      | 19/58 [00:01<00:01, 23.13it/s]Capturing num tokens (num_tokens=896 avail_mem=55.58 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.19it/s]Capturing num tokens (num_tokens=832 avail_mem=55.58 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.19it/s]Capturing num tokens (num_tokens=768 avail_mem=55.58 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.19it/s]Capturing num tokens (num_tokens=704 avail_mem=55.57 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.19it/s]Capturing num tokens (num_tokens=640 avail_mem=55.57 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.19it/s]

    Capturing num tokens (num_tokens=640 avail_mem=55.57 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.51it/s]Capturing num tokens (num_tokens=576 avail_mem=55.57 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.51it/s]Capturing num tokens (num_tokens=512 avail_mem=55.55 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.51it/s]Capturing num tokens (num_tokens=480 avail_mem=55.57 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.51it/s]Capturing num tokens (num_tokens=448 avail_mem=55.57 GB):  47%|████▋     | 27/58 [00:01<00:01, 28.51it/s]Capturing num tokens (num_tokens=448 avail_mem=55.57 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.08it/s]Capturing num tokens (num_tokens=416 avail_mem=55.57 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.08it/s]Capturing num tokens (num_tokens=384 avail_mem=55.56 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.08it/s]Capturing num tokens (num_tokens=352 avail_mem=55.56 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.08it/s]

    Capturing num tokens (num_tokens=320 avail_mem=55.55 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.08it/s]Capturing num tokens (num_tokens=320 avail_mem=55.55 GB):  60%|██████    | 35/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=288 avail_mem=55.55 GB):  60%|██████    | 35/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=256 avail_mem=55.55 GB):  60%|██████    | 35/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=240 avail_mem=55.54 GB):  60%|██████    | 35/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=224 avail_mem=55.54 GB):  60%|██████    | 35/58 [00:01<00:00, 31.49it/s]Capturing num tokens (num_tokens=224 avail_mem=55.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=208 avail_mem=55.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=192 avail_mem=55.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.28it/s]

    Capturing num tokens (num_tokens=176 avail_mem=55.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=160 avail_mem=55.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=160 avail_mem=55.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.96it/s]Capturing num tokens (num_tokens=144 avail_mem=55.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.96it/s]Capturing num tokens (num_tokens=128 avail_mem=55.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.96it/s]Capturing num tokens (num_tokens=112 avail_mem=55.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.96it/s]Capturing num tokens (num_tokens=96 avail_mem=55.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.96it/s] Capturing num tokens (num_tokens=96 avail_mem=55.52 GB):  81%|████████  | 47/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=80 avail_mem=55.51 GB):  81%|████████  | 47/58 [00:01<00:00, 33.12it/s]

    Capturing num tokens (num_tokens=64 avail_mem=55.51 GB):  81%|████████  | 47/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=48 avail_mem=55.51 GB):  81%|████████  | 47/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=32 avail_mem=55.50 GB):  81%|████████  | 47/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=32 avail_mem=55.50 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.45it/s]Capturing num tokens (num_tokens=28 avail_mem=55.50 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.45it/s]Capturing num tokens (num_tokens=24 avail_mem=55.50 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.45it/s]Capturing num tokens (num_tokens=20 avail_mem=55.49 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.45it/s]Capturing num tokens (num_tokens=16 avail_mem=55.49 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.45it/s]

    Capturing num tokens (num_tokens=16 avail_mem=55.49 GB):  95%|█████████▍| 55/58 [00:02<00:00, 33.88it/s]Capturing num tokens (num_tokens=12 avail_mem=55.49 GB):  95%|█████████▍| 55/58 [00:02<00:00, 33.88it/s]Capturing num tokens (num_tokens=8 avail_mem=55.49 GB):  95%|█████████▍| 55/58 [00:02<00:00, 33.88it/s] Capturing num tokens (num_tokens=4 avail_mem=55.48 GB):  95%|█████████▍| 55/58 [00:02<00:00, 33.88it/s]Capturing num tokens (num_tokens=4 avail_mem=55.48 GB): 100%|██████████| 58/58 [00:02<00:00, 26.28it/s]


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
    Generated text:  Jill and I am a newbie to this. I have a question for you, I got my second job, and now I'm teaching myself to play piano with a friend, is it possible to get paid for this? And if so, how can I do this?
    It's great to hear that you're taking the initiative to learn to play piano. While teaching yourself can certainly be a rewarding and educational experience, getting paid for your piano lessons can sometimes be challenging. Here are a few things you might consider:
    
    ### 1. **Try Online Platforms:**
       - **Khan Academy:** Khan Academy offers online courses that focus on
    ===============================
    Prompt: The president of the United States is
    Generated text:  to be elected by a majority of the popular vote in a specific state. If 20 states have a majority of 25%, 15 states have a majority of 40%, and 10 states have a majority of 50%, what is the smallest possible percentage of states in which the president will be elected by a majority of 5%?
    To determine the smallest possible percentage of states in which the president will be elected by a majority of 5%, we need to consider the current distribution of majority percentages in each state and the maximum possible election percentage that could be achieved under those conditions.
    
    Given:
    -
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a city in France, and it is one of the most famous and oldest cities in the world. It is situated on the right bank of the Seine river, in the heart of Paris. The Seine river runs through the heart of Paris.
    Its capital was established by Charles VII in 1484, which was then known as the “City of Kings”. The city is also famous for having the world’s oldest stone church in Paris, Sainte-Chapelle. The city is also famous for its museums, parks, and tourist sites.
    Why is Paris so special?
    Paris is one of the most
    ===============================
    Prompt: The future of AI is
    Generated text:  already here
    
    AI is the future. AI is here, and it is here to stay. AI is not a hot topic anymore, it is a point of reference for many organizations, companies, and experts in the field. Even though we are not seeing a sudden and sudden increase in the adoption of AI, it is already here. Let us go through the history and trends of AI to understand how it is here.
    
    The history of AI
    
    There is a lot of discussion about the history of AI. From the days of ancient Greece, to the 1940s, and the 1960s, we have


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a cultural and economic hub, known for its rich history, art, and cuisine. It is a popular tourist destination and a major economic and political center in Europe. The city is also home to the French Riviera, a popular tourist destination for its beaches and luxury resorts. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant culture. It is a city that has been a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars to personalized medicine. Additionally, AI is likely to continue to be used for tasks such as fraud detection, security, and autonomous decision-making. As AI becomes more integrated into our daily lives, we can expect to see more widespread adoption of AI in various industries and applications. However, there are also potential risks and challenges associated with the use of AI, such as job displacement and ethical concerns. It is important to carefully consider the
    


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
    Generated text:  [Name] and I'm a [role]! I come from [ hometown ] and I'm [ age ]. I'm passionate about [thing to share about your profession]. I enjoy [reason for love/hobby], and I'm [degree], and I'm [favorite hobby]. I'm currently [current status], and I'm [name of current project]. I'm always looking for opportunities to learn and grow, and I'm always ready to embrace new experiences. I'm a [personality]! I love [job/role]. I've always been [personality]. I'm a [ personality ]! I'm [ personality
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris, France's largest city and capital, is known for its rich history, famous landmarks, and diverse cultural scene. It is a melting pot of various ethnicities and traditions, including French, English, Italian, and Spanish. It is often referred to as the "City of Lights" for its beautiful skyline and charming boulevards. The city is also famous for its food, music, and fashion. Paris is one of the most popular tourist destinations in the world, and it continues to be a major center of European culture and diplomacy. It's a hub for many important events, including the Eiffel Tower and the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  diverse and uncertain, with many possibilities ranging from increasing automation and convenience to more personalized and ethical applications. Here are some potential trends that could emerge in the coming years:
    
    1. Increased automation and convenience: AI is already making a significant impact in various industries, including transportation, healthcare, and manufacturing. As the technology continues to evolve, we can expect to see even more automation and convenience in our daily lives.
    
    2. Personalized AI: As AI becomes more advanced, we can expect to see a rise in personalized AI. This means that we will be able to create more tailored and relevant AI systems that can adapt to individual needs and preferences.
    
    


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

     am

     a

     [

    insert

     your

     profession

     or

     role

    ].

     I

     am

     an

     [

    insert

     your

     profession

     or

     role

    ]

     with

     [

    insert

     a

     couple

     of

     sentences

     describing

     your

     background

    ,

     skills

    ,

     or

     experiences

    ]

     that

     have

     made

     me

     an

     asset

     to

     my

     organization

    .

     I

     enjoy

     [

    insert

     a

     few

     things

     you

     like

     about

     yourself

    ]

     and

     I

     am

     a

     [

    insert

     a

     few

     things

     you

     are

     passionate

     about

    ].

     I

     am

     always

     looking

     for

     opportunities

     to

     [

    insert

     a

     few

     things

     you

     are

     looking

     for

     in

     a

     job

    ],

     because

     I

     want

     to

     make

     a

     difference

     and

     bring

     about

     change

    .

     And

     I

     am

     always

     learning

     new

     things

     and

     trying

     to

     grow

     as

     a

     person

    .

     Thank

     you

     for

     asking

    !

     May

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     renowned

     for

     its

     exquisite

     art

     and

     architecture

    ,

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

     and

     the

     E

    iff

    el

     Tower

    ,

     and

     a

     diverse

     and

     vibrant

     cultural

     scene

    .
    


    Paris

    ,

     the

     cultural

     and

     artistic

     capital

     of

     France

    ,

     is

     renowned

     for

     its

     rich

     history

    ,

     exquisite

     art

     and

     architecture

    ,

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

     and

     the

     E

    iff

    el

     Tower

    ,

     and

     a

     vibrant

     and

     diverse

     cultural

     scene

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

     Middle

     Ages

     and

     is

     home

     to

     several

     UNESCO

     World

     Heritage

     Sites

    ,

     including

     the

     Lou

    vre

     Museum

     and

     the

     Palace

     of

     Vers

    ailles

    .

     The

     city

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

     music

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     diverse

     range

     of

     trends

     that

     will

     shape

     the

     way

     that

     humans

     and

     machines

     interact

     with

     each

     other

     and

     the

     world

     around

     them

    .

     Here

     are

     some

     possible

     trends

     that

     may

     be

     on

     the

     horizon

    :
    


    1

    .

     Increased

     efficiency

     and

     effectiveness

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     it

     is

     expected

     that

     it

     will

     become

     more

     efficient

     and

     effective

     at

     performing

     tasks

    ,

     such

     as

     decision

    -making

     and

     problem

    -solving

    .

     This

     will

     lead

     to

     greater

     productivity

     and

     reduced

     costs

     for

     businesses

     and

     organizations

    .
    


    2

    .

     Enhanced

     human

    -A

    I

     collaboration

    :

     As

     AI

     becomes

     more

     integrated

     into

     daily

     life

    ,

     it

     is

     possible

     that

     human

    -A

    I

     collaboration

     will

     become

     more

     common

     and

     effective

    .

     This

     could

     lead

     to

    



```python
llm.shutdown()
```
