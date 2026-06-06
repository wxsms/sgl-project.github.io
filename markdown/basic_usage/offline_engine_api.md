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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.23it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:46,  1.15it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:15,  3.22it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:15,  3.22it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:15,  3.22it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:15,  3.22it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:15,  3.22it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:15,  3.22it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:15,  3.22it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:15,  3.22it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:04<00:15,  3.22it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:05,  7.59it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:05,  7.59it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:05,  7.59it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:05,  7.59it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:05,  7.59it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:04<00:05,  7.59it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:04<00:05,  7.59it/s]Compiling num tokens (num_tokens=832):  29%|██▉       | 17/58 [00:04<00:05,  7.59it/s]Compiling num tokens (num_tokens=768):  29%|██▉       | 17/58 [00:04<00:05,  7.59it/s]

    Compiling num tokens (num_tokens=704):  29%|██▉       | 17/58 [00:04<00:05,  7.59it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=320):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:01, 20.83it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:01, 20.83it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:01, 20.83it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 20.83it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 20.83it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 20.83it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 20.83it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 20.83it/s]

    Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:01, 20.83it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:05<00:01, 20.83it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 29.08it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:05<00:00, 29.08it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 37.89it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 37.89it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 37.89it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 37.89it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 37.89it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 37.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.30 GB):   3%|▎         | 2/58 [00:00<00:04, 11.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.29 GB):   3%|▎         | 2/58 [00:00<00:04, 11.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.29 GB):   3%|▎         | 2/58 [00:00<00:04, 11.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.29 GB):   7%|▋         | 4/58 [00:00<00:04, 11.00it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.29 GB):   7%|▋         | 4/58 [00:00<00:04, 11.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.29 GB):   7%|▋         | 4/58 [00:00<00:04, 11.00it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=53.29 GB):  10%|█         | 6/58 [00:00<00:04, 11.52it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.28 GB):  10%|█         | 6/58 [00:00<00:04, 11.52it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.27 GB):  10%|█         | 6/58 [00:00<00:04, 11.52it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.27 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.27 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.27 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.73it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=53.27 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.26 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.42it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.26 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.42it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.26 GB):  21%|██        | 12/58 [00:00<00:03, 13.47it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.25 GB):  21%|██        | 12/58 [00:00<00:03, 13.47it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=53.25 GB):  21%|██        | 12/58 [00:01<00:03, 13.47it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.25 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.15it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.24 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.15it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.24 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.15it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=53.24 GB):  28%|██▊       | 16/58 [00:01<00:03, 13.84it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.24 GB):  28%|██▊       | 16/58 [00:01<00:03, 13.84it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.23 GB):  28%|██▊       | 16/58 [00:01<00:03, 13.84it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.23 GB):  31%|███       | 18/58 [00:01<00:02, 14.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.23 GB):  31%|███       | 18/58 [00:01<00:02, 14.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.23 GB):  31%|███       | 18/58 [00:01<00:02, 14.40it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=53.23 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.21 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.51it/s]Capturing num tokens (num_tokens=960 avail_mem=53.23 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.51it/s] Capturing num tokens (num_tokens=896 avail_mem=53.22 GB):  34%|███▍      | 20/58 [00:01<00:02, 15.51it/s]Capturing num tokens (num_tokens=896 avail_mem=53.22 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.56it/s]Capturing num tokens (num_tokens=832 avail_mem=53.22 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.56it/s]Capturing num tokens (num_tokens=768 avail_mem=53.22 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.56it/s]Capturing num tokens (num_tokens=704 avail_mem=53.21 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.56it/s]

    Capturing num tokens (num_tokens=704 avail_mem=53.21 GB):  45%|████▍     | 26/58 [00:01<00:01, 20.59it/s]Capturing num tokens (num_tokens=640 avail_mem=53.21 GB):  45%|████▍     | 26/58 [00:01<00:01, 20.59it/s]Capturing num tokens (num_tokens=576 avail_mem=53.21 GB):  45%|████▍     | 26/58 [00:01<00:01, 20.59it/s]Capturing num tokens (num_tokens=512 avail_mem=53.19 GB):  45%|████▍     | 26/58 [00:01<00:01, 20.59it/s]Capturing num tokens (num_tokens=512 avail_mem=53.19 GB):  50%|█████     | 29/58 [00:01<00:01, 20.78it/s]Capturing num tokens (num_tokens=480 avail_mem=52.90 GB):  50%|█████     | 29/58 [00:01<00:01, 20.78it/s]

    Capturing num tokens (num_tokens=448 avail_mem=53.18 GB):  50%|█████     | 29/58 [00:01<00:01, 20.78it/s]Capturing num tokens (num_tokens=416 avail_mem=53.17 GB):  50%|█████     | 29/58 [00:01<00:01, 20.78it/s]Capturing num tokens (num_tokens=416 avail_mem=53.17 GB):  55%|█████▌    | 32/58 [00:02<00:01, 18.75it/s]Capturing num tokens (num_tokens=384 avail_mem=53.17 GB):  55%|█████▌    | 32/58 [00:02<00:01, 18.75it/s]Capturing num tokens (num_tokens=352 avail_mem=52.94 GB):  55%|█████▌    | 32/58 [00:02<00:01, 18.75it/s]

    Capturing num tokens (num_tokens=352 avail_mem=52.94 GB):  59%|█████▊    | 34/58 [00:02<00:01, 16.75it/s]Capturing num tokens (num_tokens=320 avail_mem=53.15 GB):  59%|█████▊    | 34/58 [00:02<00:01, 16.75it/s]Capturing num tokens (num_tokens=288 avail_mem=53.15 GB):  59%|█████▊    | 34/58 [00:02<00:01, 16.75it/s]

    Capturing num tokens (num_tokens=288 avail_mem=53.15 GB):  62%|██████▏   | 36/58 [00:02<00:01, 14.18it/s]Capturing num tokens (num_tokens=256 avail_mem=53.14 GB):  62%|██████▏   | 36/58 [00:02<00:01, 14.18it/s]Capturing num tokens (num_tokens=240 avail_mem=53.13 GB):  62%|██████▏   | 36/58 [00:02<00:01, 14.18it/s]Capturing num tokens (num_tokens=240 avail_mem=53.13 GB):  66%|██████▌   | 38/58 [00:02<00:01, 13.06it/s]Capturing num tokens (num_tokens=224 avail_mem=53.12 GB):  66%|██████▌   | 38/58 [00:02<00:01, 13.06it/s]

    Capturing num tokens (num_tokens=208 avail_mem=53.12 GB):  66%|██████▌   | 38/58 [00:02<00:01, 13.06it/s]Capturing num tokens (num_tokens=208 avail_mem=53.12 GB):  69%|██████▉   | 40/58 [00:02<00:01, 12.74it/s]Capturing num tokens (num_tokens=192 avail_mem=53.11 GB):  69%|██████▉   | 40/58 [00:02<00:01, 12.74it/s]Capturing num tokens (num_tokens=176 avail_mem=52.99 GB):  69%|██████▉   | 40/58 [00:02<00:01, 12.74it/s]

    Capturing num tokens (num_tokens=176 avail_mem=52.99 GB):  72%|███████▏  | 42/58 [00:02<00:01, 13.10it/s]Capturing num tokens (num_tokens=160 avail_mem=53.10 GB):  72%|███████▏  | 42/58 [00:02<00:01, 13.10it/s]Capturing num tokens (num_tokens=144 avail_mem=53.09 GB):  72%|███████▏  | 42/58 [00:02<00:01, 13.10it/s]Capturing num tokens (num_tokens=144 avail_mem=53.09 GB):  76%|███████▌  | 44/58 [00:03<00:01, 13.16it/s]Capturing num tokens (num_tokens=128 avail_mem=53.09 GB):  76%|███████▌  | 44/58 [00:03<00:01, 13.16it/s]

    Capturing num tokens (num_tokens=112 avail_mem=53.08 GB):  76%|███████▌  | 44/58 [00:03<00:01, 13.16it/s]Capturing num tokens (num_tokens=112 avail_mem=53.08 GB):  79%|███████▉  | 46/58 [00:03<00:00, 13.46it/s]Capturing num tokens (num_tokens=96 avail_mem=53.08 GB):  79%|███████▉  | 46/58 [00:03<00:00, 13.46it/s] Capturing num tokens (num_tokens=80 avail_mem=53.07 GB):  79%|███████▉  | 46/58 [00:03<00:00, 13.46it/s]

    Capturing num tokens (num_tokens=80 avail_mem=53.07 GB):  83%|████████▎ | 48/58 [00:03<00:00, 13.94it/s]Capturing num tokens (num_tokens=64 avail_mem=53.07 GB):  83%|████████▎ | 48/58 [00:03<00:00, 13.94it/s]Capturing num tokens (num_tokens=48 avail_mem=53.06 GB):  83%|████████▎ | 48/58 [00:03<00:00, 13.94it/s]Capturing num tokens (num_tokens=48 avail_mem=53.06 GB):  86%|████████▌ | 50/58 [00:03<00:00, 15.29it/s]Capturing num tokens (num_tokens=32 avail_mem=53.05 GB):  86%|████████▌ | 50/58 [00:03<00:00, 15.29it/s]Capturing num tokens (num_tokens=28 avail_mem=53.03 GB):  86%|████████▌ | 50/58 [00:03<00:00, 15.29it/s]Capturing num tokens (num_tokens=24 avail_mem=53.04 GB):  86%|████████▌ | 50/58 [00:03<00:00, 15.29it/s]

    Capturing num tokens (num_tokens=24 avail_mem=53.04 GB):  91%|█████████▏| 53/58 [00:03<00:00, 17.47it/s]Capturing num tokens (num_tokens=20 avail_mem=53.03 GB):  91%|█████████▏| 53/58 [00:03<00:00, 17.47it/s]Capturing num tokens (num_tokens=16 avail_mem=53.03 GB):  91%|█████████▏| 53/58 [00:03<00:00, 17.47it/s]Capturing num tokens (num_tokens=12 avail_mem=53.02 GB):  91%|█████████▏| 53/58 [00:03<00:00, 17.47it/s]Capturing num tokens (num_tokens=12 avail_mem=53.02 GB):  97%|█████████▋| 56/58 [00:03<00:00, 19.26it/s]Capturing num tokens (num_tokens=8 avail_mem=53.02 GB):  97%|█████████▋| 56/58 [00:03<00:00, 19.26it/s] Capturing num tokens (num_tokens=4 avail_mem=53.01 GB):  97%|█████████▋| 56/58 [00:03<00:00, 19.26it/s]

    Capturing num tokens (num_tokens=4 avail_mem=53.01 GB): 100%|██████████| 58/58 [00:03<00:00, 15.49it/s]


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
    Generated text:  Daria. I am a student at the University of Southern Denmark. I started my university studies in October 2021. I was born in 1995. My home country is Denmark. I am interested in computers and programming languages. In my free time, I like to play basketball and watch sports. I also like playing with my toys. I enjoy spending time with my friends.
    What is your favorite programming language?
    Hello! As a computer science student at the University of Southern Denmark, I find myself drawn to programming languages that are both challenging and rewarding. Here's my favorite programming language:
    
    **C++**
    
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to use the word "win" or "winning" in an upcoming election. The president is considering the following four scenarios to test their theories.
    
    1. If the president uses "win," the probability of winning the election is 1/5, and the probability of winning with a 50/50 split is 1/10.
    2. If the president uses "winning," the probability of winning is 2/3, and the probability of winning with a 50/50 split is 1/2.
    3. If the president uses "win," the probability of
    ===============================
    Prompt: The capital of France is
    Generated text:  ________.
    A. Paris
    B. Geneva
    C. Brussels
    D. Lyon
    Answer: A
    
    Which of the following statements about the characteristics of Chinese opera is incorrect?
    A. The main color is black.
    B. The costume is mostly cotton.
    C. It has a strong flavor of realism.
    D. It has a high level of artistry.
    Answer: B
    
    When an organization decides to establish a new company, they need to compare it with similar companies and identify the unique strengths of their own organization. This type of analysis is known as ____
    A. Normative Analysis
    B. Causal Analysis
    C
    ===============================
    Prompt: The future of AI is
    Generated text:  shaping the way we live, work, and interact with each other. As AI continues to evolve, it will become increasingly important to understand the ethical implications of its use and the impact it has on society. One such area of concern is the potential misuse of AI in healthcare. AI is being used to improve patient care and treatment outcomes, but there is also a risk that it could lead to the exploitation of vulnerable populations.
    
    One example of this is the use of AI to detect diseases in real-time. This can be particularly useful in rural areas where access to medical professionals is limited. However, if the AI is not trained appropriately or if it


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your profession or role]. I enjoy [insert a short description of your hobbies or interests]. I'm [insert a short description of your personality or character]. Thank you for taking the time to meet me. I look forward to our conversation. [Name] [Company Name] [Date] [Name] [Company Name] [Date] [Name] [Company Name] [Date] [Name] [Company
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of many famous French artists, writers, and composers. Paris is a bustling metropolis with a rich cultural heritage, and it is a popular tourist destination for its beautiful architecture, delicious cuisine, and lively nightlife. The city is also home to the French Parliament, the French Parliament building, and the Eiffel Tower. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. Its unique blend of old and new has made it a popular
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased integration of AI into various industries: AI is already being used in a wide range of industries, from healthcare and finance to transportation and manufacturing. As AI becomes more integrated into these industries, we can expect to see even more applications of AI in the future.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, privacy, and transparency.
    
    3. Development
    


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
    Generated text:  Emily. I'm a busy, ambitious student working towards a degree in social work. I have a desire to help others, and I'm always looking to learn new things and expand my skills. I'm also a passionate advocate for social justice and believe that everyone deserves to have access to affordable housing and healthcare. I'm always looking to improve myself and make a positive impact in the world. And I'm always ready to connect with others and learn from them.
    
    I'm excited to meet you and contribute to the good things in the world. What kind of role are you looking to play in the social work community? Emily is excited to contribute
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    In this statement, please include the context: "Paris" refers to the capital city of France. Additionally, provide an interesting fact about Paris that would be relevant to someone who is visiting the city. 
    Here's an example response: 
    "Paris, the heart of French culture and gastronomy, is home to many iconic landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a popular tourist destination and a UNESCO World Heritage site." 
    
    Please provide a similar response for the next city you visit. 
    Also, please include an interesting fact about the city that would be
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and constantly evolving. Here are some of the potential trends we can expect to see in the near future:
    
    1. Increasing focus on ethical AI: As concerns about AI ethics and bias grow, the focus will shift towards developing AI that is more fair, transparent, and accountable. This may involve developing new ethical frameworks for AI development, using AI in ways that are more aligned with societal values, and implementing robust monitoring and accountability measures.
    
    2. AI that learns from feedback: With the rise of big data and machine learning, it's becoming increasingly possible to create AI that can learn from feedback and improve over time. This could include developing AI


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

    ].

     I

     am

     a

     [

    Your

     Profession

    ]

     who

     has

     been

     [

    Your

     Professional

     Background

     or

     Achie

    vements

    ]

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

     am

     passionate

     about

     [

    Your

     Personal

     Interest

     or

     Hobby

    ],

     and

     I

     enjoy

     [

    Your

     Mot

    ivation

     or

     Passion

    ].

     I

     am

     always

     eager

     to

     learn

     and

     expand

     my

     knowledge

    ,

     and

     I

     strive

     to

     always

     improve

     myself

    .

     I

     am

     confident

     and

     I

     believe

     in

     my

     own

     abilities

    .

     What

     is

     your

     profession

     or

     occupation

    ?

     What

     are

     your

     hobbies

     and

     interests

    ?

     How

     long

     have

     you

     been

     in

     your

     profession

     or

     achieved

     your

     goals

    ?

     What

     motiv

    ates

     you

     to

     continue

     learning

     and

     improving

    ?

     How

     do

     you

     feel

     about

     yourself

     and

     your

     achievements

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     answer

     is

     Paris

    .

     
    


    To

     elaborate

     on

     this

     statement

    :
    


    1

    .

     The

     city

     of

     Paris

     is

     the

     capital

     of

     France

    .


    2

    .

     It

     is

     the

     largest

     city

     in

     France

     by

     population

    .


    3

    .

     Paris

     is

     renowned

     for

     its

     historic

     architecture

    ,

     art

    ,

     and

     fashion

    .


    4

    .

     It

     is

     home

     to

     many

     famous

     landmarks

    ,

     such

     as

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     various

     museums

    .


    5

    .

     Paris

     is

     known

     for

     its

     vibrant

     culture

    ,

     including

     the

     city

    's

     annual

     annual

     E

    iff

    el

     Tower

     fair

    .


    6

    .

     The

     French

     Revolution

     and

     subsequent

     national

     re

    organization

     also

     took

     place

     in

     Paris

    .


    7

    .

     Paris

     is

     a

     cultural

    
    
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

    ,

     including

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     concerns

     about

     AI

    's

     potential

     impact

     on

     society

     and

     the

     environment

     grow

    ,

     there

     is

     an

     increasing

     focus

     on

     developing

     ethical

     AI

     systems

    .

     This

     could

     involve

     creating

     AI

     that

     is

     designed

     to

     minimize

     harm

     and

     ensure

     the

     protection

     of

     individual

     rights

     and

     freedoms

    .
    


    2

    .

     Development

     of

     more

     advanced

     hardware

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

     the

     need

     for

     more

     powerful

     hardware

     is

     becoming

     more

     apparent

    .

     This

     could

     lead

     to

     the

     development

     of

     even

     more

     powerful

     processors

     and

     GPUs

    ,

     as

     well

     as

     new

     forms

     of

     AI

     that

     are

     able

     to

     process

     data

     at

     scales

     that

     are

     currently

     impossible

     with

     current

    



```python
llm.shutdown()
```
