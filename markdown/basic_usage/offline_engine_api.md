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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.95it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.95it/s]


    2026-04-30 00:42:09,558 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 00:42:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:35,  4.84s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:35,  4.84s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:35,  4.84s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:11,  1.30s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:11,  1.30s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:05<01:11,  1.30s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:16,  2.95it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:16,  2.95it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:16,  2.95it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:16,  2.95it/s]

    Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:16,  2.95it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:08,  5.44it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:08,  5.44it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:08,  5.44it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:08,  5.44it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:08,  5.44it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:05<00:04,  8.50it/s]

    Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:02, 13.13it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 20.62it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 20.62it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 20.62it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 20.62it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 20.62it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 20.62it/s]

    Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 20.62it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:01, 20.62it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 28.41it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 28.41it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 28.41it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 28.41it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 28.41it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 28.41it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 28.41it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:00, 28.41it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 36.04it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 36.04it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 36.04it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 36.04it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 36.04it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 36.04it/s] 

    Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 36.04it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 36.04it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 42.76it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 42.76it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 42.76it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 42.76it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:06<00:00, 42.76it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:06<00:00, 42.76it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:06<00:00, 42.76it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:06<00:00, 42.76it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:06<00:00, 42.76it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:06<00:00, 51.17it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:06<00:00, 51.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.71 GB):   2%|▏         | 1/58 [00:00<00:05,  9.68it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.68 GB):   2%|▏         | 1/58 [00:00<00:05,  9.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.68 GB):   2%|▏         | 1/58 [00:00<00:05,  9.68it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=55.68 GB):   5%|▌         | 3/58 [00:00<00:05, 10.60it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.68 GB):   5%|▌         | 3/58 [00:00<00:05, 10.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.67 GB):   5%|▌         | 3/58 [00:00<00:05, 10.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.67 GB):   9%|▊         | 5/58 [00:00<00:04, 11.23it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.67 GB):   9%|▊         | 5/58 [00:00<00:04, 11.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.66 GB):   9%|▊         | 5/58 [00:00<00:04, 11.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.66 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.66 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.73it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=55.66 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.66 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.65 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.65 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.65 GB):  16%|█▌        | 9/58 [00:00<00:04, 10.50it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=55.65 GB):  21%|██        | 12/58 [00:00<00:03, 14.36it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.64 GB):  21%|██        | 12/58 [00:00<00:03, 14.36it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.65 GB):  21%|██        | 12/58 [00:01<00:03, 14.36it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.64 GB):  21%|██        | 12/58 [00:01<00:03, 14.36it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.64 GB):  21%|██        | 12/58 [00:01<00:03, 14.36it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.64 GB):  21%|██        | 12/58 [00:01<00:03, 14.36it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.64 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.98it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.63 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.63 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.98it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.63 GB):  29%|██▉       | 17/58 [00:01<00:01, 21.98it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=55.63 GB):  34%|███▍      | 20/58 [00:01<00:02, 12.96it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:01<00:02, 12.96it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:01<00:02, 12.96it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:01<00:02, 12.96it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:01<00:02, 12.96it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:01<00:02, 12.96it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.57it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.57it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.57it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.57it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.57it/s]

    Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 18.57it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.56it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.56it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.56it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.56it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.56it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.56it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 28.90it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 28.90it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 28.90it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 28.90it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 28.90it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  60%|██████    | 35/58 [00:01<00:00, 28.90it/s]

    Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  60%|██████    | 35/58 [00:01<00:00, 28.90it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 34.43it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 34.43it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:02<00:00, 34.43it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:02<00:00, 34.43it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:02<00:00, 34.43it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:02<00:00, 34.43it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:02<00:00, 37.85it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:02<00:00, 37.85it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:02<00:00, 37.85it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:02<00:00, 37.85it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  79%|███████▉  | 46/58 [00:02<00:00, 37.85it/s]

    Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  79%|███████▉  | 46/58 [00:02<00:00, 37.85it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:02<00:00, 40.27it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:02<00:00, 40.27it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:02<00:00, 40.27it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:02<00:00, 40.27it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:02<00:00, 40.27it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  88%|████████▊ | 51/58 [00:02<00:00, 40.27it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  88%|████████▊ | 51/58 [00:02<00:00, 40.27it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:02<00:00, 43.24it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:02<00:00, 43.24it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:02<00:00, 24.89it/s]


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
    Generated text:  Cynthia. I have a passion for the outdoors and I enjoy hiking and camping. I'm currently learning French and I would like to make friends and connect with other French speakers through online communities.
    What are some ways that Cynthia could improve her French skills and meet new French-speaking friends online? Cynthia could consider participating in online French language courses or language exchange programs. She could also try using language learning apps and websites that offer French lessons and practice. She could also join online language groups or communities, such as those on platforms like Facebook or Discord. Additionally, Cynthia could try making new friends through social media groups or language exchange platforms, such as those
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office with no fixed term and can be filled for multiple terms. In what year did the first president of the United States, George Washington, serve as president? George Washington served as president of the United States from 1789 to 1797. He was the first president of the United States and the first President of the United States since 1789. The term of office was 14 years. However, George Washington was not the first president of the United States. He was the first president from 1789 to 1797. The first president of the United
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Lille
    B. Paris
    C. Tours
    D. Nice
    Answer: B
    
    The number of students who passed the college entrance examination in 2012 was approximately ____ more than the number of students who passed in 2011.
    A. 13%
    B. 36%
    C. 25%
    D. 15%
    Answer: D
    
    The "Chinese Dream" is a dream of the people of the country, aiming to achieve the great rejuvenation of the Chinese nation, which is a Chinese Dream. What does the phrase "the great rejuvenation of
    ===============================
    Prompt: The future of AI is
    Generated text:  very promising, but it is also a lot of uncertainty. The current state of AI is a combination of new and old technologies. To make the AI a truly human-like intelligence, we need to solve the following problems:
    1. What are the core principles of AI? What should the AI evolve into? And what should its applications be? These questions are crucial to the success of AI.
    2. How should we design AI to improve efficiency and reduce errors? We must be able to consider multiple variables and anticipate different situations.
    3. What are the ethical implications of AI? We should consider the impact of AI on the environment, society,


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Skill] with [Number] years of experience in [Field]. I am passionate about [What I love to do]. I am always looking for new challenges and opportunities to grow and learn. I am a [Personality] person with [Strengths] and [Weaknesses]. I am [What I strive to be] and I am [What I aspire to be]. I am excited to meet you and learn more about you. How would you describe your personality and what makes you unique? As a [Name] with [Age
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a bustling city with a diverse population and is home to many famous landmarks and attractions. The city is also known for its cuisine, including French cuisine and its famous wine. Paris is a popular tourist destination and is home to many cultural institutions and events throughout the year. It is a city that is both a symbol of France and a major economic and cultural center in the world. The city is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more and more AI systems are used in decision-making processes, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and fairness.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, including healthcare, finance, transportation, and manufacturing. As more of these technologies become integrated, we can expect to see even more integration of AI with
    


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
    Generated text:  [Name], and I'm a [Age] year-old [Occupation]. I have always been passionate about [Your passion or interests]. I am a [job title or role] who was born [Date of Birth] in [Location]. I am always learning new things, and I am eager to keep discovering new things about my personal and professional life. What are you like? Hello! My name is [Name], and I'm a [Age] year-old [Occupation]. I have always been passionate about [Your passion or interests]. I am a [job title or role] who was born [Date of Birth] in
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic and cosmopolitan city known for its rich history, vibrant culture, and world-class museums and art galleries. 
    
    (Note: The statement provided is in French. The American English version would be: "The capital of France is Paris, a historic and cosmopolitan city known for its rich history, vibrant culture, and world-class museums and art galleries.") 
    
    It's worth noting that Paris is often referred to as the "City of a Hundred Faces" due to its historic and cultural diversity, which can be seen in the city's many museums, theaters, and neighborhoods. The city has a rich history, including a long history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a combination of progress, stagnation, and disruption. Here are some possible trends that could be expected in the coming years:
    
    1. Advancements in machine learning and deep learning: One of the most significant trends is the continued development of machine learning and deep learning. This will involve more sophisticated algorithms that can handle complex data and patterns, and improve their ability to learn from new data.
    
    2. Increased reliance on AI for decision-making: As AI becomes more integrated into various industries and applications, we are likely to see an increasing focus on AI as a tool for decision-making. This could lead to more complex, nuanced


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

     [

    Age

    ],

     [

    Occup

    ation

    ],

     [

    Field

     of

     Study

    ],

     or

     [

    School

    ,

     Major

    ].

     I

     am

     a

     [

    typeof

    ]

     student

     at

     [

    School

    ].

     I

     am

     passionate

     about

     [

    job

    ,

     hobby

    ,

     or

     sport

    ],

     [

    emotion

    ]

     my

     school

     is

     [

    context

    ]

     and

     I

     am

     very

     [

    specific

     emotion

     or

     trait

    ].

     I

     am

     always

     looking

     for

     opportunities

     to

     [

    write

     about

     an

     experience

    ,

     hobby

    ,

     or

     project

    ].

     I

     am

     a

     true

     [

    type

     of

     person

    ]

     and

     I

     am

     always

     [

    write

     about

     an

     ability

     or

     characteristic

     you

     believe

     is

     important

     to

     you

    ].

     I

     am

     confident

     in

     [

    write

     about

     an

     accomplishment

     or

     achievement

    ].

     I

     strive

     to

     [

    write

    
    
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

     the

     country

     and

     serves

     as

     the

     administrative

     and

     economic

     center

     of

     France

    .

     The

     city

     is

     home

     to

     iconic

     landmarks

     such

     as

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

     the

     Lou

    vre

     Museum

    .

     It

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     fashion

    ,

     and

     cuisine

    ,

     and

     is

     a

     major

     hub

     for

     the

     French

    -speaking

     world

    .

     Paris

     is

     also

     home

     to

     some

     of

     the

     world

    ’s

     most

     famous

     museums

    ,

     including

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     the

     Mus

    ée

     Rod

    in

    ,

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     attracts

     millions

     of

     visitors

     annually

    .

     The

     city

     is

     also

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

    ,

     and

     there

     are

     several

     potential

     trends

     that

     could

     significantly

     shape

     its

     development

    .

     Here

     are

     some

     of

     the

     most

     notable

     trends

     in

     AI

    :
    


    1

    .

     **

    Increased

     Integration

     with

     Other

     Technologies

    **:

     AI

     is

     likely

     to

     continue

     to

     integrate

     more

     seamlessly

     with

     other

     technologies

     such

     as

     the

     Internet

     of

     Things

     (

    Io

    T

    ),

     cloud

     computing

    ,

     and

     blockchain

    .

     This

     integration

     could

     lead

     to

     new

     applications

     and

     opportunities

     for

     AI

     to

     be

     more

     integrated

     into

     everyday

     life

    .
    


    2

    .

     **

    Deep

     Learning

     and

     AI

     Automation

    **:

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

     trend

     towards

     deep

     learning

    ,

     where

     AI

     systems

     become

     more

     capable

     of

     learning

     and

     solving

     complex

     problems

     without

     explicit

     programming

    .

     This

     could

     result

     in

     AI

    



```python
llm.shutdown()
```
