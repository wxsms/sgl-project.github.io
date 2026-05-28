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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.49it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:39,  3.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:39,  3.86s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:39,  3.86s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:39,  3.86s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:39,  3.86s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:31,  1.68it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:31,  1.68it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:31,  1.68it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:16,  3.04it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:16,  3.04it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:16,  3.04it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:16,  3.04it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:16,  3.04it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:08,  5.33it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:08,  5.33it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:08,  5.33it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:08,  5.33it/s]

    Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:08,  5.33it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:05,  8.19it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03, 11.41it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03, 11.41it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03, 11.41it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03, 11.41it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03, 11.41it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03, 11.41it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:02, 16.30it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:02, 16.30it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:02, 16.30it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:02, 16.30it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:02, 16.30it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:02, 16.30it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 21.15it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 21.15it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 21.15it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 21.15it/s]

    Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 21.15it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 21.15it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:00, 26.10it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:00, 26.10it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:00, 26.10it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:00, 26.10it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:04<00:00, 26.10it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:04<00:00, 26.10it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s]

    Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 30.61it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 40.85it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 40.85it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 40.85it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 40.85it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 40.85it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 40.85it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 40.85it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 40.85it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 40.85it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 40.85it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 51.60it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 51.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.46it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=70.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.59 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=70.59 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=70.58 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.58 GB):   3%|▎         | 2/58 [00:00<00:02, 18.87it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=70.58 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=5632 avail_mem=70.58 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.57 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.57 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.57 GB):   9%|▊         | 5/58 [00:00<00:02, 22.14it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=70.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.55 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.03it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=70.55 GB):  21%|██        | 12/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.55 GB):  21%|██        | 12/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=67.01 GB):  21%|██        | 12/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=2560 avail_mem=66.54 GB):  21%|██        | 12/58 [00:00<00:01, 26.15it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=66.54 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=67.00 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=66.56 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=66.99 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=66.99 GB):  31%|███       | 18/58 [00:00<00:02, 18.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=66.99 GB):  31%|███       | 18/58 [00:00<00:02, 18.52it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=66.98 GB):  31%|███       | 18/58 [00:00<00:02, 18.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=66.98 GB):  34%|███▍      | 20/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=1024 avail_mem=66.97 GB):  34%|███▍      | 20/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=960 avail_mem=66.63 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.68it/s] Capturing num tokens (num_tokens=896 avail_mem=66.97 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.68it/s]

    Capturing num tokens (num_tokens=896 avail_mem=66.97 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.92it/s]Capturing num tokens (num_tokens=832 avail_mem=66.66 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.92it/s]Capturing num tokens (num_tokens=768 avail_mem=66.96 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.92it/s]Capturing num tokens (num_tokens=704 avail_mem=66.96 GB):  40%|███▉      | 23/58 [00:01<00:01, 18.92it/s]Capturing num tokens (num_tokens=704 avail_mem=66.96 GB):  45%|████▍     | 26/58 [00:01<00:01, 19.72it/s]Capturing num tokens (num_tokens=640 avail_mem=66.95 GB):  45%|████▍     | 26/58 [00:01<00:01, 19.72it/s]Capturing num tokens (num_tokens=576 avail_mem=66.72 GB):  45%|████▍     | 26/58 [00:01<00:01, 19.72it/s]

    Capturing num tokens (num_tokens=512 avail_mem=66.93 GB):  45%|████▍     | 26/58 [00:01<00:01, 19.72it/s]Capturing num tokens (num_tokens=512 avail_mem=66.93 GB):  50%|█████     | 29/58 [00:01<00:01, 20.95it/s]Capturing num tokens (num_tokens=480 avail_mem=66.95 GB):  50%|█████     | 29/58 [00:01<00:01, 20.95it/s]Capturing num tokens (num_tokens=448 avail_mem=66.94 GB):  50%|█████     | 29/58 [00:01<00:01, 20.95it/s]Capturing num tokens (num_tokens=416 avail_mem=66.74 GB):  50%|█████     | 29/58 [00:01<00:01, 20.95it/s]Capturing num tokens (num_tokens=416 avail_mem=66.74 GB):  55%|█████▌    | 32/58 [00:01<00:01, 22.25it/s]Capturing num tokens (num_tokens=384 avail_mem=66.75 GB):  55%|█████▌    | 32/58 [00:01<00:01, 22.25it/s]

    Capturing num tokens (num_tokens=352 avail_mem=66.91 GB):  55%|█████▌    | 32/58 [00:01<00:01, 22.25it/s]Capturing num tokens (num_tokens=320 avail_mem=66.92 GB):  55%|█████▌    | 32/58 [00:01<00:01, 22.25it/s]Capturing num tokens (num_tokens=320 avail_mem=66.92 GB):  60%|██████    | 35/58 [00:01<00:00, 23.26it/s]Capturing num tokens (num_tokens=288 avail_mem=66.90 GB):  60%|██████    | 35/58 [00:01<00:00, 23.26it/s]Capturing num tokens (num_tokens=256 avail_mem=66.89 GB):  60%|██████    | 35/58 [00:01<00:00, 23.26it/s]Capturing num tokens (num_tokens=240 avail_mem=66.89 GB):  60%|██████    | 35/58 [00:01<00:00, 23.26it/s]Capturing num tokens (num_tokens=240 avail_mem=66.89 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.59it/s]Capturing num tokens (num_tokens=224 avail_mem=66.88 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.59it/s]

    Capturing num tokens (num_tokens=208 avail_mem=66.87 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.59it/s]Capturing num tokens (num_tokens=192 avail_mem=66.84 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.59it/s]Capturing num tokens (num_tokens=176 avail_mem=66.84 GB):  66%|██████▌   | 38/58 [00:01<00:00, 24.59it/s]Capturing num tokens (num_tokens=176 avail_mem=66.84 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.23it/s]Capturing num tokens (num_tokens=160 avail_mem=66.86 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.23it/s]Capturing num tokens (num_tokens=144 avail_mem=66.84 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.23it/s]Capturing num tokens (num_tokens=128 avail_mem=66.85 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.23it/s]Capturing num tokens (num_tokens=112 avail_mem=66.84 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.23it/s]

    Capturing num tokens (num_tokens=112 avail_mem=66.84 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.49it/s]Capturing num tokens (num_tokens=96 avail_mem=66.80 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.49it/s] Capturing num tokens (num_tokens=80 avail_mem=66.79 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.49it/s]Capturing num tokens (num_tokens=64 avail_mem=66.80 GB):  79%|███████▉  | 46/58 [00:02<00:00, 29.49it/s]Capturing num tokens (num_tokens=48 avail_mem=66.79 GB):  79%|███████▉  | 46/58 [00:02<00:00, 29.49it/s]Capturing num tokens (num_tokens=48 avail_mem=66.79 GB):  86%|████████▌ | 50/58 [00:02<00:00, 30.92it/s]Capturing num tokens (num_tokens=32 avail_mem=66.79 GB):  86%|████████▌ | 50/58 [00:02<00:00, 30.92it/s]Capturing num tokens (num_tokens=28 avail_mem=66.80 GB):  86%|████████▌ | 50/58 [00:02<00:00, 30.92it/s]Capturing num tokens (num_tokens=24 avail_mem=66.79 GB):  86%|████████▌ | 50/58 [00:02<00:00, 30.92it/s]Capturing num tokens (num_tokens=20 avail_mem=66.78 GB):  86%|████████▌ | 50/58 [00:02<00:00, 30.92it/s]

    Capturing num tokens (num_tokens=16 avail_mem=66.78 GB):  86%|████████▌ | 50/58 [00:02<00:00, 30.92it/s]Capturing num tokens (num_tokens=16 avail_mem=66.78 GB):  95%|█████████▍| 55/58 [00:02<00:00, 33.80it/s]Capturing num tokens (num_tokens=12 avail_mem=66.78 GB):  95%|█████████▍| 55/58 [00:02<00:00, 33.80it/s]Capturing num tokens (num_tokens=8 avail_mem=66.76 GB):  95%|█████████▍| 55/58 [00:02<00:00, 33.80it/s] Capturing num tokens (num_tokens=4 avail_mem=66.75 GB):  95%|█████████▍| 55/58 [00:02<00:00, 33.80it/s]Capturing num tokens (num_tokens=4 avail_mem=66.75 GB): 100%|██████████| 58/58 [00:02<00:00, 25.45it/s]


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
    Generated text:  Taz. I'm a 21 year old male who is also a certified CFA and a fellow of the Society of CPAs. I'm a hybrid of everything - a fighter, an influencer, a maniac, a man, a lover, an athlete, an alcoholic, and a genius. As an influencer, I am a fan of whatever it is that I'm not using to make a living, and what ever I do I try to do it in a way that my friends and followers can understand and connect to. I'm a maniac because I love what I do. I love making people laugh and
    ===============================
    Prompt: The president of the United States is
    Generated text:  attempting to improve relations with their neighbors, and in order to accomplish this, they have decided to go to a neighboring country and have a summit. The president is about to leave the country and needs to make sure he has all the necessary documents and documents to go. He needs to have a passport, a visa, and also a proof of the purpose of visit in the country. 
    
    The president also needs to make sure that the documents are clear and properly formatted. To do this, he will need to check the format of the documents to ensure that they are in compliance with the country's regulations. 
    
    To make sure that the documents are
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a city of history, culture and art. It has a long and rich history. The streets and buildings are very old and have been standing for centuries. The city has a long history of being ruled by kings and queens. Today, Paris is the capital of France. It is the second largest city in the European Union. It is the 12th most populous city in the world. Paris has a long history and a rich culture. Many people consider Paris the most beautiful city in the world. Paris is a very important city. It is a city that is famous for its art, architecture, and history.
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly changing. With advancements in machine learning and natural language processing, AI is becoming more capable of understanding and processing data. This, in turn, has opened up a wide range of opportunities for businesses to leverage AI technology for their purposes. However, it is important to be aware of the potential risks and challenges associated with AI and to take steps to mitigate them.
    One potential risk of AI is that it may lead to job displacement. As AI becomes more capable of processing data and making decisions, it may become increasingly difficult for humans to keep up with the task. This could lead to job losses for individuals who are unable to adapt to the changing


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a brief description of your profession or experience here]. I enjoy [insert a brief description of your hobbies or interests here]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie? I'm always looking for new ways to learn and grow, and I'm always eager
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and the Middle Ages. Paris is a popular tourist destination and a major hub for business and finance in Europe. It is home to many world-renowned museums, art galleries, and theaters. The city is also known for its cuisine, with dishes
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will likely lead to increased efficiency, cost savings, and job displacement, but it will also create new opportunities for AI developers and entrepreneurs.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, we will need to ensure that it is used in a way that respects privacy and security. This will
    


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
    Generated text:  [Name], and I'm a [role] at [Company Name]. [Name] is a [role] at [Company Name]. I'm a [age] year old who grew up in [location] and have always been passionate about [interest]. I'm a [skill] with [number of experiences] years of experience in [industry]. I have a [job title] at [Company Name] and I enjoy [job title] at [Company Name]. What brings you to [Company Name]?
    
    I'm thrilled to be here, and I look forward to building a career with [Company Name]. What can I do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Paris is known for its beautiful architecture, iconic landmarks such as the Eiffel Tower, and the city's history, including its many historical sites and museums. It is also a cultural and educational hub, with numerous universities and art galleries. The city's cuisine and wine production are also renowned, making it a popular destination for tourists and locals alike. Overall, Paris is a vibrant and dynamic city that is steeped in tradition and innovation. 
    
    For those who are interested in French culture and food, Paris is a must-visit destination. The city has many traditional and popular restaurants, cafes, and street food vendors that are
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating, and it is difficult to predict exactly what will happen in the coming years. However, we can look at some potential trends that are likely to shape the AI landscape in the coming years.
    
    1. Improved Privacy and Security: One of the most significant trends in AI is the increasing importance of privacy and security. As AI becomes more integrated into our daily lives, it is essential to ensure that personal data is protected. This includes measures such as data encryption, biometric authentication, and the use of artificial intelligence to detect and prevent data breaches.
    
    2. Automation and AI in Healthcare: With the growing importance of AI in healthcare, it is


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

    'm

     a

     [

    Your

     Occupation

     or

     field

     of

     work

    ].

     I

    'm

     a

     [

    Your

     favorite

     genre

    ]

     author

    .

     And

     I

    'm

     constantly

     on

     the

     lookout

     for

     new

     opportunities

     to

     create

     characters

     and

     stories

     for

     fans

     around

     the

     world

    .

     I

    'm

     passionate

     about

     writing

     stories

     that

     are

     both

     interesting

     and

     rel

    atable

    ,

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     expand

     my

     creative

     vision

    .

     So

     if

     you

    'd

     like

     to

     know

     more

     about

     my

     life

     and

     how

     I

     became

     an

     author

    ,

     feel

     free

     to

     ask

     me

     anything

    ,

     and

     I

    'll

     be

     happy

     to

     share

     my

     thoughts

     and

     experiences

    .

     I

     hope

     you

     enjoy

     the

     story

    .

     How

     about

     you

    ?

     What

    's

     your

     name

    ,

     occupation

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     largest

     city

     in

     Europe

    ,

     with

     a

     population

     of

     approximately

     

    2

    .

    1

     million

     as

     of

     

    2

    0

    2

    1

    .

     It

     is

     also

     the

     seat

     of

     the

     French

     government

     and

     the

     seat

     of

     the

     French

     parliament

    .

     Paris

     is

     known

     for

     its

     historical

     architecture

    ,

     art

    ,

     food

    ,

     and

     fashion

    .

     The

     city

     is

     also

     famous

     for

     its

     annual

     cultural

     events

     and

     festivals

    .

     Paris

     is

     also

     one

     of

     the

     most

     visited

     cities

     in

     the

     world

    ,

     with

     millions

     of

     visitors

     each

     year

    .

     Its

     cultural

     heritage

    ,

     rich

     history

    ,

     and

     vibrant

     street

     life

     make

     it

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     The

     city

     is

     also

     home

     to

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     some

     potential

     trends

     that

     are

     emerging

     include

    :
    


    1

    .

     Increased

     automation

     and

     robotics

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

     we

     may

     see

     more

     robots

     and

     automation

     systems

     take

     over

     tasks

     that

     were

     previously

     done

     by

     humans

    .

     This

     could

     lead

     to

     job

     displacement

    ,

     but

     also

     potential

     new

     opportunities

     for

     creative

     problem

    -solving

     and

     innovation

    .
    


    2

    .

     Enhanced

     cognitive

     functions

    :

     AI

     may

     be

     able

     to

     learn

     and

     adapt

     to

     new

     situations

    ,

     providing

     even

     more

     advanced

     and

     sophisticated

     artificial

     intelligence

    .

     This

     could

     lead

     to

     breakthrough

    s

     in

     areas

     like

     artificial

     general

     intelligence

    ,

     where

     AI

     can

     think

     and

     learn

     in

     ways

     that

     would

     be

     impossible

     for

     humans

     to

     do

    .
    


    3

    .

     AI

     in

     healthcare

    



```python
llm.shutdown()
```
