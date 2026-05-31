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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:06<06:18,  6.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:06<06:18,  6.64s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:06<06:18,  6.64s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:06<06:18,  6.64s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:06<06:18,  6.64s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:53,  1.01s/it]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:53,  1.01s/it]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:06<00:53,  1.01s/it]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:06<00:53,  1.01s/it]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:06<00:53,  1.01s/it]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:06<00:53,  1.01s/it]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:06<00:53,  1.01s/it]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:17,  2.71it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:17,  2.71it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:17,  2.71it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:06<00:17,  2.71it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:06<00:17,  2.71it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:06<00:17,  2.71it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:06<00:17,  2.71it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:06<00:17,  2.71it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:06<00:17,  2.71it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:06<00:06,  5.77it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:06<00:06,  5.77it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:06<00:06,  5.77it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:06<00:06,  5.77it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:07<00:06,  5.77it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:07<00:06,  5.77it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:07<00:06,  5.77it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:07<00:06,  5.77it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:07<00:06,  5.77it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:07<00:03,  9.72it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:07<00:03,  9.72it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:07<00:03,  9.72it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:07<00:03,  9.72it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:07<00:03,  9.72it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:07<00:03,  9.72it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:07<00:03,  9.72it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:07<00:03,  9.72it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:07<00:03,  9.72it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:07<00:03,  9.72it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:01, 14.92it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:01, 14.92it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:01, 14.92it/s]

    Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:01, 14.92it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:01, 14.92it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:01, 14.92it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:01, 14.92it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:01, 14.92it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:01, 14.92it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:07<00:01, 14.92it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:07<00:00, 21.51it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:07<00:00, 21.51it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:07<00:00, 21.51it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:07<00:00, 21.51it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:07<00:00, 21.51it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:07<00:00, 21.51it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:07<00:00, 21.51it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:07<00:00, 21.51it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:07<00:00, 21.51it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:07<00:00, 21.51it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:07<00:00, 29.12it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:07<00:00, 29.12it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:07<00:00, 29.12it/s]

    Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:07<00:00, 29.12it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:07<00:00, 29.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.35it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.35it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.35it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 28.88it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 28.88it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 28.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.88it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.75it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.75it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.75it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.75it/s] Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.62it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.62it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.62it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.62it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.62it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.62it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.82it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.11it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.11it/s]

    Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 37.07it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 37.07it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 37.07it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 37.07it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 37.07it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 37.07it/s]

    Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.72it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.72it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.22it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.22it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.22it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.22it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.22it/s]

    Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.22it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.28it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.28it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.28it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 36.30it/s]


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
    Generated text:  Nelly and I am a young woman from New York City. I am an undergraduate student in the College of Liberal Arts at Harvard University. I am a social studies major and have taken many courses in the History Department. My passion is history and I love learning. I will never be a historian, but I will always love to learn about the past. I have an interest in the history of African Americans, Native Americans, and the Civil Rights Movement. What do you like to do on a weekend? I love hiking and biking through the hills of the Catskills. I also love to read, especially the works of Barack Obama. As
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposing to increase the minimum wage for all workers in the United States. You are the government representative in your state. Write a letter to the governor of the state requesting their support for this proposal.
    [State Governor's Name], [State] [City],
    My esteemed [State] Governor, I am writing to request your support in advocating for the increase of the minimum wage in our state. As the governor, my state has seen a significant increase in unemployment rates and a decrease in the standard of living of many. I am concerned about the impact this increase in minimum wage may have on the average worker, and I urge you to support this
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the capital of France. Paris is the capital of France, and the capital of France is Paris. Paris is the capital of France, and the capital of France is Paris. The capital of France is Paris. The capital of France is Paris.
    Therefore, the answer is yes. The first sentence is true because the capital of France is Paris. The second sentence is true because there is only one capital in France. The third sentence is true because the second sentence is a complete sentence and the first sentence is a fragment. The fourth sentence is true because the first sentence is a fragment and the second sentence is a complete sentence
    ===============================
    Prompt: The future of AI is
    Generated text:  steeped in uncertainty. Already, a wave of companies have made bold statements promising a future powered by AI, but just how big will the change be? What will the impact be on the future of work, on society, and on the planet? An AI revolution is in the air and as a result the future of AI is a huge question mark. The language is changing. The stakes are escalating. The global community is taking action to shape the path. AI, the language is now speaking. This language will guide the design, manufacture and deployment of the next-generation AI.
    AI is a concept that has been around for a long time


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast. I'm always looking for new experiences and learning new things. I'm always eager to try new things and push myself to do better. I'm a [Favorite Activity] lover. I love to explore new places and try new foods. I'm always looking for new adventures and new experiences. I'm a [Favorite Book or Movie] fan. I love to read and watch movies. I'm always looking for new ways to learn and grow as a person. I'm a [Favorite Music] lover. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is also home to the French Parliament, the National Library, and the French Academy of Sciences. The city is known for its rich history, art, and cuisine, and is a major tourist destination for visitors from around the world. Paris is a city of contrasts, with its modern architecture and historical landmarks
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation: One of the most significant trends in AI is the increasing automation of tasks that are currently performed by humans. This could include tasks such as data analysis, image recognition, and natural language processing. As AI becomes more capable of performing these tasks, it is likely to become more efficient and less prone to errors.
    
    2. Improved privacy and security: As AI systems become more advanced, there is a risk
    


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
    Generated text:  ___________ and I'm a/an _____________________. I come from _______________. I have been living in _______________ since _______________ year. In my free time, I enjoy _______________. I am ____________________. I am ____________________. I am ____________________. I am ____________________. I am ____________________. What do you know about yourself? I'm ____________________. I am ____________________. I am ____________________. I am ____________________. What do you know about yourself? I'm ____________________. I am ____________________. I am ____________________. I am ____________________. I am ____________________. I am ____________________. I am ____________________. I am ____________________. I am ____________________. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the most populous city in the European Union and the second-largest city in the world by population. Paris is known for its romantic architecture, historical landmarks, and delicious cuisine. It is also famous for its annual Eiffel Tower exhibition, fashion shows, and artistic works. With its stunning views of the city skyline and its rich history, Paris is a major hub of global culture and diplomacy. It is a beautiful city that attracts millions of tourists each year.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and constantly evolving, and there are many potential trends that could shape the technology in the coming decades. Here are some of the most likely trends that could shape the future of AI:
    
    1. Improved Privacy and Security: As AI becomes more integrated into our lives, it's likely that we will see greater emphasis on privacy and security. Developers will need to find ways to protect user data and ensure that it's not abused or misused.
    
    2. Increased Use of AI for Personalized Learning: As AI becomes more integrated into education, we may see increased use of AI-powered tools and technologies to personalize learning experiences. This could mean offering more


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

     [

    Age

    ]

     years

     old

    .

     I

     currently

     work

     as

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

     have

     been

     in

     the

     field

     of

     [

    industry

    ]

     for

     [

    number

     of

     years

    ]

     years

    ,

     and

     my

     career

     has

     taken

     me

     from

     [

    starting

     position

    ],

     where

     I

     helped

     [

    description

     of

     the

     job

     role

    ],

     to

     [

    current

     position

    ]

     where

     I

     work

     on

     [

    current

     job

    ].

     I

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     learn

     more

     about

     the

     industry

     and

     share

     what

     I

    've

     learned

    .

     I

    'm

     a

     [

    character

     trait

     or

     ability

    ],

     and

     I

    'm

     passionate

     about

     [

    your

     field

     of

     interest

    ].

     If

     you

     have

     any

     questions

     or

     need

     help

    
    
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

     the

     second

     largest

     city

     in

     the

     world

     by

     population

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

    ,

     the

     French

     parliament

    ,

     and

     the

     headquarters

     of

     the

     French

     state

    .

     Paris

     is

     home

     to

     many

     iconic

     landmarks

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Place

     de

     la

     Con

    cor

    de

    .

     The

     city

     is

     known

     for

     its

     rich

     cultural

     heritage

    ,

     including

     its

     vibrant

     arts

     scene

    ,

     diverse

     cuisine

    ,

     and

     beautiful

     parks

    .

     It

     is

     a

     major

     tourist

     destination

     and

     a

     global

     economic

     power

    ,

     with

     a

     strong

     economy

     and

     a

     thriving

     business

     community

    .

     The

     city

     is

     also

     home

     to

     many

     popular

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     here

     are

     some

     possible

     trends

     that

     could

     emerge

    :
    


    1

    .

     Increased

     accuracy

     and

     precision

    :

     AI

     is

     becoming

     more

     accurate

     and

     precise

    ,

     making

     it

     more

     difficult

     for

     humans

     to

     predict

     outcomes

    .

     This

     could

     lead

     to

     more

     efficient

     decision

    -making

     and

     better

     healthcare

     outcomes

    .
    


    2

    .

     Natural

     language

     processing

    :

     AI

     is

     becoming

     increasingly

     capable

     of

     processing

     and

     understanding

     natural

     language

    ,

     leading

     to

     the

     development

     of

     chat

    bots

    ,

     language

     translation

     software

    ,

     and

     virtual

     assistants

    .
    


    3

    .

     Autonomous

     vehicles

    :

     AI

     is

     being

     used

     in

     autonomous

     vehicles

    ,

     making

     them

     safer

     and

     more

     reliable

     than

     traditional

     vehicles

    .

     This

     could

     lead

     to

     a

     decrease

     in

     accidents

     and

     improve

     traffic

     flow

    .
    


    4

    .

     AI

     in

     healthcare

    :

     AI

    



```python
llm.shutdown()
```
