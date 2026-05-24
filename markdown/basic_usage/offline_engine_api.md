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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.19it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.18it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.58it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.58it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.58it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.58it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.58it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.58it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.58it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:04,  8.58it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:04<00:04,  8.58it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:02, 13.98it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:02, 13.98it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:02, 13.98it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:02, 13.98it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:02, 13.98it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:02, 13.98it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:04<00:02, 13.98it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:04<00:02, 13.98it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:04<00:02, 13.98it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:01, 20.46it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:01, 20.46it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:01, 20.46it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:01, 20.46it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:04<00:01, 20.46it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:04<00:01, 20.46it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:04<00:01, 20.46it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:04<00:01, 20.46it/s]

    Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:04<00:01, 20.46it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:04<00:01, 20.46it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:04<00:00, 28.85it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:04<00:00, 28.85it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:04<00:00, 28.85it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:04<00:00, 28.85it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:04<00:00, 28.85it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:04<00:00, 28.85it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:04<00:00, 28.85it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:04<00:00, 28.85it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:04<00:00, 28.85it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 36.46it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 36.46it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 36.46it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 36.46it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 36.46it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 36.46it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 36.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.57it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.14 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.14 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  31%|███       | 18/58 [00:00<00:01, 35.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  31%|███       | 18/58 [00:00<00:01, 35.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.12 GB):  31%|███       | 18/58 [00:00<00:01, 35.33it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.10 GB):  31%|███       | 18/58 [00:00<00:01, 35.33it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  31%|███       | 18/58 [00:00<00:01, 35.33it/s] Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  31%|███       | 18/58 [00:00<00:01, 35.33it/s]Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.02it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.02it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.02it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.02it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.02it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.82it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.82it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.12it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.12it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.12it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 31.20it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 31.20it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 31.20it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 31.20it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 31.20it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 31.20it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.44it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.44it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.44it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.44it/s]

    Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.44it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.44it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.44it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 36.44it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.24it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.24it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.24it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.24it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.24it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.24it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.36it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.36it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.36it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.36it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.36it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 35.37it/s]


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
    Generated text:  Jessica and I am a PhD student studying Mathematics with an emphasis on Number Theory, Computational Number Theory and Cryptography. I am also a Computer Science major. I have taken many courses, but it is my final year of the Computer Science major that I am most proud of as a student. I was the first student to complete both the Computer Science and Mathematics major and I hope to continue to excel in both fields. If you're interested, I'd be happy to get to know you! You can reach me at email@example.com. Thank you! Jessica Let me know if you have any other questions! ✍️✨✨📚💼
    ===============================
    Prompt: The president of the United States is
    Generated text:  a highly important person. He or she is the head of the government, and in charge of the whole country. The president has the power to make major decisions and to appoint (任命) important people to high positions. He or she also has to sign (签名) presidential (总统) pardons (减刑) and keep records of all the decisions made by the government. He or she also has to vote for (投票) all the senators and representatives (代表). The president is elected (选举) to a two-year term (任期). Once in office, he or she must make decisions that are important and will affect the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a historical city, filled with beautifully landscaped squares, beautiful architecture and lively streets. The city is known for its art, music, festivals, food, nightlife and traditions.
    The capital of France is a historic city and its historic center has been built on the site of ancient Roman and Greek city-state of Rome. Today, the French government has chosen the site of the Roman Forum as the capital.
    The city of Paris was founded by Emperor Charles V in 1562 and was renamed Paris in 1791 when Napoleon conquered France. During the French Revolution, the city was renamed "Quebec".
    ===============================
    Prompt: The future of AI is
    Generated text:  currently in an uncertain state. In order to ensure a secure and reliable future, we must adopt a multi-layered approach to develop and implement AI technologies. This includes setting specific and measurable goals, conducting rigorous testing, and continuously monitoring and analyzing the performance and impact of the AI systems. In addition, we must ensure that the AI systems are transparent and accountable, and that they are designed and implemented in a manner that is transparent and accountable, and that they are designed and implemented in a manner that is transparent and accountable. Finally, we must ensure that the AI systems are designed and implemented in a manner that is designed and implemented in a manner


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [Age], [Gender], [Nationality], [Occupation], and I have [Number] years of experience in [Field of Work]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [Favorite Activity], and I enjoy [Favorite Hobby]. I'm always looking for ways to stay active and healthy, and I'm always eager to learn new things. What's your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant nightlife and is a popular tourist destination. The city is home to many museums, art galleries, and theaters, and is a major center for business and finance. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that has played a significant role in French culture and history, and continues to be a major cultural and economic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to diagnose and treat diseases, and it has the potential to revolutionize the field by improving patient outcomes and reducing costs.
    
    2. AI in manufacturing: AI is being used to optimize production processes, reduce waste, and improve quality control. This trend is expected to continue as AI becomes more integrated into manufacturing processes.
    
    3. AI in finance: AI is being used to analyze financial data, identify fraud, and make investment decisions.
    


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
    Generated text:  [Name] and I am a [Occupation] [Position]. I have been working in the [Industry/Field] for [Number of Years] years now, and I enjoy the [Reward/Financial Reward] I receive from my work. I am always willing to learn and expand my skills, and I strive to be the best version of myself. What can you tell me about yourself? [Name] has been working in the [Industry/Field] for [Number of Years] years now, and they enjoy the [Reward/Financial Reward] they receive from their work. They are always willing to learn and expand their skills
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its historical landmarks, vibrant culture, and fashion scene. It is the second-largest city in Europe and the largest in the French region of the same name. The city has a population of over 2 million people and is a major cultural and economic center of France. It is also known for its annual "Beaunecessité" fair, a day when millions of people gather to celebrate the beauty of Paris. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, which draw millions of visitors each year. Its rich history and cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and encompasses a range of possibilities that will shape the way we live and work in the coming decades. Here are some possible trends in AI that could shape the industry in the coming years:
    
    1. Increased automation and robotics: AI is already enabling the development of robots and automation in many industries, from manufacturing to service, but it is expected that this trend will continue as AI will enable more machines to perform tasks that previously required human intervention.
    
    2. More personalized and context-aware AI: With the rise of big data and machine learning, AI will be able to learn from users' data and generate more accurate predictions and recommendations. This will result


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

     John

    .

     I

     am

     a

     software

     developer

     with

     over

     

    1

    0

     years

     of

     experience

     in

     building

     and

     maintaining

     software

     applications

    .

     I

    'm

     known

     for

     my

     ability

     to

     design

     and

     develop

     efficient

     and

     user

    -friendly

     software

     solutions

     that

     meet

     the

     needs

     of

     businesses

     and

     individuals

     alike

    .

     I

     thrive

     in

     a

     fast

    -paced

     environment

     and

     enjoy

     collaborating

     with

     cross

    -functional

     teams

     to

     deliver

     projects

     that

     exceed

     client

     expectations

    .

     I

     am

     an

     adaptable

     and

     empath

    etic

     person

     who

     is

     always

     looking

     for

     ways

     to

     improve

     my

     skills

     and

     knowledge

     in

     my

     field

    .

     As

     a

     software

     engineer

    ,

     I

     am

     committed

     to

     using

     my

     skills

     to

     make

     a

     positive

     impact

     in

     the

     world

     through

     my

     work

    .

     
    


    I

     am

     passionate

     about

     exploring

     new

     technologies

     and

     learning

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     most

     populous

     city

     in

     the

     country

    .

     It

     is

     located

     in

     the

     Î

    le

     de

     la

     C

    ité

     region

     in

     the

     Rh

    ô

    ne

     Valley

    ,

     southwest

     of

     the

     French

     city

     of

     Marseille

    .

     Paris

     is

     also

     known

     as

     "

    le

     Ch

    amps

    -

    É

    lys

    ées

    "

     and

     is

     the

     main

     hub

     of

     the

     French

     economy

    .

     It

     is

     home

     to

     several

     landmarks

    ,

     including

     Notre

     Dame

     Cathedral

    ,

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

     Palace

     of

     Vers

    ailles

    .

     The

     city

     has

     a

     diverse

     cultural

     scene

     and

     a

     vibrant

     nightlife

    ,

     with

     many

     popular

     bars

    ,

     clubs

    ,

     and

     restaurants

    .

     France

    's

     capital

     is

     also

     home

     to

     many

     museums

     and

     cultural

     institutions

    ,

     including

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     poised

     for

     significant

     advancements

     in

     several

     key

     areas

    ,

     driven

     by

     advancements

     in

     computing

     power

    ,

     data

     availability

    ,

     and

     the

     growing

     use

     of

     AI

     in

     industries

     like

     healthcare

    ,

     finance

    ,

     and

     transportation

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Larger

    ,

     more

     capable

     AI

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     there

     is

     a

     growing

     possibility

     that

     AI

     will

     become

     even

     more

     powerful

     and

     capable

    .

     This

     will

     lead

     to

     breakthrough

    s

     in

     fields

     like

     natural

     language

     processing

    ,

     robotics

    ,

     and

     autonomous

     vehicles

     that

     could

     revolution

    ize

     industries

     and

     impact

     our

     way

     of

     life

    .
    


    2

    .

     Enhanced

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     there

     is

     a

     growing

     concern

     about

     the

     potential

     for

    



```python
llm.shutdown()
```
