# OpenAI APIs - Vision

SGLang provides OpenAI-compatible APIs to enable a smooth transition from OpenAI services to self-hosted local models.
A complete reference for the API is available in the [OpenAI API Reference](https://platform.openai.com/docs/guides/vision).
This tutorial covers the vision APIs for vision language models.

SGLang supports various vision language models such as Llama 3.2, LLaVA-OneVision, Qwen2.5-VL, Gemma3 and [more](../supported_models/text_generation/multimodal_language_models.md).

As an alternative to the OpenAI API, you can also use the [SGLang offline engine](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/offline_batch_inference_vlm.py).

## Launch A Server

Launch the server in your terminal and wait for it to initialize.


```python
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

vision_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}", process=vision_process)
```

    [2026-02-15 07:39:06] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-15 07:39:06] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-15 07:39:06] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-15 07:39:12] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-15 07:39:12] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-15 07:39:12] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-15 07:39:15] INFO server_args.py:1832: Attention backend not specified. Use fa3 backend by default.
    [2026-02-15 07:39:15] INFO server_args.py:2867: Set soft_watchdog_timeout since in CI


    [2026-02-15 07:39:19] Ignore import error when loading sglang.srt.multimodal.processors.glm4v: No module named 'transformers.models.glm_ocr'
    [2026-02-15 07:39:19] Ignore import error when loading sglang.srt.multimodal.processors.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-02-15 07:39:22] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-15 07:39:22] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-15 07:39:22] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-15 07:39:22] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-15 07:39:22] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-15 07:39:22] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-15 07:39:30] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-15 07:39:30] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-15 07:39:30] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:00<00:03,  1.29it/s]


    Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:01<00:02,  1.31it/s]


    Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:02<00:01,  1.30it/s]


    Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:03<00:00,  1.35it/s]


    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.78it/s]
    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.54it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=59.19 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=59.19 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.61it/s]Capturing batches (bs=2 avail_mem=59.16 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.61it/s]Capturing batches (bs=1 avail_mem=59.16 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.61it/s]Capturing batches (bs=1 avail_mem=59.16 GB): 100%|██████████| 3/3 [00:00<00:00,  4.36it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


## Using cURL

Once the server is up, you can send test requests using curl or requests.


```python
import subprocess

curl_command = f"""
curl -s http://localhost:{port}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
      {{
        "role": "user",
        "content": [
          {{
            "type": "text",
            "text": "What’s in this image?"
          }},
          {{
            "type": "image_url",
            "image_url": {{
              "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
            }}
          }}
        ]
      }}
    ],
    "max_tokens": 300
  }}'
"""

response = subprocess.check_output(curl_command, shell=True).decode()
print_highlight(response)


response = subprocess.check_output(curl_command, shell=True).decode()
print_highlight(response)
```


<strong style='color: #00008B;'>{"id":"28d7b7f332ca441ea30cd42613ce62b1","object":"chat.completion","created":1771141185,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows a man standing on the back of a yellow taxi, using an iron to iron a piece of clothing. The taxi is parked on a city street, and there are other taxis and buildings in the background. The man appears to be balancing on the taxi's rear bumper while ironing, which is an unusual and somewhat humorous scene.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":307,"total_tokens":377,"completion_tokens":70,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}</strong>



<strong style='color: #00008B;'>{"id":"2cea9feb08de48ea80eddd92e96dd2dc","object":"chat.completion","created":1771141186,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows a man standing on the back of a yellow taxi, using an iron to iron a piece of clothing. The taxi is parked on a city street, and there are other taxis and buildings in the background. The man appears to be balancing on the taxi's rear bumper while ironing, which is an unusual and somewhat humorous scene.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":307,"total_tokens":377,"completion_tokens":70,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}</strong>


## Using Python Requests


```python
import requests

url = f"http://localhost:{port}/v1/chat/completions"

data = {
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print_highlight(response.text)
```


<strong style='color: #00008B;'>{"id":"678d9e381ad843d29c678f7c1051bfd5","object":"chat.completion","created":1771141187,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows a man standing on the back of a yellow taxi, using an iron to iron a pair of blue jeans. The taxi is parked on a city street, and there are other taxis and buildings in the background. The man appears to be balancing on the taxi's rear bumper while ironing the jeans.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":307,"total_tokens":371,"completion_tokens":64,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}</strong>


## Using OpenAI Python Client


```python
from openai import OpenAI

client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>The image shows a man standing on the back of a yellow taxi, using an iron to iron a piece of clothing. The taxi is parked on a city street, and there are other taxis and buildings in the background. The man appears to be balancing on the taxi's rear bumper while ironing, which is an unusual and somewhat humorous scene.</strong>


## Multiple-Image Inputs

The server also supports multiple images and interleaved text and images if the model supports it.


```python
from openai import OpenAI

client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png",
                    },
                },
                {
                    "type": "text",
                    "text": "I have two very different images. They are not related at all. "
                    "Please describe the first image in one sentence, and then describe the second image in another sentence.",
                },
            ],
        }
    ],
    temperature=0,
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>The first image shows a man ironing clothes on the back of a yellow taxi in an urban setting. The second image is a stylized logo featuring the letters "SGL" in orange with a book and a computer icon as part of the design.</strong>



```python
terminate_process(vision_process)
```
