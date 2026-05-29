# Tool Parser

This guide demonstrates how to use SGLang’s [Function calling](https://platform.openai.com/docs/guides/function-calling) functionality.

## Currently supported parsers:

| Parser | Supported Models | Notes |
|---|---|---|
| `deepseekv3` | DeepSeek-v3 (e.g., `deepseek-ai/DeepSeek-V3-0324`) | Recommend adding `--chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja` to launch command. |
| `deepseekv31` | DeepSeek-V3.1 and DeepSeek-V3.2-Exp (e.g. `deepseek-ai/DeepSeek-V3.1`, `deepseek-ai/DeepSeek-V3.2-Exp`) | Recommend adding `--chat-template ./examples/chat_template/tool_chat_template_deepseekv31.jinja` (Or ..deepseekv32.jinja for DeepSeek-V3.2) to launch command. |
| `deepseekv32` | DeepSeek-V3.2 (`deepseek-ai/DeepSeek-V3.2`) | |
| `glm` | GLM series (e.g. `zai-org/GLM-4.6`) | |
| `gpt-oss` | GPT-OSS (e.g., `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `lmsys/gpt-oss-120b-bf16`, `lmsys/gpt-oss-20b-bf16`) | The gpt-oss tool parser filters out analysis channel events and only preserves normal text. This can cause the content to be empty when explanations are in the analysis channel. To work around this, complete the tool round by returning tool results as `role="tool"` messages, which enables the model to generate the final content. |
| `kimi_k2` | `moonshotai/Kimi-K2-Instruct` | |
| `llama3` | Llama 3.1 / 3.2 / 3.3 (e.g. `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`) | |
| `llama4` | Llama 4 (e.g. `meta-llama/Llama-4-Scout-17B-16E-Instruct`) | |
| `mistral` | Mistral (e.g. `mistralai/Mistral-7B-Instruct-v0.3`, `mistralai/Mistral-Nemo-Instruct-2407`, `mistralai/Mistral-7B-v0.3`) | |
| `pythonic` | Llama-3.2 / Llama-3.3 / Llama-4 | Model outputs function calls as Python code. Requires `--tool-call-parser pythonic` and is recommended to use with a specific chat template. |
| `qwen` | Qwen series (e.g. `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3-VL-30B-A3B-Thinking`) except Qwen3-Coder| |
| `qwen3_coder` | Qwen3-Coder (e.g. `Qwen/Qwen3-Coder-30B-A3B-Instruct`) | |
| `step3` | Step-3 | |


## OpenAI Compatible API

### Launching the Server


```python
import json
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process
from openai import OpenAI

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --tool-call-parser qwen25 --host 0.0.0.0 --log-level warning"  # qwen25
)
wait_for_server(f"http://localhost:{port}", process=server_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(
    [2026-05-29 18:18:08] The tool_call_parser 'qwen25' is deprecated. Please use 'qwen' instead.


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  1.58it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.45it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.35it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.29it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:02,  5.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:02,  5.31s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:09,  2.31s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:09,  2.31s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:14,  1.36s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:14,  1.36s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:48,  1.12it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:48,  1.12it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:33,  1.60it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:33,  1.60it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:24,  2.15it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:24,  2.15it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.81it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.81it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.54it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.54it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.39it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.39it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.39it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.08it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.08it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.08it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.64it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.64it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:05,  7.64it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.27it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.27it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.27it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.23it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.23it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.23it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:07<00:03, 11.72it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:07<00:03, 11.72it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:07<00:03, 11.72it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:07<00:03, 11.66it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:07<00:03, 11.66it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:07<00:03, 11.66it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:07<00:02, 12.89it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:07<00:02, 12.89it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:07<00:02, 12.89it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:07<00:02, 12.89it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:07<00:02, 12.89it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:07<00:02, 12.89it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:07<00:01, 21.01it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:07<00:01, 21.01it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:07<00:01, 21.01it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:07<00:01, 21.01it/s]

    Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:07<00:01, 21.01it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:07<00:01, 21.01it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:07<00:01, 21.01it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:07<00:00, 30.09it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:07<00:00, 30.09it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:07<00:00, 30.09it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:07<00:00, 30.09it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:07<00:00, 30.09it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:07<00:00, 30.09it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:07<00:00, 30.09it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:07<00:00, 30.09it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:07<00:00, 30.09it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:07<00:00, 41.31it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:07<00:00, 41.31it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:07<00:00, 41.31it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:07<00:00, 41.31it/s]

    Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:07<00:00, 41.31it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:08<00:00, 41.31it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:08<00:00, 41.31it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:08<00:00, 41.31it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:08<00:00, 41.31it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:08<00:00, 50.02it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:08<00:00, 50.02it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:08<00:00, 50.02it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:08<00:00, 50.02it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:08<00:00, 50.02it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:08<00:00, 50.02it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:08<00:00, 50.02it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:08<00:00, 50.02it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:08<00:00, 50.02it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=28.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=28.59 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=28.56 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=28.56 GB):   3%|▎         | 2/58 [00:00<00:15,  3.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=28.56 GB):   3%|▎         | 2/58 [00:00<00:15,  3.56it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=28.56 GB):   5%|▌         | 3/58 [00:00<00:14,  3.79it/s]Capturing num tokens (num_tokens=6656 avail_mem=28.56 GB):   5%|▌         | 3/58 [00:00<00:14,  3.79it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=28.56 GB):   7%|▋         | 4/58 [00:01<00:13,  4.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=28.56 GB):   7%|▋         | 4/58 [00:01<00:13,  4.07it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=28.56 GB):   9%|▊         | 5/58 [00:01<00:12,  4.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=28.56 GB):   9%|▊         | 5/58 [00:01<00:12,  4.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=28.56 GB):  10%|█         | 6/58 [00:01<00:11,  4.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=28.56 GB):  10%|█         | 6/58 [00:01<00:11,  4.68it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=28.56 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=28.56 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.09it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=28.56 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=28.56 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.70it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=28.56 GB):  16%|█▌        | 9/58 [00:02<00:13,  3.53it/s]Capturing num tokens (num_tokens=3840 avail_mem=28.56 GB):  16%|█▌        | 9/58 [00:02<00:13,  3.53it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=28.56 GB):  17%|█▋        | 10/58 [00:02<00:13,  3.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=28.56 GB):  17%|█▋        | 10/58 [00:02<00:13,  3.45it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=28.56 GB):  19%|█▉        | 11/58 [00:02<00:13,  3.47it/s]Capturing num tokens (num_tokens=3328 avail_mem=28.56 GB):  19%|█▉        | 11/58 [00:02<00:13,  3.47it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=28.56 GB):  21%|██        | 12/58 [00:03<00:12,  3.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=28.55 GB):  21%|██        | 12/58 [00:03<00:12,  3.59it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=28.55 GB):  22%|██▏       | 13/58 [00:03<00:12,  3.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=28.55 GB):  22%|██▏       | 13/58 [00:03<00:12,  3.71it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=28.55 GB):  24%|██▍       | 14/58 [00:03<00:11,  3.89it/s]Capturing num tokens (num_tokens=2560 avail_mem=28.55 GB):  24%|██▍       | 14/58 [00:03<00:11,  3.89it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=28.55 GB):  26%|██▌       | 15/58 [00:03<00:10,  4.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=28.55 GB):  26%|██▌       | 15/58 [00:03<00:10,  4.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=28.55 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.35it/s]Capturing num tokens (num_tokens=2048 avail_mem=28.54 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.35it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=28.54 GB):  29%|██▉       | 17/58 [00:04<00:08,  4.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=28.54 GB):  29%|██▉       | 17/58 [00:04<00:08,  4.66it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=28.54 GB):  31%|███       | 18/58 [00:04<00:08,  4.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=28.54 GB):  31%|███       | 18/58 [00:04<00:08,  4.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=28.54 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.10it/s]Capturing num tokens (num_tokens=1280 avail_mem=28.54 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.10it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=28.54 GB):  34%|███▍      | 20/58 [00:04<00:06,  5.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=28.53 GB):  34%|███▍      | 20/58 [00:04<00:06,  5.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=28.53 GB):  36%|███▌      | 21/58 [00:04<00:05,  6.26it/s]Capturing num tokens (num_tokens=960 avail_mem=28.52 GB):  36%|███▌      | 21/58 [00:04<00:05,  6.26it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=28.52 GB):  36%|███▌      | 21/58 [00:04<00:05,  6.26it/s]Capturing num tokens (num_tokens=896 avail_mem=28.52 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.24it/s]Capturing num tokens (num_tokens=832 avail_mem=28.52 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.24it/s]

    Capturing num tokens (num_tokens=832 avail_mem=28.52 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.58it/s]Capturing num tokens (num_tokens=768 avail_mem=28.51 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.58it/s]Capturing num tokens (num_tokens=704 avail_mem=28.51 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.58it/s]Capturing num tokens (num_tokens=704 avail_mem=28.51 GB):  45%|████▍     | 26/58 [00:05<00:03,  8.73it/s]Capturing num tokens (num_tokens=640 avail_mem=28.50 GB):  45%|████▍     | 26/58 [00:05<00:03,  8.73it/s]

    Capturing num tokens (num_tokens=576 avail_mem=28.50 GB):  45%|████▍     | 26/58 [00:05<00:03,  8.73it/s]Capturing num tokens (num_tokens=576 avail_mem=28.50 GB):  48%|████▊     | 28/58 [00:05<00:02, 10.65it/s]Capturing num tokens (num_tokens=512 avail_mem=28.49 GB):  48%|████▊     | 28/58 [00:05<00:02, 10.65it/s]Capturing num tokens (num_tokens=480 avail_mem=28.49 GB):  48%|████▊     | 28/58 [00:05<00:02, 10.65it/s]Capturing num tokens (num_tokens=480 avail_mem=28.49 GB):  52%|█████▏    | 30/58 [00:05<00:02, 12.37it/s]Capturing num tokens (num_tokens=448 avail_mem=28.49 GB):  52%|█████▏    | 30/58 [00:05<00:02, 12.37it/s]

    Capturing num tokens (num_tokens=416 avail_mem=28.48 GB):  52%|█████▏    | 30/58 [00:05<00:02, 12.37it/s]Capturing num tokens (num_tokens=416 avail_mem=28.48 GB):  55%|█████▌    | 32/58 [00:05<00:01, 13.88it/s]Capturing num tokens (num_tokens=384 avail_mem=28.48 GB):  55%|█████▌    | 32/58 [00:05<00:01, 13.88it/s]Capturing num tokens (num_tokens=352 avail_mem=28.47 GB):  55%|█████▌    | 32/58 [00:05<00:01, 13.88it/s]Capturing num tokens (num_tokens=352 avail_mem=28.47 GB):  59%|█████▊    | 34/58 [00:05<00:01, 14.40it/s]Capturing num tokens (num_tokens=320 avail_mem=28.47 GB):  59%|█████▊    | 34/58 [00:05<00:01, 14.40it/s]

    Capturing num tokens (num_tokens=288 avail_mem=28.48 GB):  59%|█████▊    | 34/58 [00:05<00:01, 14.40it/s]Capturing num tokens (num_tokens=256 avail_mem=28.47 GB):  59%|█████▊    | 34/58 [00:05<00:01, 14.40it/s]Capturing num tokens (num_tokens=240 avail_mem=28.47 GB):  59%|█████▊    | 34/58 [00:05<00:01, 14.40it/s]Capturing num tokens (num_tokens=240 avail_mem=28.47 GB):  66%|██████▌   | 38/58 [00:05<00:00, 20.18it/s]Capturing num tokens (num_tokens=224 avail_mem=28.46 GB):  66%|██████▌   | 38/58 [00:05<00:00, 20.18it/s]Capturing num tokens (num_tokens=208 avail_mem=28.46 GB):  66%|██████▌   | 38/58 [00:06<00:00, 20.18it/s]Capturing num tokens (num_tokens=192 avail_mem=28.46 GB):  66%|██████▌   | 38/58 [00:06<00:00, 20.18it/s]Capturing num tokens (num_tokens=176 avail_mem=28.45 GB):  66%|██████▌   | 38/58 [00:06<00:00, 20.18it/s]Capturing num tokens (num_tokens=176 avail_mem=28.45 GB):  72%|███████▏  | 42/58 [00:06<00:00, 24.69it/s]Capturing num tokens (num_tokens=160 avail_mem=28.45 GB):  72%|███████▏  | 42/58 [00:06<00:00, 24.69it/s]

    Capturing num tokens (num_tokens=144 avail_mem=28.44 GB):  72%|███████▏  | 42/58 [00:06<00:00, 24.69it/s]Capturing num tokens (num_tokens=128 avail_mem=28.45 GB):  72%|███████▏  | 42/58 [00:06<00:00, 24.69it/s]Capturing num tokens (num_tokens=112 avail_mem=28.44 GB):  72%|███████▏  | 42/58 [00:06<00:00, 24.69it/s]Capturing num tokens (num_tokens=112 avail_mem=28.44 GB):  79%|███████▉  | 46/58 [00:06<00:00, 28.05it/s]Capturing num tokens (num_tokens=96 avail_mem=28.44 GB):  79%|███████▉  | 46/58 [00:06<00:00, 28.05it/s] Capturing num tokens (num_tokens=80 avail_mem=28.43 GB):  79%|███████▉  | 46/58 [00:06<00:00, 28.05it/s]Capturing num tokens (num_tokens=64 avail_mem=28.43 GB):  79%|███████▉  | 46/58 [00:06<00:00, 28.05it/s]Capturing num tokens (num_tokens=48 avail_mem=28.43 GB):  79%|███████▉  | 46/58 [00:06<00:00, 28.05it/s]Capturing num tokens (num_tokens=48 avail_mem=28.43 GB):  86%|████████▌ | 50/58 [00:06<00:00, 30.69it/s]Capturing num tokens (num_tokens=32 avail_mem=28.42 GB):  86%|████████▌ | 50/58 [00:06<00:00, 30.69it/s]

    Capturing num tokens (num_tokens=28 avail_mem=27.29 GB):  86%|████████▌ | 50/58 [00:06<00:00, 30.69it/s]Capturing num tokens (num_tokens=24 avail_mem=27.28 GB):  86%|████████▌ | 50/58 [00:06<00:00, 30.69it/s]

    Capturing num tokens (num_tokens=20 avail_mem=27.22 GB):  86%|████████▌ | 50/58 [00:06<00:00, 30.69it/s]Capturing num tokens (num_tokens=20 avail_mem=27.22 GB):  93%|█████████▎| 54/58 [00:06<00:00, 20.28it/s]Capturing num tokens (num_tokens=16 avail_mem=28.31 GB):  93%|█████████▎| 54/58 [00:06<00:00, 20.28it/s]Capturing num tokens (num_tokens=12 avail_mem=27.32 GB):  93%|█████████▎| 54/58 [00:06<00:00, 20.28it/s]

    Capturing num tokens (num_tokens=8 avail_mem=27.32 GB):  93%|█████████▎| 54/58 [00:06<00:00, 20.28it/s] Capturing num tokens (num_tokens=8 avail_mem=27.32 GB):  98%|█████████▊| 57/58 [00:06<00:00, 16.87it/s]Capturing num tokens (num_tokens=4 avail_mem=27.32 GB):  98%|█████████▊| 57/58 [00:06<00:00, 16.87it/s]Capturing num tokens (num_tokens=4 avail_mem=27.32 GB): 100%|██████████| 58/58 [00:07<00:00,  8.28it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


Note that `--tool-call-parser` defines the parser used to interpret responses.

### Define Tools for Function Call
Below is a Python snippet that shows how to define a tool as a dictionary. The dictionary includes a tool name, a description, and property defined Parameters.


```python
# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that the city is"
                        " in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]
```

### Define Messages


```python
def get_messages():
    return [
        {
            "role": "user",
            "content": "What's the weather like in Boston today? Output a reasoning before act, then use the tools to help you.",
        }
    ]


messages = get_messages()
```

### Initialize the Client


```python
# Initialize OpenAI-like client
client = OpenAI(api_key="None", base_url=f"http://0.0.0.0:{port}/v1")
model_name = client.models.list().data[0].id
```

###  Non-Streaming Request


```python
# Non-streaming mode test
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0,
    top_p=0.95,
    max_tokens=1024,
    stream=False,  # Non-streaming
    tools=tools,
)
print_highlight("Non-stream response:")
print_highlight(response_non_stream)
print_highlight("==== content ====")
print_highlight(response_non_stream.choices[0].message.content)
print_highlight("==== tool_calls ====")
print_highlight(response_non_stream.choices[0].message.tool_calls)
```


<strong style='color: #00008B;'>Non-stream response:</strong>



<strong style='color: #00008B;'>ChatCompletion(id='d472b3177c0c4a9ab6617adfafd3db83', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content='To determine the current weather in Boston, I will use the `get_current_weather` function by providing the city name "Boston", state "MA" (which is the two-letter abbreviation for Massachusetts), and specifying the unit of temperature. Since no specific unit was requested, I\'ll assume Celsius for a more universally understood temperature scale.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_cc9261f9e1984b8cad5e168e', function=Function(arguments='{"city": "Boston", "state": "MA", "unit": "celsius"}', name='get_current_weather'), type='function', index=0)], reasoning_content=None), matched_stop=None)], created=1780078732, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=100, prompt_tokens=296, total_tokens=396, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



<strong style='color: #00008B;'>==== content ====</strong>



<strong style='color: #00008B;'>To determine the current weather in Boston, I will use the `get_current_weather` function by providing the city name "Boston", state "MA" (which is the two-letter abbreviation for Massachusetts), and specifying the unit of temperature. Since no specific unit was requested, I'll assume Celsius for a more universally understood temperature scale.</strong>



<strong style='color: #00008B;'>==== tool_calls ====</strong>



<strong style='color: #00008B;'>[ChatCompletionMessageFunctionToolCall(id='call_cc9261f9e1984b8cad5e168e', function=Function(arguments='{"city": "Boston", "state": "MA", "unit": "celsius"}', name='get_current_weather'), type='function', index=0)]</strong>


#### Handle Tools
When the engine determines it should call a particular tool, it will return arguments or partial arguments through the response. You can parse these arguments and later invoke the tool accordingly.


```python
name_non_stream = response_non_stream.choices[0].message.tool_calls[0].function.name
arguments_non_stream = (
    response_non_stream.choices[0].message.tool_calls[0].function.arguments
)

print_highlight(f"Final streamed function call name: {name_non_stream}")
print_highlight(f"Final streamed function call arguments: {arguments_non_stream}")
```


<strong style='color: #00008B;'>Final streamed function call name: get_current_weather</strong>



<strong style='color: #00008B;'>Final streamed function call arguments: {"city": "Boston", "state": "MA", "unit": "celsius"}</strong>


### Streaming Request


```python
# Streaming mode test
print_highlight("Streaming response:")
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0,
    top_p=0.95,
    max_tokens=1024,
    stream=True,  # Enable streaming
    tools=tools,
)

texts = ""
tool_calls = []
name = ""
arguments = ""
for chunk in response_stream:
    if chunk.choices[0].delta.content:
        texts += chunk.choices[0].delta.content
    if chunk.choices[0].delta.tool_calls:
        tool_calls.append(chunk.choices[0].delta.tool_calls[0])
print_highlight("==== Text ====")
print_highlight(texts)

print_highlight("==== Tool Call ====")
for tool_call in tool_calls:
    print_highlight(tool_call)
```


<strong style='color: #00008B;'>Streaming response:</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>To determine the current weather in Boston, I will use the `get_current_weather` function by providing the city name "Boston", state "MA" (which is the two-letter abbreviation for Massachusetts), and specifying the unit of temperature. Since no specific unit was requested, I'll assume Celsius for a more universally understood temperature scale.<br><br></strong>



<strong style='color: #00008B;'>==== Tool Call ====</strong>



<strong style='color: #00008B;'>ChoiceDeltaToolCall(index=0, id='call_895eaf9cab19464a93e0fa60', function=ChoiceDeltaToolCallFunction(arguments='', name='get_current_weather'), type='function')</strong>



<strong style='color: #00008B;'>ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='{"city": "', name=None), type='function')</strong>



<strong style='color: #00008B;'>ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='Boston"', name=None), type='function')</strong>



<strong style='color: #00008B;'>ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments=', "state": "', name=None), type='function')</strong>



<strong style='color: #00008B;'>ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='MA"', name=None), type='function')</strong>



<strong style='color: #00008B;'>ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments=', "unit": "', name=None), type='function')</strong>



<strong style='color: #00008B;'>ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='c', name=None), type='function')</strong>



<strong style='color: #00008B;'>ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='elsius"}', name=None), type='function')</strong>


#### Handle Tools
When the engine determines it should call a particular tool, it will return arguments or partial arguments through the response. You can parse these arguments and later invoke the tool accordingly.


```python
# Parse and combine function call arguments
arguments = []
for tool_call in tool_calls:
    if tool_call.function.name:
        print_highlight(f"Streamed function call name: {tool_call.function.name}")

    if tool_call.function.arguments:
        arguments.append(tool_call.function.arguments)

# Combine all fragments into a single JSON string
full_arguments = "".join(arguments)
print_highlight(f"streamed function call arguments: {full_arguments}")
```


<strong style='color: #00008B;'>Streamed function call name: get_current_weather</strong>



<strong style='color: #00008B;'>streamed function call arguments: {"city": "Boston", "state": "MA", "unit": "celsius"}</strong>


### Define a Tool Function


```python
# This is a demonstration, define real function according to your usage.
def get_current_weather(city: str, state: str, unit: "str"):
    return (
        f"The weather in {city}, {state} is 85 degrees {unit}. It is "
        "partly cloudly, with highs in the 90's."
    )


available_tools = {"get_current_weather": get_current_weather}
```


### Execute the Tool


```python
messages.append(response_non_stream.choices[0].message)

# Call the corresponding tool function
tool_call = messages[-1].tool_calls[0]
tool_name = tool_call.function.name
tool_to_call = available_tools[tool_name]
result = tool_to_call(**(json.loads(tool_call.function.arguments)))
print_highlight(f"Function call result: {result}")
# messages.append({"role": "tool", "content": result, "name": tool_name})
messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result),
        "name": tool_name,
    }
)

print_highlight(f"Updated message history: {messages}")
```


<strong style='color: #00008B;'>Function call result: The weather in Boston, MA is 85 degrees celsius. It is partly cloudly, with highs in the 90's.</strong>



<strong style='color: #00008B;'>Updated message history: [{'role': 'user', 'content': "What's the weather like in Boston today? Output a reasoning before act, then use the tools to help you."}, ChatCompletionMessage(content='To determine the current weather in Boston, I will use the `get_current_weather` function by providing the city name "Boston", state "MA" (which is the two-letter abbreviation for Massachusetts), and specifying the unit of temperature. Since no specific unit was requested, I\'ll assume Celsius for a more universally understood temperature scale.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_cc9261f9e1984b8cad5e168e', function=Function(arguments='{"city": "Boston", "state": "MA", "unit": "celsius"}', name='get_current_weather'), type='function', index=0)], reasoning_content=None), {'role': 'tool', 'tool_call_id': 'call_cc9261f9e1984b8cad5e168e', 'content': "The weather in Boston, MA is 85 degrees celsius. It is partly cloudly, with highs in the 90's.", 'name': 'get_current_weather'}]</strong>


### Send Results Back to Model


```python
final_response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0,
    top_p=0.95,
    stream=False,
    tools=tools,
)
print_highlight("Non-stream response:")
print_highlight(final_response)

print_highlight("==== Text ====")
print_highlight(final_response.choices[0].message.content)
```


<strong style='color: #00008B;'>Non-stream response:</strong>



<strong style='color: #00008B;'>ChatCompletion(id='f15364b516984e4e912d90589300f696', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content="There seems to be an error in the response as 85 degrees Celsius is extremely high for a typical temperature in Boston. Let's correct this by rechecking with the function and ensuring we get the right unit of measurement.", refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_3bfcb5c9357f437b8f12a632', function=Function(arguments='{"city": "Boston", "state": "MA", "unit": "celsius"}', name='get_current_weather'), type='function', index=0)], reasoning_content=None), matched_stop=None)], created=1780078735, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=79, prompt_tokens=442, total_tokens=521, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'>There seems to be an error in the response as 85 degrees Celsius is extremely high for a typical temperature in Boston. Let's correct this by rechecking with the function and ensuring we get the right unit of measurement.</strong>


## Native API and SGLang Runtime (SRT)


```python
from transformers import AutoTokenizer
import requests

# generate an answer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

messages = get_messages()

input = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, tools=tools, return_dict=False
)

gen_url = f"http://localhost:{port}/generate"
gen_data = {
    "text": input,
    "sampling_params": {
        "skip_special_tokens": False,
        "max_new_tokens": 1024,
        "temperature": 0,
        "top_p": 0.95,
    },
}
gen_response = requests.post(gen_url, json=gen_data).json()["text"]
print_highlight("==== Response ====")
print_highlight(gen_response)

# parse the response
parse_url = f"http://localhost:{port}/parse_function_call"

function_call_input = {
    "text": gen_response,
    "tool_call_parser": "qwen25",
    "tools": tools,
}

function_call_response = requests.post(parse_url, json=function_call_input)
function_call_response_json = function_call_response.json()

print_highlight("==== Text ====")
print(function_call_response_json["normal_text"])
print_highlight("==== Calls ====")
print("function name: ", function_call_response_json["calls"][0]["name"])
print("function arguments: ", function_call_response_json["calls"][0]["parameters"])
```


<strong style='color: #00008B;'>==== Response ====</strong>



<strong style='color: #00008B;'>To provide you with the current weather in Boston, I will use the `get_current_weather` function. Since you didn't specify the state, I'll assume you're referring to Boston, Massachusetts, which is commonly abbreviated as MA. The unit of temperature can be either Celsius or Fahrenheit; since it's not specified, I'll provide the information in both units for a comprehensive answer.<br><br>Reasoning: The user wants to know the current weather in Boston. By using the `get_current_weather` function, we can obtain the necessary information.<br><br><tool_call><br>{"name": "get_current_weather", "arguments": {"city": "Boston", "state": "MA", "unit": "celsius"}}<br></tool_call><br><tool_call><br>{"name": "get_current_weather", "arguments": {"city": "Boston", "state": "MA", "unit": "fahrenheit"}}<br></tool_call></strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Text ====</strong>


    To provide you with the current weather in Boston, I will use the `get_current_weather` function. Since you didn't specify the state, I'll assume you're referring to Boston, Massachusetts, which is commonly abbreviated as MA. The unit of temperature can be either Celsius or Fahrenheit; since it's not specified, I'll provide the information in both units for a comprehensive answer.
    
    Reasoning: The user wants to know the current weather in Boston. By using the `get_current_weather` function, we can obtain the necessary information.



<strong style='color: #00008B;'>==== Calls ====</strong>


    function name:  get_current_weather
    function arguments:  {"city": "Boston", "state": "MA", "unit": "celsius"}



```python
terminate_process(server_process)
```

## Offline Engine API


```python
import sglang as sgl
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Tool, Function

llm = sgl.Engine(model_path="Qwen/Qwen2.5-7B-Instruct")
tokenizer = llm.tokenizer_manager.tokenizer
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, tools=tools, return_dict=False
)

# Note that for gpt-oss tool parser, adding "no_stop_trim": True
# to make sure the tool call token <call> is not trimmed.

sampling_params = {
    "max_new_tokens": 1024,
    "temperature": 0,
    "top_p": 0.95,
    "skip_special_tokens": False,
}

# 1) Offline generation
result = llm.generate(input_ids=input_ids, sampling_params=sampling_params)
generated_text = result["text"]  # Assume there is only one prompt

print_highlight("=== Offline Engine Output Text ===")
print_highlight(generated_text)


# 2) Parse using FunctionCallParser
def convert_dict_to_tool(tool_dict: dict) -> Tool:
    function_dict = tool_dict.get("function", {})
    return Tool(
        type=tool_dict.get("type", "function"),
        function=Function(
            name=function_dict.get("name"),
            description=function_dict.get("description"),
            parameters=function_dict.get("parameters"),
        ),
    )


tools = [convert_dict_to_tool(raw_tool) for raw_tool in tools]

parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")
normal_text, calls = parser.parse_non_stream(generated_text)

print_highlight("=== Parsing Result ===")
print("Normal text portion:", normal_text)
print_highlight("Function call portion:")
for call in calls:
    # call: ToolCallItem
    print_highlight(f"  - tool name: {call.name}")
    print_highlight(f"    parameters: {call.parameters}")

# 3) If needed, perform additional logic on the parsed functions, such as automatically calling the corresponding function to obtain a return value, etc.
```

    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.39it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.33it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.29it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.29it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:11,  5.46s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:11,  5.46s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:19,  2.50s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:19,  2.50s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:24,  1.54s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:24,  1.54s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:58,  1.08s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:58,  1.08s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:42,  1.25it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:42,  1.25it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:33,  1.57it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:33,  1.57it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:26,  1.95it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:26,  1.95it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:20,  2.43it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:20,  2.43it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:16,  3.00it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:16,  3.00it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:12,  3.71it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:12,  3.71it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:11,  4.00it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:11,  4.00it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:10,  4.32it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:10,  4.32it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:09,  4.68it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:09,  4.68it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:08,  5.03it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:08,  5.03it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:07,  5.49it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:07,  5.49it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:06,  6.01it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:06,  6.01it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:06,  6.58it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:06,  6.58it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:05,  7.17it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:05,  7.17it/s]

    Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:09<00:05,  7.17it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:04,  8.53it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:04,  8.53it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:09<00:04,  8.53it/s] 

    Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:09<00:03, 10.19it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:09<00:03, 10.19it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:09<00:03, 10.19it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:09<00:02, 11.78it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:09<00:02, 11.78it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:09<00:02, 11.78it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:09<00:02, 13.37it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:09<00:02, 13.37it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:09<00:02, 13.37it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:09<00:02, 14.90it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:09<00:02, 14.90it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:09<00:02, 14.90it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:09<00:02, 14.90it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:10<00:01, 17.57it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:10<00:01, 17.57it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:10<00:01, 17.57it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:10<00:01, 18.01it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:10<00:01, 18.01it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:10<00:01, 18.01it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:10<00:01, 18.01it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:10<00:01, 20.09it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:10<00:01, 20.09it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:10<00:01, 20.09it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:10<00:01, 20.09it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:10<00:01, 20.09it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:10<00:00, 23.18it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:10<00:00, 23.18it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:10<00:00, 23.18it/s]

    Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:10<00:00, 23.18it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:10<00:00, 24.27it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:10<00:00, 24.27it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:10<00:00, 24.27it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:10<00:00, 24.27it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:10<00:00, 24.72it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:10<00:00, 24.72it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:10<00:00, 24.72it/s]

    Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:10<00:00, 24.72it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:10<00:00, 25.72it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:10<00:00, 25.72it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:10<00:00, 25.72it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:10<00:00, 25.72it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:10<00:00, 25.72it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:10<00:00, 27.21it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:10<00:00, 27.21it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:10<00:00, 27.21it/s]

    Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:10<00:00, 27.21it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:10<00:00, 27.21it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:10<00:00, 29.85it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:10<00:00, 29.85it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=24.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=24.37 GB):   2%|▏         | 1/58 [00:00<00:23,  2.42it/s]Capturing num tokens (num_tokens=7680 avail_mem=24.34 GB):   2%|▏         | 1/58 [00:00<00:23,  2.42it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=24.34 GB):   3%|▎         | 2/58 [00:00<00:18,  3.01it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.69 GB):   3%|▎         | 2/58 [00:00<00:18,  3.01it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.69 GB):   5%|▌         | 3/58 [00:00<00:16,  3.41it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.68 GB):   5%|▌         | 3/58 [00:00<00:16,  3.41it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.68 GB):   7%|▋         | 4/58 [00:01<00:14,  3.80it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.68 GB):   7%|▋         | 4/58 [00:01<00:14,  3.80it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.68 GB):   9%|▊         | 5/58 [00:01<00:12,  4.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.68 GB):   9%|▊         | 5/58 [00:01<00:12,  4.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.68 GB):  10%|█         | 6/58 [00:01<00:11,  4.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.65 GB):  10%|█         | 6/58 [00:01<00:11,  4.51it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.65 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.65 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.92it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=42.65 GB):  14%|█▍        | 8/58 [00:02<00:12,  4.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.65 GB):  14%|█▍        | 8/58 [00:02<00:12,  4.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.65 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.65 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.61it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=42.65 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.65 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.65 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.65 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.97it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=42.65 GB):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.65 GB):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.65 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.44it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.65 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.44it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.64 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.44it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=42.64 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.64 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.64 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.64 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.64 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.51it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=42.63 GB):  29%|██▉       | 17/58 [00:03<00:03, 10.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.63 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.63 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.24it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.62 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.24it/s]Capturing num tokens (num_tokens=960 avail_mem=42.62 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.24it/s] Capturing num tokens (num_tokens=960 avail_mem=42.62 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.53it/s]Capturing num tokens (num_tokens=896 avail_mem=42.61 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.53it/s]

    Capturing num tokens (num_tokens=832 avail_mem=42.61 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.53it/s]Capturing num tokens (num_tokens=768 avail_mem=42.60 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.53it/s]Capturing num tokens (num_tokens=768 avail_mem=42.60 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.57it/s]Capturing num tokens (num_tokens=704 avail_mem=42.60 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.57it/s]Capturing num tokens (num_tokens=640 avail_mem=42.60 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.57it/s]Capturing num tokens (num_tokens=576 avail_mem=42.59 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.57it/s]Capturing num tokens (num_tokens=512 avail_mem=42.59 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.57it/s]

    Capturing num tokens (num_tokens=512 avail_mem=42.59 GB):  50%|█████     | 29/58 [00:03<00:01, 22.39it/s]Capturing num tokens (num_tokens=480 avail_mem=42.59 GB):  50%|█████     | 29/58 [00:03<00:01, 22.39it/s]Capturing num tokens (num_tokens=448 avail_mem=42.58 GB):  50%|█████     | 29/58 [00:03<00:01, 22.39it/s]Capturing num tokens (num_tokens=416 avail_mem=42.58 GB):  50%|█████     | 29/58 [00:03<00:01, 22.39it/s]Capturing num tokens (num_tokens=384 avail_mem=42.58 GB):  50%|█████     | 29/58 [00:03<00:01, 22.39it/s]Capturing num tokens (num_tokens=384 avail_mem=42.58 GB):  57%|█████▋    | 33/58 [00:03<00:00, 25.36it/s]Capturing num tokens (num_tokens=352 avail_mem=42.57 GB):  57%|█████▋    | 33/58 [00:03<00:00, 25.36it/s]Capturing num tokens (num_tokens=320 avail_mem=42.57 GB):  57%|█████▋    | 33/58 [00:03<00:00, 25.36it/s]Capturing num tokens (num_tokens=288 avail_mem=42.57 GB):  57%|█████▋    | 33/58 [00:03<00:00, 25.36it/s]

    Capturing num tokens (num_tokens=256 avail_mem=42.57 GB):  57%|█████▋    | 33/58 [00:03<00:00, 25.36it/s]Capturing num tokens (num_tokens=256 avail_mem=42.57 GB):  64%|██████▍   | 37/58 [00:03<00:00, 28.14it/s]Capturing num tokens (num_tokens=240 avail_mem=42.56 GB):  64%|██████▍   | 37/58 [00:03<00:00, 28.14it/s]Capturing num tokens (num_tokens=224 avail_mem=42.56 GB):  64%|██████▍   | 37/58 [00:03<00:00, 28.14it/s]Capturing num tokens (num_tokens=208 avail_mem=42.56 GB):  64%|██████▍   | 37/58 [00:03<00:00, 28.14it/s]Capturing num tokens (num_tokens=192 avail_mem=42.55 GB):  64%|██████▍   | 37/58 [00:03<00:00, 28.14it/s]Capturing num tokens (num_tokens=192 avail_mem=42.55 GB):  71%|███████   | 41/58 [00:03<00:00, 30.55it/s]Capturing num tokens (num_tokens=176 avail_mem=42.55 GB):  71%|███████   | 41/58 [00:03<00:00, 30.55it/s]Capturing num tokens (num_tokens=160 avail_mem=42.55 GB):  71%|███████   | 41/58 [00:03<00:00, 30.55it/s]Capturing num tokens (num_tokens=144 avail_mem=42.54 GB):  71%|███████   | 41/58 [00:03<00:00, 30.55it/s]

    Capturing num tokens (num_tokens=128 avail_mem=42.54 GB):  71%|███████   | 41/58 [00:03<00:00, 30.55it/s]Capturing num tokens (num_tokens=128 avail_mem=42.54 GB):  78%|███████▊  | 45/58 [00:03<00:00, 32.36it/s]Capturing num tokens (num_tokens=112 avail_mem=42.54 GB):  78%|███████▊  | 45/58 [00:03<00:00, 32.36it/s]Capturing num tokens (num_tokens=96 avail_mem=42.54 GB):  78%|███████▊  | 45/58 [00:03<00:00, 32.36it/s] Capturing num tokens (num_tokens=80 avail_mem=42.53 GB):  78%|███████▊  | 45/58 [00:03<00:00, 32.36it/s]Capturing num tokens (num_tokens=64 avail_mem=42.53 GB):  78%|███████▊  | 45/58 [00:03<00:00, 32.36it/s]Capturing num tokens (num_tokens=64 avail_mem=42.53 GB):  84%|████████▍ | 49/58 [00:04<00:00, 33.42it/s]Capturing num tokens (num_tokens=48 avail_mem=42.52 GB):  84%|████████▍ | 49/58 [00:04<00:00, 33.42it/s]Capturing num tokens (num_tokens=32 avail_mem=42.52 GB):  84%|████████▍ | 49/58 [00:04<00:00, 33.42it/s]Capturing num tokens (num_tokens=28 avail_mem=42.52 GB):  84%|████████▍ | 49/58 [00:04<00:00, 33.42it/s]

    Capturing num tokens (num_tokens=24 avail_mem=42.52 GB):  84%|████████▍ | 49/58 [00:04<00:00, 33.42it/s]Capturing num tokens (num_tokens=24 avail_mem=42.52 GB):  91%|█████████▏| 53/58 [00:04<00:00, 33.70it/s]Capturing num tokens (num_tokens=20 avail_mem=42.51 GB):  91%|█████████▏| 53/58 [00:04<00:00, 33.70it/s]Capturing num tokens (num_tokens=16 avail_mem=42.51 GB):  91%|█████████▏| 53/58 [00:04<00:00, 33.70it/s]Capturing num tokens (num_tokens=12 avail_mem=42.50 GB):  91%|█████████▏| 53/58 [00:04<00:00, 33.70it/s]Capturing num tokens (num_tokens=8 avail_mem=42.50 GB):  91%|█████████▏| 53/58 [00:04<00:00, 33.70it/s] Capturing num tokens (num_tokens=8 avail_mem=42.50 GB):  98%|█████████▊| 57/58 [00:04<00:00, 32.00it/s]Capturing num tokens (num_tokens=4 avail_mem=42.50 GB):  98%|█████████▊| 57/58 [00:04<00:00, 32.00it/s]

    Capturing num tokens (num_tokens=4 avail_mem=42.50 GB): 100%|██████████| 58/58 [00:04<00:00, 13.52it/s]



<strong style='color: #00008B;'>=== Offline Engine Output Text ===</strong>



<strong style='color: #00008B;'>To provide you with the current weather in Boston, I will use the `get_current_weather` function. Since you didn't specify the unit for temperature, I'll assume you prefer Fahrenheit, which is commonly used in the United States.<br><br>Reasoning: The user asked for the current weather in Boston, so we need to fetch the weather data for this specific location. Using the `get_current_weather` function with the city set to "Boston" and the state set to "MA" (the two-letter abbreviation for Massachusetts), and the unit set to "fahrenheit" will give us the required information.<br><br><tool_call><br>{"name": "get_current_weather", "arguments": {"city": "Boston", "state": "MA", "unit": "fahrenheit"}}<br></tool_call></strong>



<strong style='color: #00008B;'>=== Parsing Result ===</strong>


    Normal text portion: To provide you with the current weather in Boston, I will use the `get_current_weather` function. Since you didn't specify the unit for temperature, I'll assume you prefer Fahrenheit, which is commonly used in the United States.
    
    Reasoning: The user asked for the current weather in Boston, so we need to fetch the weather data for this specific location. Using the `get_current_weather` function with the city set to "Boston" and the state set to "MA" (the two-letter abbreviation for Massachusetts), and the unit set to "fahrenheit" will give us the required information.



<strong style='color: #00008B;'>Function call portion:</strong>



<strong style='color: #00008B;'>  - tool name: get_current_weather</strong>



<strong style='color: #00008B;'>    parameters: {"city": "Boston", "state": "MA", "unit": "fahrenheit"}</strong>



```python
llm.shutdown()
```

## Tool Choice Mode

SGLang supports OpenAI's `tool_choice` parameter to control when and which tools the model should call. This feature is implemented using EBNF (Extended Backus-Naur Form) grammar to ensure reliable tool calling behavior.

### Supported Tool Choice Options

- **`tool_choice="required"`**: Forces the model to call at least one tool
- **`tool_choice={"type": "function", "function": {"name": "specific_function"}}`**: Forces the model to call a specific function

### Backend Compatibility

Tool choice is fully supported with the **Xgrammar backend**, which is the default grammar backend (`--grammar-backend xgrammar`). However, it may not be fully supported with other backends such as `outlines`.

### Example: Required Tool Choice


```python
from openai import OpenAI
from sglang.utils import wait_for_server, print_highlight, terminate_process
from sglang.test.doc_patch import launch_server_cmd

# Start a new server session for tool choice examples
server_process_tool_choice, port_tool_choice = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --tool-call-parser qwen25 --host 0.0.0.0  --log-level warning"
)
wait_for_server(
    f"http://localhost:{port_tool_choice}", process=server_process_tool_choice
)

# Initialize client for tool choice examples
client_tool_choice = OpenAI(
    api_key="None", base_url=f"http://0.0.0.0:{port_tool_choice}/v1"
)
model_name_tool_choice = client_tool_choice.models.list().data[0].id

# Example with tool_choice="required" - forces the model to call a tool
messages_required = [
    {"role": "user", "content": "Hello, what is the capital of France?"}
]

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "unit"],
            },
        },
    }
]

response_required = client_tool_choice.chat.completions.create(
    model=model_name_tool_choice,
    messages=messages_required,
    temperature=0,
    max_tokens=1024,
    tools=tools,
    tool_choice="required",  # Force the model to call a tool
)

print_highlight("Response with tool_choice='required':")
print("Content:", response_required.choices[0].message.content)
print("Tool calls:", response_required.choices[0].message.tool_calls)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(
    [2026-05-29 18:19:47] The tool_call_parser 'qwen25' is deprecated. Please use 'qwen' instead.


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  1.82it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.85it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:01<00:00,  1.70it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.54it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:14,  5.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:14,  5.52s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:14,  2.40s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:14,  2.40s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:16,  1.40s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:16,  1.40s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:49,  1.09it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:49,  1.09it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:25,  2.08it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:25,  2.08it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.72it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.72it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.45it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.45it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.30it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.30it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.30it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  5.99it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  5.99it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:07,  5.99it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.47it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.47it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  7.47it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.16it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.16it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.16it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.06it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.06it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.06it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.06it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.36it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.36it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.36it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.36it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.36it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.55it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.55it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.55it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.55it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.55it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.55it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.52it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.52it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.52it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.52it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.52it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.52it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 26.52it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]

    Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:07<00:00, 44.76it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:07<00:00, 44.76it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:07<00:00, 44.76it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:07<00:00, 44.76it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:08<00:00, 44.76it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:08<00:00, 44.76it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:08<00:00, 44.76it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:08<00:00, 44.76it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:08<00:00, 44.76it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:08<00:00, 52.68it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:08<00:00, 52.68it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:08<00:00, 52.68it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:08<00:00, 52.68it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:08<00:00, 52.68it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:08<00:00, 52.68it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:08<00:00, 52.68it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:08<00:00, 52.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=21.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=21.77 GB):   2%|▏         | 1/58 [00:00<00:17,  3.35it/s]Capturing num tokens (num_tokens=7680 avail_mem=21.71 GB):   2%|▏         | 1/58 [00:00<00:17,  3.35it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=21.71 GB):   3%|▎         | 2/58 [00:00<00:16,  3.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=21.70 GB):   3%|▎         | 2/58 [00:00<00:16,  3.47it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=21.70 GB):   5%|▌         | 3/58 [00:00<00:15,  3.61it/s]Capturing num tokens (num_tokens=6656 avail_mem=21.70 GB):   5%|▌         | 3/58 [00:00<00:15,  3.61it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=21.70 GB):   7%|▋         | 4/58 [00:01<00:20,  2.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=20.77 GB):   7%|▋         | 4/58 [00:01<00:20,  2.68it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=20.77 GB):   9%|▊         | 5/58 [00:01<00:17,  3.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=20.77 GB):   9%|▊         | 5/58 [00:01<00:17,  3.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=20.77 GB):  10%|█         | 6/58 [00:01<00:14,  3.63it/s]Capturing num tokens (num_tokens=5120 avail_mem=20.76 GB):  10%|█         | 6/58 [00:01<00:14,  3.63it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=20.76 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=20.77 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=20.77 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=20.77 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.77it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=20.77 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.38it/s]Capturing num tokens (num_tokens=3840 avail_mem=20.77 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.38it/s]Capturing num tokens (num_tokens=3840 avail_mem=20.77 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Capturing num tokens (num_tokens=3584 avail_mem=20.76 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=20.76 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.58it/s]Capturing num tokens (num_tokens=3328 avail_mem=20.76 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.58it/s]Capturing num tokens (num_tokens=3328 avail_mem=20.76 GB):  21%|██        | 12/58 [00:02<00:06,  7.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=20.76 GB):  21%|██        | 12/58 [00:02<00:06,  7.25it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=20.76 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.88it/s]Capturing num tokens (num_tokens=2816 avail_mem=20.76 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=20.76 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=20.76 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.35it/s]Capturing num tokens (num_tokens=2304 avail_mem=20.75 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.35it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=20.75 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.35it/s]Capturing num tokens (num_tokens=2048 avail_mem=20.75 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.91it/s]Capturing num tokens (num_tokens=1792 avail_mem=20.75 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.91it/s]Capturing num tokens (num_tokens=1536 avail_mem=20.74 GB):  29%|██▉       | 17/58 [00:03<00:03, 10.91it/s]Capturing num tokens (num_tokens=1536 avail_mem=20.74 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=20.74 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.73it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=20.73 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.73it/s]Capturing num tokens (num_tokens=960 avail_mem=20.73 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.73it/s] Capturing num tokens (num_tokens=960 avail_mem=20.73 GB):  38%|███▊      | 22/58 [00:03<00:02, 16.06it/s]Capturing num tokens (num_tokens=896 avail_mem=20.73 GB):  38%|███▊      | 22/58 [00:03<00:02, 16.06it/s]Capturing num tokens (num_tokens=832 avail_mem=20.72 GB):  38%|███▊      | 22/58 [00:03<00:02, 16.06it/s]Capturing num tokens (num_tokens=768 avail_mem=20.72 GB):  38%|███▊      | 22/58 [00:03<00:02, 16.06it/s]Capturing num tokens (num_tokens=768 avail_mem=20.72 GB):  43%|████▎     | 25/58 [00:03<00:01, 19.10it/s]Capturing num tokens (num_tokens=704 avail_mem=20.71 GB):  43%|████▎     | 25/58 [00:03<00:01, 19.10it/s]

    Capturing num tokens (num_tokens=640 avail_mem=20.71 GB):  43%|████▎     | 25/58 [00:03<00:01, 19.10it/s]Capturing num tokens (num_tokens=576 avail_mem=20.71 GB):  43%|████▎     | 25/58 [00:03<00:01, 19.10it/s]Capturing num tokens (num_tokens=576 avail_mem=20.71 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.63it/s]Capturing num tokens (num_tokens=512 avail_mem=20.70 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.63it/s]Capturing num tokens (num_tokens=480 avail_mem=20.70 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.63it/s]Capturing num tokens (num_tokens=448 avail_mem=20.70 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.63it/s]Capturing num tokens (num_tokens=448 avail_mem=20.70 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.66it/s]Capturing num tokens (num_tokens=416 avail_mem=20.69 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.66it/s]

    Capturing num tokens (num_tokens=384 avail_mem=20.69 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.66it/s]Capturing num tokens (num_tokens=352 avail_mem=20.68 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.66it/s]Capturing num tokens (num_tokens=320 avail_mem=20.68 GB):  53%|█████▎    | 31/58 [00:03<00:01, 23.66it/s]Capturing num tokens (num_tokens=320 avail_mem=20.68 GB):  60%|██████    | 35/58 [00:03<00:00, 27.14it/s]Capturing num tokens (num_tokens=288 avail_mem=20.69 GB):  60%|██████    | 35/58 [00:03<00:00, 27.14it/s]Capturing num tokens (num_tokens=256 avail_mem=20.68 GB):  60%|██████    | 35/58 [00:03<00:00, 27.14it/s]Capturing num tokens (num_tokens=240 avail_mem=20.68 GB):  60%|██████    | 35/58 [00:03<00:00, 27.14it/s]Capturing num tokens (num_tokens=224 avail_mem=20.67 GB):  60%|██████    | 35/58 [00:03<00:00, 27.14it/s]Capturing num tokens (num_tokens=224 avail_mem=20.67 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.32it/s]Capturing num tokens (num_tokens=208 avail_mem=20.67 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.32it/s]

    Capturing num tokens (num_tokens=192 avail_mem=20.67 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.32it/s]Capturing num tokens (num_tokens=176 avail_mem=20.66 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.32it/s]Capturing num tokens (num_tokens=160 avail_mem=20.66 GB):  67%|██████▋   | 39/58 [00:03<00:00, 30.32it/s]Capturing num tokens (num_tokens=160 avail_mem=20.66 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.74it/s]Capturing num tokens (num_tokens=144 avail_mem=20.65 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.74it/s]Capturing num tokens (num_tokens=128 avail_mem=20.66 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.74it/s]Capturing num tokens (num_tokens=112 avail_mem=20.65 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.74it/s]Capturing num tokens (num_tokens=96 avail_mem=20.65 GB):  74%|███████▍  | 43/58 [00:03<00:00, 32.74it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=20.65 GB):  81%|████████  | 47/58 [00:04<00:00, 27.20it/s]Capturing num tokens (num_tokens=80 avail_mem=20.64 GB):  81%|████████  | 47/58 [00:04<00:00, 27.20it/s]Capturing num tokens (num_tokens=64 avail_mem=20.64 GB):  81%|████████  | 47/58 [00:04<00:00, 27.20it/s]Capturing num tokens (num_tokens=48 avail_mem=20.62 GB):  81%|████████  | 47/58 [00:04<00:00, 27.20it/s]

    Capturing num tokens (num_tokens=48 avail_mem=20.62 GB):  86%|████████▌ | 50/58 [00:04<00:00, 18.87it/s]Capturing num tokens (num_tokens=32 avail_mem=20.61 GB):  86%|████████▌ | 50/58 [00:04<00:00, 18.87it/s]Capturing num tokens (num_tokens=28 avail_mem=20.61 GB):  86%|████████▌ | 50/58 [00:04<00:00, 18.87it/s]Capturing num tokens (num_tokens=24 avail_mem=20.61 GB):  86%|████████▌ | 50/58 [00:04<00:00, 18.87it/s]

    Capturing num tokens (num_tokens=24 avail_mem=20.61 GB):  91%|█████████▏| 53/58 [00:04<00:00, 15.38it/s]Capturing num tokens (num_tokens=20 avail_mem=20.61 GB):  91%|█████████▏| 53/58 [00:04<00:00, 15.38it/s]Capturing num tokens (num_tokens=16 avail_mem=20.60 GB):  91%|█████████▏| 53/58 [00:04<00:00, 15.38it/s]

    Capturing num tokens (num_tokens=16 avail_mem=20.60 GB):  95%|█████████▍| 55/58 [00:04<00:00, 13.64it/s]Capturing num tokens (num_tokens=12 avail_mem=20.60 GB):  95%|█████████▍| 55/58 [00:04<00:00, 13.64it/s]Capturing num tokens (num_tokens=8 avail_mem=20.59 GB):  95%|█████████▍| 55/58 [00:05<00:00, 13.64it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=20.59 GB):  98%|█████████▊| 57/58 [00:05<00:00, 10.41it/s]Capturing num tokens (num_tokens=4 avail_mem=20.59 GB):  98%|█████████▊| 57/58 [00:05<00:00, 10.41it/s]Capturing num tokens (num_tokens=4 avail_mem=20.59 GB): 100%|██████████| 58/58 [00:05<00:00, 10.78it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



<strong style='color: #00008B;'>Response with tool_choice='required':</strong>


    Content: None
    Tool calls: [ChatCompletionMessageFunctionToolCall(id='call_8be93b3e1711445889968b80', function=Function(arguments='{}', name='get_current_weather'), type='function', index=0)]


### Example: Specific Function Choice



```python
# Example with specific function choice - forces the model to call a specific function
messages_specific = [
    {"role": "user", "content": "What are the most attactive places in France?"}
]

response_specific = client_tool_choice.chat.completions.create(
    model=model_name_tool_choice,
    messages=messages_specific,
    temperature=0,
    max_tokens=1024,
    tools=tools,
    tool_choice={
        "type": "function",
        "function": {"name": "get_current_weather"},
    },  # Force the model to call the specific get_current_weather function
)

print_highlight("Response with specific function choice:")
print("Content:", response_specific.choices[0].message.content)
print("Tool calls:", response_specific.choices[0].message.tool_calls)

if response_specific.choices[0].message.tool_calls:
    tool_call = response_specific.choices[0].message.tool_calls[0]
    print_highlight(f"Called function: {tool_call.function.name}")
    print_highlight(f"Arguments: {tool_call.function.arguments}")
```


<strong style='color: #00008B;'>Response with specific function choice:</strong>


    Content: None
    Tool calls: [ChatCompletionMessageFunctionToolCall(id='call_bee4ebb245c64fc9b76fa558', function=Function(arguments='{"city": "Paris", "unit": "celsius"}', name='get_current_weather'), type='function', index=0)]



<strong style='color: #00008B;'>Called function: get_current_weather</strong>



<strong style='color: #00008B;'>Arguments: {"city": "Paris", "unit": "celsius"}</strong>



```python
terminate_process(server_process_tool_choice)
```

## Pythonic Tool Call Format (Llama-3.2 / Llama-3.3 / Llama-4)

Some Llama models (such as Llama-3.2-1B, Llama-3.2-3B, Llama-3.3-70B, and Llama-4) support a "pythonic" tool call format, where the model outputs function calls as Python code, e.g.:

```python
[get_current_weather(city="San Francisco", state="CA", unit="celsius")]
```

- The output is a Python list of function calls, with arguments as Python literals (not JSON).
- Multiple tool calls can be returned in the same list:
```python
[get_current_weather(city="San Francisco", state="CA", unit="celsius"),
 get_current_weather(city="New York", state="NY", unit="fahrenheit")]
```

For more information, refer to Meta’s documentation on  [Zero shot function calling](https://github.com/meta-llama/llama-models/blob/main/models/llama4/prompt_format.md#zero-shot-function-calling---system-message).

Note that this feature is still under development on Blackwell.

### How to enable
- Launch the server with `--tool-call-parser pythonic`
- You may also specify --chat-template with the improved template for the model (e.g., `--chat-template=examples/chat_template/tool_chat_template_llama4_pythonic.jinja`).
This is recommended because the model expects a special prompt format to reliably produce valid pythonic tool call outputs. The template ensures that the prompt structure (e.g., special tokens, message boundaries like `<|eom|>`, and function call delimiters) matches what the model was trained or fine-tuned on. If you do not use the correct chat template, tool calling may fail or produce inconsistent results.

#### Forcing Pythonic Tool Call Output Without a Chat Template
If you don't want to specify a chat template, you must give the model extremely explicit instructions in your messages to enforce pythonic output. For example, for `Llama-3.2-1B-Instruct`, you need:


```python
import openai

server_process, port = launch_server_cmd(
    " python3 -m sglang.launch_server --model-path meta-llama/Llama-3.2-1B-Instruct --tool-call-parser pythonic --tp 1  --log-level warning"  # llama-3.2-1b-instruct
)
wait_for_server(f"http://localhost:{port}", process=server_process)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The name of the city or location.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tourist_attractions",
            "description": "Get a list of top tourist attractions for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to find attractions for.",
                    }
                },
                "required": ["city"],
            },
        },
    },
]


def get_messages():
    return [
        {
            "role": "system",
            "content": (
                "You are a travel assistant. "
                "When asked to call functions, ALWAYS respond ONLY with a python list of function calls, "
                "using this format: [func_name1(param1=value1, param2=value2), func_name2(param=value)]. "
                "Do NOT use JSON, do NOT use variables, do NOT use any other format. "
                "Here is an example:\n"
                '[get_weather(location="Paris"), get_tourist_attractions(city="Paris")]'
            ),
        },
        {
            "role": "user",
            "content": (
                "I'm planning a trip to Tokyo next week. What's the weather like and what are some top tourist attractions? "
                "Propose parallel tool calls at once, using the python list of function calls format as shown above."
            ),
        },
    ]


messages = get_messages()

client = openai.Client(base_url=f"http://localhost:{port}/v1", api_key="xxxxxx")
model_name = client.models.list().data[0].id


response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0,
    top_p=0.9,
    stream=False,  # Non-streaming
    tools=tools,
)
print_highlight("Non-stream response:")
print_highlight(response_non_stream)

response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0,
    top_p=0.9,
    stream=True,
    tools=tools,
)
texts = ""
tool_calls = []
name = ""
arguments = ""

for chunk in response_stream:
    if chunk.choices[0].delta.content:
        texts += chunk.choices[0].delta.content
    if chunk.choices[0].delta.tool_calls:
        tool_calls.append(chunk.choices[0].delta.tool_calls[0])

print_highlight("Streaming Response:")
print_highlight("==== Text ====")
print_highlight(texts)

print_highlight("==== Tool Call ====")
for tool_call in tool_calls:
    print_highlight(tool_call)

terminate_process(server_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.29it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:01,  3.18s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:01,  3.18s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:01,  3.18s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:01,  3.18s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:34,  1.57it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:13,  3.74it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:13,  3.74it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:13,  3.74it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:13,  3.74it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:03<00:13,  3.74it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:03<00:13,  3.74it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:03<00:13,  3.74it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:03<00:05,  7.93it/s] 

    Compiling num tokens (num_tokens=896):  24%|██▍       | 14/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=832):  24%|██▍       | 14/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=768):  24%|██▍       | 14/58 [00:03<00:05,  7.93it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=352):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=320):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=288):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=256):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=240):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=224):  43%|████▎     | 25/58 [00:03<00:01, 17.76it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 32.28it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 32.28it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 32.28it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 32.28it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 32.28it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 32.28it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 32.28it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 32.28it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 32.28it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 32.28it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:03<00:00, 32.28it/s]Compiling num tokens (num_tokens=48):  67%|██████▋   | 39/58 [00:03<00:00, 32.28it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 43.51it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 43.51it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 43.51it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 43.51it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 43.51it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 43.51it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 43.51it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 43.51it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 43.51it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=51.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=51.63 GB):   2%|▏         | 1/58 [00:00<00:08,  6.62it/s]Capturing num tokens (num_tokens=7680 avail_mem=51.58 GB):   2%|▏         | 1/58 [00:00<00:08,  6.62it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=51.57 GB):   2%|▏         | 1/58 [00:00<00:08,  6.62it/s]Capturing num tokens (num_tokens=7168 avail_mem=51.57 GB):   5%|▌         | 3/58 [00:00<00:05,  9.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=51.09 GB):   5%|▌         | 3/58 [00:00<00:05,  9.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.93 GB):   5%|▌         | 3/58 [00:00<00:05,  9.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.93 GB):   9%|▊         | 5/58 [00:00<00:04, 12.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=50.93 GB):   9%|▊         | 5/58 [00:00<00:04, 12.60it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=50.93 GB):   9%|▊         | 5/58 [00:00<00:04, 12.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=50.93 GB):   9%|▊         | 5/58 [00:00<00:04, 12.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=50.93 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.93 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=50.93 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.32it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=50.92 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=50.92 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.40it/s]Capturing num tokens (num_tokens=3328 avail_mem=50.92 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=50.92 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=50.92 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=50.92 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=50.92 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=50.92 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=50.91 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.64it/s]Capturing num tokens (num_tokens=1792 avail_mem=50.91 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=50.91 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.64it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=50.90 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=50.90 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.64it/s]Capturing num tokens (num_tokens=960 avail_mem=50.90 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.64it/s] Capturing num tokens (num_tokens=896 avail_mem=50.90 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.64it/s]Capturing num tokens (num_tokens=896 avail_mem=50.90 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.50it/s]Capturing num tokens (num_tokens=832 avail_mem=50.89 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.50it/s]Capturing num tokens (num_tokens=768 avail_mem=50.90 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.50it/s]Capturing num tokens (num_tokens=704 avail_mem=50.89 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.50it/s]Capturing num tokens (num_tokens=640 avail_mem=50.89 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.50it/s]Capturing num tokens (num_tokens=576 avail_mem=50.89 GB):  40%|███▉      | 23/58 [00:00<00:00, 36.50it/s]Capturing num tokens (num_tokens=512 avail_mem=50.87 GB):  40%|███▉      | 23/58 [00:01<00:00, 36.50it/s]Capturing num tokens (num_tokens=480 avail_mem=50.88 GB):  40%|███▉      | 23/58 [00:01<00:00, 36.50it/s]Capturing num tokens (num_tokens=448 avail_mem=50.88 GB):  40%|███▉      | 23/58 [00:01<00:00, 36.50it/s]Capturing num tokens (num_tokens=448 avail_mem=50.88 GB):  53%|█████▎    | 31/58 [00:01<00:00, 46.95it/s]Capturing num tokens (num_tokens=416 avail_mem=50.88 GB):  53%|█████▎    | 31/58 [00:01<00:00, 46.95it/s]Capturing num tokens (num_tokens=384 avail_mem=50.88 GB):  53%|█████▎    | 31/58 [00:01<00:00, 46.95it/s]

    Capturing num tokens (num_tokens=352 avail_mem=50.88 GB):  53%|█████▎    | 31/58 [00:01<00:00, 46.95it/s]Capturing num tokens (num_tokens=320 avail_mem=50.88 GB):  53%|█████▎    | 31/58 [00:01<00:00, 46.95it/s]Capturing num tokens (num_tokens=288 avail_mem=50.87 GB):  53%|█████▎    | 31/58 [00:01<00:00, 46.95it/s]Capturing num tokens (num_tokens=256 avail_mem=50.87 GB):  53%|█████▎    | 31/58 [00:01<00:00, 46.95it/s]Capturing num tokens (num_tokens=256 avail_mem=50.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 49.01it/s]Capturing num tokens (num_tokens=240 avail_mem=50.88 GB):  64%|██████▍   | 37/58 [00:01<00:00, 49.01it/s]Capturing num tokens (num_tokens=224 avail_mem=50.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 49.01it/s]Capturing num tokens (num_tokens=208 avail_mem=50.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 49.01it/s]Capturing num tokens (num_tokens=192 avail_mem=50.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 49.01it/s]Capturing num tokens (num_tokens=176 avail_mem=50.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 49.01it/s]Capturing num tokens (num_tokens=160 avail_mem=50.86 GB):  64%|██████▍   | 37/58 [00:01<00:00, 49.01it/s]Capturing num tokens (num_tokens=160 avail_mem=50.86 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.89it/s]Capturing num tokens (num_tokens=144 avail_mem=50.86 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.89it/s]

    Capturing num tokens (num_tokens=128 avail_mem=50.86 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.89it/s]Capturing num tokens (num_tokens=112 avail_mem=50.86 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.89it/s]Capturing num tokens (num_tokens=96 avail_mem=50.86 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.89it/s] Capturing num tokens (num_tokens=80 avail_mem=50.86 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.89it/s]Capturing num tokens (num_tokens=64 avail_mem=50.85 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.89it/s]Capturing num tokens (num_tokens=48 avail_mem=50.85 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.89it/s]Capturing num tokens (num_tokens=48 avail_mem=50.85 GB):  86%|████████▌ | 50/58 [00:01<00:00, 55.11it/s]Capturing num tokens (num_tokens=32 avail_mem=50.85 GB):  86%|████████▌ | 50/58 [00:01<00:00, 55.11it/s]Capturing num tokens (num_tokens=28 avail_mem=50.85 GB):  86%|████████▌ | 50/58 [00:01<00:00, 55.11it/s]Capturing num tokens (num_tokens=24 avail_mem=50.84 GB):  86%|████████▌ | 50/58 [00:01<00:00, 55.11it/s]Capturing num tokens (num_tokens=20 avail_mem=50.84 GB):  86%|████████▌ | 50/58 [00:01<00:00, 55.11it/s]Capturing num tokens (num_tokens=16 avail_mem=50.84 GB):  86%|████████▌ | 50/58 [00:01<00:00, 55.11it/s]Capturing num tokens (num_tokens=12 avail_mem=50.84 GB):  86%|████████▌ | 50/58 [00:01<00:00, 55.11it/s]Capturing num tokens (num_tokens=8 avail_mem=50.83 GB):  86%|████████▌ | 50/58 [00:01<00:00, 55.11it/s] Capturing num tokens (num_tokens=4 avail_mem=50.83 GB):  86%|████████▌ | 50/58 [00:01<00:00, 55.11it/s]Capturing num tokens (num_tokens=4 avail_mem=50.83 GB): 100%|██████████| 58/58 [00:01<00:00, 60.86it/s]Capturing num tokens (num_tokens=4 avail_mem=50.83 GB): 100%|██████████| 58/58 [00:01<00:00, 38.92it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



<strong style='color: #00008B;'>Non-stream response:</strong>



<strong style='color: #00008B;'>ChatCompletion(id='7052f7fe144e4409b58ccc52adaab55d', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_699ce7af69e44ca78420e6da', function=Function(arguments='{"location": "Tokyo"}', name='get_weather'), type='function', index=0), ChatCompletionMessageFunctionToolCall(id='call_fe66865174f04b37bb08da2c', function=Function(arguments='{"city": "Tokyo"}', name='get_tourist_attractions'), type='function', index=1)], reasoning_content=None), matched_stop=None)], created=1780078870, model='meta-llama/Llama-3.2-1B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=20, prompt_tokens=449, total_tokens=469, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



<strong style='color: #00008B;'>Streaming Response:</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'></strong>



<strong style='color: #00008B;'>==== Tool Call ====</strong>



<strong style='color: #00008B;'>ChoiceDeltaToolCall(index=0, id='call_9083803eb1394319b852c505', function=ChoiceDeltaToolCallFunction(arguments='{"location": "Tokyo"}', name='get_weather'), type='function')</strong>



<strong style='color: #00008B;'>ChoiceDeltaToolCall(index=1, id='call_bfea7bd8cfd544938e8742fa', function=ChoiceDeltaToolCallFunction(arguments='{"city": "Tokyo"}', name='get_tourist_attractions'), type='function')</strong>


> **Note:**  
> The model may still default to JSON if it was heavily finetuned on that format. Prompt engineering (including examples) is the only way to increase the chance of pythonic output if you are not using a chat template.

## How to support a new model?
1. Update the TOOLS_TAG_LIST in sglang/srt/function_call_parser.py with the model’s tool tags. Currently supported tags include:
```
	TOOLS_TAG_LIST = [
	    “<|plugin|>“,
	    “<function=“,
	    “<tool_call>“,
	    “<|python_tag|>“,
	    “[TOOL_CALLS]”
	]
```
2. Create a new detector class in sglang/srt/function_call_parser.py that inherits from BaseFormatDetector. The detector should handle the model’s specific function call format. For example:
```
    class NewModelDetector(BaseFormatDetector):
```
3. Add the new detector to the MultiFormatParser class that manages all the format detectors.
