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
    [2026-05-19 03:48:34] The tool_call_parser 'qwen25' is deprecated. Please use 'qwen' instead.


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:01<00:05,  1.76s/it]

    Multi-thread loading shards:  50% Completed | 2/4 [00:03<00:02,  1.47s/it]

    Multi-thread loading shards:  75% Completed | 3/4 [00:04<00:01,  1.39s/it]

    Multi-thread loading shards: 100% Completed | 4/4 [00:05<00:00,  1.35s/it]Multi-thread loading shards: 100% Completed | 4/4 [00:05<00:00,  1.40s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:14,  5.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:14,  5.52s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:14,  2.39s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:14,  2.39s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:16,  1.39s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:16,  1.39s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:49,  1.10it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:49,  1.10it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:34,  1.55it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:25,  2.05it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:25,  2.05it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:19,  2.65it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:19,  2.65it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:15,  3.33it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:15,  3.33it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.19it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.19it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:09,  5.08it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:09,  5.08it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:07<00:09,  5.08it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:06,  6.80it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:06,  6.80it/s]

    Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:07<00:06,  6.80it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:07<00:05,  8.23it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:07<00:05,  8.23it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:07<00:05,  8.23it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:04,  9.84it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:04,  9.84it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:07<00:04,  9.84it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:07<00:04,  9.84it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:07<00:03, 12.95it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:07<00:03, 12.95it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:07<00:03, 12.95it/s]

    Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:07<00:03, 12.95it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:07<00:03, 12.95it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:07<00:01, 18.04it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:07<00:01, 18.04it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:07<00:01, 18.04it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:07<00:01, 18.04it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:07<00:01, 18.04it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:07<00:01, 18.04it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:07<00:01, 24.92it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:07<00:01, 24.92it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:07<00:01, 24.92it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:07<00:01, 24.92it/s]

    Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:07<00:01, 24.92it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:07<00:01, 24.92it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:07<00:01, 24.92it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:07<00:01, 24.92it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 34.40it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 34.40it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 34.40it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 34.40it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 34.40it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 34.40it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:07<00:00, 34.40it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:07<00:00, 34.40it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:08<00:00, 34.40it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:08<00:00, 44.29it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:08<00:00, 44.29it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:08<00:00, 44.29it/s]

    Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:08<00:00, 44.29it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:08<00:00, 44.29it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:08<00:00, 44.29it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:08<00:00, 44.29it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:08<00:00, 44.29it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:08<00:00, 44.29it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:08<00:00, 51.70it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:08<00:00, 51.70it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:08<00:00, 51.70it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:08<00:00, 51.70it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:08<00:00, 51.70it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:08<00:00, 51.70it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:08<00:00, 51.70it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:08<00:00, 51.70it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.07it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=48.23 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=48.23 GB):   2%|▏         | 1/58 [00:00<00:16,  3.47it/s]Capturing num tokens (num_tokens=7680 avail_mem=48.19 GB):   2%|▏         | 1/58 [00:00<00:16,  3.47it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=48.19 GB):   3%|▎         | 2/58 [00:00<00:15,  3.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=48.19 GB):   3%|▎         | 2/58 [00:00<00:15,  3.64it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=48.19 GB):   5%|▌         | 3/58 [00:00<00:14,  3.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=48.19 GB):   5%|▌         | 3/58 [00:00<00:14,  3.88it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=48.19 GB):   7%|▋         | 4/58 [00:01<00:12,  4.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=48.19 GB):   7%|▋         | 4/58 [00:01<00:12,  4.16it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=48.19 GB):   9%|▊         | 5/58 [00:01<00:11,  4.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=48.19 GB):   9%|▊         | 5/58 [00:01<00:11,  4.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=48.19 GB):  10%|█         | 6/58 [00:01<00:10,  4.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=48.19 GB):  10%|█         | 6/58 [00:01<00:10,  4.82it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=48.19 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=48.19 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=48.19 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=48.19 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.69it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=48.19 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=48.19 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=48.19 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.68it/s]Capturing num tokens (num_tokens=3584 avail_mem=48.19 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.68it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=48.19 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=48.18 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=48.18 GB):  21%|██        | 12/58 [00:02<00:05,  7.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=48.18 GB):  21%|██        | 12/58 [00:02<00:05,  7.79it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=48.18 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.33it/s]Capturing num tokens (num_tokens=2816 avail_mem=48.18 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.33it/s]Capturing num tokens (num_tokens=2816 avail_mem=48.18 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=48.18 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.34it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=48.18 GB):  26%|██▌       | 15/58 [00:02<00:05,  7.46it/s]Capturing num tokens (num_tokens=2304 avail_mem=48.18 GB):  26%|██▌       | 15/58 [00:02<00:05,  7.46it/s]Capturing num tokens (num_tokens=2304 avail_mem=48.18 GB):  28%|██▊       | 16/58 [00:02<00:05,  7.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=48.17 GB):  28%|██▊       | 16/58 [00:02<00:05,  7.77it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=48.17 GB):  29%|██▉       | 17/58 [00:02<00:05,  8.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=48.17 GB):  29%|██▉       | 17/58 [00:02<00:05,  8.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=48.17 GB):  29%|██▉       | 17/58 [00:02<00:05,  8.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=48.17 GB):  33%|███▎      | 19/58 [00:02<00:04,  9.35it/s]Capturing num tokens (num_tokens=1280 avail_mem=48.17 GB):  33%|███▎      | 19/58 [00:02<00:04,  9.35it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=48.16 GB):  33%|███▎      | 19/58 [00:03<00:04,  9.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=48.16 GB):  36%|███▌      | 21/58 [00:03<00:03, 11.04it/s]Capturing num tokens (num_tokens=960 avail_mem=48.15 GB):  36%|███▌      | 21/58 [00:03<00:03, 11.04it/s] Capturing num tokens (num_tokens=896 avail_mem=48.15 GB):  36%|███▌      | 21/58 [00:03<00:03, 11.04it/s]Capturing num tokens (num_tokens=896 avail_mem=48.15 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.53it/s]Capturing num tokens (num_tokens=832 avail_mem=48.15 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.53it/s]

    Capturing num tokens (num_tokens=768 avail_mem=48.14 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.53it/s]Capturing num tokens (num_tokens=704 avail_mem=48.14 GB):  40%|███▉      | 23/58 [00:03<00:02, 12.53it/s]Capturing num tokens (num_tokens=704 avail_mem=48.14 GB):  45%|████▍     | 26/58 [00:03<00:01, 16.39it/s]Capturing num tokens (num_tokens=640 avail_mem=48.13 GB):  45%|████▍     | 26/58 [00:03<00:01, 16.39it/s]Capturing num tokens (num_tokens=576 avail_mem=57.20 GB):  45%|████▍     | 26/58 [00:03<00:01, 16.39it/s]

    Capturing num tokens (num_tokens=576 avail_mem=57.20 GB):  48%|████▊     | 28/58 [00:03<00:02, 14.99it/s]Capturing num tokens (num_tokens=512 avail_mem=57.19 GB):  48%|████▊     | 28/58 [00:03<00:02, 14.99it/s]Capturing num tokens (num_tokens=480 avail_mem=57.19 GB):  48%|████▊     | 28/58 [00:03<00:02, 14.99it/s]Capturing num tokens (num_tokens=448 avail_mem=57.19 GB):  48%|████▊     | 28/58 [00:03<00:02, 14.99it/s]Capturing num tokens (num_tokens=448 avail_mem=57.19 GB):  53%|█████▎    | 31/58 [00:03<00:01, 18.45it/s]Capturing num tokens (num_tokens=416 avail_mem=57.18 GB):  53%|█████▎    | 31/58 [00:03<00:01, 18.45it/s]Capturing num tokens (num_tokens=384 avail_mem=57.18 GB):  53%|█████▎    | 31/58 [00:03<00:01, 18.45it/s]Capturing num tokens (num_tokens=352 avail_mem=57.17 GB):  53%|█████▎    | 31/58 [00:03<00:01, 18.45it/s]Capturing num tokens (num_tokens=320 avail_mem=57.17 GB):  53%|█████▎    | 31/58 [00:03<00:01, 18.45it/s]

    Capturing num tokens (num_tokens=320 avail_mem=57.17 GB):  60%|██████    | 35/58 [00:03<00:01, 22.87it/s]Capturing num tokens (num_tokens=288 avail_mem=57.17 GB):  60%|██████    | 35/58 [00:03<00:01, 22.87it/s]Capturing num tokens (num_tokens=256 avail_mem=57.17 GB):  60%|██████    | 35/58 [00:03<00:01, 22.87it/s]Capturing num tokens (num_tokens=240 avail_mem=57.16 GB):  60%|██████    | 35/58 [00:03<00:01, 22.87it/s]Capturing num tokens (num_tokens=224 avail_mem=57.16 GB):  60%|██████    | 35/58 [00:03<00:01, 22.87it/s]Capturing num tokens (num_tokens=224 avail_mem=57.16 GB):  67%|██████▋   | 39/58 [00:03<00:00, 26.71it/s]Capturing num tokens (num_tokens=208 avail_mem=57.16 GB):  67%|██████▋   | 39/58 [00:03<00:00, 26.71it/s]Capturing num tokens (num_tokens=192 avail_mem=57.15 GB):  67%|██████▋   | 39/58 [00:03<00:00, 26.71it/s]

    Capturing num tokens (num_tokens=176 avail_mem=57.15 GB):  67%|██████▋   | 39/58 [00:03<00:00, 26.71it/s]Capturing num tokens (num_tokens=176 avail_mem=57.15 GB):  72%|███████▏  | 42/58 [00:03<00:00, 24.47it/s]Capturing num tokens (num_tokens=160 avail_mem=57.15 GB):  72%|███████▏  | 42/58 [00:03<00:00, 24.47it/s]Capturing num tokens (num_tokens=144 avail_mem=57.14 GB):  72%|███████▏  | 42/58 [00:04<00:00, 24.47it/s]Capturing num tokens (num_tokens=128 avail_mem=57.14 GB):  72%|███████▏  | 42/58 [00:04<00:00, 24.47it/s]Capturing num tokens (num_tokens=128 avail_mem=57.14 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.60it/s]Capturing num tokens (num_tokens=112 avail_mem=57.14 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.60it/s]Capturing num tokens (num_tokens=96 avail_mem=57.14 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.60it/s] Capturing num tokens (num_tokens=80 avail_mem=57.13 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.60it/s]

    Capturing num tokens (num_tokens=64 avail_mem=57.13 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.60it/s]Capturing num tokens (num_tokens=64 avail_mem=57.13 GB):  84%|████████▍ | 49/58 [00:04<00:00, 28.87it/s]Capturing num tokens (num_tokens=48 avail_mem=57.12 GB):  84%|████████▍ | 49/58 [00:04<00:00, 28.87it/s]Capturing num tokens (num_tokens=32 avail_mem=57.12 GB):  84%|████████▍ | 49/58 [00:04<00:00, 28.87it/s]Capturing num tokens (num_tokens=28 avail_mem=57.12 GB):  84%|████████▍ | 49/58 [00:04<00:00, 28.87it/s]Capturing num tokens (num_tokens=24 avail_mem=57.12 GB):  84%|████████▍ | 49/58 [00:04<00:00, 28.87it/s]Capturing num tokens (num_tokens=24 avail_mem=57.12 GB):  91%|█████████▏| 53/58 [00:04<00:00, 31.67it/s]Capturing num tokens (num_tokens=20 avail_mem=57.11 GB):  91%|█████████▏| 53/58 [00:04<00:00, 31.67it/s]Capturing num tokens (num_tokens=16 avail_mem=57.11 GB):  91%|█████████▏| 53/58 [00:04<00:00, 31.67it/s]Capturing num tokens (num_tokens=12 avail_mem=57.11 GB):  91%|█████████▏| 53/58 [00:04<00:00, 31.67it/s]

    Capturing num tokens (num_tokens=8 avail_mem=57.10 GB):  91%|█████████▏| 53/58 [00:04<00:00, 31.67it/s] Capturing num tokens (num_tokens=8 avail_mem=57.10 GB):  98%|█████████▊| 57/58 [00:04<00:00, 33.63it/s]Capturing num tokens (num_tokens=4 avail_mem=57.10 GB):  98%|█████████▊| 57/58 [00:04<00:00, 33.63it/s]Capturing num tokens (num_tokens=4 avail_mem=57.10 GB): 100%|██████████| 58/58 [00:04<00:00, 13.13it/s]


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



<strong style='color: #00008B;'>ChatCompletion(id='e1d14c20d51648faaa5f05ddc11c60f6', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content='To determine the current weather in Boston, I will use the `get_current_weather` function by providing the city name "Boston", state "MA" (which is the two-letter abbreviation for Massachusetts), and specifying the unit of temperature. Since no specific unit was requested, I\'ll assume Celsius for a more universally understood temperature scale.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_4ba97d07ddd24fbf9fbd4302', function=Function(arguments='{"city": "Boston", "state": "MA", "unit": "celsius"}', name='get_current_weather'), type='function', index=0)], reasoning_content=None), matched_stop=None)], created=1779162563, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=100, prompt_tokens=296, total_tokens=396, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



<strong style='color: #00008B;'>==== content ====</strong>



<strong style='color: #00008B;'>To determine the current weather in Boston, I will use the `get_current_weather` function by providing the city name "Boston", state "MA" (which is the two-letter abbreviation for Massachusetts), and specifying the unit of temperature. Since no specific unit was requested, I'll assume Celsius for a more universally understood temperature scale.</strong>



<strong style='color: #00008B;'>==== tool_calls ====</strong>



<strong style='color: #00008B;'>[ChatCompletionMessageFunctionToolCall(id='call_4ba97d07ddd24fbf9fbd4302', function=Function(arguments='{"city": "Boston", "state": "MA", "unit": "celsius"}', name='get_current_weather'), type='function', index=0)]</strong>


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



<strong style='color: #00008B;'>ChoiceDeltaToolCall(index=0, id='call_f3bd236a7cf7468c8e531ae2', function=ChoiceDeltaToolCallFunction(arguments='', name='get_current_weather'), type='function')</strong>



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



<strong style='color: #00008B;'>Updated message history: [{'role': 'user', 'content': "What's the weather like in Boston today? Output a reasoning before act, then use the tools to help you."}, ChatCompletionMessage(content='To determine the current weather in Boston, I will use the `get_current_weather` function by providing the city name "Boston", state "MA" (which is the two-letter abbreviation for Massachusetts), and specifying the unit of temperature. Since no specific unit was requested, I\'ll assume Celsius for a more universally understood temperature scale.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_4ba97d07ddd24fbf9fbd4302', function=Function(arguments='{"city": "Boston", "state": "MA", "unit": "celsius"}', name='get_current_weather'), type='function', index=0)], reasoning_content=None), {'role': 'tool', 'tool_call_id': 'call_4ba97d07ddd24fbf9fbd4302', 'content': "The weather in Boston, MA is 85 degrees celsius. It is partly cloudly, with highs in the 90's.", 'name': 'get_current_weather'}]</strong>


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



<strong style='color: #00008B;'>ChatCompletion(id='ffb5ea48c348442c878a5612a0a249ad', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content="There seems to be an error in the response as 85 degrees Celsius is extremely high for a typical temperature in Boston. Let's correct this by rechecking with the function and ensuring we get the right unit of measurement.", refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_6d5ac0c016004a57b3afcb45', function=Function(arguments='{"city": "Boston", "state": "MA", "unit": "celsius"}', name='get_current_weather'), type='function', index=0)], reasoning_content=None), matched_stop=None)], created=1779162565, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=79, prompt_tokens=442, total_tokens=521, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



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

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  2.03it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.92it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:01<00:00,  1.84it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.86it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:02,  5.31s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:02,  5.31s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:09,  2.31s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:09,  2.31s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:13,  1.34s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:13,  1.34s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:32,  1.61it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:32,  1.61it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:24,  2.11it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:24,  2.11it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:21,  2.40it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:21,  2.40it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:16,  2.96it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:16,  2.96it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:12,  3.77it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:12,  3.77it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:12,  3.77it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:08,  5.44it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:08,  5.44it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:08,  5.44it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.05it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.05it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  7.05it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  8.70it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  8.70it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  8.70it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.66it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.66it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 10.66it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 10.66it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 13.93it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 13.93it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 13.93it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 13.93it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 13.93it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.30it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.30it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.30it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.30it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.30it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.30it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.50it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.50it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.50it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.50it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.50it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.50it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 26.50it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]

    Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:07<00:00, 34.62it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:07<00:00, 44.95it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:07<00:00, 44.95it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:08<00:00, 44.95it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:08<00:00, 44.95it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:08<00:00, 44.95it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:08<00:00, 44.95it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:08<00:00, 44.07it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:08<00:00, 44.07it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:08<00:00, 44.07it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:08<00:00, 44.07it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:08<00:00, 44.07it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:08<00:00, 44.07it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:08<00:00, 44.07it/s]

    Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:08<00:00, 44.07it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:08<00:00, 44.07it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:08<00:00, 44.07it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:08<00:00, 55.68it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:08<00:00, 55.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=44.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=44.29 GB):   2%|▏         | 1/58 [00:00<00:22,  2.52it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.25 GB):   2%|▏         | 1/58 [00:00<00:22,  2.52it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=44.25 GB):   3%|▎         | 2/58 [00:00<00:28,  1.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=44.25 GB):   3%|▎         | 2/58 [00:00<00:28,  1.99it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=44.25 GB):   5%|▌         | 3/58 [00:01<00:28,  1.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=44.25 GB):   5%|▌         | 3/58 [00:01<00:28,  1.95it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=44.25 GB):   7%|▋         | 4/58 [00:01<00:26,  2.01it/s]Capturing num tokens (num_tokens=6144 avail_mem=44.25 GB):   7%|▋         | 4/58 [00:01<00:26,  2.01it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=44.25 GB):   9%|▊         | 5/58 [00:02<00:25,  2.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=44.25 GB):   9%|▊         | 5/58 [00:02<00:25,  2.09it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=44.25 GB):  10%|█         | 6/58 [00:02<00:23,  2.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=44.25 GB):  10%|█         | 6/58 [00:02<00:23,  2.23it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=44.25 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.25 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.37it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=44.25 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=44.25 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.56it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=44.25 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.25 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.78it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=44.25 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.00it/s]Capturing num tokens (num_tokens=3584 avail_mem=44.25 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.00it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=44.25 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=44.25 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.27it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=44.25 GB):  21%|██        | 12/58 [00:04<00:12,  3.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=44.24 GB):  21%|██        | 12/58 [00:04<00:12,  3.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=44.24 GB):  22%|██▏       | 13/58 [00:04<00:11,  4.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=44.24 GB):  22%|██▏       | 13/58 [00:04<00:11,  4.01it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=44.24 GB):  24%|██▍       | 14/58 [00:04<00:09,  4.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=44.24 GB):  24%|██▍       | 14/58 [00:04<00:09,  4.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=44.24 GB):  26%|██▌       | 15/58 [00:04<00:08,  5.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.24 GB):  26%|██▌       | 15/58 [00:04<00:08,  5.14it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=44.24 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.24 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.88it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.23 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.88it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.23 GB):  31%|███       | 18/58 [00:05<00:05,  7.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.23 GB):  31%|███       | 18/58 [00:05<00:05,  7.39it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=44.23 GB):  31%|███       | 18/58 [00:05<00:05,  7.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=44.23 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=44.21 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.80it/s]Capturing num tokens (num_tokens=960 avail_mem=44.21 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.80it/s] Capturing num tokens (num_tokens=896 avail_mem=44.21 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.80it/s]Capturing num tokens (num_tokens=896 avail_mem=44.21 GB):  40%|███▉      | 23/58 [00:05<00:02, 12.43it/s]Capturing num tokens (num_tokens=832 avail_mem=44.20 GB):  40%|███▉      | 23/58 [00:05<00:02, 12.43it/s]

    Capturing num tokens (num_tokens=768 avail_mem=44.20 GB):  40%|███▉      | 23/58 [00:05<00:02, 12.43it/s]Capturing num tokens (num_tokens=704 avail_mem=44.19 GB):  40%|███▉      | 23/58 [00:05<00:02, 12.43it/s]Capturing num tokens (num_tokens=704 avail_mem=44.19 GB):  45%|████▍     | 26/58 [00:05<00:02, 15.82it/s]Capturing num tokens (num_tokens=640 avail_mem=44.19 GB):  45%|████▍     | 26/58 [00:05<00:02, 15.82it/s]Capturing num tokens (num_tokens=576 avail_mem=44.19 GB):  45%|████▍     | 26/58 [00:05<00:02, 15.82it/s]Capturing num tokens (num_tokens=512 avail_mem=44.18 GB):  45%|████▍     | 26/58 [00:05<00:02, 15.82it/s]Capturing num tokens (num_tokens=480 avail_mem=44.18 GB):  45%|████▍     | 26/58 [00:05<00:02, 15.82it/s]Capturing num tokens (num_tokens=480 avail_mem=44.18 GB):  52%|█████▏    | 30/58 [00:05<00:01, 20.13it/s]Capturing num tokens (num_tokens=448 avail_mem=44.18 GB):  52%|█████▏    | 30/58 [00:05<00:01, 20.13it/s]

    Capturing num tokens (num_tokens=416 avail_mem=44.17 GB):  52%|█████▏    | 30/58 [00:05<00:01, 20.13it/s]Capturing num tokens (num_tokens=384 avail_mem=44.17 GB):  52%|█████▏    | 30/58 [00:05<00:01, 20.13it/s]Capturing num tokens (num_tokens=384 avail_mem=44.17 GB):  57%|█████▋    | 33/58 [00:05<00:01, 19.61it/s]Capturing num tokens (num_tokens=352 avail_mem=43.03 GB):  57%|█████▋    | 33/58 [00:05<00:01, 19.61it/s]

    Capturing num tokens (num_tokens=320 avail_mem=43.03 GB):  57%|█████▋    | 33/58 [00:06<00:01, 19.61it/s]Capturing num tokens (num_tokens=288 avail_mem=44.42 GB):  57%|█████▋    | 33/58 [00:06<00:01, 19.61it/s]Capturing num tokens (num_tokens=288 avail_mem=44.42 GB):  62%|██████▏   | 36/58 [00:06<00:01, 16.19it/s]Capturing num tokens (num_tokens=256 avail_mem=44.13 GB):  62%|██████▏   | 36/58 [00:06<00:01, 16.19it/s]

    Capturing num tokens (num_tokens=240 avail_mem=43.14 GB):  62%|██████▏   | 36/58 [00:06<00:01, 16.19it/s]Capturing num tokens (num_tokens=240 avail_mem=43.14 GB):  66%|██████▌   | 38/58 [00:06<00:01, 14.81it/s]Capturing num tokens (num_tokens=224 avail_mem=43.14 GB):  66%|██████▌   | 38/58 [00:06<00:01, 14.81it/s]

    Capturing num tokens (num_tokens=208 avail_mem=43.13 GB):  66%|██████▌   | 38/58 [00:06<00:01, 14.81it/s]Capturing num tokens (num_tokens=208 avail_mem=43.13 GB):  69%|██████▉   | 40/58 [00:06<00:01, 13.25it/s]Capturing num tokens (num_tokens=192 avail_mem=44.11 GB):  69%|██████▉   | 40/58 [00:06<00:01, 13.25it/s]Capturing num tokens (num_tokens=176 avail_mem=43.19 GB):  69%|██████▉   | 40/58 [00:06<00:01, 13.25it/s]

    Capturing num tokens (num_tokens=176 avail_mem=43.19 GB):  72%|███████▏  | 42/58 [00:06<00:01, 12.21it/s]Capturing num tokens (num_tokens=160 avail_mem=43.19 GB):  72%|███████▏  | 42/58 [00:06<00:01, 12.21it/s]Capturing num tokens (num_tokens=144 avail_mem=44.10 GB):  72%|███████▏  | 42/58 [00:06<00:01, 12.21it/s]Capturing num tokens (num_tokens=144 avail_mem=44.10 GB):  76%|███████▌  | 44/58 [00:06<00:01, 11.77it/s]Capturing num tokens (num_tokens=128 avail_mem=43.25 GB):  76%|███████▌  | 44/58 [00:06<00:01, 11.77it/s]

    Capturing num tokens (num_tokens=112 avail_mem=43.23 GB):  76%|███████▌  | 44/58 [00:07<00:01, 11.77it/s]Capturing num tokens (num_tokens=112 avail_mem=43.23 GB):  79%|███████▉  | 46/58 [00:07<00:01, 11.07it/s]Capturing num tokens (num_tokens=96 avail_mem=44.08 GB):  79%|███████▉  | 46/58 [00:07<00:01, 11.07it/s] Capturing num tokens (num_tokens=80 avail_mem=43.28 GB):  79%|███████▉  | 46/58 [00:07<00:01, 11.07it/s]

    Capturing num tokens (num_tokens=80 avail_mem=43.28 GB):  83%|████████▎ | 48/58 [00:07<00:00, 10.61it/s]Capturing num tokens (num_tokens=64 avail_mem=43.58 GB):  83%|████████▎ | 48/58 [00:07<00:00, 10.61it/s]Capturing num tokens (num_tokens=48 avail_mem=43.41 GB):  83%|████████▎ | 48/58 [00:07<00:00, 10.61it/s]Capturing num tokens (num_tokens=48 avail_mem=43.41 GB):  86%|████████▌ | 50/58 [00:07<00:00, 11.03it/s]Capturing num tokens (num_tokens=32 avail_mem=42.68 GB):  86%|████████▌ | 50/58 [00:07<00:00, 11.03it/s]

    Capturing num tokens (num_tokens=28 avail_mem=43.41 GB):  86%|████████▌ | 50/58 [00:07<00:00, 11.03it/s]Capturing num tokens (num_tokens=28 avail_mem=43.41 GB):  90%|████████▉ | 52/58 [00:07<00:00, 11.32it/s]Capturing num tokens (num_tokens=24 avail_mem=42.74 GB):  90%|████████▉ | 52/58 [00:07<00:00, 11.32it/s]Capturing num tokens (num_tokens=20 avail_mem=42.74 GB):  90%|████████▉ | 52/58 [00:07<00:00, 11.32it/s]

    Capturing num tokens (num_tokens=20 avail_mem=42.74 GB):  93%|█████████▎| 54/58 [00:07<00:00, 11.64it/s]Capturing num tokens (num_tokens=16 avail_mem=43.36 GB):  93%|█████████▎| 54/58 [00:07<00:00, 11.64it/s]Capturing num tokens (num_tokens=12 avail_mem=42.80 GB):  93%|█████████▎| 54/58 [00:08<00:00, 11.64it/s]Capturing num tokens (num_tokens=12 avail_mem=42.80 GB):  97%|█████████▋| 56/58 [00:08<00:00, 11.79it/s]Capturing num tokens (num_tokens=8 avail_mem=43.39 GB):  97%|█████████▋| 56/58 [00:08<00:00, 11.79it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=42.86 GB):  97%|█████████▋| 56/58 [00:08<00:00, 11.79it/s]Capturing num tokens (num_tokens=4 avail_mem=42.86 GB): 100%|██████████| 58/58 [00:08<00:00, 12.07it/s]Capturing num tokens (num_tokens=4 avail_mem=42.86 GB): 100%|██████████| 58/58 [00:08<00:00,  7.05it/s]



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
    [2026-05-19 03:50:06] The tool_call_parser 'qwen25' is deprecated. Please use 'qwen' instead.


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  1.93it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.79it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:01<00:00,  1.72it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.79it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:29,  5.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:29,  5.78s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:20,  2.51s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:20,  2.51s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:19,  1.45s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:19,  1.45s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:51,  1.05it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:51,  1.05it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:25,  2.04it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:25,  2.04it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:19,  2.67it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:19,  2.67it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.39it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.39it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:11,  4.22it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:11,  4.22it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:07<00:11,  4.22it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:07,  5.90it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:07,  5.90it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:07,  5.90it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.47it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.47it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  7.47it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.11it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.11it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.11it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.97it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.97it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 10.97it/s]

    Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 10.97it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.06it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.06it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.06it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.06it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.06it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.51it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.51it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.51it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.51it/s]

    Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.51it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.51it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 19.51it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.28it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.28it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:08<00:00, 28.28it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:08<00:00, 28.28it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:08<00:00, 28.28it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:08<00:00, 28.28it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:08<00:00, 28.28it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:08<00:00, 35.91it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:08<00:00, 35.91it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:08<00:00, 35.91it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:08<00:00, 35.91it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:08<00:00, 35.91it/s]

    Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:08<00:00, 35.91it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:08<00:00, 35.91it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:08<00:00, 35.91it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:08<00:00, 35.91it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:08<00:00, 46.20it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:08<00:00, 46.20it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:08<00:00, 46.20it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:08<00:00, 46.20it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:08<00:00, 46.20it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:08<00:00, 46.20it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:08<00:00, 46.20it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:08<00:00, 46.20it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:08<00:00, 46.20it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:08<00:00, 54.60it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:08<00:00, 54.60it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:08<00:00, 54.60it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:08<00:00, 54.60it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:08<00:00, 54.60it/s]

    Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:08<00:00, 54.60it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:08<00:00, 54.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=45.39 GB):   2%|▏         | 1/58 [00:00<00:16,  3.44it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.35 GB):   2%|▏         | 1/58 [00:00<00:16,  3.44it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=45.35 GB):   3%|▎         | 2/58 [00:00<00:15,  3.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=45.35 GB):   3%|▎         | 2/58 [00:00<00:15,  3.54it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=45.35 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=45.35 GB):   5%|▌         | 3/58 [00:00<00:14,  3.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=45.35 GB):   7%|▋         | 4/58 [00:01<00:13,  4.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.35 GB):   7%|▋         | 4/58 [00:01<00:13,  4.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=45.35 GB):   9%|▊         | 5/58 [00:01<00:12,  4.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.35 GB):   9%|▊         | 5/58 [00:01<00:12,  4.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.35 GB):  10%|█         | 6/58 [00:01<00:10,  4.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.35 GB):  10%|█         | 6/58 [00:01<00:10,  4.77it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=45.35 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.35 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.14it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.35 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.65it/s]Capturing num tokens (num_tokens=4096 avail_mem=45.35 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.65it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=45.35 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.35 GB):  16%|█▌        | 9/58 [00:01<00:07,  6.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.35 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=45.35 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.62it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=45.35 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.35 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.35 GB):  21%|██        | 12/58 [00:02<00:05,  7.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=45.34 GB):  21%|██        | 12/58 [00:02<00:05,  7.74it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=45.34 GB):  21%|██        | 12/58 [00:02<00:05,  7.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.34 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.72it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.34 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.72it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=45.34 GB):  26%|██▌       | 15/58 [00:02<00:05,  7.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.24 GB):  26%|██▌       | 15/58 [00:02<00:05,  7.72it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=44.24 GB):  28%|██▊       | 16/58 [00:02<00:06,  6.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.24 GB):  28%|██▊       | 16/58 [00:02<00:06,  6.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.24 GB):  28%|██▊       | 16/58 [00:02<00:06,  6.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.24 GB):  31%|███       | 18/58 [00:02<00:04,  8.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.23 GB):  31%|███       | 18/58 [00:02<00:04,  8.40it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=44.23 GB):  31%|███       | 18/58 [00:02<00:04,  8.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=44.23 GB):  34%|███▍      | 20/58 [00:03<00:03, 10.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=44.22 GB):  34%|███▍      | 20/58 [00:03<00:03, 10.32it/s]Capturing num tokens (num_tokens=960 avail_mem=44.21 GB):  34%|███▍      | 20/58 [00:03<00:03, 10.32it/s] Capturing num tokens (num_tokens=960 avail_mem=44.21 GB):  38%|███▊      | 22/58 [00:03<00:03, 11.81it/s]Capturing num tokens (num_tokens=896 avail_mem=44.21 GB):  38%|███▊      | 22/58 [00:03<00:03, 11.81it/s]

    Capturing num tokens (num_tokens=832 avail_mem=44.21 GB):  38%|███▊      | 22/58 [00:03<00:03, 11.81it/s]Capturing num tokens (num_tokens=832 avail_mem=44.21 GB):  41%|████▏     | 24/58 [00:03<00:03, 10.31it/s]Capturing num tokens (num_tokens=768 avail_mem=44.18 GB):  41%|████▏     | 24/58 [00:03<00:03, 10.31it/s]

    Capturing num tokens (num_tokens=704 avail_mem=44.18 GB):  41%|████▏     | 24/58 [00:03<00:03, 10.31it/s]Capturing num tokens (num_tokens=704 avail_mem=44.18 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.32it/s]Capturing num tokens (num_tokens=640 avail_mem=44.18 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.32it/s]Capturing num tokens (num_tokens=576 avail_mem=44.17 GB):  45%|████▍     | 26/58 [00:03<00:02, 11.32it/s]Capturing num tokens (num_tokens=576 avail_mem=44.17 GB):  48%|████▊     | 28/58 [00:03<00:02, 12.59it/s]Capturing num tokens (num_tokens=512 avail_mem=44.17 GB):  48%|████▊     | 28/58 [00:03<00:02, 12.59it/s]

    Capturing num tokens (num_tokens=480 avail_mem=44.16 GB):  48%|████▊     | 28/58 [00:03<00:02, 12.59it/s]Capturing num tokens (num_tokens=480 avail_mem=44.16 GB):  52%|█████▏    | 30/58 [00:03<00:02, 12.66it/s]Capturing num tokens (num_tokens=448 avail_mem=44.15 GB):  52%|█████▏    | 30/58 [00:03<00:02, 12.66it/s]Capturing num tokens (num_tokens=416 avail_mem=43.67 GB):  52%|█████▏    | 30/58 [00:03<00:02, 12.66it/s]Capturing num tokens (num_tokens=416 avail_mem=43.67 GB):  55%|█████▌    | 32/58 [00:03<00:01, 13.61it/s]Capturing num tokens (num_tokens=384 avail_mem=43.57 GB):  55%|█████▌    | 32/58 [00:03<00:01, 13.61it/s]

    Capturing num tokens (num_tokens=352 avail_mem=43.50 GB):  55%|█████▌    | 32/58 [00:03<00:01, 13.61it/s]Capturing num tokens (num_tokens=352 avail_mem=43.50 GB):  59%|█████▊    | 34/58 [00:04<00:01, 13.95it/s]Capturing num tokens (num_tokens=320 avail_mem=43.49 GB):  59%|█████▊    | 34/58 [00:04<00:01, 13.95it/s]Capturing num tokens (num_tokens=288 avail_mem=43.50 GB):  59%|█████▊    | 34/58 [00:04<00:01, 13.95it/s]Capturing num tokens (num_tokens=288 avail_mem=43.50 GB):  62%|██████▏   | 36/58 [00:04<00:01, 15.12it/s]Capturing num tokens (num_tokens=256 avail_mem=43.49 GB):  62%|██████▏   | 36/58 [00:04<00:01, 15.12it/s]

    Capturing num tokens (num_tokens=240 avail_mem=43.49 GB):  62%|██████▏   | 36/58 [00:04<00:01, 15.12it/s]Capturing num tokens (num_tokens=224 avail_mem=43.49 GB):  62%|██████▏   | 36/58 [00:04<00:01, 15.12it/s]Capturing num tokens (num_tokens=208 avail_mem=43.48 GB):  62%|██████▏   | 36/58 [00:04<00:01, 15.12it/s]Capturing num tokens (num_tokens=208 avail_mem=43.48 GB):  69%|██████▉   | 40/58 [00:04<00:00, 20.58it/s]Capturing num tokens (num_tokens=192 avail_mem=43.48 GB):  69%|██████▉   | 40/58 [00:04<00:00, 20.58it/s]Capturing num tokens (num_tokens=176 avail_mem=43.48 GB):  69%|██████▉   | 40/58 [00:04<00:00, 20.58it/s]Capturing num tokens (num_tokens=160 avail_mem=43.47 GB):  69%|██████▉   | 40/58 [00:04<00:00, 20.58it/s]Capturing num tokens (num_tokens=144 avail_mem=43.47 GB):  69%|██████▉   | 40/58 [00:04<00:00, 20.58it/s]

    Capturing num tokens (num_tokens=144 avail_mem=43.47 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.22it/s]Capturing num tokens (num_tokens=128 avail_mem=43.47 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.22it/s]Capturing num tokens (num_tokens=112 avail_mem=43.47 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.22it/s]Capturing num tokens (num_tokens=96 avail_mem=43.46 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.22it/s] Capturing num tokens (num_tokens=80 avail_mem=43.46 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.22it/s]Capturing num tokens (num_tokens=80 avail_mem=43.46 GB):  83%|████████▎ | 48/58 [00:04<00:00, 27.82it/s]Capturing num tokens (num_tokens=64 avail_mem=43.45 GB):  83%|████████▎ | 48/58 [00:04<00:00, 27.82it/s]Capturing num tokens (num_tokens=48 avail_mem=43.45 GB):  83%|████████▎ | 48/58 [00:04<00:00, 27.82it/s]Capturing num tokens (num_tokens=32 avail_mem=43.45 GB):  83%|████████▎ | 48/58 [00:04<00:00, 27.82it/s]Capturing num tokens (num_tokens=28 avail_mem=43.45 GB):  83%|████████▎ | 48/58 [00:04<00:00, 27.82it/s]

    Capturing num tokens (num_tokens=28 avail_mem=43.45 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.93it/s]Capturing num tokens (num_tokens=24 avail_mem=43.44 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.93it/s]Capturing num tokens (num_tokens=20 avail_mem=43.44 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.93it/s]Capturing num tokens (num_tokens=16 avail_mem=43.43 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.93it/s]Capturing num tokens (num_tokens=12 avail_mem=43.43 GB):  90%|████████▉ | 52/58 [00:04<00:00, 30.93it/s]Capturing num tokens (num_tokens=12 avail_mem=43.43 GB):  97%|█████████▋| 56/58 [00:04<00:00, 32.65it/s]Capturing num tokens (num_tokens=8 avail_mem=43.43 GB):  97%|█████████▋| 56/58 [00:04<00:00, 32.65it/s] Capturing num tokens (num_tokens=4 avail_mem=43.42 GB):  97%|█████████▋| 56/58 [00:04<00:00, 32.65it/s]Capturing num tokens (num_tokens=4 avail_mem=43.42 GB): 100%|██████████| 58/58 [00:04<00:00, 12.12it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



<strong style='color: #00008B;'>Response with tool_choice='required':</strong>


    Content: None
    Tool calls: [ChatCompletionMessageFunctionToolCall(id='call_1f0a23ba949644e59d110e30', function=Function(arguments='{}', name='get_current_weather'), type='function', index=0)]


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
    Tool calls: [ChatCompletionMessageFunctionToolCall(id='call_87d4a8c513c14923ad8586d4', function=Function(arguments='{"city": "Paris", "unit": "celsius"}', name='get_current_weather'), type='function', index=0)]



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


    [2026-05-19 03:50:51] Tokenizer loaded as generic TokenizersBackend for meta-llama/Llama-3.2-1B-Instruct, retrying with use_fast=False


    [2026-05-19 03:50:53] Tokenizer for meta-llama/Llama-3.2-1B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [2026-05-19 03:50:57] Tokenizer loaded as generic TokenizersBackend for meta-llama/Llama-3.2-1B-Instruct, retrying with use_fast=False


    [2026-05-19 03:50:57] Tokenizer loaded as generic TokenizersBackend for meta-llama/Llama-3.2-1B-Instruct, retrying with use_fast=False


    [2026-05-19 03:50:59] Tokenizer for meta-llama/Llama-3.2-1B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.
    [2026-05-19 03:50:59] Tokenizer for meta-llama/Llama-3.2-1B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:57,  3.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:57,  3.12s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<02:57,  3.12s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<02:57,  3.12s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.60it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.60it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:33,  1.60it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:33,  1.60it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:33,  1.60it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:13,  3.81it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:13,  3.81it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:13,  3.81it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:13,  3.81it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:03<00:13,  3.81it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:03<00:13,  3.81it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:03<00:13,  3.81it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:05,  8.05it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:05,  8.05it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:05,  8.05it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:05,  8.05it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:05,  8.05it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:03<00:05,  8.05it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:03<00:05,  8.05it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:03<00:05,  8.05it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:03<00:05,  8.05it/s] 

    Compiling num tokens (num_tokens=896):  24%|██▍       | 14/58 [00:03<00:05,  8.05it/s]Compiling num tokens (num_tokens=832):  24%|██▍       | 14/58 [00:03<00:05,  8.05it/s]Compiling num tokens (num_tokens=768):  24%|██▍       | 14/58 [00:03<00:05,  8.05it/s]Compiling num tokens (num_tokens=704):  24%|██▍       | 14/58 [00:03<00:05,  8.05it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Compiling num tokens (num_tokens=320):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Compiling num tokens (num_tokens=288):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Compiling num tokens (num_tokens=256):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 29.98it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 29.98it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 29.98it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 29.98it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 29.98it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 29.98it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 29.98it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 29.98it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:03<00:00, 29.98it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:03<00:00, 29.98it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:03<00:00, 29.98it/s] Compiling num tokens (num_tokens=80):  64%|██████▍   | 37/58 [00:03<00:00, 29.98it/s]Compiling num tokens (num_tokens=64):  64%|██████▍   | 37/58 [00:03<00:00, 29.98it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 42.85it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 42.85it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 42.85it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 42.85it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 42.85it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 42.85it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 42.85it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 42.85it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 42.85it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 42.85it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.38 GB):   3%|▎         | 2/58 [00:00<00:03, 14.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.38 GB):   3%|▎         | 2/58 [00:00<00:03, 14.32it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.38 GB):   3%|▎         | 2/58 [00:00<00:03, 14.32it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=58.38 GB):   7%|▋         | 4/58 [00:00<00:03, 16.00it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.38 GB):   7%|▋         | 4/58 [00:00<00:03, 16.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.38 GB):   7%|▋         | 4/58 [00:00<00:03, 16.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.38 GB):   7%|▋         | 4/58 [00:00<00:03, 16.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.38 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.38 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.38 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.96it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=58.38 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.38 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.38 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.38 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.38 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.38 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.14it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.14it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=58.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.36 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s]Capturing num tokens (num_tokens=960 avail_mem=58.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s] Capturing num tokens (num_tokens=896 avail_mem=58.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s]Capturing num tokens (num_tokens=832 avail_mem=58.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s]Capturing num tokens (num_tokens=768 avail_mem=58.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s]Capturing num tokens (num_tokens=704 avail_mem=58.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s]Capturing num tokens (num_tokens=640 avail_mem=58.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.60it/s]Capturing num tokens (num_tokens=640 avail_mem=58.35 GB):  47%|████▋     | 27/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=576 avail_mem=58.35 GB):  47%|████▋     | 27/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=512 avail_mem=58.32 GB):  47%|████▋     | 27/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=480 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=448 avail_mem=58.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 45.58it/s]

    Capturing num tokens (num_tokens=416 avail_mem=58.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=384 avail_mem=58.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 45.58it/s]Capturing num tokens (num_tokens=352 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:00<00:00, 52.62it/s]Capturing num tokens (num_tokens=320 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:00<00:00, 52.62it/s]Capturing num tokens (num_tokens=288 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:00<00:00, 52.62it/s]Capturing num tokens (num_tokens=256 avail_mem=58.32 GB):  59%|█████▊    | 34/58 [00:00<00:00, 52.62it/s]Capturing num tokens (num_tokens=240 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:00<00:00, 52.62it/s]Capturing num tokens (num_tokens=224 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:00<00:00, 52.62it/s]Capturing num tokens (num_tokens=208 avail_mem=58.33 GB):  59%|█████▊    | 34/58 [00:00<00:00, 52.62it/s]Capturing num tokens (num_tokens=208 avail_mem=58.33 GB):  69%|██████▉   | 40/58 [00:01<00:00, 53.51it/s]Capturing num tokens (num_tokens=192 avail_mem=58.32 GB):  69%|██████▉   | 40/58 [00:01<00:00, 53.51it/s]Capturing num tokens (num_tokens=176 avail_mem=58.32 GB):  69%|██████▉   | 40/58 [00:01<00:00, 53.51it/s]Capturing num tokens (num_tokens=160 avail_mem=58.32 GB):  69%|██████▉   | 40/58 [00:01<00:00, 53.51it/s]Capturing num tokens (num_tokens=144 avail_mem=58.32 GB):  69%|██████▉   | 40/58 [00:01<00:00, 53.51it/s]

    Capturing num tokens (num_tokens=128 avail_mem=58.32 GB):  69%|██████▉   | 40/58 [00:01<00:00, 53.51it/s]Capturing num tokens (num_tokens=112 avail_mem=58.32 GB):  69%|██████▉   | 40/58 [00:01<00:00, 53.51it/s]Capturing num tokens (num_tokens=96 avail_mem=58.31 GB):  69%|██████▉   | 40/58 [00:01<00:00, 53.51it/s] Capturing num tokens (num_tokens=96 avail_mem=58.31 GB):  81%|████████  | 47/58 [00:01<00:00, 57.65it/s]Capturing num tokens (num_tokens=80 avail_mem=58.31 GB):  81%|████████  | 47/58 [00:01<00:00, 57.65it/s]Capturing num tokens (num_tokens=64 avail_mem=58.31 GB):  81%|████████  | 47/58 [00:01<00:00, 57.65it/s]Capturing num tokens (num_tokens=48 avail_mem=58.31 GB):  81%|████████  | 47/58 [00:01<00:00, 57.65it/s]Capturing num tokens (num_tokens=32 avail_mem=58.30 GB):  81%|████████  | 47/58 [00:01<00:00, 57.65it/s]Capturing num tokens (num_tokens=28 avail_mem=58.30 GB):  81%|████████  | 47/58 [00:01<00:00, 57.65it/s]Capturing num tokens (num_tokens=24 avail_mem=58.30 GB):  81%|████████  | 47/58 [00:01<00:00, 57.65it/s]Capturing num tokens (num_tokens=20 avail_mem=58.30 GB):  81%|████████  | 47/58 [00:01<00:00, 57.65it/s]Capturing num tokens (num_tokens=20 avail_mem=58.30 GB):  93%|█████████▎| 54/58 [00:01<00:00, 60.86it/s]Capturing num tokens (num_tokens=16 avail_mem=58.30 GB):  93%|█████████▎| 54/58 [00:01<00:00, 60.86it/s]Capturing num tokens (num_tokens=12 avail_mem=58.29 GB):  93%|█████████▎| 54/58 [00:01<00:00, 60.86it/s]Capturing num tokens (num_tokens=8 avail_mem=58.29 GB):  93%|█████████▎| 54/58 [00:01<00:00, 60.86it/s] Capturing num tokens (num_tokens=4 avail_mem=58.29 GB):  93%|█████████▎| 54/58 [00:01<00:00, 60.86it/s]

    Capturing num tokens (num_tokens=4 avail_mem=58.29 GB): 100%|██████████| 58/58 [00:01<00:00, 44.87it/s]


    [2026-05-19 03:51:15] Tokenizer loaded as generic TokenizersBackend for meta-llama/Llama-3.2-1B-Instruct, retrying with use_fast=False


    [2026-05-19 03:51:17] Tokenizer for meta-llama/Llama-3.2-1B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



<strong style='color: #00008B;'>Non-stream response:</strong>



<strong style='color: #00008B;'>ChatCompletion(id='570aaed8a16741a8be9bbfaa6cbd9ef4', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_cd73927fd42f4057a921ddee', function=Function(arguments='{"location": "Tokyo"}', name='get_weather'), type='function', index=0), ChatCompletionMessageFunctionToolCall(id='call_f762474f50624205933bcb51', function=Function(arguments='{"city": "Tokyo"}', name='get_tourist_attractions'), type='function', index=1)], reasoning_content=None), matched_stop=None)], created=1779162684, model='meta-llama/Llama-3.2-1B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=20, prompt_tokens=449, total_tokens=469, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>



<strong style='color: #00008B;'>Streaming Response:</strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'></strong>



<strong style='color: #00008B;'>==== Tool Call ====</strong>



<strong style='color: #00008B;'>ChoiceDeltaToolCall(index=0, id='call_4b6bf9731b1f46d0bae09e7a', function=ChoiceDeltaToolCallFunction(arguments='{"location": "Tokyo"}', name='get_weather'), type='function')</strong>



<strong style='color: #00008B;'>ChoiceDeltaToolCall(index=1, id='call_8d094670f8e74fcd8e17d9b2', function=ChoiceDeltaToolCallFunction(arguments='{"city": "Tokyo"}', name='get_tourist_attractions'), type='function')</strong>


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
