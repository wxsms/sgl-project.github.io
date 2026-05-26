# SGLang Frontend Language

SGLang frontend language can be used to define simple and easy prompts in a convenient, structured way.

## Launch A Server

Launch the server in your terminal and wait for it to initialize.


```python
from sglang import assistant_begin, assistant_end
from sglang import assistant, function, gen, system, user
from sglang import image
from sglang import RuntimeEndpoint
from sglang.lang.api import set_default_backend
from sglang.srt.utils import load_image
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import print_highlight, terminate_process, wait_for_server

server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
print(f"Server started on http://localhost:{port}")
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:01<00:05,  1.86s/it]

    Multi-thread loading shards:  50% Completed | 2/4 [00:03<00:03,  1.58s/it]

    Multi-thread loading shards:  75% Completed | 3/4 [00:04<00:01,  1.65s/it]

    Multi-thread loading shards: 100% Completed | 4/4 [00:06<00:00,  1.49s/it]Multi-thread loading shards: 100% Completed | 4/4 [00:06<00:00,  1.56s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:09,  5.43s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:09,  5.43s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:20,  2.52s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:20,  2.52s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:26,  1.57s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:26,  1.57s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:59,  1.11s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:59,  1.11s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:44,  1.19it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:44,  1.19it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:34,  1.49it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:34,  1.49it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:27,  1.83it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:27,  1.83it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:23,  2.16it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:23,  2.16it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:19,  2.55it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:19,  2.55it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:16,  2.97it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:16,  2.97it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:14,  3.36it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:14,  3.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:12,  3.78it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:12,  3.78it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:09<00:10,  4.25it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:09<00:10,  4.25it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:09,  4.69it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:09,  4.69it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:08,  5.24it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:08,  5.24it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:07,  5.86it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:07,  5.86it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:06,  6.51it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:06,  6.51it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:09<00:06,  6.51it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:09<00:04,  7.94it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:09<00:04,  7.94it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:09<00:04,  7.94it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:09<00:03,  9.46it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:09<00:03,  9.46it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:09<00:03,  9.46it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:10<00:03, 11.55it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:10<00:03, 11.55it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:10<00:03, 11.55it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:10<00:03, 11.55it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:10<00:02, 14.47it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:10<00:02, 14.47it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:10<00:02, 14.47it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:10<00:02, 14.47it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:10<00:01, 17.29it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:10<00:01, 17.29it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:10<00:01, 17.29it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:10<00:01, 17.29it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:10<00:01, 20.06it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:10<00:01, 20.06it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:10<00:01, 20.06it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:10<00:01, 20.06it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:10<00:01, 20.06it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:10<00:00, 23.84it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:10<00:00, 23.84it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:10<00:00, 23.84it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:10<00:00, 23.84it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:10<00:00, 23.84it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:10<00:00, 23.84it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:10<00:00, 29.05it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:10<00:00, 29.05it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:10<00:00, 29.05it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:10<00:00, 29.05it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:10<00:00, 29.05it/s]

    Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:10<00:00, 29.05it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:10<00:00, 32.67it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:10<00:00, 32.67it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:10<00:00, 32.67it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:10<00:00, 32.67it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:10<00:00, 32.67it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:10<00:00, 32.67it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:10<00:00, 35.29it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:10<00:00, 35.29it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:10<00:00, 35.29it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:10<00:00, 35.29it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:10<00:00, 35.29it/s]

    Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:10<00:00, 35.29it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:10<00:00, 35.29it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:10<00:00, 40.62it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:10<00:00, 40.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  5.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=44.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=44.38 GB):   2%|▏         | 1/58 [00:00<00:35,  1.62it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.32 GB):   2%|▏         | 1/58 [00:00<00:35,  1.62it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=44.32 GB):   3%|▎         | 2/58 [00:01<00:33,  1.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=44.31 GB):   3%|▎         | 2/58 [00:01<00:33,  1.68it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=44.31 GB):   5%|▌         | 3/58 [00:01<00:31,  1.77it/s]Capturing num tokens (num_tokens=6656 avail_mem=44.31 GB):   5%|▌         | 3/58 [00:01<00:31,  1.77it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=44.31 GB):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Capturing num tokens (num_tokens=6144 avail_mem=44.31 GB):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=44.31 GB):   9%|▊         | 5/58 [00:02<00:26,  2.03it/s]Capturing num tokens (num_tokens=5632 avail_mem=44.31 GB):   9%|▊         | 5/58 [00:02<00:26,  2.03it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=44.31 GB):  10%|█         | 6/58 [00:02<00:23,  2.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=44.31 GB):  10%|█         | 6/58 [00:02<00:23,  2.23it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=44.31 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.31 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.41it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=44.31 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=44.31 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.66it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=44.31 GB):  16%|█▌        | 9/58 [00:03<00:16,  3.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.79 GB):  16%|█▌        | 9/58 [00:03<00:16,  3.03it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=43.79 GB):  17%|█▋        | 10/58 [00:04<00:14,  3.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.81 GB):  17%|█▋        | 10/58 [00:04<00:14,  3.24it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.81 GB):  19%|█▉        | 11/58 [00:04<00:13,  3.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.84 GB):  19%|█▉        | 11/58 [00:04<00:13,  3.44it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=43.84 GB):  21%|██        | 12/58 [00:04<00:12,  3.72it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.87 GB):  21%|██        | 12/58 [00:04<00:12,  3.72it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=43.87 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.90 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.90 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.35it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.93 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.35it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=43.93 GB):  26%|██▌       | 15/58 [00:05<00:09,  4.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.96 GB):  26%|██▌       | 15/58 [00:05<00:09,  4.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.96 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.29it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.99 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.29it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=44.27 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.27 GB):  31%|███       | 18/58 [00:05<00:06,  6.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.27 GB):  31%|███       | 18/58 [00:05<00:06,  6.64it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=44.27 GB):  33%|███▎      | 19/58 [00:05<00:05,  6.87it/s]Capturing num tokens (num_tokens=1280 avail_mem=44.27 GB):  33%|███▎      | 19/58 [00:05<00:05,  6.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=44.08 GB):  33%|███▎      | 19/58 [00:05<00:05,  6.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=44.08 GB):  36%|███▌      | 21/58 [00:05<00:04,  8.35it/s]Capturing num tokens (num_tokens=960 avail_mem=44.08 GB):  36%|███▌      | 21/58 [00:05<00:04,  8.35it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=44.08 GB):  38%|███▊      | 22/58 [00:05<00:04,  8.63it/s]Capturing num tokens (num_tokens=896 avail_mem=44.24 GB):  38%|███▊      | 22/58 [00:05<00:04,  8.63it/s]Capturing num tokens (num_tokens=832 avail_mem=44.23 GB):  38%|███▊      | 22/58 [00:05<00:04,  8.63it/s]Capturing num tokens (num_tokens=832 avail_mem=44.23 GB):  41%|████▏     | 24/58 [00:06<00:03, 10.22it/s]Capturing num tokens (num_tokens=768 avail_mem=44.22 GB):  41%|████▏     | 24/58 [00:06<00:03, 10.22it/s]

    Capturing num tokens (num_tokens=704 avail_mem=44.22 GB):  41%|████▏     | 24/58 [00:06<00:03, 10.22it/s]Capturing num tokens (num_tokens=704 avail_mem=44.22 GB):  45%|████▍     | 26/58 [00:06<00:02, 11.78it/s]Capturing num tokens (num_tokens=640 avail_mem=44.21 GB):  45%|████▍     | 26/58 [00:06<00:02, 11.78it/s]Capturing num tokens (num_tokens=576 avail_mem=44.20 GB):  45%|████▍     | 26/58 [00:06<00:02, 11.78it/s]Capturing num tokens (num_tokens=576 avail_mem=44.20 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.29it/s]Capturing num tokens (num_tokens=512 avail_mem=44.19 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.29it/s]

    Capturing num tokens (num_tokens=480 avail_mem=44.19 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.29it/s]Capturing num tokens (num_tokens=448 avail_mem=44.18 GB):  48%|████▊     | 28/58 [00:06<00:02, 13.29it/s]Capturing num tokens (num_tokens=448 avail_mem=44.18 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.44it/s]Capturing num tokens (num_tokens=416 avail_mem=44.17 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.44it/s]Capturing num tokens (num_tokens=384 avail_mem=44.17 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.44it/s]Capturing num tokens (num_tokens=352 avail_mem=44.16 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.44it/s]

    Capturing num tokens (num_tokens=352 avail_mem=44.16 GB):  59%|█████▊    | 34/58 [00:06<00:01, 17.31it/s]Capturing num tokens (num_tokens=320 avail_mem=44.15 GB):  59%|█████▊    | 34/58 [00:06<00:01, 17.31it/s]Capturing num tokens (num_tokens=288 avail_mem=44.15 GB):  59%|█████▊    | 34/58 [00:06<00:01, 17.31it/s]Capturing num tokens (num_tokens=256 avail_mem=44.14 GB):  59%|█████▊    | 34/58 [00:06<00:01, 17.31it/s]Capturing num tokens (num_tokens=256 avail_mem=44.14 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.42it/s]Capturing num tokens (num_tokens=240 avail_mem=44.14 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.42it/s]Capturing num tokens (num_tokens=224 avail_mem=44.14 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.42it/s]Capturing num tokens (num_tokens=208 avail_mem=44.13 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.42it/s]

    Capturing num tokens (num_tokens=208 avail_mem=44.13 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.24it/s]Capturing num tokens (num_tokens=192 avail_mem=44.12 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.24it/s]Capturing num tokens (num_tokens=176 avail_mem=44.08 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.24it/s]Capturing num tokens (num_tokens=160 avail_mem=44.11 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.24it/s]Capturing num tokens (num_tokens=160 avail_mem=44.11 GB):  74%|███████▍  | 43/58 [00:06<00:00, 23.26it/s]Capturing num tokens (num_tokens=144 avail_mem=44.10 GB):  74%|███████▍  | 43/58 [00:06<00:00, 23.26it/s]Capturing num tokens (num_tokens=128 avail_mem=44.10 GB):  74%|███████▍  | 43/58 [00:06<00:00, 23.26it/s]Capturing num tokens (num_tokens=112 avail_mem=44.09 GB):  74%|███████▍  | 43/58 [00:06<00:00, 23.26it/s]

    Capturing num tokens (num_tokens=112 avail_mem=44.09 GB):  79%|███████▉  | 46/58 [00:07<00:00, 24.40it/s]Capturing num tokens (num_tokens=96 avail_mem=44.08 GB):  79%|███████▉  | 46/58 [00:07<00:00, 24.40it/s] Capturing num tokens (num_tokens=80 avail_mem=44.08 GB):  79%|███████▉  | 46/58 [00:07<00:00, 24.40it/s]Capturing num tokens (num_tokens=64 avail_mem=44.08 GB):  79%|███████▉  | 46/58 [00:07<00:00, 24.40it/s]Capturing num tokens (num_tokens=64 avail_mem=44.08 GB):  84%|████████▍ | 49/58 [00:07<00:00, 25.64it/s]Capturing num tokens (num_tokens=48 avail_mem=44.07 GB):  84%|████████▍ | 49/58 [00:07<00:00, 25.64it/s]Capturing num tokens (num_tokens=32 avail_mem=44.06 GB):  84%|████████▍ | 49/58 [00:07<00:00, 25.64it/s]Capturing num tokens (num_tokens=28 avail_mem=44.06 GB):  84%|████████▍ | 49/58 [00:07<00:00, 25.64it/s]

    Capturing num tokens (num_tokens=28 avail_mem=44.06 GB):  90%|████████▉ | 52/58 [00:07<00:00, 26.65it/s]Capturing num tokens (num_tokens=24 avail_mem=44.05 GB):  90%|████████▉ | 52/58 [00:07<00:00, 26.65it/s]Capturing num tokens (num_tokens=20 avail_mem=44.04 GB):  90%|████████▉ | 52/58 [00:07<00:00, 26.65it/s]Capturing num tokens (num_tokens=16 avail_mem=44.01 GB):  90%|████████▉ | 52/58 [00:07<00:00, 26.65it/s]Capturing num tokens (num_tokens=12 avail_mem=44.03 GB):  90%|████████▉ | 52/58 [00:07<00:00, 26.65it/s]Capturing num tokens (num_tokens=12 avail_mem=44.03 GB):  97%|█████████▋| 56/58 [00:07<00:00, 28.21it/s]Capturing num tokens (num_tokens=8 avail_mem=44.02 GB):  97%|█████████▋| 56/58 [00:07<00:00, 28.21it/s] Capturing num tokens (num_tokens=4 avail_mem=44.02 GB):  97%|█████████▋| 56/58 [00:07<00:00, 28.21it/s]Capturing num tokens (num_tokens=4 avail_mem=44.02 GB): 100%|██████████| 58/58 [00:07<00:00,  7.83it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:36168


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-26 22:04:08] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


## Basic Usage

The most simple way of using SGLang frontend language is a simple question answer dialog between a user and an assistant.


```python
@function
def basic_qa(s, question):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user(question)
    s += assistant(gen("answer", max_tokens=512))
```


```python
state = basic_qa("List 3 countries and their capitals.")
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>Certainly! Here are three countries along with their capitals:<br><br>1. **France** - Paris<br>2. **Japan** - Tokyo<br>3. **Brazil** - Brasília</strong>


## Multi-turn Dialog

SGLang frontend language can also be used to define multi-turn dialogs.


```python
@function
def multi_turn_qa(s):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user("Please give me a list of 3 countries and their capitals.")
    s += assistant(gen("first_answer", max_tokens=512))
    s += user("Please give me another list of 3 countries and their capitals.")
    s += assistant(gen("second_answer", max_tokens=512))
    return s


state = multi_turn_qa()
print_highlight(state["first_answer"])
print_highlight(state["second_answer"])
```


<strong style='color: #00008B;'>Certainly! Here are three countries along with their capitals:<br><br>1. **France** - Paris<br>2. **Germany** - Berlin<br>3. **Spain** - Madrid</strong>



<strong style='color: #00008B;'>Of course! Here's another list of three countries along with their capitals:<br><br>1. **Italy** - Rome<br>2. **Japan** - Tokyo<br>3. **Australia** - Canberra</strong>


## Control flow

You may use any Python code within the function to define more complex control flows.


```python
@function
def tool_use(s, question):
    s += assistant(
        "To answer this question: "
        + question
        + ". I need to use a "
        + gen("tool", choices=["calculator", "search engine"])
        + ". "
    )

    if s["tool"] == "calculator":
        s += assistant("The math expression is: " + gen("expression"))
    elif s["tool"] == "search engine":
        s += assistant("The key word to search is: " + gen("word"))


state = tool_use("What is 2 * 2?")
print_highlight(state["tool"])
print_highlight(state["expression"])
```


<strong style='color: #00008B;'>calculator</strong>



<strong style='color: #00008B;'>2 * 2<br><br>When we perform the multiplication:<br><br>2 * 2 = 4<br><br>Therefore, the value of 2 * 2 is 4.<br><br>You didn't really need a calculator for this one, but I'm happy to confirm the result!</strong>


## Parallelism

Use `fork` to launch parallel prompts. Because `sgl.gen` is non-blocking, the for loop below issues two generation calls in parallel.


```python
@function
def tip_suggestion(s):
    s += assistant(
        "Here are two tips for staying healthy: "
        "1. Balanced Diet. 2. Regular Exercise.\n\n"
    )

    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += assistant(
            f"Now, expand tip {i+1} into a paragraph:\n"
            + gen("detailed_tip", max_tokens=256, stop="\n\n")
        )

    s += assistant("Tip 1:" + forks[0]["detailed_tip"] + "\n")
    s += assistant("Tip 2:" + forks[1]["detailed_tip"] + "\n")
    s += assistant(
        "To summarize the above two tips, I can say:\n" + gen("summary", max_tokens=512)
    )


state = tip_suggestion()
print_highlight(state["summary"])
```


<strong style='color: #00008B;'>1. **Balanced Diet**: A balanced diet involves consuming a variety of nutritious foods in appropriate portions to ensure you are receiving all the necessary vitamins, minerals, and macronutrients. Focus on a diet rich in fruits, vegetables, whole grains, lean proteins, and healthy fats, and stay hydrated with plenty of water.<br><br>2. **Regular Exercise**: Regular exercise is essential for maintaining good health and preventing various health conditions. It should include both aerobic and strength training activities. Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week, plus strength training exercises at least two days a week.<br><br>By combining these two practices, you can significantly enhance your overall health and well-being.</strong>


## Constrained Decoding

Use `regex` to specify a regular expression as a decoding constraint. This is only supported for local models.


```python
@function
def regular_expression_gen(s):
    s += user("What is the IP address of the Google DNS servers?")
    s += assistant(
        gen(
            "answer",
            temperature=0,
            regex=r"((25[0-5]|2[0-4]\d|[01]?\d\d?).){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
        )
    )


state = regular_expression_gen()
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>208.67.222.222</strong>


Use `regex` to define a `JSON` decoding schema.


```python
character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
    + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
    + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
    + r"""    "wand": \{\n"""
    + r"""        "wood": "[\w\d\s]{1,16}",\n"""
    + r"""        "core": "[\w\d\s]{1,16}",\n"""
    + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "alive": "(Alive|Deceased)",\n"""
    + r"""    "patronus": "[\w\d\s]{1,16}",\n"""
    + r"""    "bogart": "[\w\d\s]{1,16}"\n"""
    + r"""\}"""
)


@function
def character_gen(s, name):
    s += user(
        f"{name} is a character in Harry Potter. Please fill in the following information about this character."
    )
    s += assistant(gen("json_output", max_tokens=256, regex=character_regex))


state = character_gen("Harry Potter")
print_highlight(state["json_output"])
```


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Holly",<br>        "core": "Phoenix feather",<br>        "length": 11.0<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Directory of All"<br>}</strong>


## Batching 

Use `run_batch` to run a batch of prompts.


```python
@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


states = text_qa.run_batch(
    [
        {"question": "What is the capital of the United Kingdom?"},
        {"question": "What is the capital of France?"},
        {"question": "What is the capital of Japan?"},
    ],
    progress_bar=True,
)

for i, state in enumerate(states):
    print_highlight(f"Answer {i+1}: {states[i]['answer']}")
```

      0%|          | 0/3 [00:00<?, ?it/s]

    100%|██████████| 3/3 [00:00<00:00, 26.53it/s]

    100%|██████████| 3/3 [00:00<00:00, 26.36it/s]

    



<strong style='color: #00008B;'>Answer 1: The capital of the United Kingdom is London.</strong>



<strong style='color: #00008B;'>Answer 2: The capital of France is Paris.</strong>



<strong style='color: #00008B;'>Answer 3: The capital of Japan is Tokyo.</strong>


## Streaming 

Use `stream` to stream the output to the user.


```python
@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


state = text_qa.run(
    question="What is the capital of France?", temperature=0.1, stream=True
)

for out in state.text_iter():
    print(out, end="", flush=True)
```

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What is the capital of France?<|im_end|>
    <|im_start|>assistant


    The

     capital

     of

     France

     is

     Paris

    .

    <|im_end|>


## Complex Prompts

You may use `{system|user|assistant}_{begin|end}` to define complex prompts.


```python
@function
def chat_example(s):
    s += system("You are a helpful assistant.")
    # Same as: s += s.system("You are a helpful assistant.")

    with s.user():
        s += "Question: What is the capital of France?"

    s += assistant_begin()
    s += "Answer: " + gen("answer", max_tokens=100, stop="\n")
    s += assistant_end()


state = chat_example()
print_highlight(state["answer"])
```


<strong style='color: #00008B;'> The capital of France is Paris.</strong>



```python
terminate_process(server_process)
```

## Multi-modal Generation

You may use SGLang frontend language to define multi-modal prompts.
See [here](https://docs.sglang.io/supported_models/text_generation/multimodal_language_models.html) for supported models.


```python
server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
print(f"Server started on http://localhost:{port}")
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-26 22:04:28] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-26 22:04:33] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:01<00:04,  1.07s/it]

    Multi-thread loading shards:  40% Completed | 2/5 [00:01<00:02,  1.23it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:02<00:01,  1.36it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:02<00:00,  1.44it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.91it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.56it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:32590



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-26 22:04:50] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


Ask a question about an image.


```python
@function
def image_qa(s, image_file, question):
    s += user(image(image_file) + question)
    s += assistant(gen("answer", max_tokens=256))


image_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
image_bytes, _ = load_image(image_url)
state = image_qa(image_bytes, "What is in the image?")
print_highlight(state["answer"])
```

    /actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py:893: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1586.)
      encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)



<strong style='color: #00008B;'>The image shows a man standing on the tailgate of a loaded taxi, using an iron to iron clothes freshly taken off the staircase, indicating he is a taxi driver who takes a break to iron his clothes on public transport. The taxi appears to be yellow, which is the typical color of taxis in New York City. The background includes tall buildings, a tree, and a yellow taxi on the street, suggesting an urban setting.</strong>



```python
terminate_process(server_process)
```
