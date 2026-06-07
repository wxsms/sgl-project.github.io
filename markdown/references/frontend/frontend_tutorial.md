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

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  1.51it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.41it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.43it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.50it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.47it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:40,  4.92s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:40,  4.92s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:09,  2.31s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:09,  2.31s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:20,  1.46s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:20,  1.46s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:55,  1.03s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:55,  1.03s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:41,  1.27it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:41,  1.27it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:32,  1.58it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:32,  1.58it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:26,  1.92it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:26,  1.92it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:22,  2.27it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:22,  2.27it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:18,  2.65it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:18,  2.65it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:15,  3.06it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:15,  3.06it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:13,  3.47it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:13,  3.47it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:11,  3.92it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:11,  3.92it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:10,  4.38it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:10,  4.38it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:09,  4.82it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:09,  4.82it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:08,  5.30it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:08,  5.30it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:07,  5.92it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:07,  5.92it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:06,  6.68it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:06,  6.68it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:05,  7.29it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:05,  7.29it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:09<00:05,  7.29it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:04,  8.88it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:04,  8.88it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:09<00:04,  8.88it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:09<00:03, 10.36it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:09<00:03, 10.36it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:09<00:03, 10.36it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:09<00:03, 10.36it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:09<00:02, 13.21it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:09<00:02, 13.21it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:09<00:02, 13.21it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:09<00:02, 13.21it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:09<00:01, 16.08it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:09<00:01, 16.08it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:09<00:01, 16.08it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:09<00:01, 16.08it/s]

    Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:09<00:01, 16.08it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:09<00:01, 19.61it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:09<00:01, 19.61it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:09<00:01, 19.61it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:09<00:01, 19.61it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:09<00:01, 19.61it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:09<00:00, 22.77it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:09<00:00, 22.77it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:09<00:00, 22.77it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:10<00:00, 22.77it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:10<00:00, 22.77it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:10<00:00, 22.77it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:10<00:00, 27.98it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:10<00:00, 27.98it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:10<00:00, 27.98it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:10<00:00, 27.98it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:10<00:00, 27.98it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:10<00:00, 30.85it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:10<00:00, 30.85it/s]

    Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:10<00:00, 30.85it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:10<00:00, 30.85it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:10<00:00, 30.85it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:10<00:00, 30.85it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:10<00:00, 34.28it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:10<00:00, 34.28it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:10<00:00, 34.28it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:10<00:00, 34.28it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:10<00:00, 34.28it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:10<00:00, 34.28it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:10<00:00, 34.28it/s]

    Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:10<00:00, 40.21it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:10<00:00, 40.21it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:10<00:00, 40.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.33 GB):   2%|▏         | 1/58 [00:00<00:30,  1.87it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.33 GB):   2%|▏         | 1/58 [00:00<00:30,  1.87it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.33 GB):   3%|▎         | 2/58 [00:01<00:29,  1.92it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.45 GB):   3%|▎         | 2/58 [00:01<00:29,  1.92it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.45 GB):   5%|▌         | 3/58 [00:01<00:27,  2.01it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.27 GB):   5%|▌         | 3/58 [00:01<00:27,  2.01it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.27 GB):   7%|▋         | 4/58 [00:01<00:25,  2.08it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.27 GB):   7%|▋         | 4/58 [00:01<00:25,  2.08it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.27 GB):   9%|▊         | 5/58 [00:02<00:24,  2.16it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.56 GB):   9%|▊         | 5/58 [00:02<00:24,  2.16it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=42.56 GB):  10%|█         | 6/58 [00:02<00:22,  2.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.61 GB):  10%|█         | 6/58 [00:02<00:22,  2.32it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.61 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.66 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.41it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=42.66 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.27 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.62it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.27 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.82it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.27 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.82it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=43.27 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.06it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.77 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.06it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.77 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.82 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.33it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=42.82 GB):  21%|██        | 12/58 [00:04<00:12,  3.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.84 GB):  21%|██        | 12/58 [00:04<00:12,  3.64it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=42.84 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.91it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.16 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.91it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=43.16 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.25 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.25 GB):  26%|██▌       | 15/58 [00:05<00:09,  4.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.89 GB):  26%|██▌       | 15/58 [00:05<00:09,  4.54it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=42.89 GB):  28%|██▊       | 16/58 [00:05<00:08,  5.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.91 GB):  28%|██▊       | 16/58 [00:05<00:08,  5.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.91 GB):  29%|██▉       | 17/58 [00:05<00:07,  5.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.24 GB):  29%|██▉       | 17/58 [00:05<00:07,  5.68it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=43.24 GB):  31%|███       | 18/58 [00:05<00:06,  6.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.24 GB):  31%|███       | 18/58 [00:05<00:06,  6.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.24 GB):  33%|███▎      | 19/58 [00:05<00:05,  6.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.03 GB):  33%|███▎      | 19/58 [00:05<00:05,  6.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.97 GB):  33%|███▎      | 19/58 [00:05<00:05,  6.57it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=42.97 GB):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Capturing num tokens (num_tokens=960 avail_mem=43.21 GB):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s] Capturing num tokens (num_tokens=896 avail_mem=43.21 GB):  36%|███▌      | 21/58 [00:05<00:04,  8.28it/s]Capturing num tokens (num_tokens=896 avail_mem=43.21 GB):  40%|███▉      | 23/58 [00:05<00:03,  9.50it/s]Capturing num tokens (num_tokens=832 avail_mem=43.01 GB):  40%|███▉      | 23/58 [00:05<00:03,  9.50it/s]

    Capturing num tokens (num_tokens=768 avail_mem=43.03 GB):  40%|███▉      | 23/58 [00:05<00:03,  9.50it/s]Capturing num tokens (num_tokens=768 avail_mem=43.03 GB):  43%|████▎     | 25/58 [00:06<00:03, 10.80it/s]Capturing num tokens (num_tokens=704 avail_mem=43.19 GB):  43%|████▎     | 25/58 [00:06<00:03, 10.80it/s]Capturing num tokens (num_tokens=640 avail_mem=43.19 GB):  43%|████▎     | 25/58 [00:06<00:03, 10.80it/s]

    Capturing num tokens (num_tokens=640 avail_mem=43.19 GB):  47%|████▋     | 27/58 [00:06<00:02, 12.05it/s]Capturing num tokens (num_tokens=576 avail_mem=43.17 GB):  47%|████▋     | 27/58 [00:06<00:02, 12.05it/s]Capturing num tokens (num_tokens=512 avail_mem=43.16 GB):  47%|████▋     | 27/58 [00:06<00:02, 12.05it/s]Capturing num tokens (num_tokens=512 avail_mem=43.16 GB):  50%|█████     | 29/58 [00:06<00:02, 13.44it/s]Capturing num tokens (num_tokens=480 avail_mem=43.05 GB):  50%|█████     | 29/58 [00:06<00:02, 13.44it/s]Capturing num tokens (num_tokens=448 avail_mem=43.06 GB):  50%|█████     | 29/58 [00:06<00:02, 13.44it/s]

    Capturing num tokens (num_tokens=448 avail_mem=43.06 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.80it/s]Capturing num tokens (num_tokens=416 avail_mem=43.14 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.80it/s]Capturing num tokens (num_tokens=384 avail_mem=43.13 GB):  53%|█████▎    | 31/58 [00:06<00:01, 14.80it/s]Capturing num tokens (num_tokens=384 avail_mem=43.13 GB):  57%|█████▋    | 33/58 [00:06<00:01, 16.05it/s]Capturing num tokens (num_tokens=352 avail_mem=43.12 GB):  57%|█████▋    | 33/58 [00:06<00:01, 16.05it/s]Capturing num tokens (num_tokens=320 avail_mem=43.12 GB):  57%|█████▋    | 33/58 [00:06<00:01, 16.05it/s]Capturing num tokens (num_tokens=288 avail_mem=43.12 GB):  57%|█████▋    | 33/58 [00:06<00:01, 16.05it/s]

    Capturing num tokens (num_tokens=288 avail_mem=43.12 GB):  62%|██████▏   | 36/58 [00:06<00:01, 18.07it/s]Capturing num tokens (num_tokens=256 avail_mem=43.11 GB):  62%|██████▏   | 36/58 [00:06<00:01, 18.07it/s]Capturing num tokens (num_tokens=240 avail_mem=43.10 GB):  62%|██████▏   | 36/58 [00:06<00:01, 18.07it/s]Capturing num tokens (num_tokens=224 avail_mem=43.06 GB):  62%|██████▏   | 36/58 [00:06<00:01, 18.07it/s]Capturing num tokens (num_tokens=224 avail_mem=43.06 GB):  67%|██████▋   | 39/58 [00:06<00:00, 20.17it/s]Capturing num tokens (num_tokens=208 avail_mem=43.05 GB):  67%|██████▋   | 39/58 [00:06<00:00, 20.17it/s]Capturing num tokens (num_tokens=192 avail_mem=43.08 GB):  67%|██████▋   | 39/58 [00:06<00:00, 20.17it/s]Capturing num tokens (num_tokens=176 avail_mem=43.08 GB):  67%|██████▋   | 39/58 [00:06<00:00, 20.17it/s]

    Capturing num tokens (num_tokens=176 avail_mem=43.08 GB):  72%|███████▏  | 42/58 [00:06<00:00, 21.60it/s]Capturing num tokens (num_tokens=160 avail_mem=43.07 GB):  72%|███████▏  | 42/58 [00:06<00:00, 21.60it/s]Capturing num tokens (num_tokens=144 avail_mem=43.06 GB):  72%|███████▏  | 42/58 [00:06<00:00, 21.60it/s]Capturing num tokens (num_tokens=128 avail_mem=43.06 GB):  72%|███████▏  | 42/58 [00:06<00:00, 21.60it/s]Capturing num tokens (num_tokens=128 avail_mem=43.06 GB):  78%|███████▊  | 45/58 [00:07<00:00, 22.94it/s]Capturing num tokens (num_tokens=112 avail_mem=43.05 GB):  78%|███████▊  | 45/58 [00:07<00:00, 22.94it/s]Capturing num tokens (num_tokens=96 avail_mem=43.04 GB):  78%|███████▊  | 45/58 [00:07<00:00, 22.94it/s] Capturing num tokens (num_tokens=80 avail_mem=43.03 GB):  78%|███████▊  | 45/58 [00:07<00:00, 22.94it/s]

    Capturing num tokens (num_tokens=80 avail_mem=43.03 GB):  83%|████████▎ | 48/58 [00:07<00:00, 24.16it/s]Capturing num tokens (num_tokens=64 avail_mem=43.03 GB):  83%|████████▎ | 48/58 [00:07<00:00, 24.16it/s]Capturing num tokens (num_tokens=48 avail_mem=43.02 GB):  83%|████████▎ | 48/58 [00:07<00:00, 24.16it/s]Capturing num tokens (num_tokens=32 avail_mem=42.99 GB):  83%|████████▎ | 48/58 [00:07<00:00, 24.16it/s]Capturing num tokens (num_tokens=28 avail_mem=43.01 GB):  83%|████████▎ | 48/58 [00:07<00:00, 24.16it/s]Capturing num tokens (num_tokens=28 avail_mem=43.01 GB):  90%|████████▉ | 52/58 [00:07<00:00, 26.23it/s]Capturing num tokens (num_tokens=24 avail_mem=43.00 GB):  90%|████████▉ | 52/58 [00:07<00:00, 26.23it/s]Capturing num tokens (num_tokens=20 avail_mem=42.99 GB):  90%|████████▉ | 52/58 [00:07<00:00, 26.23it/s]Capturing num tokens (num_tokens=16 avail_mem=42.99 GB):  90%|████████▉ | 52/58 [00:07<00:00, 26.23it/s]

    Capturing num tokens (num_tokens=16 avail_mem=42.99 GB):  95%|█████████▍| 55/58 [00:07<00:00, 27.03it/s]Capturing num tokens (num_tokens=12 avail_mem=42.99 GB):  95%|█████████▍| 55/58 [00:07<00:00, 27.03it/s]Capturing num tokens (num_tokens=8 avail_mem=42.98 GB):  95%|█████████▍| 55/58 [00:07<00:00, 27.03it/s] Capturing num tokens (num_tokens=4 avail_mem=42.98 GB):  95%|█████████▍| 55/58 [00:07<00:00, 27.03it/s]Capturing num tokens (num_tokens=4 avail_mem=42.98 GB): 100%|██████████| 58/58 [00:07<00:00, 27.59it/s]Capturing num tokens (num_tokens=4 avail_mem=42.98 GB): 100%|██████████| 58/58 [00:07<00:00,  7.78it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:36680


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-06-07 03:09:37] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Sure! Here are three countries and their capitals:<br><br>1. France - Paris<br>2. Brazil - Brasília<br>3. Japan - Tokyo</strong>


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


<strong style='color: #00008B;'>Certainly! Here is a list of three countries along with their capitals:<br><br>1. Japan – Tokyo<br>2. France – Paris<br>3. Brazil – Brasília</strong>



<strong style='color: #00008B;'>Of course! Here is another list of three countries along with their capitals:<br><br>1. Germany – Berlin<br>2. Italy – Rome<br>3. Canada – Ottawa</strong>


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



<strong style='color: #00008B;'>2 * 2. <br><br>While it's simple enough to calculate mentally (2 * 2 = 4), you can use a calculator if you prefer. Here's a simple way to do it:<br><br>1. Turn on your calculator.<br>2. Enter the number 2.<br>3. Press the multiplication (x) button.<br>4. Enter the number 2 again.<br>5. Press the equals (=) button.<br><br>The calculator will display 4 as the result. <br><br>So, 2 * 2 = 4.</strong>


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


<strong style='color: #00008B;'>### Tip 1: Balanced Diet<br>Maintaining a balanced diet is essential for overall health and well-being. Key points include:<br>- **Diverse Food Groups:** Include a variety of foods from each food group, such as fruits, vegetables, whole grains, lean proteins, and healthy fats.<br>- **Limit Processed Foods:** Reduce the intake of processed foods, sugars, and saturated fats.<br>- **Stay Hydrated:** Drink adequate amounts of water to support your body's functions.<br><br>### Tip 2: Regular Exercise<br>Staying healthy through regular exercise is crucial for maintaining a strong and vibrant lifestyle. Key points include:<br>- **Choose Activities You Enjoy:** Engage in activities that you find enjoyable and sustainable.<br>- **Set Goals:** Aim for at least 150 minutes of moderate-intensity exercise or 75 minutes of vigorous exercise each week.<br>- **Incorporate Strength Training:** Include strength training exercises to build muscle and improve bone density.<br>- **Promote Cardiovascular Health:** Regular exercise improves cardiovascular health and reduces the risk of chronic diseases.<br><br>By following these tips, you can significantly enhance your overall health and well-being.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Ash",<br>        "core": "Phoenix feather",<br>        "length": 11.0<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Dementor"<br>}</strong>


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

    100%|██████████| 3/3 [00:00<00:00, 28.92it/s]

    100%|██████████| 3/3 [00:00<00:00, 28.71it/s]

    



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
    [2026-06-07 03:09:54] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-06-07 03:09:57] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:00<00:03,  1.29it/s]

    Multi-thread loading shards:  40% Completed | 2/5 [00:01<00:02,  1.29it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:02<00:01,  1.29it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:03<00:00,  1.29it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.66it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.47it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:33451



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-06-07 03:10:13] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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

    /actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py:903: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1586.)
      encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)



<strong style='color: #00008B;'>The image shows a man on the back of a taxi, holding an iron and a blue shirt. He appears to be ironing the shirt while standing against the open flatbed of the taxi. This scene seems to symbolize efficiency, as the taxi itself is used as a work station while it is parked on a busy street. The background includes other taxis and urban architecture, suggesting this was taken in a city setting.</strong>



```python
terminate_process(server_process)
```
