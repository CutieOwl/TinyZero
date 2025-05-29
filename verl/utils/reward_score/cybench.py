import fcntl
import glob
import os
import json
import re
from pathlib import Path
import shutil
import time
import uuid


def compute_score(rollout_id, log_dir):
    backoff = 0.01
    max_backoff = 0.1
    while True:
        input_files = glob.glob(os.path.join(log_dir, "cybench", f"{rollout_id}_*_score.txt"))
        if not input_files:
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
        else:
            file_name = input_files[0]
            with open(
                file_name,
                "r",
            ) as file:
                fcntl.flock(file, fcntl.LOCK_SH) 
                score = float(file.read().strip())
                fcntl.flock(file, fcntl.LOCK_UN) 
            break

    return score    
