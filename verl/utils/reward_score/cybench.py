import os
import json
import re
from pathlib import Path
import shutil
import uuid


def compute_score(solution_str, ground_truth, has_submission_score=0.1, valid_submission_score=0.2):
    print(f"solution str {solution_str}")
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[WARN] wrong assistant format")
        return 0

    # ignore 
    return 1

    # code = extract_code(solution_str)
    # nl_text = extract_text_up_to_code(solution_str)

    # if not code or not nl_text:
    #     print("[WARN] wrong plan + code format")
    #     return 0

    # res = interpreter.run(code)
    # interpreter.cleanup_session()

    # submission_file = run_dir / "submission.csv"
    # if not submission_file.exists():
    #     print("REWARD", "no submission.csv")
    #     return 0

    # cmd = f"mlebench grade-sample {submission_file} {ground_truth}"
    # result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    # try:
    #     result = '{' + result.stderr.split('{')[1]
    #     result = json.loads(result)
    # except:
    #     print("REWARD", "error parsing grade.json")
    #     return 0

    # #shutil.rmtree(run_dir, ignore_errors=True)

    # if result["score"]:
    #     print("REWARD", result["score"])
    #     return result["score"]
    # if result["valid_submission"]:
    #     print("REWARD", "valid_submission")
    #     return valid_submission_score
    # print("REWARD", "has_submission")
    # return has_submission_score