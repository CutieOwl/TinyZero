import re
import random
import ast
import operator


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    pattern = r"Answer:\s*(.*?)(?=\n|$|\<)"
    match = re.finditer(pattern, content)
    matches = list(match)
    
    if matches:
        # Return the last occurrence
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

def matches_answer_format(output_str: str, answer_format: str):
    output_str = output_str.strip()
    answer_format = answer_format.strip()
    
    # Quick length check - if lengths don't match after replacing wildcards, can't be valid
    if len(output_str) != len(answer_format):
        return False
    
    # Check character by character
    pattern = "^" + re.escape(answer_format).replace("\\*", ".") + "$"
    
    return bool(re.match(pattern, output_str))


def compute_score(solution_str, ground_truth, output_score=0.1, format_score=0.2, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    ans = ground_truth['answer']
    ans_format = ground_truth['format']
    
    output = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {ans} | Format: {ans_format}")
        print(f"Extracted output: {output}")

    if output is None:
        if do_print:
            print(f"No output found")
        return 0
    
    # Validate equation uses correct numbers
    if not matches_answer_format(output, ans_format):
        if do_print:
            print(f"Invalid answer format")
        return output_score
        
    if output == ans:  
        if do_print:
            print(f"Correct answer: {output}")
        return score
    else:
        if do_print:
            print(f"Wrong answer: output = {output}, target = {ans}")
        return format_score