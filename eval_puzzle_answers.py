from utils import start_vllm_server, stop_vllm_server, chat_completion, write_jsonl, read_jsonl
import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor

def extract_rating(response):
    """
    Extract a rating (1 to 5) from the LLM response.
    Assumes the response contains a phrase like "Rating: X" (case insensitive).
    """
    m = re.search(r'Rating\s*:\s*([1-5])', response, re.IGNORECASE)
    if m:
        return int(m.group(1))
    else:
        return None

# System message for puzzle evaluation
PUZZLE_EVAL_SYS_MSG = """You are an expert puzzle evaluator. Your task is to evaluate the quality of a puzzle solution by comparing it with the reference solution.
You are provided with:
    1. A puzzle description
    2. A solution attempt
    3. The reference solution
Please perform the following steps:
1. Analyze the solution attempt for:
   - Correctness of the answer
   - Clarity of explanation
   - Logical reasoning
   - Step-by-step approach
   - Mathematical accuracy (if applicable)
2. Compare with the reference solution
3. Assign a rating from 1 to 5 based on the following criteria:
   - **Rating 5:** The solution is perfect. It provides a clear, step-by-step explanation that matches or exceeds the reference solution in clarity and completeness.
   - **Rating 4:** The solution is very good. It reaches the correct answer with minor issues in explanation or presentation.
   - **Rating 3:** The solution is acceptable. It shows understanding but may have some gaps in explanation or minor errors.
   - **Rating 2:** The solution is poor. It may have the right answer but lacks proper explanation, or has significant errors in reasoning.
   - **Rating 1:** The solution is incorrect or completely misses the point of the puzzle.
4. Output your evaluation in the exact format: "Rating: X. Explanation: <your explanation>."
Ensure your explanation clearly justifies the assigned rating and provides specific feedback on what was done well and what could be improved."""

def eval_puzzle_jsonl(path_to_jsonl, api_base, model_name, max_tokens=512, temperature=0.7, threads=10, output_file=None):
    def process_data(data_item, api_base, model_name, max_tokens=512, temperature=0.7):
        puzzle_title = data_item.get("title", "")
        puzzle_content = data_item.get("content", "")
        llm_solution = data_item.get("llm_answer", "")
        reference_solution = data_item.get("answer", "")
        
        user_prompt = f"""Puzzle Title: {puzzle_title}

Puzzle Description:
{puzzle_content}

Solution Attempt:
{llm_solution}

Reference Solution:
{reference_solution}"""

        messages = [
            {"role": "system", "content": PUZZLE_EVAL_SYS_MSG},
            {"role": "user", "content": user_prompt}
        ]
        
        response = chat_completion(api_base=api_base, model_name=model_name, messages=messages,
                                   max_tokens=max_tokens, temperature=temperature)
        
        rating = extract_rating(response)
        
        return {
            "puzzle_title": puzzle_title,
            "puzzle_content": puzzle_content,
            "llm_solution": llm_solution,
            "reference_solution": reference_solution,
            "eval_feedback": response,
            "eval_rating": rating
        }
    
    data_list = list(read_jsonl(path_to_jsonl))
    total_counter = len(data_list)
    file_name = os.path.splitext(os.path.basename(path_to_jsonl))[0]
    
    if output_file is None:
        output_file = os.path.join("eval_results", file_name + "_eval.jsonl")
    
    output_list = []
    total_rating = 0
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_data, data_item, api_base, model_name, max_tokens, temperature)
                   for data_item in data_list]
        for future in futures:
            result_json = future.result()
            if result_json["eval_rating"] is not None:
                total_rating += result_json["eval_rating"]
            output_list.append(result_json)
   
    write_jsonl(output_file, output_list)
    print(f'[INFO] Evaluation results have been saved to {output_file}')
    if total_counter > 0:
        avg_rating = total_rating / total_counter
        print(f'[INFO] Average Rating: {avg_rating:.2f}/5.00')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate puzzle solutions')
    parser.add_argument('--api_base', type=str, default="https://api.openai.com", help='API base URL')
    parser.add_argument('--model_name', type=str, default="text-davinci-003", help='Model name')
    parser.add_argument('--path_to_jsonl_list', type=str, help='List of paths to the input JSONL files')
    parser.add_argument('--max_tokens', type=int, default=512, help='Max tokens')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--port', type=int, default=8000, help='Port')
    parser.add_argument('--gpu', type=int, default=1, help='GPU')
    parser.add_argument('--threads', type=int, default=10, help='Threads')
    parser.add_argument('--output_file_list', type=str, default=None, help='List of output file paths')
    
    args = parser.parse_args()
    
    if args.model_path:
        process_id = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
        path_json_list = args.path_to_jsonl_list.split(',')
        output_file_list = args.output_file_list.split(',')
        for path_to_jsonl, output_path in zip(path_json_list, output_file_list):
            eval_puzzle_jsonl(path_to_jsonl, args.api_base, args.model_name, args.max_tokens,
                       args.temperature, args.threads, output_path)
        stop_vllm_server(process_id)
    else:
        path_json_list = args.path_to_jsonl_list.split(',')
        output_file_list = args.output_file_list.split(',')
        for path_to_jsonl, output_path in zip(path_json_list, output_file_list):
            eval_puzzle_jsonl(path_to_jsonl, args.api_base, args.model_name, args.max_tokens,
                       args.temperature, args.threads, output_path) 