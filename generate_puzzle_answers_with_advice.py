from utils import start_vllm_server, stop_vllm_server, chat_completion, write_jsonl, read_jsonl
import argparse
from concurrent.futures import ThreadPoolExecutor
import os

# Mapping from dataset type to tailored system prompts
SYSTEM_PROMPTS = {
    "logic": """You are a Student Logic Puzzle Solver. Your task is to:
1. Follow the provided solving advice carefully
2. Apply the suggested approach to solve the puzzle
3. Show your work step by step
4. Verify your solution

Remember to:
- Use the advice as a guide for your approach
- Break down the problem as suggested
- Show your reasoning clearly
- Check your solution against all conditions""",

    "mathematical": """You are a Student Mathematical Puzzle Solver. Your task is to:
1. Follow the provided solving advice carefully
2. Apply the suggested mathematical approach
3. Show your work step by step
4. Verify your solution

Remember to:
- Use the advice as a guide for your approach
- Show all mathematical steps clearly
- Apply any suggested formulas or methods
- Verify your solution works"""
}

def gen_answers_with_advice(input_file, advice_file, output_file, api_base, model_name, max_tokens=1024, temperature=0.7, threads=10):
    """
    Generates answers for puzzle datasets using advice from a larger LLM.
    The dataset type is derived from the input file name.
    """
    # Derive the puzzle type from the input filename
    base_name = os.path.basename(input_file).lower()
    if "fantiasic_logic" in base_name:
        puzzle_type = "logic"
    else:
        puzzle_type = "mathematical"
    
    system_prompt = SYSTEM_PROMPTS[puzzle_type]
    
    # Load both input data and advice
    input_data_list = list(read_jsonl(input_file))
    advice_data = {item.get("title", ""): item.get("solving_advice", "") 
                  for item in read_jsonl(advice_file)}
    
    output_data_list = []

    def process_data(data_item, api_base, model_name, max_tokens=1024, temperature=0.7):
        # Get the puzzle content and advice
        title = data_item.get("title", "")
        content = data_item.get("content", "")
        advice = advice_data.get(title, "No specific advice available for this puzzle.")
        
        # Create a prompt that includes both the puzzle and the advice
        prompt = f"""Title: {title}

Puzzle:
{content}

Solving Advice:
{advice}

Please solve this puzzle by following the advice above. Show your work step by step."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Call the chat completion helper
        response = chat_completion(api_base=api_base, model_name=model_name, messages=messages,
                                 max_tokens=max_tokens, temperature=temperature)
        
        # Store the original data and add the LLM's response
        output_item = data_item.copy()
        output_item["llm_answer"] = response
        return output_item

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(process_data, data_item, api_base, model_name, max_tokens, temperature)
            for data_item in input_data_list
        ]
        for future in futures:
            output_data_list.append(future.result())

    write_jsonl(output_file, output_data_list)
    print(f"[INFO] Generation complete. Results saved to {output_file}.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers for puzzle datasets using advice from a larger LLM.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--advice_file", type=str, help="Path to the JSONL file containing solving advice.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file.")
    parser.add_argument("--api_base", type=str, help="Base URL for the OpenAI API.")
    parser.add_argument("--model_name", type=str, help="Name of the model to use.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model.")
    parser.add_argument("--port", type=int, default=8000, help="Port to host the model on.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--threads", type=int, default=10, help="Number of threads to use for generation.")
    
    args = parser.parse_args()
    
    if "," in args.input_file:
        input_files = args.input_file.split(",")
    else:
        input_files = [args.input_file]

    if "," in args.output_file:
        output_files = args.output_file.split(",")
    else:
        output_files = [args.output_file]

    if "," in args.advice_file:
        advice_files = args.advice_file.split(",")
    else:
        advice_files = [args.advice_file]
        
    
    if args.model_path:
        process_id = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
        for input_file, output_file, advice_file in zip(input_files, output_files, advice_files):
            gen_answers_with_advice(input_file, advice_file, output_file, 
                                  args.api_base, args.model_name, args.max_tokens, 
                                  args.temperature, args.threads)
        stop_vllm_server(process_id)
    else:
        for input_file, output_file, advice_file in zip(input_files, output_files, advice_files):
            gen_answers_with_advice(input_file, advice_file, output_file, 
                                  args.api_base, args.model_name, args.max_tokens, 
                                  args.temperature, args.threads)
        
    # if args.model_path:
    #     process_id = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
    #     gen_answers_with_advice(args.input_file, args.advice_file, args.output_file, 
    #                           args.api_base, args.model_name, args.max_tokens, 
    #                           args.temperature, args.threads)
    #     stop_vllm_server(process_id)
    # else:
    #     gen_answers_with_advice(args.input_file, args.advice_file, args.output_file, 
    #                           args.api_base, args.model_name, args.max_tokens, 
    #                           args.temperature, args.threads) 