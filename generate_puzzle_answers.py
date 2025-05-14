from utils import start_vllm_server, stop_vllm_server, chat_completion, write_jsonl, read_jsonl
import argparse
from concurrent.futures import ThreadPoolExecutor
import os

# Mapping from dataset type to tailored system prompts
SYSTEM_PROMPTS = {
    "logic": """You are an Expert Logic Puzzle Solver. Your task is to:
1. First provide a clear, step-by-step solution to the puzzle
2. Then explain your reasoning process in detail
3. Finally, verify your answer by checking if it satisfies all given conditions

Remember to:
- Break down complex problems into smaller parts
- Consider all possible scenarios
- Use logical deduction to eliminate impossible options
- Double-check your solution against all given constraints
- Look for patterns and relationships
- Consider both direct and indirect implications""",

    "mathematical": """You are an Expert Mathematical Puzzle Solver. Your task is to:
1. First provide a clear, step-by-step solution to the puzzle
2. Then explain your mathematical reasoning in detail
3. Finally, verify your answer by checking if it satisfies all given conditions

Remember to:
- Break down complex problems into smaller parts
- Show all mathematical steps clearly
- Consider all possible scenarios
- Verify your solution works for all cases
- Include any relevant formulas or equations
- Check for edge cases and special conditions"""
}

def gen_answers(input_file, output_file, api_base, model_name, max_tokens=1024, temperature=0.7, threads=10):
    """
    Generates answers for puzzle datasets using tailored system prompts.
    The dataset type is derived from the input file name.
    """
    # Derive the puzzle type from the input filename
    base_name = os.path.basename(input_file).lower()
    if "fantiasic_logic" in base_name:
        puzzle_type = "logic"
    else:
        puzzle_type = "mathematical"
    
    system_prompt = SYSTEM_PROMPTS[puzzle_type]
    
    input_data_list = list(read_jsonl(input_file))
    output_data_list = []

    def process_data(data_item, api_base, model_name, max_tokens=1024, temperature=0.7):
        # Get the puzzle content
        title = data_item.get("title", "")
        content = data_item.get("content", "")
        
        # Create a clear prompt that includes both title and content
        prompt = f"Title: {title}\n\nPuzzle:\n{content}\n\nPlease solve this puzzle step by step."
        
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
    parser = argparse.ArgumentParser(description="Generate answers for puzzle datasets using vLLM.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file.")
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
    
    if args.model_path:
        process_id = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
        
        if ',' in args.input_file and ',' in args.output_file:
            input_files = [f.strip() for f in args.input_file.split(',')]
            output_files = [f.strip() for f in args.output_file.split(',')]
            for i in range(len(input_files)):
                gen_answers(input_files[i], output_files[i], args.api_base, args.model_name,
                          args.max_tokens, args.temperature, args.threads)
        else:
            gen_answers(args.input_file, args.output_file, args.api_base, args.model_name,
                      args.max_tokens, args.temperature, args.threads)
        stop_vllm_server(process_id)
    else:
        if ',' in args.input_file and ',' in args.output_file:
            input_files = [f.strip() for f in args.input_file.split(',')]
            output_files = [f.strip() for f in args.output_file.split(',')]
            for i in range(len(input_files)):
                gen_answers(input_files[i], output_files[i], args.api_base, args.model_name,
                          args.max_tokens, args.temperature, args.threads)
        else:
            gen_answers(args.input_file, args.output_file, args.api_base, args.model_name,
                      args.max_tokens, args.temperature, args.threads) 