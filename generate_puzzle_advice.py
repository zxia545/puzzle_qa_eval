from utils import start_vllm_server, stop_vllm_server, chat_completion_qwen3, write_jsonl, read_jsonl
import argparse
from concurrent.futures import ThreadPoolExecutor
import os

SYSTEM_PROMPT = """You are an Expert Puzzle Solving Guide. Your task is to:
1. Analyze the provided puzzle to understand its underlying principles and common logical patterns.
2. Formulate general, step-by-step strategic advice that can be applied to a *broad category* of similar puzzles. This advice should focus on *how to think* about such puzzles, not how to solve the specific example.
3. Highlight common pitfalls, and discuss general problem-solving techniques or heuristics that are useful for this puzzle type.

**CRITICAL INSTRUCTIONS:**
- **DO NOT provide any information, hints, or steps that directly lead to or simplify the solution of the *specific example puzzle* given in the prompt.**
- **Your advice MUST be abstract and general enough to help someone solve *other, different* puzzles of the same type, without making the provided example easier.**
- **Focus on transferable skills and logical reasoning, not on the features or solution of the example.**
- **IMPORTANT: Keep your advice concise and focused. Limit your response to 300-400 words maximum.**
- **Provide only 2-3 key strategies or approaches, rather than an exhaustive list.**

Your advice should:
- Be concise yet comprehensive.
- Emphasize general problem-solving approaches and analytical techniques.
- Discuss logical reasoning patterns and critical thinking skills applicable to the puzzle category.
- Mention relevant mathematical concepts or formulas in a general way, if applicable to the puzzle type.
- Be written in clear, accessible language for students.
"""

def gen_advice(input_file, output_file, api_base, model_name, max_tokens=512, temperature=0.7, threads=10):
    """
    Generates puzzle-solving advice using a larger LLM.
    """
    input_data_list = list(read_jsonl(input_file))
    output_data_list = []

    def process_data(data_item, api_base, model_name, max_tokens=1024, temperature=0.7):
        # Get the puzzle content
        title = data_item.get("title", "")
        content = data_item.get("content", "")
        
        # Create a prompt that asks for general advice
        prompt = f"Title: {title}\n\nPuzzle:\n{content}\n\nPlease provide brief, focused advice on how to approach and solve this type of puzzle. Focus on the logical structure and 2-3 key concepts that would help someone solve similar puzzles. Be concise (300-400 words maximum)."
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Call the chat completion helper
        response = chat_completion_qwen3(api_base=api_base, model_name=model_name, messages=messages,
                                 max_tokens=max_tokens, temperature=temperature)
        
        # Store the original data and add the advice
        output_item = data_item.copy()
        output_item["solving_advice"] = response
        return output_item

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(process_data, data_item, api_base, model_name, max_tokens, temperature)
            for data_item in input_data_list
        ]
        for future in futures:
            output_data_list.append(future.result())

    write_jsonl(output_file, output_data_list)
    print(f"[INFO] Advice generation complete. Results saved to {output_file}.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate puzzle-solving advice using vLLM.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file.")
    parser.add_argument("--api_base", type=str, help="Base URL for the OpenAI API.")
    parser.add_argument("--model_name", type=str, help="Name of the model to use.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model.")
    parser.add_argument("--port", type=int, default=8000, help="Port to host the model on.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--threads", type=int, default=10, help="Number of threads to use for generation.")
    
    args = parser.parse_args()
    
    if args.model_path:
        process_id = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
        gen_advice(args.input_file, args.output_file, args.api_base, args.model_name,
                  args.max_tokens, args.temperature, args.threads)
        stop_vllm_server(process_id)
    else:
        gen_advice(args.input_file, args.output_file, args.api_base, args.model_name,
                  args.max_tokens, args.temperature, args.threads) 