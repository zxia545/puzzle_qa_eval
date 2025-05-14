import re
import json
import os

# Define filenames
question_filename = 'question.txt'
solution_filename = 'solution.txt'
output_filename = 'output.jsonl'

# Function to read file content
def read_file(filename):
    """Reads the content of a file, normalizing line endings."""
    try:
        # Ensure UTF-8 encoding is specified for reading
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().replace('\r\n', '\n').replace('\r', '\n')
            if content.startswith('\ufeff'): # Remove BOM
                content = content[1:]
            return content
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found in directory '{os.getcwd()}'.")
        return None
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        return None

# Function to clean text content/answer
def clean_text(text):
    """Removes illustrations, references, and extra whitespace."""
    # Remove [Illustration] tags and surrounding whitespace, replace with single newline
    cleaned = re.sub(r'\s*\[Illustration\]\s*', '\n', text).strip()
    # Remove common reference patterns more generally (often at the end)
    cleaned = re.sub(r'\n*See No\. \d+.*$', '', cleaned, flags=re.MULTILINE | re.DOTALL).strip()
    cleaned = re.sub(r'\n*See also .*$', '', cleaned, flags=re.MULTILINE | re.DOTALL).strip()
    cleaned = re.sub(r'\n*Compare with .*$', '', cleaned, flags=re.MULTILINE | re.DOTALL).strip()
    cleaned = re.sub(r'\n*\[[A-Z]\] .*$', '', cleaned, flags=re.MULTILINE | re.DOTALL).strip() # Remove lines like [B] Mr. Oscar...
    # Reduce multiple blank lines (3 or more newlines) to a single blank line (2 newlines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned

# --- Main Processing Logic ---
question_text_full = read_file(question_filename)
solution_text_full = read_file(solution_filename)

if question_text_full is not None and solution_text_full is not None:
    print("Files read successfully. Parsing using heading locations...")

    # Regex to find the STARTING headings only
    heading_regex = re.compile(
        r"^(?P<idx>\d+)\.--?_(?P<title>.*?)_?\._", # Match the heading line structure
        re.MULTILINE # Ensure ^ matches start of lines
    )

    # --- Process Questions ---
    questions = {}
    print("\nProcessing Questions:")
    question_matches = list(heading_regex.finditer(question_text_full))
    print(f"  Found {len(question_matches)} question headings.")

    for i, match in enumerate(question_matches):
        data = match.groupdict()
        try:
            idx = int(data['idx'])
            title = data['title'].strip()
            content_start_index = match.end()
            if i + 1 < len(question_matches):
                content_end_index = question_matches[i+1].start()
            else:
                content_end_index = len(question_text_full)
            raw_content = question_text_full[content_start_index:content_end_index]
            content = clean_text(raw_content)
            questions[idx] = {'title': title, 'content': content}
            if idx <= 5 or idx == 32 or idx % 50 == 0 or idx >= 110:
                 print(f"  Extracted Question: idx={idx}, title='{title}' (Content length: {len(content)})")
        except ValueError:
            print(f"  Error converting question idx '{data.get('idx')}' to int. Skipping.")
        except Exception as e:
            print(f"  Error processing question match idx={data.get('idx', 'N/A')}: {e}")

    # --- Process Solutions ---
    solutions = {}
    print("\nProcessing Solutions:")
    solution_matches = list(heading_regex.finditer(solution_text_full))
    print(f"  Found {len(solution_matches)} solution headings.")

    for i, match in enumerate(solution_matches):
        data = match.groupdict()
        try:
            idx = int(data['idx'])
            solution_title_check = data['title'].strip() # For checking
            answer_start_index = match.end()
            if i + 1 < len(solution_matches):
                answer_end_index = solution_matches[i+1].start()
            else:
                answer_end_index = len(solution_text_full)
            raw_answer = solution_text_full[answer_start_index:answer_end_index]
            answer = clean_text(raw_answer)
            solutions[idx] = {'answer': answer}
            if idx <= 5 or idx == 32 or idx % 50 == 0 or idx >= 110:
                 print(f"  Extracted Solution: idx={idx}, title_check='{solution_title_check}' (Answer length: {len(answer)})")
        except ValueError:
            print(f"  Error converting solution idx '{data.get('idx')}' to int. Skipping.")
        except Exception as e:
             print(f"  Error processing solution match idx={data.get('idx', 'N/A')}: {e}")


    # --- Combine and Generate JSONL ---
    output_jsonl_lines = []
    sorted_indices = sorted(questions.keys())
    missing_solutions = []
    missing_questions = set(solutions.keys()) - set(questions.keys())

    print(f"\nCombining data for {len(sorted_indices)} questions found...")
    if len(questions) < 114 or len(solutions) < 114:
         print(f"Warning: Expected 114 items, found {len(questions)} questions and {len(solutions)} solutions. Check input files and intermediate headings.")

    for idx in sorted_indices:
        if idx in solutions:
            q_content = questions[idx]['content']
            s_answer = solutions[idx]['answer']
            if not q_content:
                 print(f"Warning: Empty content for question idx={idx}")
            if not s_answer:
                 print(f"Warning: Empty answer for solution idx={idx}")

            item = {
                'idx': idx,
                'title': questions[idx]['title'],
                'content': q_content,
                'answer': s_answer
            }
            # Modified json.dumps call
            output_jsonl_lines.append(json.dumps(item, ensure_ascii=False))
        else:
            missing_solutions.append(idx)
            item = {
                'idx': idx,
                'title': questions[idx]['title'],
                'content': questions[idx]['content'],
                'answer': None
            }
             # Modified json.dumps call
            output_jsonl_lines.append(json.dumps(item, ensure_ascii=False))

    # Report any mismatches
    if missing_solutions:
        print(f"\nWarning: No solutions found for question indices: {sorted(missing_solutions)}")
    if missing_questions:
         print(f"Warning: Solutions found for indices without matching questions: {sorted(list(missing_questions))}")

    final_output = "\n".join(output_jsonl_lines)

    # Write the result to the output file
    try:
        # Ensure UTF-8 encoding is specified for writing
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_output)
        print(f"\n--- Successfully combined {len(output_jsonl_lines)} items and wrote to '{output_filename}' ---")

        # --- Verification Prints ---
        print("\n--- Verification Sample (idx=1) ---")
        found_idx1 = False
        for line in output_jsonl_lines:
            # Need to load back to check, as it's already a string
            data = json.loads(line) # Load the JSON string back to dict
            if data['idx'] == 1:
                 # Print re-dumped with indent for readability
                print(json.dumps(data, indent=2, ensure_ascii=False))
                found_idx1 = True
                break
        if not found_idx1: print("Item with idx=1 not found in output.")

        print("\n--- Verification Sample (idx=32) ---")
        found_idx32 = False
        for line in output_jsonl_lines:
             data = json.loads(line) # Load the JSON string back to dict
             if data['idx'] == 32:
                  # Print re-dumped with indent for readability
                 print(json.dumps(data, indent=2, ensure_ascii=False))
                 found_idx32 = True
                 break
        if not found_idx32: print("Item with idx=32 not found in output.")


    except Exception as e:
        print(f"Error writing output file '{output_filename}': {e}")

else:
    print("\nScript aborted due to file reading errors. Please ensure 'question.txt' and 'solution.txt' exist in the correct location and are readable.")