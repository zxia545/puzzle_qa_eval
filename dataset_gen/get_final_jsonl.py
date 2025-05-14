import pandas as pd
import json
import re

def clean_text(text):
    """Remove special characters that may not print correctly."""
    return re.sub(r'[^\x20-\x7E]', '', str(text))

def main():
    # Load the Excel file
    input_file = 'fantiasic_logic_puzzles.xlsx'
    df = pd.read_excel(input_file)

    # Prepare the output JSONL file
    output_file = 'final_data.jsonl'

    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for _, row in df.iterrows():
            # Skip rows where 'idx' is NaN
            if pd.isna(row['idx']):
                continue

            # Create the item dictionary
            item = {
                'idx': int(row['idx']),
                'title': clean_text(row['title']),
                'content': clean_text(row['content']),
                'answer': clean_text(row['answer']),
            }
            # Write the item as a JSON object to the JSONL file
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()