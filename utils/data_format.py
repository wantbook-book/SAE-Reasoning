import json
import pandas as pd
import json
import os
import numpy as np
def convert_parquet_to_jsonl(file_path: str, output_path: str = None) -> str:
    """Convert a parquet file to JSONL format.
    
    Args:
        file_path: Path to the parquet file.
        output_path: Path to save the JSONL file. If None, saves as '{filename}.jsonl' in the same directory.
        
    Returns:
        str: Path to the saved JSONL file.
        
    Raises:
        FileNotFoundError: If the parquet file doesn't exist.
        ValueError: If the parquet file is empty.
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    try:
        # Load parquet file
        df = pd.read_parquet(file_path)
        
        if len(df) == 0:
            raise ValueError("Parquet file is empty")
        
        # Determine output path
        if output_path is None:
            dir_path = os.path.dirname(file_path)
            filename = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(dir_path, f"{filename}.jsonl")
            example_path = os.path.join(dir_path, f"example.json")
        # Convert DataFrame to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                # Convert row to dictionary
                example = row.to_dict()
                
                # Convert any non-serializable types
                for key, value in example.items():
                    if key == 'extra_info':
                        example[key] = None
                    elif isinstance(value, np.ndarray):
                        example[key] = value.tolist()
                    elif pd.isna(value):
                        example[key] = None
                    elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                        example[key] = str(value)
                
                # Write as JSON line
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        
        # Save first row as example
        if len(df) > 0:
            first_row = df.iloc[0].to_dict()
            # Convert any non-serializable types for the example
            for key, value in first_row.items():
                if key == 'extra_info':
                    first_row[key] = None
                elif isinstance(value, np.ndarray):
                    first_row[key] = value.tolist()
                elif pd.isna(value):
                    first_row[key] = None
                elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    first_row[key] = str(value)
            
            # Write example to JSON file
            with open(example_path, 'w', encoding='utf-8') as f:
                json.dump(first_row, f, ensure_ascii=False, indent=2)
        
        print(f"Parquet file converted to JSONL: {output_path}")
        print(f"Total records: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        return output_path
        
    except Exception as e:
        raise ValueError(f"Error processing parquet file: {e}")


def format_rlpr(input_file, output_file):
    """Format RLPR dataset from JSONL to a simplified format.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
    """
    with open(input_file, 'r', encoding='utf-8') as rf:
        with open(output_file, 'w', encoding='utf-8') as wf:
            idx = 0
            for line in rf:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    item = json.loads(line)
                    formatted_item = {
                        'idx': idx,
                        'problem': item['prompt'][1]['content'],
                        'answer': item['reward_model']['ground_truth'],
                        'ability': item['ability'],
                        'data_source': item['data_source']
                    }
                    wf.write(json.dumps(formatted_item, ensure_ascii=False) + '\n')
                    idx += 1
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Error processing line {idx}: {e}")
                    continue
        dir_path = os.path.dirname(input_file)
        example_file = os.path.join(dir_path, 'example_format.json')
        with open(output_file, 'r') as f:
            first_line = f.readline()
            if first_line:
                example = json.loads(first_line)
                with open(example_file, 'w') as ef:
                    json.dump(example, ef, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # file_path = "/pubshare/fwk/code/sae/SAE-Reasoning2/sft/dataset/datasets--openbmb--RLPR-Train-Dataset/snapshots/922eeb37157ebce90c1ee8233cdf04f04fbad728/rlpr_train.parquet"
    # convert_parquet_to_jsonl(file_path=file_path)
    input_file = '/pubshare/fwk/code/sae/SAE-Reasoning2/sft/dataset/datasets--openbmb--RLPR-Train-Dataset/snapshots/922eeb37157ebce90c1ee8233cdf04f04fbad728/rlpr_train.jsonl'
    output_file = '/pubshare/fwk/code/sae/SAE-Reasoning2/sft/dataset/datasets--openbmb--RLPR-Train-Dataset/snapshots/922eeb37157ebce90c1ee8233cdf04f04fbad728/rlpr_train_format.jsonl'
    format_rlpr(
        input_file=input_file,
        output_file=output_file
    )
