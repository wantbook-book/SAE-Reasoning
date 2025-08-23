import json
def append_boxed(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    with open(output_file, 'w') as f:
        for line in lines:
            item = json.loads(line)
            item['answer'] = f'\\boxed{{{item["answer"]}}}'
            f.write(json.dumps(item) + '\n')

def convert_record(source_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    转换单条记录
    """
    doc = source_record.get("doc", {})
    
    # 提取响应内容
    responses = source_record.get("resps", [[]])[0]
    
    # 构建目标格式
    target_record = {
        "idx": doc.get("id", source_record.get("doc_id", 0)),
        "question": doc.get("problem", ""),
        "answer": "\\boxed{"+source_record.get("target", doc.get("answer", ""))+"}",
        "level": 3,  # 默认难度级别
        "code": responses,
    }
    
    return target_record

def convert_format(input_file: str, output_file: str):
    """
    转换整个文件
    """
    converted_count = 0
    error_count = 0
    
    print(f"开始转换文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                source_record = json.loads(line)
                target_record = convert_record(source_record)
                
                # 写入转换后的记录
                json.dump(target_record, outfile, ensure_ascii=False)
                outfile.write('\n')
                
                converted_count += 1
                
                if converted_count % 100 == 0:
                    print(f"已转换 {converted_count} 条记录...")
                    
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行JSON解析错误: {e}")
                error_count += 1
            except Exception as e:
                print(f"第 {line_num} 行处理错误: {e}")
                error_count += 1
    
    print(f"\n转换完成!")
    print(f"成功转换: {converted_count} 条记录")
    print(f"错误记录: {error_count} 条")
    print(f"输出文件: {output_file}")

if __name__ == '__main__':
    input_file = '/angel/fwk/code/SAE-Reasoning/evaluation/deepseek-llama-8b-math-500/deepseek-ai__DeepSeek-R1-Distill-Llama-8B/samples_math-500_2025-08-13T08-53-03.659236_eval.jsonl'
    output_file = '/angel/fwk/code/SAE-Reasoning/evaluation/deepseek-llama-8b-math-500/deepseek-ai__DeepSeek-R1-Distill-Llama-8B/samples_math-500_2025-08-13T08-53-03.659236_eval2.jsonl'
    append_boxed(input_file, output_file)