import gradio as gr
import json
import os
choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

class JSONLComparator:
    def __init__(self):
        self.file1_data = []
        self.file2_data = []
        self.current_idx = 0
        self.filtered_indices = []
        self.filter_mode = "all"  # "all", "file1_correct_file2_wrong"
        
    def load_jsonl_file(self, file_path):
        """加载JSONL文件"""
        if not file_path or not os.path.exists(file_path):
            return None, f"文件路径无效或文件不存在: {file_path}"
        
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data, f"成功加载 {len(data)} 条记录"
        except Exception as e:
            return None, f"加载文件失败: {str(e)}"
    
    def load_files(self, file1_path, file2_path):
        """加载两个文件"""
        self.file1_data, msg1 = self.load_jsonl_file(file1_path)
        self.file2_data, msg2 = self.load_jsonl_file(file2_path)
        
        if self.file1_data is None or self.file2_data is None:
            return f"加载失败:\n文件1: {msg1}\n文件2: {msg2}"
        
        # 创建索引映射
        self.create_index_mapping()
        self.current_idx = 0
        self.filter_mode = "all"
        self.apply_filter()
        
        return f"加载成功:\n文件1: {msg1}\n文件2: {msg2}"
    
    def create_index_mapping(self):
        """创建索引映射"""
        self.file1_dict = {item['idx']: item for item in self.file1_data}
        self.file2_dict = {item['idx']: item for item in self.file2_data}
        
        # 找到共同的索引
        common_indices = set(self.file1_dict.keys()) & set(self.file2_dict.keys())
        self.all_indices = sorted(list(common_indices))
    
    def apply_filter(self):
        """应用筛选条件"""
        if self.filter_mode == "all":
            self.filtered_indices = self.all_indices.copy()
        elif self.filter_mode == "file1_correct_file2_wrong":
            self.filtered_indices = []
            for idx in self.all_indices:
                item1 = self.file1_dict.get(idx)
                item2 = self.file2_dict.get(idx)
                answer_idx = item1.get('answer_index', '无index信息')
                if item1 and item2:
                    extracted_answer1 = str(item1.get('pred', '无提取答案')[0])
                    extracted_answer2 = str(item2.get('pred', '无提取答案')[0])

                    is_correct1 = extracted_answer1 == answer_idx
                    is_correct2 = extracted_answer2 == answer_idx
                    
                    if is_correct1 and not is_correct2:
                        self.filtered_indices.append(idx)
        elif self.filter_mode == "file1_wrong_file2_correct":
            self.filtered_indices = []
            for idx in self.all_indices:
                item1 = self.file1_dict.get(idx)
                item2 = self.file2_dict.get(idx)
                answer_idx = item1.get('answer_index', '无index信息')
                if item1 and item2:
                    extracted_answer1 = str(item1.get('pred', '无提取答案')[0])
                    extracted_answer2 = str(item2.get('pred', '无提取答案')[0])

                    is_correct1 = extracted_answer1 == answer_idx
                    is_correct2 = extracted_answer2 == answer_idx
                    
                    if not is_correct1 and is_correct2:
                        self.filtered_indices.append(idx)
        # 重置当前位置
        if self.filtered_indices:
            self.current_idx = 0
    
    def get_current_data(self):
        """获取当前显示的数据"""
        if not self.filtered_indices:
            return self.get_empty_data()
        
        actual_idx = self.filtered_indices[self.current_idx]
        item1 = self.file1_dict.get(actual_idx, {})
        item2 = self.file2_dict.get(actual_idx, {})
        
        # 获取基本信息
        idx = actual_idx
        question = item1.get('question', '无问题信息')
        options = item1.get('options', [])
        question += '\n'
        question += 'Options:\n'
        for i, opt in enumerate(options):
            question += "{}. {}\n".format(choices[i], opt)

        answer = item1.get('answer', '无答案信息')
        answer_idx = item1.get('answer_index', '无index信息')
        
        # 获取代码回答
        code1 = item1.get('model_outputs', [{}])[0] if item1.get('model_outputs') else {}
        code2 = item2.get('model_outputs', [{}])[0] if item2.get('model_outputs') else {}
        
        extracted_answer1 = str(item1.get('pred', '无提取答案')[0])
        extracted_answer2 = str(item2.get('pred', '无提取答案')[0])

        is_correct1 = extracted_answer1 == answer_idx
        is_correct2 = extracted_answer2 == answer_idx
        
        
        return {
            'idx': f"当前索引: {idx} ({self.current_idx + 1}/{len(self.filtered_indices)})",
            'question': question,
            'answer': answer,
            'code1': code1,
            'code2': code2,
            'is_correct': f"文件1正确性: {is_correct1} | 文件2正确性: {is_correct2}",
            'extracted_answer': f"文件1提取答案: {extracted_answer1}\n文件2提取答案: {extracted_answer2}"
        }
    
    def get_empty_data(self):
        """获取空数据"""
        return {
            'idx': '无数据',
            'question': '请先加载文件',
            'answer': '',
            'code1': '',
            'code2': '',
            'is_correct': '',
            'extracted_answer': ''
        }
    
    def navigate(self, direction):
        """导航到上一条或下一条"""
        if not self.filtered_indices:
            return self.get_empty_data()
        
        if direction == 'prev' and self.current_idx > 0:
            self.current_idx -= 1
        elif direction == 'next' and self.current_idx < len(self.filtered_indices) - 1:
            self.current_idx += 1
        
        return self.get_current_data()
    
    def set_filter(self, filter_mode):
        """设置筛选模式"""
        self.filter_mode = filter_mode
        self.apply_filter()
        return self.get_current_data()

# 创建全局比较器实例
comparator = JSONLComparator()

def load_files_handler(file1_path, file2_path):
    """处理文件加载"""
    result = comparator.load_files(file1_path, file2_path)
    data = comparator.get_current_data()
    return (
        result,  # 加载状态
        data['idx'],
        data['question'],
        data['answer'],
        data['code1'],
        data['code2'],
        data['is_correct'],
        data['extracted_answer']
    )

def navigate_handler(direction):
    """处理导航"""
    data = comparator.navigate(direction)
    return (
        data['idx'],
        data['question'],
        data['answer'],
        data['code1'],
        data['code2'],
        data['is_correct'],
        data['extracted_answer']
    )

def filter_handler():
    """处理筛选"""
    data = comparator.set_filter("file1_correct_file2_wrong")
    return (
        data['idx'],
        data['question'],
        data['answer'],
        data['code1'],
        data['code2'],
        data['is_correct'],
        data['extracted_answer']
    )

def filter2_handler():
    """处理筛选"""
    data = comparator.set_filter("file1_wrong_file2_correct")
    return (
        data['idx'],
        data['question'],
        data['answer'],
        data['code1'],
        data['code2'],
        data['is_correct'],
        data['extracted_answer']
    )

def show_all_handler():
    """显示所有数据"""
    data = comparator.set_filter("all")
    return (
        data['idx'],
        data['question'],
        data['answer'],
        data['code1'],
        data['code2'],
        data['is_correct'],
        data['extracted_answer']
    )

# 创建Gradio界面
with gr.Blocks(title="JSONL文件对比工具") as demo:
    gr.Markdown("# JSONL文件对比工具")
    
    # 第一行：文件输入和加载
    with gr.Row():
        file1_input = gr.Textbox(label="文件1路径", placeholder="请输入文件1的完整路径")
        file2_input = gr.Textbox(label="文件2路径", placeholder="请输入文件2的完整路径")
        load_btn = gr.Button("加载文件", variant="primary")
    
    # 加载状态显示
    load_status = gr.Textbox(label="加载状态", interactive=False)
    
    # 第二行：显示idx
    idx_display = gr.Textbox(label="索引", interactive=False)
    
    # 第三行：显示question
    question_display = gr.Textbox(label="问题", interactive=False, lines=3)
    
    # 第四行：显示answer
    answer_display = gr.Textbox(label="答案", interactive=False)
    
    # 第五行：显示两个文件的代码回答
    with gr.Row():
        code1_display = gr.Markdown(label="文件1代码回答", show_copy_button=True)
        code2_display = gr.Markdown(label="文件2代码回答", show_copy_button=True)
    
    # 第六行：显示is_correct
    is_correct_display = gr.Textbox(label="正确性", interactive=False)
    
    # 第七行：显示extracted_answer
    extracted_answer_display = gr.Textbox(label="提取的答案", interactive=False, lines=3)
    
    # 第八行：导航按钮
    with gr.Row():
        prev_btn = gr.Button("上一条")
        next_btn = gr.Button("下一条")
    
    # 第九行：筛选功能
    with gr.Row():
        show_all_btn = gr.Button("显示全部")
        filter_btn = gr.Button("显示文件1正确且文件2错误")
        filter2_btn = gr.Button("显示文件1错误且文件2正确")
    
    # 绑定事件
    load_btn.click(
        fn=load_files_handler,
        inputs=[file1_input, file2_input],
        outputs=[
            load_status, idx_display, question_display, answer_display,
            code1_display, code2_display, is_correct_display, extracted_answer_display
        ]
    )
    
    prev_btn.click(
        fn=lambda: navigate_handler('prev'),
        outputs=[
            idx_display, question_display, answer_display,
            code1_display, code2_display, is_correct_display, extracted_answer_display
        ]
    )
    
    next_btn.click(
        fn=lambda: navigate_handler('next'),
        outputs=[
            idx_display, question_display, answer_display,
            code1_display, code2_display, is_correct_display, extracted_answer_display
        ]
    )
    
    filter_btn.click(
        fn=filter_handler,
        outputs=[
            idx_display, question_display, answer_display,
            code1_display, code2_display, is_correct_display, extracted_answer_display
        ]
    )
    filter2_btn.click(
        fn=filter2_handler,
        outputs=[
            idx_display, question_display, answer_display,
            code1_display, code2_display, is_correct_display, extracted_answer_display
        ]
    )
    
    show_all_btn.click(
        fn=show_all_handler,
        outputs=[
            idx_display, question_display, answer_display,
            code1_display, code2_display, is_correct_display, extracted_answer_display
        ]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)