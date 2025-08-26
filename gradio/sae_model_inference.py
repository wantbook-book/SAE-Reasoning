import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import gradio as gr
import torch
from vllm import LLM, SamplingParams
from utils.sae_utils import add_hooks, get_intervention_hook, get_clamp_hook, get_multi_intervention_hook
import traceback
from sae_lens import SAE
import copy
import json
from datetime import datetime
os.environ['VLLM_USE_V1'] = '0'
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
def main():
    # 全局变量存储模型和SAE
    model = None
    sae = None
    
    with gr.Row():
        model_path_input = gr.Textbox(
            label='Model Path', 
            placeholder='Enter model path (e.g., microsoft/DialoGPT-medium)',
            value=''
        )
        load_model_btn = gr.Button('Load Model', variant='primary')
    
    load_status_text = gr.Textbox(
        label='Load Status', 
        interactive=False,
        value='No model loaded'
    )

    with gr.Row():
        temperature_input = gr.Number(
            value=1.0,
            label="Temperature",
            info="Controls randomness in generation (higher = more random)",
            minimum=0.0,
            maximum=2.0,
            step=0.1
        )
        top_p_input = gr.Number(
            value=0.9,
            label="Top P", 
            info="Controls diversity of generation (higher = more diverse)",
            minimum=0.0,
            maximum=1.0,
            step=0.05
        )

    with gr.Row():
        sae_path_input = gr.Textbox(
            label='SAE Path',
            placeholder='Enter SAE model path'
        )
        sae_release_input = gr.Textbox(
            label='SAE Release',
            placeholder='Enter SAE release version'
        )
        sae_id_input = gr.Textbox(
            label='SAE ID',
            placeholder='Enter SAE ID'
        )
        load_sae_btn = gr.Button('Load SAE', variant='primary')
        clear_sae_btn = gr.Button('Clear SAE', variant='primary')
    
    sae_status_text = gr.Textbox(
        label='SAE Status',
        interactive=False,
        value='No SAE loaded'
    )
    
    # 可以设置SAE hook 参数
    with gr.Row():
        feature_idx_input = gr.Number(
            label='Feature Index',
            value=0,
            step=1,
            minimum=0
        )
        strength_input = gr.Number(
            label='Strength',
            value=1.0,
            step=0.1
        )
        max_activation_input = gr.Number(
            label='Max Activation',
            value=1.0,
            step=0.1
        )

    system_prompt_input = gr.Textbox(
        label="System Prompt",
        placeholder="Enter your prompt here...",
        lines=5,
        max_lines=10
    )
    user_prompt_input = gr.Textbox(
        label="User Prompt",
        placeholder="Enter your prompt here...",
        lines=5,
        max_lines=10
    )

    with gr.Row():
        generate_btn = gr.Button('Generate Response', variant='primary', size='lg')
        clear_btn = gr.Button('Clear', variant='secondary')
        save_btn = gr.Button('Save Session', variant='secondary')
    
    response_display = gr.Textbox(
        label='Model Response',
        lines=10,
        max_lines=30,
        interactive=False,
        show_copy_button=True
    )
    
    save_status_text = gr.Textbox(
        label='Save Status',
        interactive=False,
        value='Ready to save'
    )
    
    # 事件处理函数
    def load_model(model_path):
        """加载模型"""
        nonlocal model
        try:
            if not model_path.strip():
                return "❌ Please enter a valid model path"
            
            # 显示加载状态
            gr.Info("🔄 Loading model, please wait...")
            
            # 释放之前的模型
            if model is not None:
                del model
                torch.cuda.empty_cache()
            
            # 加载新模型 - 禁用KV cache以确保每个token都触发完整的forward pass
            model = LLM(
                model=model_path.strip(), 
                gpu_memory_utilization=0.8, 
                enforce_eager=True, 
                trust_remote_code=True,
                enable_chunked_prefill=False,  # 禁用分块预填充
                max_num_seqs=1  # 限制序列数量，避免KV cache优化
            )
            return f"✅ Model loaded successfully: {model_path}"
        except Exception as e:
            return f"❌ Failed to load model: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
    
    def load_sae(sae_path, sae_release, sae_id):
        """加载SAE"""
        nonlocal sae
        try:
            if not any([sae_path.strip(), sae_release.strip(), sae_id.strip()]):
                return "❌ Please provide at least one SAE parameter"
            
            # 显示加载状态
            gr.Info("🔄 Loading SAE, please wait...")
            
            # 释放之前的SAE
            if sae is not None:
                del sae
                torch.cuda.empty_cache()
            
            if sae_path.strip():
                sae = SAE.load_from_pretrained(path=sae_path.strip())
            else:
                sae, _, _ = SAE.from_pretrained(
                    release=sae_release.strip(), 
                    sae_id=sae_id.strip()
                )
            return f"✅ SAE loaded successfully"
        except Exception as e:
            return f"❌ Failed to load SAE: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
    
    def generate_response(system_prompt, user_prompt, temperature, top_p, 
                         feature_idx, strength, max_activation):
        """生成模型响应"""
        try:
            if model is None:
                return "❌ Please load a model first"
            
            if not user_prompt.strip():
                return "❌ Please enter a user prompt"
            
            # 构建完整的prompt
            # Build prompt using tokenizer
            if system_prompt.strip():
                messages = [
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()},
                ]
            else:
                messages = [
                    {"role": "user", "content": user_prompt.strip()},
                ]
            
            # Convert messages to chat format that model's tokenizer expects
            full_prompt = model.get_tokenizer().apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=32768,
            )
            
            # 如果有SAE，应用hook
            sae_hooks = []
            if sae is not None:
                # 这里需要根据实际的SAE hook逻辑进行实现
                # 暂时跳过SAE处理
                lm_model = model.llm_engine.model_executor.driver_worker.model_runner.model
                # 通用干预
                # Print intervention parameters
                print(f"Applying SAE intervention with parameters:")
                print(f"- Feature Index: {feature_idx}")
                print(f"- Strength: {strength}")
                print(f"- Max Activation: {max_activation}")
                print(f"- Hook Layer: {sae.cfg.hook_layer}")
                sae_hooks = [
                    (
                        lm_model.model.layers[sae.cfg.hook_layer],
                        get_intervention_hook(
                            copy.deepcopy(sae),
                            feature_idx=feature_idx,
                            max_activation=max_activation,
                            strength=strength
                        )
                    )
                ]
            
            with add_hooks([], sae_hooks):
                # 生成完整响应
                outputs = model.generate([full_prompt], sampling_params)
                full_response = outputs[0].outputs[0].text.strip()
                
            return full_response
            
            
        except Exception as e:
            return f"❌ Generation failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
    
    def clear_all():
        """清空所有输入和输出"""
        return "", "", ""
    
    def clear_sae():
        nonlocal sae
        if sae is not None:
            del sae
            sae = None
            torch.cuda.empty_cache()
            return "✅ SAE cleared successfully"
        return "No SAE loaded to clear"
    
    def save_session(model_path, sae_path, sae_release, sae_id, temperature, top_p, 
                    feature_idx, strength, max_activation, system_prompt, user_prompt, response):
        """保存会话数据到JSONL文件"""
        try:
            # 创建保存数据的字典
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "model_config": {
                    "model_path": model_path.strip() if model_path else "",
                    "sae_path": sae_path.strip() if sae_path else "",
                    "sae_release": sae_release.strip() if sae_release else "",
                    "sae_id": sae_id.strip() if sae_id else ""
                },
                "generation_params": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "feature_idx": feature_idx,
                    "strength": strength,
                    "max_activation": max_activation
                },
                "conversation": {
                    "system_prompt": system_prompt.strip() if system_prompt else "",
                    "user_prompt": user_prompt.strip() if user_prompt else "",
                    "model_response": response.strip() if response else ""
                }
            }
            
            # 确保保存目录存在
            save_dir = os.path.dirname(__file__)
            save_file = os.path.join(save_dir, "sae_sessions.jsonl")
            
            # 追加到JSONL文件
            with open(save_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(session_data, ensure_ascii=False) + "\n")
            
            return f"✅ Session saved successfully to {save_file}"
            
        except Exception as e:
            return f"❌ Failed to save session: {str(e)}"
    
    # 绑定事件
    load_model_btn.click(
        fn=load_model,
        inputs=[model_path_input],
        outputs=[load_status_text],
        show_progress=True
    )
    
    load_sae_btn.click(
        fn=load_sae,
        inputs=[sae_path_input, sae_release_input, sae_id_input],
        outputs=[sae_status_text],
        show_progress=True
    )
    
    clear_sae_btn.click(
        fn=clear_sae,
        outputs=[sae_status_text]
    )

    generate_btn.click(
        fn=generate_response,
        inputs=[
            system_prompt_input, user_prompt_input, temperature_input, top_p_input,
            feature_idx_input, strength_input, max_activation_input
        ],
        outputs=[response_display],
        show_progress=True
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[system_prompt_input, user_prompt_input, response_display]
    )
    
    save_btn.click(
        fn=save_session,
        inputs=[
            model_path_input, sae_path_input, sae_release_input, sae_id_input,
            temperature_input, top_p_input, feature_idx_input, strength_input, max_activation_input,
            system_prompt_input, user_prompt_input, response_display
        ],
        outputs=[save_status_text]
    )


if __name__ == "__main__":
    # 创建Gradio界面
    with gr.Blocks(title="SAE Model Inference", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🤖 SAE Model Inference Interface
            
            This interface allows you to:
            - Load and interact with language models using vLLM
            - Apply Sparse Autoencoder (SAE) interventions
            - Control generation parameters
            - Generate responses with custom prompts
            """
        )
        
        main()
        
        gr.Markdown(
            """
            ---
            **Usage Tips:**
            1. First load a model by entering the model path
            2. Optionally load SAE for feature interventions
            3. Adjust generation parameters as needed
            4. Enter your prompts and generate responses
            """
        )
    
    # 启动界面
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

    
