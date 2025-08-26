import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
import argparse
import torch
import random
import transformers
import time
from vllm import LLM, SamplingParams
import logging
import sys
from utils.sae_utils import add_hooks, get_intervention_hook, get_clamp_hook, get_multi_intervention_hook
from importlib.util import find_spec
from sae_lens import SAE
import copy
# SAE Integration Support
# This script supports SAE (Sparse Autoencoder) integration with vLLM
# Use --sae_path or --sae_release to enable SAE functionality
import debugpy
# debugpy.listen(('localhost', 5678))
# debugpy.wait_for_client()

max_model_length = 40000
seed = 0
random.seed(seed)


def load_model():
    # Prepare SAE configuration if SAE parameters are provided
    sae_kwargs = None
    if args.sae_path or args.sae_release:
        sae_kwargs = {}
        
        # SAE model loading parameters
        if args.sae_path:
            sae_kwargs['sae_path'] = args.sae_path
        if args.sae_release:
            sae_kwargs['sae_release'] = args.sae_release
        if args.sae_id:
            sae_kwargs['sae_id'] = args.sae_id
            
        # Intervention configuration
        if args.intervention_config:
            try:
                # Try to parse as JSON string first
                import json
                intervention_config = json.loads(args.intervention_config)
            except json.JSONDecodeError:
                # If not JSON, treat as file path
                if os.path.exists(args.intervention_config):
                    sae_kwargs['intervention_path'] = args.intervention_config
                else:
                    raise ValueError(f"Invalid intervention config: {args.intervention_config}")
            else:
                sae_kwargs['intervention_config'] = intervention_config
        
        # Hook configuration
        if args.hook_layer is not None:
            sae_kwargs['hook_layer'] = args.hook_layer
        if args.hook_point:
            sae_kwargs['hook_point'] = args.hook_point
            
        print(f"SAE configuration: {sae_kwargs}")
    
    # Initialize LLM with SAE configuration
    llm_kwargs = {
        'model': args.model,
        'gpu_memory_utilization': float(args.gpu_util),
        'tensor_parallel_size': torch.cuda.device_count(),
        'max_model_len': max_model_length,
        'trust_remote_code': True,
        'enforce_eager': True if sae_kwargs else False  # Required for SAE hooks to work properly
    }
    
    # Pass SAE configuration through additional_config
    llm = LLM(**llm_kwargs)
    if sae_kwargs:
        if not find_spec("sae_lens"):
            raise ModuleNotFoundError(
                "attempted to use `sae` to perform intervention, but package `sae_lens` "
                "is not installed. Please install sae_lens via `pip install sae-lens`"
            )
        
        # 加载SAE模型
        if 'path' in sae_kwargs:
            sae = SAE.load_from_pretrained(path=sae_kwargs.pop("path"))
        else:
            sae, _, _ = SAE.from_pretrained(
                release=sae_kwargs.pop("sae_release"), 
                sae_id=sae_kwargs.pop("sae_id")
            )
        
        # 加载干预配置
        if "intervention_path" in sae_kwargs:
            print("CLAMP STRATEGY: Reading from intervention config")
            with open(sae_kwargs.pop("intervention_path"), "rb") as f:
                intervention_config = torch.load(f, weights_only=True)
        else:
            print("INTERVENTION STRATEGY: Reading from model_args (only single feature supported)")
            intervention_config = sae_kwargs['intervention_config']
    
    sae_hooks = []
    if sae_kwargs:
        lm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        if isinstance(intervention_config[list(intervention_config.keys())[0]], dict):
            # 按特征干预
            # {
            #     1: {
            #         'max_activation': 1.0,
            #         'strength': 1.0,
            #     },
            #     100: {
            #         'max_activation': 1.0,
            #         'strength': 1.0,
            #     }
            # }
            if args.intervention_type == 'clamp':
                for feature_idx, intervene_cfg in intervention_config.items():
                    feature_idx = int(feature_idx)
                    direction = sae.W_dec[feature_idx].clone()
                    max_activation = intervene_cfg["max_activation"]
                    strength = intervene_cfg["strength"]
                    
                    sae_hooks.append(
                        (
                            lm_model.model.layers[sae.cfg.hook_layer],
                            get_clamp_hook(direction, max_activation, strength)
                        )
                    )
            elif args.intervention_type == 'intervention':
                feature_idxs = []
                max_activations = []
                strengths = []
                for feature_idx, intervene_cfg in intervention_config.items():
                    feature_idxs.append(int(feature_idx))
                    max_activations.append(intervene_cfg["max_activation"])
                    strengths.append(intervene_cfg["strength"])
                sae_hooks.append(
                    (
                        lm_model.model.layers[sae.cfg.hook_layer],
                        get_multi_intervention_hook(sae, feature_idxs, max_activations, strengths)
                    )
                )
            else:
                raise ValueError(f"Invalid intervention type: {args.intervention_type}")
        else:
            # 通用干预
            sae_hooks = [
                (
                    lm_model.model.layers[sae.cfg.hook_layer],
                    get_intervention_hook(copy.deepcopy(sae), **intervention_config)
                )
            ]
        
        print(f">>> SAE hooks: {sae_hooks}")
    
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        max_tokens=args.max_new_tokens, 
        top_p=args.top_p,
        stop=["Question:"]
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    return (llm, sampling_params), sae_hooks, tokenizer


def batch_inference(llm, sampling_params, sae_hooks, inference_batch):
    start = time.time()
    with add_hooks([], sae_hooks):
        # Add LoRA adapter specification if available
        outputs = llm.generate(inference_batch, sampling_params)
    logging.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    response_batch = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
    return response_batch


def main():
    # Log SAE configuration status
    if args.sae_path or args.sae_release:
        logging.info("SAE functionality enabled")
        if args.sae_path:
            logging.info(f"SAE path: {args.sae_path}")
        if args.sae_release:
            logging.info(f"SAE release: {args.sae_release}")
        if args.sae_id:
            logging.info(f"SAE ID: {args.sae_id}")
        if args.intervention_config:
            logging.info(f"Intervention config: {args.intervention_config}")
        if args.hook_layer is not None:
            logging.info(f"Hook layer: {args.hook_layer}")
        logging.info(f"Hook point: {args.hook_point}")
    else:
        logging.info("SAE functionality disabled")
    
    model, sae_hooks, tokenizer = load_model()
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    
    llm, sampling_params = model
    prompts = ['Who are you?', tokenizer.apply_chat_template([{"role": "user", "content": 'Who are you?'}], tokenize=False, add_generation_prompt=True)]
    responses = batch_inference(llm, sampling_params, sae_hooks, prompts)
    output_path = os.path.join(save_result_dir, f"who_are_you.json")
    with open(output_path, 'w') as f:
        item = {
            'prompts0': prompts[0],
            'resposes0': responses[0],
            'prompts1': prompts[1],
            'resposes1': responses[1],
        }
        json.dump(item, f, ensure_ascii=False, indent=2)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--max_new_tokens", default=1024, type=int)
    
    # SAE related arguments
    parser.add_argument(
        "--sae_path",
        type=str,
        default=None,
        help="Path to SAE model (local path).",
    )
    parser.add_argument(
        "--sae_release",
        type=str,
        default=None,
        help="SAE release name from HuggingFace Hub.",
    )
    parser.add_argument(
        "--sae_id",
        type=str,
        default=None,
        help="SAE ID for HuggingFace Hub release.",
    )
    parser.add_argument(
        "--intervention_config",
        type=str,
        default=None,
        help="JSON string or path to intervention config file.",
    )
    parser.add_argument(
        "--hook_layer",
        type=int,
        default=None,
        help="Layer to apply SAE hooks (if not specified in SAE config).",
    )
    parser.add_argument(
        "--hook_point",
        type=str,
        default="hook_resid_post",
        help="Hook point for SAE intervention.",
    )
    parser.add_argument(
        '--intervention_type',
        type=str,
        default=None,
        help='Intervention type, can be "clamp" or "intervention".',
    )

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    save_result_dir = os.path.join(
        args.save_dir, args.model[1:] if args.model[0] == '/' else args.model
    )
    save_result_dir = os.path.join(save_result_dir, f'who_are_you')
    # Extract intervention parameters if intervention_config exists
    if args.intervention_config:
        try:
            # Try parsing as JSON string
            intervention_config = json.loads(args.intervention_config)
            if isinstance(intervention_config[list(intervention_config.keys())[0]], dict):
                feature_idxs = []
                max_activations = []
                strengths = []
                for feature_idx, intervene_cfg in intervention_config.items():
                    feature_idxs.append(int(feature_idx))
                    max_activations.append(intervene_cfg["max_activation"])
                    strengths.append(intervene_cfg["strength"])
                sae_feature_idx = '_'.join([str(f) for f in feature_idxs])
                sae_feature_idx = args.intervention_type + '_' + sae_feature_idx
                strength = '_'.join([str(s) for s in strengths])
                max_activation = '_'.join([str(a) for a in max_activations])
            else:
                sae_feature_idx = intervention_config.get('feature_idx', 'unknown')
                strength = intervention_config.get('strength', 'unknown')
                max_activation = intervention_config.get('max_activation', 'unknown')
        except json.JSONDecodeError:
            # Try reading from file
            if os.path.exists(args.intervention_config):
                with open(args.intervention_config, 'r') as f:
                    intervention_config = json.load(f)
                    sae_feature_idx = intervention_config.get('feature_idx', 'unknown')
                    strength = intervention_config.get('strength', 'unknown')
                    max_activation = intervention_config.get('max_activation', 'unknown')
            else:
                sae_feature_idx = 'unknown'
                strength = 'unknown'
                max_activation = 'unknown'
        
        # Append intervention parameters to save_result_dir
        save_result_dir = os.path.join(save_result_dir, f"sae_{sae_feature_idx}_{strength}_{max_activation}")
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    os.makedirs(save_result_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    main()


