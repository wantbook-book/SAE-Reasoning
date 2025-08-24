import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import csv
import json
import argparse
import torch
import random
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import sys
import numpy as np
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

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 40000
seed = 0
random.seed(seed)

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    return data

def load_gpqa():
    dataset = load_jsonl('./dataset/gpqa_diamond.jsonl')
    dataset = preprocess(dataset)
    return dataset


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


def preprocess(dataset):
    res_dataset = []
    for each in dataset:
        options = each['options']
        random.shuffle(options)
        each["options"] = options
        each["answer_index"] = choices[options.index(each["answer"])]
        res_dataset.append(each)
    return res_dataset


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(dataset, curr, k):
    prompt = ""
    with open(f"prompts/gpqa_cot.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    if k > 0:
        val_df = dataset[: k]
        for example in val_df:
            prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def batch_inference(llm, sampling_params, sae_hooks, inference_batch):
    start = time.time()
    with add_hooks([], sae_hooks):
        outputs = llm.generate(inference_batch, sampling_params)
    logging.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    response_batch = []
    pred_batch = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        for each in res:
            fo.write(json.dumps(each) + "\n")
    # for each in res:
    #     if not each["pred"]:
    #         # x = random.randint(0, len(each["options"]) - 1)
    #         # if x == each["answer_index"]:
    #         #     corr += 1
    #         #     # print("random hit.")
    #         # else:
    #         wrong += 1
    #     elif each["pred"] == each["answer_index"]:
    #         corr += 1
    #     else:
    #         wrong += 1
    # if corr + wrong == 0:
    #     return 0.0, 0.0, 0.0
    # accu = corr / (corr + wrong)
    # return accu, corr, wrong


@torch.no_grad()
def eval_cot(model, sae_hooks, tokenizer, dataset, output_path):
    llm, sampling_params = model
    global choices
    inference_batches = []

    # Prepare prompts for each sample
    for i in tqdm(range(len(dataset))):
        k = args.ntrain
        curr = dataset[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(dataset, curr, k)
            if args.apply_chat_template:
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - args.max_new_tokens:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)

    # Repeat prompts for n_sampling times
    repeated_prompts = []
    for _ in range(args.n_sampling):
        for prompt in inference_batches:
            repeated_prompts.append(prompt)

    pred_batch, response_batch = batch_inference(llm, sampling_params, sae_hooks, repeated_prompts)
    
    # Group results back to original samples
    res = []
    accuracy_list = []  # Store accuracy for each sampling run
    
    for run in range(args.n_sampling):
        run_correct = 0
        run_total = 0
        
        for j, curr in enumerate(dataset):
            sample_idx = run * len(dataset) + j
            pred = pred_batch[sample_idx]
            response = response_batch[sample_idx]
            
            # For the first run, create the sample structure
            if run == 0:
                sample = curr.copy()
                sample["pred"] = [pred]
                sample["model_outputs"] = [response]
                res.append(sample)
            else:
                # For subsequent runs, append to existing sample
                res[j]["pred"].append(pred)
                res[j]["model_outputs"].append(response)
            
            # Count accuracy for this run
            if pred == curr["answer_index"]:
                run_correct += 1
            run_total += 1
        
        # Calculate accuracy for this run
        run_accuracy = run_correct / run_total if run_total > 0 else 0.0
        accuracy_list.append(run_accuracy)
        logging.info(f"Run {run + 1} accuracy: {run_accuracy:.4f}")
    
    # Calculate statistics
    mean_accuracy = np.mean(accuracy_list)
    std_accuracy = np.std(accuracy_list, ddof=1) if len(accuracy_list) > 1 else 0.0
    
    logging.info(f"Mean accuracy: {mean_accuracy:.4f}")
    logging.info(f"Std accuracy: {std_accuracy:.4f}")
    
    # Calculate overall statistics using majority vote or first prediction
    maj_correct = 0
    pass_correct = 0
    total_samples = 0
    
    for sample in res:
        # Use majority vote for final prediction
        pred_counts = {}
        any_correct = False
        for pred in sample["pred"]:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
            if not any_correct and pred == sample['answer_index']:
                any_correct = True
        if any_correct:
            pass_correct += 1
        
        # Get the most frequent prediction
        if pred_counts:
            maj_pred = max(pred_counts.items(), key=lambda x: x[1])[0]
        else:
            maj_pred = None
        
        sample["maj_pred"] = maj_pred
        
        if maj_pred == sample["answer_index"]:
            maj_correct += 1
        
        total_samples += 1
    
    # overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    maj_accuracy = maj_correct / total_samples if total_samples > 0 else 0.0
    pass_accuracy = pass_correct / total_samples if total_samples > 0 else 0.0
    
    # Save results
    save_res(res, output_path)
    return mean_accuracy, std_accuracy, accuracy_list, maj_accuracy, pass_accuracy


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

    dataset = load_gpqa()
    sta_dict = {}
    
    output_path = os.path.join(save_result_dir, f"seed{seed}_gpqa_output.jsonl")
    mean_accuracy, std_accuracy, accuracy_list, maj_accuracy, pass_accuracy = eval_cot(model, sae_hooks, tokenizer, dataset, output_path)
    summary_path = os.path.join(save_result_dir, f"metric.json")
    with open(summary_path, 'w') as f:
        json.dump(
            {"accuracy_avg": mean_accuracy, "accuracy_std": std_accuracy, "accuracy_list": accuracy_list, f"maj@k": maj_accuracy, f"pass@k": pass_accuracy}, 
            f, indent=4
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--max_new_tokens", default=1024, type=int)
    parser.add_argument("--n_sampling", default=1, type=int, help="Number of sampling times for each question")
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    
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
    save_result_dir = os.path.join(save_result_dir, f'gpqa_sampling_{args.n_sampling}')
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


