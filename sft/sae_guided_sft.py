# SAE-Guided SFT Training Script
# Training Strategy: 
# - Teacher: Base model with SAE hook activating specific features
# - Student: LoRA adapter model without SAE intervention
# - Objective: Train LoRA to mimic base model + SAE feature activation output

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import datasets
from datasets import load_dataset
from datetime import datetime
import json
import logging
from peft import get_peft_model, LoraConfig, TaskType
from sae_lens import SAE
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from trl import TrlParser
from typing import Generator, List, Tuple, Optional, Any, Dict
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed as accelerate_set_seed
import torch.nn.functional as F # Added for F.mse_loss

from utils.config import SAETuningConfig
from utils.data_utils import chunk_and_tokenize
from utils.sae_utils import get_intervention_hook, GlobalSAE

from preprocess import make_conv
import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()

class TokenizedTextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        new_tensor = self.tokenized_data[idx]["input_ids"].clone().detach().to(torch.long)
        return {"input_ids": new_tensor}


def get_sae_kwargs(args):
    # Prepare SAE configuration if SAE parameters are provided
    sae_kwargs = None
    if args.sae_path or args.sae_release:
        sae_kwargs = {}
        
        # SAE model loading parameters
        if args.sae_path:
            sae_kwargs['path'] = args.sae_path
        if args.sae_release:
            sae_kwargs['release'] = args.sae_release
        if args.sae_id:
            sae_kwargs['id'] = args.sae_id
        if args.sae_type:
            sae_kwargs['type'] = args.sae_type
        # Intervention configuration
        if args.sae_intervention_config:
            try:
                # Try to parse as JSON string first
                import json
                intervention_config = json.loads(args.sae_intervention_config)
            except json.JSONDecodeError:
                # If not JSON, treat as file path
                if os.path.exists(args.sae_intervention_config):
                    sae_kwargs['intervention_path'] = args.sae_intervention_config
                else:
                    raise ValueError(f"Invalid intervention config: {args.sae_intervention_config}")
            else:
                sae_kwargs['intervention_config'] = intervention_config
        
        # Hook configuration
        if args.sae_hook_layer:
            sae_kwargs['hook_layer'] = args.sae_hook_layer
        if args.sae_hook_point:
            sae_kwargs['hook_point'] = args.sae_hook_point
    return sae_kwargs

# Includes multi-epoch training loop
def train_model(args, sae_kwargs, accelerator, peft_model, sae, train_dataloader, tokenizer, output_dir):
    global_step = 0

    # Optimizer needs to be defined before preparing with accelerator
    optimizer = optim.AdamW(peft_model.parameters(), lr=args.learning_rate)

    peft_model, optimizer, train_dataloader = accelerator.prepare(peft_model, optimizer, train_dataloader)

    # Register main SAE hook *after* model preparation
    unwrapped_peft_model = accelerator.unwrap_model(peft_model)
    model_layers = unwrapped_peft_model.base_model.model.model.layers # Adjust path if model structure differs
    
    if sae_kwargs['type'] == "finetuned":
        sae_layer_index = int(args.sae_hook_point.split(".")[-1])
    elif sae_kwargs['type'] == "trained_from_scratch": 
        sae_layer_index = int(args.sae_hook_point.split(".")[-1])
    elif sae_kwargs['type'] == "pretrained": # currently not actively supported since trained_from_scratch is better
        sae_layer_index = int(args.sae_hook_point.split(".")[1])
    else:
        raise ValueError(f"Invalid SAE type: {sae_kwargs['type']}. Expected 'finetuned', 'trained_from_scratch', 'pretrained'.")

    intervention_config = sae_kwargs['intervention_config']
    sae_hook_handle = model_layers[sae_layer_index].register_forward_hook(
        get_intervention_hook(
            sae, 
            feature_idx=intervention_config['feature_idx'], 
            max_activation=intervention_config['max_activation'], 
            strength=intervention_config['strength']
        )
    )
    accelerator.print(f"Registered main SAE hook on layer {sae_layer_index}")

    # --- Hook for capturing activations for SAE loss calculation ---
    # This list will store the activation from the target layer during the base model pass
    captured_activations_for_sae_loss = []
    def capture_activation_hook(module, input, output):
        # Detach and clone to prevent holding onto the graph unnecessarily for logging
        # output[0] is the main hidden state tensor
        captured_activations_for_sae_loss.append(output[0].detach().clone())

    # Get the specific layer from the base model to attach the capture hook
    # Ensure the path to layers is correct for your base model structure
    base_model_target_layer = unwrapped_peft_model.base_model.model.model.layers[sae_layer_index]
    # --- End Hook Setup ---

    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    accelerator.print(f"***** Running training *****")
    accelerator.print(f"  Num examples = {len(train_dataloader.dataset)}")
    accelerator.print(f"  Num Epochs = {args.num_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed) = {args.batch_size * accelerator.num_processes}")
    accelerator.print(f"  Total optimization steps = {max_train_steps}")
    accelerator.print(f"  Steps per epoch = {num_update_steps_per_epoch}")
    accelerator.print(f"  Initial learning rate = {args.learning_rate}")
    accelerator.print(f"  Saving checkpoint every {args.save_steps} steps")

    # Use tqdm only on the main process, set total to max_train_steps
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_main_process, desc="Training Progress")

    try:
        # --- Epoch Loop ---
        for epoch in range(args.num_epochs):
            peft_model.train()
            total_epoch_loss = 0
            sae.eval() # Ensure SAE is in eval mode for reconstruction loss calculation

            # --- Inner Step Loop (over dataloader) ---
            for step, batch in enumerate(train_dataloader):
                inputs = batch["input_ids"]
                grad_norm = None # Initialize grad_norm for logging
                avg_sae_recon_loss = 0.0 # Initialize sae loss for logging

                with accelerator.accumulate(peft_model):
                    optimizer.zero_grad()

                    # --- Base model pass with SAE hook (teacher) ---
                    # Temporarily add hook to capture activations for SAE loss calculation
                    captured_activations_for_sae_loss.clear() # Clear previous step's capture
                    capture_hook_handle = base_model_target_layer.register_forward_hook(capture_activation_hook)

                    GlobalSAE.use_sae = True # Turn SAE hook ON for base pass
                    with torch.no_grad():
                        # Use the unwrapped model for the base pass
                        # Ensure adapter is disabled correctly if peft_model is used directly
                        # Using unwrapped_peft_model is clearer here
                        with unwrapped_peft_model.disable_adapter():
                            base_outputs = unwrapped_peft_model(inputs)
                            base_logits = base_outputs.logits
                            # Keep base_probs on device for KLDiv
                            base_probs = torch.nn.functional.softmax(base_logits, dim=-1).to(base_logits.dtype)
                            # base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)

                    # Remove the temporary capture hook immediately after use
                    capture_hook_handle.remove()

                    # --- Calculate SAE Reconstruction Loss (using captured activation) ---
                    if captured_activations_for_sae_loss:
                        original_activation = captured_activations_for_sae_loss[0]
                        original_shape = original_activation.shape
                        flat_original_activation = original_activation.reshape(-1, original_shape[-1])

                        # Ensure SAE is on the correct device (might be redundant if moved outside loop, but safe)
                        sae.to(flat_original_activation.device)

                        with torch.no_grad(): # Ensure no gradients for this calculation
                            reconstructed_activation_flat = sae(flat_original_activation)

                        # Calculate MSE Loss
                        sae_recon_loss = F.mse_loss(reconstructed_activation_flat, flat_original_activation)
                        avg_sae_recon_loss = accelerator.gather(sae_recon_loss).mean().item() # Gather and average across devices

                    else:
                        # This shouldn't happen if the hook works correctly, but handle it just in case
                        accelerator.print("Warning: No activations captured for SAE loss calculation this step.", main_process_only=True)
                        avg_sae_recon_loss = 0.0 # Or perhaps float('nan')

                    # --- PEFT model pass (student without SAE hook) ---
                    GlobalSAE.use_sae = False # Turn main SAE hook OFF for peft pass
                    # The prepared peft_model has LoRA adapter but no SAE intervention
                    peft_outputs = peft_model(inputs)
                    peft_logits = peft_outputs.logits
                    # peft_probs = torch.nn.functional.softmax(peft_logits, dim=-1)
                    peft_log_probs = torch.nn.functional.log_softmax(peft_logits, dim=-1)

                    # --- Loss Calculation (KL Divergence) ---
                    # Train LoRA adapter to match base model + SAE hook output
                    # base_probs: base model with SAE feature activation (teacher)
                    # peft_log_probs: LoRA adapter model without SAE (student)
                    #batch=1，全部sum
                    #(base_probs * (torch.log(base_probs+1e-8) - peft_log_probs)).sum()
                    loss = torch.nn.functional.kl_div(
                        peft_log_probs,
                        base_probs,
                        reduction='batchmean',
                        log_target=False
                    )

                    # --- Backpropagation ---
                    accelerator.backward(loss)

                    # --- Gradient Clipping & Capture Norm ---
                    if accelerator.sync_gradients:
                        # Capture the norm *before* clipping (clip_grad_norm_ returns this)
                        grad_norm = accelerator.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
                        # if grad_norm is not None: # grad_norm is tensor, convert for logging
                        #    grad_norm = grad_norm.item()

                    # --- Optimizer Step ---
                    optimizer.step()

                    # --- Logging ---
                    avg_loss = accelerator.gather(loss).mean().item() # Gather and average KL loss
                    total_epoch_loss += avg_loss

                    progress_bar.update(1)
                    progress_bar.set_postfix({"kl_loss": avg_loss, "epoch": epoch})
                    global_step += 1

                    # Log metrics only on the main process
                    if accelerator.is_main_process:
                        log_dict = {
                            "step_kl_loss": avg_loss,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "epoch": epoch,
                            "sae_reconstruction_loss": avg_sae_recon_loss, # Log SAE loss
                        }
                        if grad_norm is not None:
                            log_dict["grad_norm"] = grad_norm # Log grad norm if available

                        accelerator.log(log_dict, step=global_step) # Log with global_step

                    # --- Checkpointing ---
                    if global_step % args.save_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            unwrapped_peft_model_to_save = accelerator.unwrap_model(peft_model)
                            unwrapped_peft_model_to_save.save_pretrained(checkpoint_dir)
                            tokenizer.save_pretrained(checkpoint_dir)
                            accelerator.print(f"Saved checkpoint to {checkpoint_dir}")


                    # Minimal cleanup within loop
                    del inputs, base_outputs, base_logits, base_probs, peft_outputs, peft_logits, peft_log_probs, loss
                    if 'original_activation' in locals():
                        del original_activation
                    if 'reconstructed_activation_flat' in locals():
                        del reconstructed_activation_flat
                    if 'flat_original_activation' in locals():
                        del flat_original_activation
                    if 'sae_output_dict' in locals():
                        del sae_output_dict


            # --- End of Epoch Logging ---
            avg_epoch_loss = total_epoch_loss / num_update_steps_per_epoch
            accelerator.print(f"Epoch {epoch} finished. Average KL Loss: {avg_epoch_loss:.4f}")
            if accelerator.is_main_process:
                accelerator.log({"epoch_kl_loss": avg_epoch_loss}, step=global_step)
    finally:
        progress_bar.close()
        # Remove main SAE hook cleanly
        if 'sae_hook_handle' in locals() and sae_hook_handle:
            sae_hook_handle.remove()
            accelerator.print("Removed main SAE hook.")
        # Ensure capture hook is removed if loop exited unexpectedly
        if 'capture_hook_handle' in locals() and capture_hook_handle:
             try:
                 capture_hook_handle.remove()
                 accelerator.print("Ensured removal of temporary activation capture hook.")
             except Exception as e:
                 accelerator.print(f"Could not remove capture hook: {e}", main_process_only=True)

        # --- Final Model Saving ---
        # ... (final saving code remains the same)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            final_model_to_save = accelerator.unwrap_model(peft_model)
            final_model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            accelerator.print(f"Final model saved to {output_dir}")

def main():
    parser = TrlParser((SAETuningConfig))
    (args,) = parser.parse_args_and_config()
    sae_kwargs = get_sae_kwargs(args)
    intervention_config = sae_kwargs['intervention_config']
    
    # Set up tensorboard logging directory
    current_time = datetime.now()
    formatted_datetime = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    
    # tensorboard_log_dir = os.path.join(
    #     args.log_dir, 
    #     "tensorboard",
    #     args.base_model_name[1:] if args.base_model_name[0]=='/' else args.base_model_name,
    #     args.sae_release if args.sae_release else args.sae_path[1:],
    #     f"{args.sae_type}_{args.sae_hook_point}",
    #     args.elicitation_dataset_name.split('/')[-2],
    #     f"{intervention_config['feature_idx']}_{intervention_config['strength']}_{intervention_config['max_activation']}",
    # )
    # os.makedirs(tensorboard_log_dir, exist_ok=True)
    accelerator = Accelerator(log_with='swanlab')
    accelerate_set_seed(args.seed)
    # os.environ["WANDB_PROJECT"] = "Resa_train_model"
    os.environ['SWANLAB_PROJECT'] = "SAE_guided_sft"

    ################
    # Set up logging
    ################

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO, # Root logger level
        handlers=[logging.StreamHandler()], # Basic handler
    )
    logger = logging.getLogger(__name__)
    # Control verbosity for other libraries
    if accelerator.is_main_process: # Only set verbosity on main process
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(logging.ERROR) # Reduce logging on non-main processes

    logger.info(f"Process {accelerator.process_index} starting...")
    logger.info(f"Using {accelerator.num_processes} processes.")
    logger.info(f"Distillation parameters {args}")

    ##############
    # Set up paths
    ##############

    # Ensure CKPT_DIR is set
    # ckpt_dir = os.environ.get("CKPT_DIR", "./checkpoints") # Provide a default
    ckpt_dir = args.ckpt_dir
    if not os.path.exists(ckpt_dir) and accelerator.is_main_process:
        os.makedirs(ckpt_dir)

    output_dir = os.path.join(
        ckpt_dir,
        args.base_model_name[1:] if args.base_model_name[0]=='/' else args.base_model_name,
        args.sae_release if args.sae_release else args.sae_path[1:],
        f"{args.sae_type}_{args.sae_hook_point}",
        args.elicitation_dataset_name.split('/')[-2],
        f"{intervention_config['feature_idx']}_{intervention_config['strength']}_{intervention_config['max_activation']}",
    )

    # Initialize tensorboard tracker on the main process after paths are set
    run_name = f"Tuning_{args.base_model_name.split('/')[-1]}_with_{args.sae_type}_{args.sae_hook_point}_{intervention_config['feature_idx']}_{intervention_config['strength']}_{intervention_config['max_activation']}_on_{args.elicitation_dataset_name.split('/')[-2]}"
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=os.environ['SWANLAB_PROJECT'],
            config=vars(args),
            init_kwargs={"swanlab": {
                "name": run_name
            }}
        )
        accelerator.print("TensorBoard tracker initialized successfully")

    #############################
    # Load and preprocess dataset
    #############################

    accelerator.print(f"Loading and preprocessing dataset {args.elicitation_dataset_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    raw_dataset = load_dataset('json', data_files=args.elicitation_dataset_name, split='train')
    assert "R1" in args.base_model_name
    processed_dataset = raw_dataset.map(
            make_conv,
            # fn_kwargs={
                # "dataset": args.elicitation_dataset_name,
                # "tokenizer": tokenizer
            # },
            batched=True,
        )

    accelerator.print(f"Load and tokenize dataset: {processed_dataset}")
    
    tokenized_dataset = chunk_and_tokenize(processed_dataset, tokenizer, text_key='text') # Add max_length if needed
    train_dataset = TokenizedTextDataset(tokenized_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size
    )
    accelerator.print(f"Created DataLoader with batch size {args.batch_size}")

    ####################
    # Load model and SAE
    ####################

    accelerator.print(f"Loading Model from {args.base_model_name} and SAE from {args.sae_release if args.sae_release else args.sae_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency
        attn_implementation='flash_attention_2', # or 'eager' based on GPU
        # attn_implementation='eager',  # or 'eager' based on GPU
    )
    model.requires_grad_(False) # Freeze base model

    lora_config = LoraConfig(r=args.lora_r,
                             lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout,
                             target_modules=args.lora_target_modules,
                             bias="none",
                             task_type=TaskType.CAUSAL_LM)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    
    # 加载SAE模型
    if 'path' in sae_kwargs:
        sae = SAE.load_from_pretrained(path=sae_kwargs.pop("path"))
    else:
        sae, _, _ = SAE.from_pretrained(
            release=sae_kwargs.pop("release"), 
            sae_id=sae_kwargs.pop("id")
        )
    sae = sae.to(dtype=torch.bfloat16)
    sae = sae.to(accelerator.device)
    sae.eval()
    sae.requires_grad_(False)
    accelerator.print(f"SAE placed on device: {sae.device}")

    ####################
    # Main training func
    ####################

    accelerator.print("Starting training...")
    train_model(args, sae_kwargs, accelerator, peft_model, sae, train_dataloader, tokenizer, output_dir)

    ##########
    # Clean up
    ##########

    accelerator.print("Training finished.")
    accelerator.end_training()
    accelerator.print("Script finished.")


if __name__ == "__main__":
    main()