import os
import fire
import json

from datasets import load_dataset
import torch

from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE, ActivationsStore
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.data_writing_fns import save_feature_centric_vis


def compute_sae_dashboard(
    model_path: str,
    sae_path: str,
    dataset_path: str,
    features_path: str,
    output_dir: str,
    sae_id: str = None,
    column_name: str = "text",
    minibatch_size_features: int = 256,
    minibatch_size_tokens: int = 64,
    n_samples: int = 5000
):
    torch.set_num_threads(50)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(">>> Loading SAE and LLM")
    if sae_id is None:
        sae = SAE.load_from_pretrained(sae_path, device=device)
    else:
        sae, _, _ = SAE.from_pretrained(sae_path, sae_id, device=device)

    model = HookedTransformer.from_pretrained_no_processing(
        model_path,
        dtype=torch.bfloat16,
        device=device,
    )
    # make pad token different from `bos` and `eos` to prevent removing `bos`/`eos` token during slicing
    if model.tokenizer.pad_token_id == model.tokenizer.eos_token_id:
        model.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    print(">>> Loading dataset")
    dataset = load_dataset(dataset_path, streaming=False, split="train")
    if column_name == "tokens":
        token_dataset = dataset
    else:
        print(">>> Tokenize dataset")
        token_dataset = tokenize_and_concatenate(
            dataset=dataset,
            tokenizer=model.tokenizer,
            streaming=False,
            max_length=sae.cfg.context_size,
            column_name=column_name,
            add_bos_token=sae.cfg.prepend_bos,
            num_proc=4
        )
    
    # we could've initialized from SAE, but this implies using the dataset specified in its configuration
    activation_store = ActivationsStore(
        model=model,
        dataset=token_dataset,
        d_in=sae.cfg.d_in,
        hook_name=sae.cfg.hook_name,
        hook_layer=sae.cfg.hook_layer,
        hook_head_index=sae.cfg.hook_head_index,
        context_size=sae.cfg.context_size,
        prepend_bos=sae.cfg.prepend_bos,
        streaming=True,
        store_batch_size_prompts=2,
        train_batch_size_tokens=16,
        n_batches_in_buffer=1,
        total_training_tokens=1_000_000,
        normalize_activations=sae.cfg.normalize_activations,
        dataset_trust_remote_code=sae.cfg.dataset_trust_remote_code,
        dtype=sae.cfg.dtype,
        device=torch.device(device),
        seqpos_slice=sae.cfg.seqpos_slice
    )
    activation_store.shuffle_input_dataset(seed=42)

    with open(features_path, "r") as file:
        features = json.load(file)
    print(">>> Processing {} features.".format(len(features)))

    dashboard_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(features_path))[0]}_{len(features)}.html')
    os.makedirs(output_dir, exist_ok=True)

    feature_vis_config = SaeVisConfig(
        hook_point=sae.cfg.hook_name,
        features=features,
        minibatch_size_features=minibatch_size_features,
        minibatch_size_tokens=minibatch_size_tokens,
        verbose=True,
        device=device
    )

    visualization_data = SaeVisRunner(
        feature_vis_config
    ).run(
        encoder=sae,
        model=model,
        tokens=token_dataset[:n_samples]["tokens"]
    )

    save_feature_centric_vis(sae_vis_data=visualization_data, filename=dashboard_path)


if __name__ == "__main__":
    fire.Fire(compute_sae_dashboard)
