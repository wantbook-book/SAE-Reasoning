"""Data processing utilities for SAE training.

This module provides alternative implementations for data processing functions
that may not be available in certain packages.
"""

from datasets import Dataset

import math
from multiprocessing import cpu_count
from typing import TypeVar, Union
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase


T = TypeVar("T", bound=Union[Dataset, DatasetDict])

def download_hf_dataset(
    repo_id: str = "openbmb/RLPR-Train-Dataset",
    cache_dir: str = None,
    revision: str = "main",
    token: str = None,
    local_files_only: bool = False,
) -> str:
    """Download RLPR dataset from Hugging Face repository.
    
    Args:
        repo_id: The repository ID on Hugging Face Hub. Defaults to "openbmb/RLPR-Train-Dataset".
        cache_dir: Directory to cache the downloaded files. If None, uses default HF cache.
        revision: The specific revision (branch, tag, or commit) to download. Defaults to "main".
        token: Hugging Face authentication token for private repositories.
        local_files_only: If True, only use local cached files without downloading.
        
    Returns:
        str: Path to the downloaded dataset directory.
        
    Raises:
        ValueError: If the repository is not found or access is denied.
        ConnectionError: If there are network issues during download.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download datasets. "
            "Install it with: pip install huggingface_hub"
        )
    
    try:
        # Download the entire repository
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
            repo_type="dataset"
        )
        
        print(f"Dataset downloaded successfully to: {local_dir}")
        return local_dir
        
    except Exception as e:
        if "Repository not found" in str(e):
            raise ValueError(f"Repository '{repo_id}' not found. Please check the repository ID.")
        elif "Access denied" in str(e) or "401" in str(e):
            raise ValueError(
                f"Access denied to repository '{repo_id}'. "
                "You may need to provide a valid token or request access."
            )
        elif "Connection" in str(e) or "Network" in str(e):
            raise ConnectionError(f"Network error while downloading: {e}")
        else:
            raise e

def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> list[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names
    
def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = cpu_count() // 2,
    text_key: str = "text",
    max_seq_len: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
) -> T:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_seq_len` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_seq_len: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.

    Returns:
        The chunked and tokenized dataset.
    """

    def _tokenize_fn(x: dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        sep = tokenizer.eos_token or "<|endoftext|>"
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output.input_ids[0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size]
                for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        return output

    data = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    return data.with_format(format, columns=["input_ids"])




if __name__ == '__main__':
    # repo_id = "openbmb/RLPR-Train-Dataset"
    # cache_dir = "/pubshare/fwk/code/sae/SAE-Reasoning2/sft/dataset"
    # download_hf_dataset(
    #     repo_id=repo_id,
    #     cache_dir=cache_dir
    # )
    pass
    
