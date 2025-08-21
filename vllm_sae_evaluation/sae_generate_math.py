import copy
import math
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from importlib.util import find_spec

import ray
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.lora.request import LoRARequest
from transformers import AutoConfig
from more_itertools import distribute
try:
    from sae_lens import SAE
    from sae_utils import add_hooks, get_intervention_hook, get_clamp_hook
except ImportError:
    SAE = None
    add_hooks = None
    get_intervention_hook = None
    get_clamp_hook = None

def undistribute(iterable):
    """
    Undoes https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distribute .

    Re-interleaves results that have been split using more_itertools.distribute:
        >>> group_1, group_2 = distribute(2, [1, 2, 3, 4, 5, 6])
        >>> list(group_1)
        [1, 3, 5]
        >>> list(group_2)
        [2, 4, 6]
        >>> undistribute([group_1, group_2])
        [1, 2, 3, 4, 5, 6]

    Handles non-uniform component lengths:

        >>> children = distribute(3, [1, 2, 3, 4, 5, 6, 7])
        >>> [list(c) for c in children]
        [[1, 4, 7], [2, 5], [3, 6]]
        >>> undistribute(children)
        [1, 2, 3, 4, 5, 6, 7]

    Also handles when some iterables are empty:

        >>> children = distribute(5, [1, 2, 3])
        >>> [list(c) for c in children]
        [[1], [2], [3], [], []]
        >>> undistribute(children)
        [1, 2, 3]

    """

    return [
        x
        for x in itertools.chain.from_iterable(
            itertools.zip_longest(*[list(x) for x in iterable])
        )
        if x is not None
    ]


class VLLMSAEGenerator:
    """VLLM模型生成器，支持SAE干预"""
    
    def __init__(self):
        self.model = None
        self.sae = None
        self.intervention_config = None
        self.model_args = None
        self.batch_size = None
        self.data_parallel_size = 1
        self.tensor_parallel_size = 1
        self._max_length = None
        self._config = None
        self.lora_request = None
    
    def load_model(self, 
                   pretrained: str,
                   max_model_len: Optional[int] = None,
                   max_length: Optional[int] = None,
                   tensor_parallel_size: int = 1,
                   data_parallel_size: int = 1,
                   gpu_memory_utilization: float = 0.9,
                   revision: Optional[str] = None,
                   dtype: str = "auto",
                   tokenizer: Optional[str] = None,
                   tokenizer_mode: str = "auto",
                   tokenizer_revision: Optional[str] = None,
                   trust_remote_code: bool = False,
                   swap_space: int = 4,
                   quantization: Optional[str] = None,
                   seed: int = 0,
                   batch_size: str = "auto",
                   **kwargs) -> None:
        """加载VLLM模型和配置
        
        Args:
            pretrained: 预训练模型路径或名称
            max_model_len: 模型最大长度
            max_length: 最大长度（备用）
            tensor_parallel_size: 张量并行大小
            data_parallel_size: 数据并行大小
            gpu_memory_utilization: GPU内存利用率
            revision: 模型版本
            dtype: 数据类型
            tokenizer: 分词器路径
            tokenizer_mode: 分词器模式
            tokenizer_revision: 分词器版本
            trust_remote_code: 是否信任远程代码
            swap_space: 交换空间大小
            quantization: 量化方法
            seed: 随机种子
            batch_size: 批次大小
            **kwargs: 其他参数，包括SAE相关参数
        """
        # 处理SAE相关参数
        self._load_sae(**kwargs)
        
        # 设置模型参数
        self._max_length = max_model_len if max_model_len is not None else max_length
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.data_parallel_size = int(data_parallel_size)
        
        self.model_args = {
            "model": pretrained,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "revision": revision,
            "dtype": dtype,
            "tokenizer": tokenizer,
            "tokenizer_mode": tokenizer_mode,
            "tokenizer_revision": tokenizer_revision,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": int(tensor_parallel_size),
            "max_model_len": int(self._max_length) if self._max_length else None,
            "swap_space": int(swap_space),
            "enforce_eager": True if self.sae is not None else False,  # SAE干预需要eager模式
            "quantization": quantization,
            "seed": int(seed),
        }
        
        # 移除SAE参数后更新模型参数
        filtered_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("sae_")}
        self.model_args.update(filtered_kwargs)
        
        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else int(batch_size)
        )
        
        # 加载模型
        if self.data_parallel_size <= 1:
            self.model = LLM(**self.model_args)
        else:
            print("Warning: You might experience occasional issues with model weight downloading when data_parallel is in use.")
            self.model_args["distributed_executor_backend"] = "ray"
            self.batch_size = "auto"
            print("Manual batching is not compatible with data parallelism.")
        
        # 加载配置
        self._config = AutoConfig.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code, revision=revision
        )
        
        print(f"Model loaded: {pretrained}")
        if self.sae is not None:
            print(f"SAE loaded with intervention config: {self.intervention_config}")
    
    def _load_sae(self, **kwargs) -> None:
        """加载SAE模型和干预配置"""
        self.sae = None
        sae_kwargs = {k.replace("sae_", ""): kwargs.pop(k) for k in list(kwargs.keys()) if k.startswith("sae_")}
        
        if sae_kwargs:
            if not find_spec("sae_lens"):
                raise ModuleNotFoundError(
                    "attempted to use `sae` to perform intervention, but package `sae_lens` "
                    "is not installed. Please install sae_lens via `pip install sae-lens`"
                )
            
            # 加载SAE模型
            if 'path' in sae_kwargs:
                self.sae = SAE.load_from_pretrained(path=sae_kwargs.pop("path"))
            else:
                self.sae, _, _ = SAE.from_pretrained(
                    release=sae_kwargs.pop("release"), 
                    sae_id=sae_kwargs.pop("id")
                )
            
            # 加载干预配置
            if "intervention_path" in sae_kwargs:
                print("CLAMP STRATEGY: Reading from intervention config")
                with open(sae_kwargs.pop("intervention_path"), "rb") as f:
                    self.intervention_config = torch.load(f, weights_only=True)
            else:
                print("INTERVENTION STRATEGY: Reading from model_args (only single feature supported)")
                self.intervention_config = sae_kwargs
    
    def setup_sae_hooks(self) -> List:
        """设置SAE干预钩子
        
        Returns:
            List: SAE钩子列表
        """
        sae_hooks = []
        
        if self.sae is not None:
            lm_model = self.model.llm_engine.model_executor.driver_worker.model_runner.model
            
            if isinstance(self.intervention_config[list(self.intervention_config.keys())[0]], dict):
                # 按特征干预
                for feature_idx, intervene_cfg in self.intervention_config.items():
                    direction = self.sae.W_dec[feature_idx].clone()
                    max_activation = intervene_cfg["max_activation"]
                    strength = intervene_cfg["strength"]
                    
                    sae_hooks.append(
                        (
                            lm_model.model.layers[self.sae.cfg.hook_layer],
                            get_clamp_hook(direction, max_activation, strength)
                        )
                    )
            else:
                # 通用干预
                sae_hooks = [
                    (
                        lm_model.model.layers[self.sae.cfg.hook_layer],
                        get_intervention_hook(copy.deepcopy(self.sae), **self.intervention_config)
                    )
                ]
            
            print(f">>> SAE hooks: {sae_hooks}")
        
        return sae_hooks
    
    def generate(self, 
                 requests: List[List[int]],
                 sampling_params: SamplingParams,
                 cutoff_token: Optional[int] = None) -> List:
        """生成文本
        
        Args:
            requests: 输入token序列列表
            sampling_params: 采样参数
            cutoff_token: 截断token
            
        Returns:
            List: 生成结果
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        # 设置SAE钩子
        sae_hooks = self.setup_sae_hooks()
        
        # 数据并行处理
        if self.data_parallel_size > 1:
            return self._generate_data_parallel(requests, sampling_params, sae_hooks)
        
        # 单机生成
        with add_hooks([], sae_hooks):
            outputs = self.model.generate(
                prompt_token_ids=requests,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
                lora_request=self.lora_request,
            )
        
        # 处理输出统计
        self._log_generation_stats(outputs, cutoff_token)
        
        return outputs
    
    def _generate_data_parallel(self, 
                               requests: List[List[int]], 
                               sampling_params: SamplingParams,
                               sae_hooks: List) -> List:
        """数据并行生成"""
        @ray.remote
        def run_inference_one_model(
            model_args: dict,
            sampling_params: SamplingParams,
            requests: List[List[int]],
            lora_request: LoRARequest,
        ):
            llm = LLM(**model_args)
            return llm.generate(
                prompt_token_ids=requests,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
        
        # 分发请求到所有工作进程
        requests = [list(x) for x in distribute(self.data_parallel_size, requests)]
        inputs = (
            (self.model_args, sampling_params, req, self.lora_request)
            for req in requests
        )
        object_refs = [run_inference_one_model.remote(*x) for x in inputs]
        results = ray.get(object_refs)
        
        # 关闭ray防止挂起
        ray.shutdown()
        
        # 展平结果
        return undistribute(results)
    
    def _log_generation_stats(self, outputs: List, cutoff_token: Optional[int] = None) -> None:
        """记录生成统计信息"""
        completions = [out.token_ids for completions in outputs for out in completions.outputs]
        
        if cutoff_token is not None:
            completions_cutoff_lengths = [
                len(out[:out.index(cutoff_token)]) if cutoff_token in out else len(out)
                for out in completions
            ]
            completions_cutoff_mean_length = math.ceil(np.mean(completions_cutoff_lengths))
            print("#" * 30 + f" Mean Completion Length (CUTOFF): {completions_cutoff_mean_length} " + "#" * 30)
        
        completions_lengths = list(map(len, completions))
        completions_mean_length = math.ceil(np.mean(completions_lengths))
        print("#" * 30 + f" Mean Completion Length: {completions_mean_length} " + "#" * 30)


# 便捷函数
def create_generator(**kwargs) -> VLLMSAEGenerator:
    """创建VLLM SAE生成器"""
    generator = VLLMSAEGenerator()
    if kwargs:
        generator.load_model(**kwargs)
    return generator


# 示例用法
if __name__ == "__main__":
    # 创建生成器
    generator = VLLMSAEGenerator()
    
    # 加载模型（示例参数）
    generator.load_model(
        pretrained="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        max_model_len=50000,
        tensor_parallel_size=1,
        data_parallel_size=1,
        # SAE参数示例 - 启用SAE干预
        sae_release="andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts",
        sae_id="blocks.19.hook_resid_post",
        sae_feature_idx=1160,
        sae_strength=1.0,
        sae_max_activation=5.395
    )
    
    # 准备输入
    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    prompts = ["Who are you?"]
    prompt_token_ids = [tokenizer.encode(prompt) for prompt in prompts]
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=100
    )
    
    # 生成文本
    outputs = generator.generate(prompt_token_ids, sampling_params)
    
    # 打印结果
    for output in outputs:
        for completion in output.outputs:
            print(f"Generated: {tokenizer.decode(completion.token_ids)}")