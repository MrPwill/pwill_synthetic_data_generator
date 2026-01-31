from enum import Enum
from typing import Dict, Optional

class LLMProvider(str, Enum):
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"
    NVIDIA = "nvidia"
    GOOGLE = "google"
    DASHSCOPE = "dashscope"
    OPENAI = "openai"

class GeneratorModels(str, Enum):
    # Free models via OpenRouter
    GPT_OSS_120B = "openai/gpt-oss-120b:free"
    GPT_OSS_20B = "openai/gpt-oss-20b:free"
    NEMOTRON_3_NANO = "nvidia/nemotron-3-nano-30b-a3b:free"
    GEMMA_3_27B = "google/gemma-3-27b-it:free"
    LLAMA_3_3_70B = "meta-llama/llama-3.3-70b-instruct:free"
    MISTRAL_SMALL = "mistralai/mistral-small-3.1-24b-instruct:free"
    DEEPSEEK_R1 = "deepseek/deepseek-r1-0528:free"
    DEEPSEEK_R1_CHIMERA = "tngtech/deepseek-r1t2-chimera:free"
    LFM_2_5 = "liquid/lfm-2.5-1.2b-thinking:free"
    QWEN_3_4B = "qwen/qwen3-4b:free"
    QWEN_3_NEXT = "qwen/qwen3-next-80b-a3b-instruct:free"
    NEMOTRON_NANO = "nvidia/nemotron-nano-12b-v2-vl:free"
    QWEN_3_CODER = "qwen/qwen3-coder:free"

class JudgeModels(str, Enum):
    # Strong models for judging/evaluation
    LLAMA_3_1_405B = "meta-llama/llama-3.1-405b-instruct:free"
    HERMES_3_405B = "nousresearch/hermes-3-llama-3.1-405b:free"

# Map friendly names to IDs if needed for UI
MODEL_FRIENDLY_NAMES = {
    GeneratorModels.GPT_OSS_120B: "GPT OSS 120B",
    GeneratorModels.GPT_OSS_20B: "GPT OSS 20B",
    GeneratorModels.NEMOTRON_3_NANO: "Nemotron 3 Nano",
    GeneratorModels.GEMMA_3_27B: "Gemma 3 27B",
    GeneratorModels.LLAMA_3_3_70B: "Llama 3.3 70B",
    GeneratorModels.MISTRAL_SMALL: "Mistral Small",
    GeneratorModels.DEEPSEEK_R1: "DeepSeek R1",
    GeneratorModels.DEEPSEEK_R1_CHIMERA: "DeepSeek R1 Chimera",
    GeneratorModels.LFM_2_5: "Liquid LFM 2.5",
    GeneratorModels.QWEN_3_4B: "Qwen 3 4B",
    GeneratorModels.QWEN_3_NEXT: "Qwen 3 Next",
    GeneratorModels.NEMOTRON_NANO: "Nemotron Nano",
    GeneratorModels.QWEN_3_CODER: "Qwen 3 Coder",
    JudgeModels.LLAMA_3_1_405B: "Llama 3.1 405B",
    JudgeModels.HERMES_3_405B: "Hermes 3 405B",
}

# Mapping models to providers
MODEL_PROVIDER_MAP: Dict[str, LLMProvider] = {
    model.value: LLMProvider.OPENROUTER for model in list(GeneratorModels) + list(JudgeModels)
}

def get_model_name(model_enum: Enum) -> str:
    return MODEL_FRIENDLY_NAMES.get(model_enum, model_enum.value)

def get_model_provider(model_id: str) -> Optional[LLMProvider]:
    """Get the provider for a given model ID."""
    # Check if it matches any known enum values
    return MODEL_PROVIDER_MAP.get(model_id)
