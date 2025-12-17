from enum import Enum

class GeneratorModels(str, Enum):
    NEMOTRON_4_340B = "nvidia/nemotron-4-340b-instruct"
    DEEPSEEK_R1 = "deepseek/deepseek-r1"
    MISTRAL_MEDIUM = "mistralai/mistral-medium"
    LLAMA_3_70B = "meta-llama/llama-3.1-70b-instruct"

class JudgeModels(str, Enum):
    CLAUDE_3_5_SONNET = "anthropic/claude-3.5-sonnet"
    CLAUDE_3_OPUS = "anthropic/claude-3-opus"
    GEMINI_PRO_1_5 = "google/gemini-pro-1.5"
    GPT_4_TURBO = "openai/gpt-4-turbo"

# Map friendly names to IDs if needed for UI
MODEL_FRIENDLY_NAMES = {
    GeneratorModels.NEMOTRON_4_340B: "Nvidia Nemotron 4 340B",
    GeneratorModels.DEEPSEEK_R1: "DeepSeek R1",
    GeneratorModels.MISTRAL_MEDIUM: "Mistral Medium",
    GeneratorModels.LLAMA_3_70B: "Llama 3.1 70B",
    JudgeModels.CLAUDE_3_5_SONNET: "Claude 3.5 Sonnet",
    JudgeModels.CLAUDE_3_OPUS: "Claude 3 Opus",
    JudgeModels.GEMINI_PRO_1_5: "Gemini 1.5 Pro",
    JudgeModels.GPT_4_TURBO: "GPT-4 Turbo",
}

def get_model_name(model_enum: Enum) -> str:
    return MODEL_FRIENDLY_NAMES.get(model_enum, model_enum.value)
