# 🧠 Synthetic Data Generation Platform

A production-grade, extensible platform for generating high-quality synthetic data using Multi-Agent LLM Orchestration (Generator, Judge, Refiner). Built with Python (uv), Gradio, and OpenRouter.

![Status](https://img.shields.io/badge/Status-POC-green) ![Python](https://img.shields.io/badge/Python-3.14-blue)

## 🏗️ Architecture

The system uses a **Generator-Judge-Refiner** loop to ensure data quality:

1.  **Generator Agent**: Creates initial samples based on user prompts and schemas.
2.  **Judge Agent** (LLM-as-a-Judge): Evaluates samples against correctness, schema compliance, and realism criteria.
3.  **Refiner Agent**: Loops back feedback to the Generator if quality thresholds aren't met.
4.  **Memory Layer**: Persists all prompts, generations, and scores in structured SQLite.

## 🚀 Features

-   **Multi-Model Support**: Swap between Nemotron, Llama 3, Claude 3.5, Gemini, etc. via OpenRouter.
-   **Structured Output**: Generates valid JSON, CSV/Tabular, or Text.
-   **Quality Assurance**: Automated scoring and regeneration loop.
-   **History & Persistence**: All runs are saved to a local database.
-   **Export**: Export datasets to JSON, CSV, JSONL.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/pwill_synthetic_data_generator.git
    cd pwill_synthetic_data_generator
    ```

2.  **Install dependencies with `uv`**:
    ```bash
    uv sync
    ```

3.  **Configure Environment**:
    Create a `.env` file in the root:
    ```env
    OPENROUTER_API_KEY=sk-or-your-key-here
    ```

## 🎮 Usage

1.  **Run the UI**:
    ```bash
    uv run python app/main.py
    ```
2.  Open your browser at `http://127.0.0.1:7860`.
3.  Select a **Generator Model** (e.g., Mistral Medium) and **Judge Model** (e.g., Claude 3.5 Sonnet).
4.  Enter a prompt and click **Generate**.
5.  View results and history in the tabs.

## 📂 Project Structure

-   `app/`: Gradio UI and State management.
-   `core/`: Agents (Generator, Judge, Refiner).
-   `llms/`: OpenRouter client and model registry.
-   `memory/`: Database logic (SQLAlchemy).
-   `exports/`: Data export utilities.

## 🔮 Roadmap

- [ ] Advanced RAG integration for seeding context.
- [ ] Fine-tuning dataset preparation.
- [ ] Distributed generation for massive datasets.

---
*Princewill.*
