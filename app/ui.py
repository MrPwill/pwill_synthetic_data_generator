import gradio as gr
import json
import logging
import pandas as pd
from app.state import app_state
from core.schemas import GenerationRequest, EvaluationCriteria
from llms.model_registry import GeneratorModels, JudgeModels, get_model_name

logger = logging.getLogger(__name__)

def generate_data(
    prompt, 
    data_type, 
    num_samples, 
    model_name, 
    judge_model,
    correctness, 
    schema_compliance, 
    diversity
):
    try:
        req = GenerationRequest(
            prompt=prompt,
            data_type=data_type,
            num_samples=int(num_samples),
            model_name=model_name
        )
        
        criteria = EvaluationCriteria(
            correctness=correctness,
            schema_compliance=schema_compliance,
            diversity=diversity
        )
        
        # Update app refiner judge model dynamically
        app_state.refiner.judge.model_name = judge_model
        
        result = app_state.refiner.generate_verified(req, criteria=criteria)
        
        # Calculate avg score from samples if possible (Refiner doesn't store per-sample feedback in result well right now, 
        # but for simplicity we assume 100 if passed or we should drag feedback out.
        # Improv: Refiner could attach feedback to result.
        # For now, we just save.
        
        app_state.repo.save_result(req, result, avg_score=0.0) # Placeholder score
        
        # Return text representation
        output_text = ""
        for i, s in enumerate(result.samples):
            content = s.content
            if isinstance(content, (dict, list)):
                content = json.dumps(content, indent=2)
            output_text += f"Sample {i+1}:\n{content}\n" + "-"*40 + "\n"
            
        return output_text, "✅ Generation Complete"
    except Exception as e:
        logger.error(f"UI Error: {e}")
        return str(e), "❌ Error Occurred"

def get_history_df():
    items = app_state.repo.get_history()
    data = []
    for item in items:
        data.append({
            "ID": item.id,
            "Prompt": item.prompt,
            "Type": item.data_type,
            "Model": item.model_name,
            "Date": item.created_at
        })
    return pd.DataFrame(data)

def create_ui():
    with gr.Blocks(title="Synthetic Data Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧬 Synthetic Data Generator")
        
        with gr.Tabs():
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_input = gr.TextArea(label="Prompt", placeholder="Describe the data you need...", lines=5)
                        
                        with gr.Row():
                            data_type = gr.Dropdown(
                                choices=["text", "json", "tabular", "code", "reasoning"], 
                                value="text", 
                                label="Data Type"
                            )
                            num_samples = gr.Number(value=1, label="Count", precision=0, minimum=1, maximum=50)

                        with gr.Row():
                            gen_model = gr.Dropdown(
                                choices=[m.value for m in GeneratorModels],
                                value=GeneratorModels.MISTRAL_MEDIUM.value,
                                label="Generator Model"
                            )
                            judge_model = gr.Dropdown(
                                choices=[m.value for m in JudgeModels],
                                value=JudgeModels.CLAUDE_3_5_SONNET.value,
                                label="Judge Model"
                            )
                        
                        with gr.Accordion("Evaluation Criteria", open=False):
                            check_correctness = gr.Checkbox(label="Check Correctness", value=True)
                            check_schema = gr.Checkbox(label="Check Schema Compliance", value=True)
                            check_diversity = gr.Checkbox(label="Check Diversity", value=False)
                            
                        btn_gen = gr.Button("🚀 Generate", variant="primary")
                        status_box = gr.Textbox(label="Status", interactive=False)
                        
                    with gr.Column(scale=1):
                        output_display = gr.Code(label="Generated Output", language="json")
                
                btn_gen.click(
                    generate_data,
                    inputs=[prompt_input, data_type, num_samples, gen_model, judge_model, check_correctness, check_schema, check_diversity],
                    outputs=[output_display, status_box]
                )

            with gr.Tab("History"):
                gr.Markdown("## Past Generations")
                refresh_btn = gr.Button("🔄 Refresh")
                history_table = gr.Dataframe(interactive=False)
                
                refresh_btn.click(get_history_df, outputs=history_table)
                # Auto load on start
                demo.load(get_history_df, outputs=history_table)
                
            with gr.Tab("Export"):
                gr.Markdown("## Export Data")
                gr.Markdown("Select a generation from History to export (Not fully linked in this MVP, demonstrating button logic).")
                
                # ideally we select from history, but here we just show buttons 
                # that would act on the 'last result' if we stored it in session state more broadly
                # For this step, I'll just put placeholder text explaining usage.
                
                gr.Info("Export functionality is available via API. UI implementation requires complex state management for selection.")


    return demo
