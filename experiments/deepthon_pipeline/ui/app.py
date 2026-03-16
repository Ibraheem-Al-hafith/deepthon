import gradio as gr
from pathlib import Path
from ..config.loader import load_config
from ..training.runner import ExperimentRunner
from ..cli.commands import cmd_train

from ..utils.logging import setup_logging, get_logger

# 1. Initialize the global deepthon logging system
# Ideally, we pass the default config path here
setup_logging("experiments/configs/config.yaml")

# 2. Get the logger for this specific module
logger = get_logger(__name__)

def get_config_info(config_path):
    """Helper to extract dataset and model keys for the UI."""
    try:
        cfg = load_config(config_path)
        assert isinstance(cfg.datasets, dict)
        assert isinstance(cfg.model, dict)
        datasets = list(cfg.datasets.keys())
        models = list(cfg.model.keys())
        # Return updates for all relevant dropdowns
        return (
            gr.update(choices=["all"] + datasets, value="all"), 
            gr.update(choices=["all"] + models, value="all"),
            gr.update(choices=datasets), 
            gr.update(choices=models),
            gr.update(choices=datasets), 
            gr.update(choices=models)
        )
    except Exception:
        return [gr.update(choices=[], value=None)] * 6

def resolve_ckpt(config_path, dataset, model):
    """Logic to find the model path based on config and selection."""
    try:
        cfg = load_config(config_path)
        exp_name = cfg.get("experiment", "run")
        # Standard path: runs/exp_name/dataset_model/model.npz
        ckpt_path = Path("runs") / exp_name / f"{dataset}_{model}" / "best_model.pkl"
        
        if not ckpt_path.exists():
            # Fallback check for .pkl if you use that extension
            ckpt_path = ckpt_path.with_suffix(".pkl")
            
        return ckpt_path if ckpt_path.exists() else None
    except:
        return None

def run_train_gui(config_path, datasets, models, resume):
    logger.info(f"Initiating GUI Training: Datasets={datasets}, Models={models}")
    
    ds_list = "all" if "all" in datasets else datasets
    md_list = "all" if "all" in models else models
    
    try:
        # This will now use the handlers configured in setup_logging!
        cmd_train(config_path, ds_list, md_list, resume=resume)
        
        logger.info("GUI Training Matrix completed successfully.")
        return f"✅ Training Finished. Check 'runs' folder."
    except Exception as e:
        logger.error(f"GUI Training Failed: {str(e)}", exc_info=True)
        return f"❌ Error: {str(e)}"

def run_test_smart_gui(config_path, dataset, model):
    """Automated test run using resolved paths."""
    ckpt = resolve_ckpt(config_path, dataset, model)
    if not ckpt:
        return "❌ Error: Checkpoint not found. Train the model first!", None

    cfg = load_config(config_path)
    runner = ExperimentRunner(cfg, dataset, model)
    runner.build_data()
    runner.build_model()
    runner.build_optimizer()
    runner.build_trainer()
    
    results = runner.test(checkpoint_path=ckpt)
    viz_path = runner.exp_dir / "test_visualization.png"
    return f"Results: {results}", str(viz_path) if viz_path.exists() else None

def download_model_gui(config_path, dataset, model):
    """Returns the file path for the Gradio File component."""
    ckpt = resolve_ckpt(config_path, dataset, model)
    if not ckpt:
        raise gr.Error(f"No checkpoint found for {dataset} - {model}")
    return str(ckpt)

def load_all_plots(config_path):
    try:
        cfg = load_config(config_path)
        exp_root = Path("runs") / cfg.get("experiment", "run")
        if not exp_root.exists(): return [], "No runs found."
        
        items = []
        for sub in exp_root.iterdir():
            if sub.is_dir():
                for img_name in ["loss_plot.png", "test_visualization.png"]:
                    img_p = sub / img_name
                    if img_p.exists():
                        items.append((str(img_p), f"{sub.name} {img_name.split('_')[0]}"))
        return items, f"Loaded {len(items)} plots."
    except:
        return [], "Error scanning plots."

def update_logs(config_path):
    try:
        cfg = load_config(config_path)
        log_file = Path("logs") / f"{cfg.get('experiment', 'run')}.log"
        if log_file.exists():
            # Return last 20 lines
            with open(log_file, "r") as f:
                return "\n".join(f.readlines())
        return "Log file not found yet. Start training to generate logs."
    except:
        return "Error loading logs."

# --- UI Layout ---
with gr.Blocks(title="Deepthon Pipeline Dashboard") as demo:
    gr.Markdown("# 🧠 Deepthon Pipeline GUI")
    
    with gr.Row():
        config_input = gr.Textbox(label="Config Path", value="experiments/configs/config.yaml")
        load_btn = gr.Button("🔍 Sync Config", variant="secondary")

    with gr.Tabs():
        # 🚀 Training
        with gr.TabItem("🚀 Training"):
            with gr.Row():
                ds_select = gr.Dropdown(label="Datasets", choices=[], multiselect=True)
                md_select = gr.Dropdown(label="Models", choices=[], multiselect=True)
            resume_toggle = gr.Checkbox(label="Resume Training")
            train_btn = gr.Button("Start Training Matrix", variant="primary")
            train_output = gr.Textbox(label="Status")

        # 📊 Evaluation (SMART)
        with gr.TabItem("📊 Evaluation"):
            with gr.Row():
                test_ds = gr.Dropdown(label="Dataset")
                test_md = gr.Dropdown(label="Model")
            gr.Markdown("_Note: Checkpoint is resolved automatically from the experiment directory._")
            test_btn = gr.Button("Run Evaluation", variant="primary")
            with gr.Row():
                test_metrics = gr.Label(label="Metrics")
                test_plot = gr.Image(label="Results Visualization")

        # 📥 Download
        with gr.TabItem("📥 Download Model"):
            with gr.Row():
                dl_ds = gr.Dropdown(label="Dataset")
                dl_md = gr.Dropdown(label="Model")
            dl_btn = gr.Button("Find Model File")
            file_out = gr.File(label="Ready for Download")

        # 🚩 Gallery
        with gr.TabItem("🚩 Plots"):
            plot_status = gr.Markdown("Sync config to view plots")
            result_gallery = gr.Gallery(label="Results", columns=3, height="auto")
    
        with gr.TabItem("⚙️ System Logs"):
            log_output = gr.Textbox(label="Live Log Stream", lines=15)
            refresh_logs = gr.Button("🔄 Refresh Logs")

    # Interactivity
    load_btn.click(
        get_config_info, 
        inputs=config_input, 
        outputs=[ds_select, md_select, test_ds, test_md, dl_ds, dl_md]
    ).then(load_all_plots, inputs=config_input, outputs=[result_gallery, plot_status])
    
    train_btn.click(run_train_gui, [config_input, ds_select, md_select, resume_toggle], train_output)
    
    test_btn.click(run_test_smart_gui, [config_input, test_ds, test_md], [test_metrics, test_plot])
    
    dl_btn.click(download_model_gui, [config_input, dl_ds, dl_md], file_out)

    refresh_logs.click(update_logs, inputs=config_input, outputs=log_output)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())