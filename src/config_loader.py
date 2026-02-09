# config_loader.py
import os, json, argparse
from pathlib import Path
from typing import Any
from dotenv import dotenv_values
import yaml

from config_schema import AppConfig, LLMConfig, GenPlanStage, BuildKGStage, UseKGStage, DesignPaths, DynamicPromptSettings

def load_app_config(
    designs_yaml: str = "designs.yaml",
    overrides_json: str = "",
    env_prefix: str = "AF_"  # AssertionForge
) -> AppConfig:
    if not Path(designs_yaml).exists():
        raise FileNotFoundError(f"Config file {designs_yaml} not found")

    data = yaml.safe_load(Path(designs_yaml).read_text(encoding="utf-8")) or {}
    # print(data.get("build_KG", {}))
    # Build AppConfig from YAML structure
    cfg = AppConfig(
        task=data.get("task", "gen_plan"),
        design_name=data.get("design_name", "uart"),
        llm=LLMConfig(**data.get("llm", {})),
        gen_plan=GenPlanStage(**data.get("gen_plan", {})),
        build_KG=BuildKGStage(**data.get("build_KG", {})),
        use_KG=UseKGStage(**data.get("use_KG", {})),
        designs={name: DesignPaths(**entry) for name, entry in data.get("designs", {}).items()},
    )

    if isinstance(cfg.gen_plan.dynamic_prompt_settings, dict):
        cfg.gen_plan.dynamic_prompt_settings = DynamicPromptSettings(**cfg.gen_plan.dynamic_prompt_settings)
        
    # 2) Apply ENV overrides (flat)
    for k, v in os.environ.items():
        if k.startswith(env_prefix):
            key = k[len(env_prefix):]
            if key == "task": cfg.task = v
            elif key == "design_name": cfg.design_name = v
            elif key == "llm_model": cfg.llm.model = v
            elif key == "use_KG": cfg.gen_plan.use_KG = v.lower() == "true"
    
    # after building cfg
    dp = cfg.designs.get(cfg.design_name)
    if dp and cfg.build_KG.env_source_path and Path(cfg.build_KG.env_source_path).exists():
        env_vars = dotenv_values(cfg.build_KG.env_source_path)
        if "GRAPHRAG_API_KEY" in env_vars:
            cfg.llm.args["api_key"] = env_vars["GRAPHRAG_API_KEY"]

    return cfg

def build_FLAGS_from_cli() -> Any:
    def build_FLAGS_from_cli() -> Any:
    p = argparse.ArgumentParser()

    # ---------------- core task ----------------
    p.add_argument("--task", choices=["gen_plan", "build_KG", "use_KG"], required=True)
    p.add_argument("--design_name", required=True)
    p.add_argument("--designs_yaml", default="designs.yaml")
    p.add_argument("--valid_signals", nargs="+", help="List of architectural signals")

    # ---------------- KG ----------------
    p.add_argument("--KG_root")
    p.add_argument("--graphrag_method")
    p.add_argument("--query")

    # ---------------- NEW: pipeline control ----------------
    p.add_argument("--continue", dest="continue_run", action="store_true",
                   help="Resume pipeline from last incomplete step")

    p.add_argument("--restart_step", type=str, default=None,
                   help="Restart pipeline from this step (e.g. nl_plans, svas)")

    p.add_argument("--pipeline_state_dir", type=str, default=".pipeline_state",
                   help="Persistent pipeline cache/state directory")

    p.add_argument("--run_dir", type=str, default=None,
                   help="Run output directory (logs, artifacts only)")

    args, _ = p.parse_known_args()

    cfg = load_app_config(designs_yaml=args.designs_yaml)

    cfg.task = args.task
    cfg.design_name = args.design_name

    if args.valid_signals:
        cfg.gen_plan.valid_signals = args.valid_signals

    # pipeline flags
    cfg.gen_plan.continue_run = args.continue_run
    cfg.gen_plan.restart_step = args.restart_step
    cfg.gen_plan.pipeline_state_dir = args.pipeline_state_dir
    cfg.gen_plan.run_dir = args.run_dir

    if args.task == "use_KG":
        if not args.query:
            p.error("--query is required when --task use_KG")
        cfg.use_KG.query = args.query
        if args.KG_root:
            cfg.use_KG.KG_root = args.KG_root
        if args.graphrag_method:
            cfg.use_KG.graphrag_method = args.graphrag_method

    return cfg.to_FLAGS()
