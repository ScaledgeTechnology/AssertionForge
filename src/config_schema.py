# config_schema.py
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace
from typing import List, Dict, Optional, Any, Union

# -------------------
# LLM config
# -------------------
@dataclass
class LLMConfig:
    engine_type: str = "<llm_engine_type>"
    model: str = "x-ai/grok-4-fast:free"
    args: Dict[str, Any] = field(
        default_factory=lambda: {"base_url": "https://openrouter.ai/api/v1", "api_key": ""}
    )
    max_tokens_per_prompt: int = 8000

# -------------------
# Dynamic Prompt sub-blocks
# -------------------
@dataclass
class DynamicPromptRAG:
    enabled: bool = True
    baseline_full_spec_RTL: bool = False
    chunk_sizes: List[int] = field(default_factory=lambda: [50, 100, 200, 800, 3200])
    overlap_ratios: List[float] = field(default_factory=lambda: [0.2, 0.4])
    k: int = 20
    enable_rtl: bool = True

@dataclass
class DynamicPromptPathBased:
    enabled: bool = True
    max_depth: int = 5
    representation_style: str = "standard"  # "concise", "standard", "detailed", "verification_focused"

@dataclass
class DynamicPromptMotif:
    enabled: bool = False
    patterns: Dict[str, bool] = field(default_factory=lambda: {"handshake": True, "pipeline": True, "star": True})
    min_star_degree: int = 3
    max_motifs_per_type: int = 2

@dataclass
class DynamicPromptCommunity:
    enabled: bool = False
    max_communities: int = 20
    min_community_size: int = 3

@dataclass
class DynamicPromptLocalExpansion:
    enabled: bool = False
    max_depth: int = 2
    max_subgraph_size: int = 20
    min_subgraph_size: int = 5

@dataclass
class DynamicPromptGuidedRandomWalk:
    enabled: bool = False
    num_walks: int = 70
    walk_budget: int = 100
    teleport_probability: float = 0.1
    local_importance_weight: float = 0.3
    direction_weight: float = 0.5
    discovery_weight: float = 0.2
    max_targets_per_walk: int = 10
    max_contexts_per_signal: int = 50

@dataclass
class DynamicPromptPruning:
    enabled: bool = True
    use_llm_pruning: bool = True
    max_contexts_per_type: int = 50
    max_total_contexts: int = 100
    min_similarity_threshold: float = 0.3

# -------------------
# Aggregate DynamicPromptSettings
# -------------------
@dataclass
class DynamicPromptSettings:
    rag: DynamicPromptRAG = field(default_factory=DynamicPromptRAG)
    path_based: DynamicPromptPathBased = field(default_factory=DynamicPromptPathBased)
    motif: DynamicPromptMotif = field(default_factory=DynamicPromptMotif)
    community: DynamicPromptCommunity = field(default_factory=DynamicPromptCommunity)
    local_expansion: DynamicPromptLocalExpansion = field(default_factory=DynamicPromptLocalExpansion)
    guided_random_walk: DynamicPromptGuidedRandomWalk = field(default_factory=DynamicPromptGuidedRandomWalk)
    pruning: DynamicPromptPruning = field(default_factory=DynamicPromptPruning)
    # global retrieval settings
    kg_k: int = 3
    traversal_max_depth: int = 1
    retrieve_edge: bool = True

# -------------------
# Stages
# -------------------
@dataclass
class GenPlanStage:
    step: float = 0
    run_dir: str | None = None
    subtask: str = "actual_gen"
    DEBUG: bool = False
    prompt_builder: str = "dynamic"
    enable_context_enhancement: bool = False
    max_num_signals_process: float = float("inf")
    max_prompts_per_signal: int = 3
    doc_retriever: bool = True
    chunk_size: int = 100
    overlap: int = 20
    doc_k: int = 3
    kg_retriever: bool = True
    use_KG: bool = True
    refine_with_rtl: bool = True
    sva_file_path: str = None
    sva_out_dir: str = None
    gen_plan_sva_using_valid_signals: bool = True
    valid_signals: Optional[List[str]] = None
    include_internal_signals: bool = False
    hierarichal_signal_depth: int = 3
    prune_signals: bool = True
    generate_SVAs: bool = False
    generate_nlp: bool = False
    context_workers: int = 4
    signal_workers: int = 4
    load_dir: Optional[str] = None
    dynamic_prompt_settings: DynamicPromptSettings = field(default_factory=DynamicPromptSettings)

@dataclass
class BuildKGStage:
    env_source_path: str = "/content/.env"
    settings_source_path: str = "/content/settings.yaml"
    entity_extraction_prompt_source_path: str = "/content/AssertionForge/entity_extraction.txt"
    graphrag_local_dir: str = "/<path>/<to>/graphrag"

@dataclass
class UseKGStage:
    KG_root: str = "/<path>/<to>/data/apb/graph_rag/output/20240813-163015/artifacts"
    graphrag_method: str = "local"
    query: str = "What does PREADY mean?"

# -------------------
# Per-design paths
# -------------------
@dataclass
class DesignPaths:
    file_path: Optional[Union[str, List[str]]] = None
    design_dir: Optional[str] = None
    KG_path: Optional[str] = None

# -------------------
# Root config
# -------------------
@dataclass
class AppConfig:
    task: str = "gen_plan"
    design_name: str = "uart"
    llm: LLMConfig = field(default_factory=LLMConfig)
    gen_plan: GenPlanStage = field(default_factory=GenPlanStage)
    build_KG: BuildKGStage = field(default_factory=BuildKGStage)
    use_KG: UseKGStage = field(default_factory=UseKGStage)
    designs: Dict[str, DesignPaths] = field(default_factory=dict)

    def to_FLAGS(self) -> SimpleNamespace:
        d = {
            "task": self.task,
            "design_name": self.design_name,
            # LLM
            "llm_engine_type": self.llm.engine_type,
            "llm_model": self.llm.model,
            "llm_args": self.llm.args,
            "max_tokens_per_prompt": self.llm.max_tokens_per_prompt,
            # GEN PLAN
            "subtask": self.gen_plan.subtask,
            "DEBUG": self.gen_plan.DEBUG,
            "prompt_builder": self.gen_plan.prompt_builder,
            "enable_context_enhancement": self.gen_plan.enable_context_enhancement,
            "max_num_signals_process": self.gen_plan.max_num_signals_process,
            "max_prompts_per_signal": self.gen_plan.max_prompts_per_signal,
            "doc_retriever": self.gen_plan.doc_retriever,
            "chunk_size": self.gen_plan.chunk_size,
            "overlap": self.gen_plan.overlap,
            "doc_k": self.gen_plan.doc_k,
            "kg_retriever": self.gen_plan.kg_retriever,
            "use_KG": self.gen_plan.use_KG,
            "refine_with_rtl": self.gen_plan.refine_with_rtl,
            "sva_file_path": self.gen_plan.sva_file_path,
            "sva_out_dir": self.gen_plan.sva_out_dir,
            "gen_plan_sva_using_valid_signals": self.gen_plan.gen_plan_sva_using_valid_signals,
            "valid_signals": self.gen_plan.valid_signals,
            "include_internal_signals": self.gen_plan.include_internal_signals,
            "hierarichal_signal_depth": self.gen_plan.hierarichal_signal_depth,
            "prune_signals": self.gen_plan.prune_signals,
            "generate_SVAs": self.gen_plan.generate_SVAs,
            'generate_nlp' : self.gen_plan.generate_nlp,
            "context_workers" : self.gen_plan.context_workers
            "signal_workers" : self.gen_plan.signal_workers
            "load_dir": self.gen_plan.load_dir,
            # dynamic prompt internals → export as dict
            "dynamic_prompt_settings": asdict(self.gen_plan.dynamic_prompt_settings),
            "kg_k": self.gen_plan.dynamic_prompt_settings.kg_k,
            "traversal_max_depth": self.gen_plan.dynamic_prompt_settings.traversal_max_depth,
            "retrieve_edge": self.gen_plan.dynamic_prompt_settings.retrieve_edge,
            # BUILD KG
            "env_source_path": self.build_KG.env_source_path,
            "settings_source_path": self.build_KG.settings_source_path,
            "entity_extraction_prompt_source_path": self.build_KG.entity_extraction_prompt_source_path,
            "graphrag_local_dir": self.build_KG.graphrag_local_dir,
            # USE KG
            "KG_root": self.use_KG.KG_root,
            "graphrag_method": self.use_KG.graphrag_method,
            "query": self.use_KG.query,
        }

        dp = self.designs.get(self.design_name, DesignPaths())
        d["file_path"] = dp.file_path
        d["design_dir"] = dp.design_dir
        d["KG_path"] = dp.KG_path

        return SimpleNamespace(**d)
