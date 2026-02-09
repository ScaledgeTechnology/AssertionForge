# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from sva_extraction import extract_svas_from_block
from doc_KG_processor import create_context_generators
from dynamic_prompt_builder import DynamicPromptBuilder
from load_result import load_svas, load_nl_plans, load_jasper_reports, load_pdf_stats
from rtl_parsing import refine_kg_from_rtl
from utils_gen_plan import (
    extract_proof_status,
    analyze_coverage_of_proven_svas,
    count_tokens_in_file,
    find_original_tcl_file,
    resolve_pdf_inputs
)
from rtl_kg import extract_rtl_knowledge
from design_context_summarizer import DesignContextSummarizer
import os, math
import subprocess
from config import FLAGS
from saver import (
    saver,
    _save_json,
    _load_json,
    _save_text,
    _load_text,
    _nl_plan_cache_path,
    load_cached_nl_plans,
    save_cached_nl_plans,
    _sva_cache_path,
    load_cached_svas,
    save_cached_svas,
    load_cached_signal_summary,
    save_cached_signal_summary
)
from utils import OurTimer
from utils_LLM import get_llm, llm_inference
from pyverilog.vparser.parser import parse
from pyverilog.vparser.ast import ModuleDef, Decl, Wire, Reg, InstanceList, Instance, Identifier, Ioport, Input, Output, Inout


import pyslang
import networkx as nx
from typing import Tuple, List, Dict, Optional, Set, Union
from PyPDF2 import PdfReader
from doxtract.processor import preprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import re
import glob
import pandas as pd
from tabulate import tabulate
from pathlib import Path
from tqdm import tqdm

print = saver.log_info
FLAGS.max_num_signals_process = float(FLAGS.max_num_signals_process)

def gen_plan():
    """
    Main function to generate test plans and SVAs from a design specification,
    and run JasperGold for verification.
    """
    timer = OurTimer()

    from pathlib import Path
    import json

    objdir = Path(saver.get_obj_dir())
    step = int(getattr(FLAGS, "step", 0))  # 0 = run all

    # Initialize context_summarizer as None
    context_summarizer = None

    if FLAGS.subtask == 'actual_gen':
        print("Starting the test plan generation process...")

        timer.start_timing()

        # -----------------------------
        # Step 1: Read PDF (or load cache)
        # -----------------------------
        print("Step 1: Reading the PDF file(s)...")
        if step in (0, 1):
            # Use repo helper if available (your earlier version had this)
            try:
                file_path = resolve_pdf_inputs(FLAGS.file_path)
            except Exception:
                file_path = FLAGS.file_path

            spec_text, pdf_stats = read_pdf(file_path)

            # Caching Step 1 output
            _save_text(objdir / "step1_spec_text.txt", spec_text)
            _save_json(objdir / "step1_pdf_stats.json", pdf_stats)
        else:
            spec_text = _load_text(objdir / "step1_spec_text.txt")
            pdf_stats = _load_json(objdir / "step1_pdf_stats.json")

        timer.time_and_clear("Read PDF")

        # -----------------------------
        # Step 2: KG / RTL knowledge (or load cache)
        # -----------------------------
        kg_nx, kg_json, rtl_knowledge = None, None, None

        # NOTE: Step 2 depends on config flags; caching must respect them.
        if step in (0, 2):
            if FLAGS.use_KG:
                print("Step 2: Loading and processing the Knowledge Graph...")
                kg_nx, kg_json = load_and_process_kg(FLAGS.KG_path)
                timer.time_and_clear("Load and process KG")

                if FLAGS.refine_with_rtl:
                    print("Step 2b: Refining Knowledge Graph with RTL information...")
                    kg_nx, rtl_knowledge = refine_kg_from_rtl(kg_nx)
                    print(
                        f"Refined Knowledge Graph now has {len(kg_nx.nodes())} nodes and {len(kg_nx.edges())} edges"
                    )
                    # Update the JSON representation after refinement
                    kg_json = convert_nx_to_json(kg_nx)
                    timer.time_and_clear("Refine KG with RTL")
                else:
                    # Your earlier repository version had this; keep it to avoid rtl_knowledge staying None
                    rtl_knowledge = extract_rtl_knowledge(
                        FLAGS.design_dir, output_dir=None, verbose=True
                    )
            else:
                # If no KG, you might still want RTL knowledge depending on downstream flags.
                # Keep rtl_knowledge as None unless you want to always compute it.
                rtl_knowledge = extract_rtl_knowledge(
                        FLAGS.design_dir, output_dir=None, verbose=True
                    )

            # Caching Step 2 outputs
            _save_json(objdir / "step2_kg.json", kg_json if kg_json is not None else {})
            _save_json(
                objdir / "step2_rtl_knowledge.json",
                rtl_knowledge if rtl_knowledge is not None else {},
            )
        else:
            kg_json = _load_json(objdir / "step2_kg.json", default={})
            rtl_knowledge = _load_json(objdir / "step2_rtl_knowledge.json", default={})
            # kg_nx isn't required for later steps; keep it None
            kg_nx = None

        # -----------------------------
        # Step 3: LLM init (cannot "cache" model object, but we can skip earlier steps)
        # -----------------------------
        print("Step 3: Initializing the language model...")
        llm_agent = get_llm(model_name=FLAGS.llm_model, **FLAGS.llm_args)
        timer.time_and_clear("Initialize LLM")

        # -----------------------------
        # Step 4: Extract valid signals (or load cache)
        # -----------------------------
        print("Step 4: Extracting valid signal names...")
        if step in (0, 4):
            if not FLAGS.valid_signals:
                _, valid_signals, signal_hierarchy = write_svas_to_file(
                    []
                )  # We pass an empty list just to extract valid signals
                if FLAGS.include_internal_signals:
                    total_signals = 0
                    for key, value in signal_hierarchy.items():
                        total_signals += len(value)
                    print(
                        f"Extracted signal names: {signal_hierarchy} \nTotal signals: {total_signals}"
                    )
                    print(
                        f"Pruned and cleaned signals: {valid_signals} \nSelected signals: {len(valid_signals)}"
                    )
                else:
                    print(f"Valid signal names: {', '.join(sorted(valid_signals))}")
            else:
                valid_signals = FLAGS.valid_signals
                print(f"Valid signal names: {', '.join(sorted(valid_signals))}")

            # Caching Step 4 outputs
            _save_json(objdir / "step4_valid_signals.json", sorted(list(valid_signals)))
        else:
            valid_signals = set(_load_json(objdir / "step4_valid_signals.json"))
            print(f"Valid signal names (loaded): {', '.join(sorted(valid_signals))}")

        timer.time_and_clear("Extract valid signals")

        # -----------------------------
        # Step 4b: Context Summarizer (optional; expensive) - not fully cached here
        # -----------------------------
        # If you want this resumable too, you'd need to persist summaries per signal.
        if FLAGS.enable_context_enhancement:
            if step in (0, 4.1):
                print("Step 4b: Initializing Design Context Summarizer...")

                # Initialize the context summarizer once
                context_summarizer = DesignContextSummarizer(llm_agent=llm_agent)

                # Extract RTL text from rtl_knowledge if available - simplified approach
                rtl_text = (
                    rtl_knowledge['combined_content']
                    if rtl_knowledge is not None
                    and isinstance(rtl_knowledge, dict)
                    and 'combined_content' in rtl_knowledge
                    else ""
                )

                # Generate the global summary once
                context_summarizer.generate_parallel_global_summary(
                    spec_text, rtl_text, list(valid_signals), timer
                )

                # Pre-generate summaries for all signals we'll process
                signals_to_process = sorted(valid_signals)

                if not math.isinf(FLAGS.max_num_signals_process):
                    signals_to_process = signals_to_process[: FLAGS.max_num_signals_process]

                def process_signal(signal_name):
                    cached = load_cached_signal_summary(objdir, signal_name)
                    if cached is not None:
                        return cached
                
                    signal_rtl = (
                        rtl_knowledge.get(signal_name, "")
                        if isinstance(rtl_knowledge, dict)
                        else ""
                    )
                
                    summary = context_summarizer.get_signal_specific_summary(
                        signal_name, spec_text, signal_rtl
                    )
                
                    save_cached_signal_summary(objdir, signal_name, summary)
                    return summary
                    
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [
                        executor.submit(process_signal, s)
                        for s in signals_to_process
                    ]
                    results = [f.result() for f in as_completed(futures)]
            
            # for signal_name in signals_to_process:
            #     # Get signal-specific RTL if available
            #     signal_rtl = (
            #         rtl_knowledge.get(signal_name, "")
            #         if isinstance(rtl_knowledge, dict)
            #         else ""
            #     )
            #     context_summarizer.get_signal_specific_summary(
            #         signal_name, spec_text, signal_rtl
            #     )

            timer.time_and_clear("Initialize Context Summarizer")

        # -----------------------------
        # Step 5: NL plans (or load cache)
        # -----------------------------
        print("Step 5: Generating natural language test plans...")
        if FLAGS.generate_nlp:
          if step in (0, 5):
              nl_plans = generate_nl_plans(
                  spec_text,
                  kg_json,
                  llm_agent,
                  valid_signals if FLAGS.gen_plan_sva_using_valid_signals else None,
                  rtl_knowledge,
                  context_summarizer,  # Pass the context_summarizer
              )

              # Caching Step 5 output
              _save_json(objdir / "step5_nl_plans.json", nl_plans)
        else:
            nl_plans = _load_json(objdir / "step5_nl_plans.json")
            print("Loaded NL plans from cache.")

        # Keep your existing nl_plans.txt write (do not remove)
        with open(
            Path(saver.logdir) / 'nl_plans.txt',
            'w',
            encoding="cp1252",
            errors="replace",
        ) as f:
            c = 1
            for signal_name, plans in nl_plans.items():
                f.write(f'Signal {signal_name}:\n')
                for plan in plans:
                    f.write(f'Plan {c}: {plan}\n')
                    c += 1
                f.write('\n')

        timer.time_and_clear("Generate NL plans")

        # -----------------------------
        # Step 6/7: SVAs (or load cache) + write to files
        # -----------------------------
        if FLAGS.generate_SVAs:
            print("Step 6: Generating SVAs...")
            if step in (0, 6):
                svas = generate_svas(
                    spec_text,
                    nl_plans,
                    kg_json,
                    llm_agent,
                    valid_signals if FLAGS.gen_plan_sva_using_valid_signals else None,
                    rtl_knowledge,
                    context_summarizer,  # Pass the context_summarizer
                )

                # Caching Step 6 output
                _save_json(objdir / "step6_svas.json", svas)
            else:
                svas = _load_json(objdir / "step6_svas.json", default=[])
                print("Loaded SVAs from cache.")

            if len(svas) == 0:
                raise RuntimeError(f'No SVA generated/extracted')
            timer.time_and_clear("Generate SVAs")

            # Print generated SVAs
            print("\nGenerated SVAs:")
            for i, sva in enumerate(svas, 1):
                print(f"{i}. {sva}")
            print('')  # Add a blank line for readability

            print("Step 7: Writing SVAs to files...")
            sva_file_paths, _, _ = write_svas_to_file(svas)
            timer.time_and_clear("Write SVAs to files")

        if FLAGS.generate_SVAs:
            print('Test plan and Assertion generation process completed.')
            print(f"<[!]> nl test plans saved to {Path(saver.logdir)}/nl_plans.txt")
            print(f"<[!]> svas saved to {sva_file_paths}")
        else:
            print('Test plan generation process completed.')
            print(f"<[!]> nl test plans saved to {Path(saver.logdir)}/nl_plans.txt")

            # print("Step 8: Generating TCL scripts...")
            # tcl_file_paths = generate_tcl_scripts(sva_file_paths)
            # timer.time_and_clear("Generate TCL scripts")

            # print("Step 9: Running JasperGold...")
            # jasper_reports = run_jaspergold(tcl_file_paths)
            # timer.time_and_clear("Run JasperGold")

            # print("Step 10: Analyzing coverage of proven SVAs...")
            # coverage_report = analyze_coverage_of_proven_svas(svas, jasper_reports)
            # timer.time_and_clear("Analyze coverage of proven SVAs")

            # print("Step 11: Analyzing and printing results...")
            # analyze_results(pdf_stats, nl_plans, svas, jasper_reports, coverage_report)
            # timer.time_and_clear("Analyze results")

            # print('Test plan generation and coverage evaluation process completed.')

    elif FLAGS.subtask == 'parse_result':
        print("Parsing results from a previous run...")
        load_dir = FLAGS.load_dir

        print("Loading PDF statistics...")
        pdf_stats = load_pdf_stats(load_dir)

        print("Loading natural language test plans...")
        nl_plans = load_nl_plans(load_dir)

        print("Loading SVAs...")
        svas = load_svas(load_dir)

        print("Loading Jasper reports...")
        jasper_reports = load_jasper_reports(load_dir)

        print("Analyzing results...")
        analyze_results(pdf_stats, nl_plans, svas, jasper_reports)

        timer.time_and_clear("parse_result")
        print('parse_result completed.')

    else:
        raise NotImplementedError()

    # Print the durations log
    timer.print_durations_log(print_func=print)


def read_pdf(file_path: Union[str, List[str]]) -> Tuple[str, dict]:
    """
    Read one or multiple PDF files and extract their content.

    Args:
        file_path (Union[str, List[str]]): Path to a single PDF file or a list of paths to multiple PDF files.

    Returns:
        Tuple[str, dict]: A tuple containing the extracted text and file statistics.
    """
    if isinstance(file_path, str):
        file_paths = [file_path]
    elif isinstance(file_path, list):
        file_paths = file_path
    else:
        raise ValueError("file_path must be a string or a list of strings")

    all_text = ""
    total_pages = 0
    total_tokens = 0
    total_file_size = 0

    output = preprocess(
        file_paths,
        markdown=True,
        extract_vectors=False,
        extract_images=False,
        strip_headers_footers=True,
        preserve_layout=True,
        as_dataset=False,
        verbose=False,
    )
    for path in file_paths:
        pdf_reader = output[os.path.basename(path)]
        text = ""
        for page in pdf_reader:
            text += page['page_content']

        all_text += text + "\n\n"  # Add some separation between different PDFs
        total_pages += len(output[os.path.basename(path)])

        # Create a temporary file to store the extracted text
        temp_file_path = f"temp_{os.path.basename(path)}.txt"
        with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
            temp_file.write(text)

        # Count tokens using the helper function
        total_tokens += count_tokens_in_file(temp_file_path)

        # Remove the temporary file
        os.remove(temp_file_path)

        total_file_size += os.path.getsize(path)

    stats = {
        "num_pages": total_pages,
        "num_tokens": total_tokens,
        "file_size": total_file_size,
        "num_files": len(file_paths),
    }

    # Print only the first few lines of spec_text
    num_lines_to_print = 5  # You can adjust this number as needed
    lines = all_text.splitlines()
    first_few_lines = '\n'.join(lines[:num_lines_to_print])
    print(
        f'First {num_lines_to_print} lines of spec_text (truncated):\n{first_few_lines}\n...'
    )
    print(f'Total number of lines in spec_text: {len(lines)}')

    return all_text.strip(), stats


def load_and_process_kg(kg_path: str) -> Tuple[nx.Graph, Dict]:
    """
    Load the Knowledge Graph from a GraphML file and process it into both a NetworkX graph
    and a JSON format suitable for prompting. Prints detailed information about the graph structure.

    Args:
        kg_path (str): Path to the GraphML file containing the Knowledge Graph.

    Returns:
        Tuple[nx.Graph, Dict]: A tuple containing the original NetworkX graph and the processed JSON format.
    """
    # Load the graph from GraphML file
    G = nx.read_graphml(kg_path)

    # Convert the graph to JSON format
    json_graph = convert_nx_to_json(G)

    # Print detailed information about the graph
    print(f"Knowledge Graph loaded from {kg_path}")
    print(f"Number of nodes: {len(G.nodes())}")
    print(f"Number of edges: {len(G.edges())}")

    # Sample and print some node attributes
    if G.nodes:
        sample_node = random.choice(list(G.nodes))
        print("\nExample node attributes:")
        for k, v in G.nodes[sample_node].items():
            print(f"  {k}: {v}")

    # Sample and print some edge attributes
    if G.edges:
        sample_edge = random.choice(list(G.edges))
        print("\nExample edge attributes:")
        for k, v in G.edges[sample_edge].items():
            print(f"  {k}: {v}")

    # Print information about attribute keys
    node_attr_keys = set().union(*(data.keys() for _, data in G.nodes(data=True)))
    edge_attr_keys = set().union(*(data.keys() for _, _, data in G.edges(data=True)))

    print("\nNode attribute keys:")
    print(", ".join(node_attr_keys))

    print("\nEdge attribute keys:")
    print(", ".join(edge_attr_keys))

    return G, json_graph


# Helper function to convert NetworkX graph to JSON format


def convert_nx_to_json(G: nx.Graph) -> Dict:
    """
    Convert a NetworkX graph to a JSON-friendly dictionary format.

    Args:
        G (nx.Graph): The NetworkX graph to convert.

    Returns:
        Dict: A dictionary representation of the graph.
    """
    json_graph = {"nodes": [], "edges": []}
    for node, data in G.nodes(data=True):
        json_graph["nodes"].append(
            {
                "id": node,
                "attributes": {k: str(v) for k, v in data.items()},
            }
        )
    for u, v, data in G.edges(data=True):
        json_graph["edges"].append(
            {
                "source": u,
                "target": v,
                "attributes": {k: str(v) for k, v in data.items()},
            }
        )
    return json_graph


def generate_nl_plans(
    spec_text: str,
    kg: Optional[Dict],
    llm_agent,
    valid_signals: Optional[Set[str]],
    rtl_knowledge,
    context_summarizer,
) -> Dict[str, List[str]]:
    """
    Generate natural language test plans using the design specification,
    optionally using a Knowledge Graph, LLM, and valid signal names.

    Args:
        spec_text (str): The design specification text.
        kg (Optional[Dict]): The processed Knowledge Graph, if available.
        llm_agent: The language model agent.
        valid_signals (Optional[Set[str]]): Set of valid signal names, if using valid signals.

    Returns:
        Dict[str, List[str]]: A dictionary mapping signal names to lists of generated natural language test plans.
    """
    if FLAGS.prompt_builder == 'dynamic':
        return generate_dynamic_nl_plans(
            spec_text, kg, llm_agent, valid_signals, rtl_knowledge, context_summarizer
        )
    elif FLAGS.prompt_builder == 'dynamic_threaded':
        return generate_dynamic_nl_plans_threaded(
            spec_text, kg, llm_agent, valid_signals, rtl_knowledge, context_summarizer
        )
    elif FLAGS.prompt_builder == 'static':
        return generate_static_nl_plans(spec_text, kg, llm_agent, valid_signals)
    else:
        raise NotImplementedError("Unsupported prompt builder type")


def generate_dynamic_nl_plans(
    spec_text: str,
    kg: Optional[Dict],
    llm_agent,
    valid_signals: Optional[Set[str]],
    rtl_knowledge,
    context_summarizer,
) -> Dict[str, List[str]]:
    """
    Generate natural language test plans using dynamic context synthesis.
    """
    # Create context generators using factory function
    context_generators = create_context_generators(
        spec_text, kg, valid_signals, rtl_knowledge
    )

    # Initialize the prompt builder with context_summarizer
    prompt_builder = DynamicPromptBuilder(
        context_generators=context_generators,
        pruning_config=FLAGS.dynamic_prompt_settings['pruning'],
        llm_agent=llm_agent,
        context_summarizer=context_summarizer,  # Pass the summarizer
    )

    nl_plans = {}
    for i, signal_name in enumerate(sorted(valid_signals)):  # sorted is key!
        if i >= FLAGS.max_num_signals_process:
            print(
                f'Reached max signals limit ({FLAGS.max_num_signals_process}), stopping.'
            )
            break

        print(f'Processing signal {i+1}/{len(valid_signals)}: {signal_name}')
        query = f"{signal_name}"

        # Get dynamic contexts with enhancement integrated if enabled
        dynamic_context_list = prompt_builder.build_prompt(
            query=query,
            base_prompt="",
            signal_name=signal_name,
            enable_context_enhancement=FLAGS.enable_context_enhancement,  # Pass the enhancement flag
        )

        print(
            f'Generated {len(dynamic_context_list)} dynamic contexts for signal {signal_name}'
        )

        assert len(dynamic_context_list) <= FLAGS.max_prompts_per_signal

        all_signal_plans = []

        # Process each dynamic context separately
        for context_idx, dynamic_context in enumerate(dynamic_context_list):
            print(
                f'Processing dynamic context {context_idx+1}/{len(dynamic_context_list)} for {signal_name}'
            )

            # Context enhancement is now integrated in prompt builder
            # No need to call add_enhanced_context here anymore

            # Rest of the existing code...
            full_prompt = construct_static_nl_prompt(
                dynamic_context,
                kg=None,  # KG info already included in dynamic context
                valid_signals=valid_signals,
            )
            full_prompt += f"\n\nGenerate diverse test plans for the signal '{signal_name}'. Each test plan should be on a new line and start with 'Plan: '."

            try:
                # Get LLM response for this context prompt
                result = llm_inference(
                    llm_agent, full_prompt, f"NL_Plans_{signal_name}_{context_idx+1}"
                )

                # Extract plans from the result
                context_plans = []
                for line in result.split('\n'):
                    if line.strip().startswith('Plan:'):
                        plan = line.split(':', 1)[-1].strip()
                        context_plans.append(plan)

                all_signal_plans.extend(context_plans)
                print(
                    f"Generated {len(context_plans)} plans from context {context_idx+1} for signal {signal_name}"
                )

            except Exception as e:
                print(
                    f"Error generating NL plans for signal {signal_name} context {context_idx+1}: {str(e)}"
                )
                print(f"Continuing with other contexts for this signal")
                continue

        # Deduplicate plans
        unique_plans = []
        plan_set = set()
        for plan in all_signal_plans:
            # Use a simplified version of the plan for deduplication
            # Remove extra spaces and convert to lowercase
            simplified_plan = ' '.join(plan.lower().split())
            if simplified_plan not in plan_set:
                plan_set.add(simplified_plan)
                unique_plans.append(plan)

        nl_plans[signal_name] = unique_plans
        print(
            f"Generated {len(unique_plans)} unique plans for signal {signal_name} from {len(all_signal_plans)} total plans"
        )

    return nl_plans

def _generate_plans_for_context(
    *,
    ctx: str,
    idx: int,
    signal_name: str,
    valid_signals: Set[str],
    llm_agent,
) -> List[str]:

    prompt = construct_static_nl_prompt(
        ctx,
        kg=None,  # KG already embedded in dynamic context
        valid_signals=valid_signals,
    )

    prompt += (
        f"\n\nGenerate diverse test plans for the signal '{signal_name}'. "
        "Each test plan should be on a new line and start with 'Plan:'."
    )

    result = llm_inference(
        llm_agent,
        prompt,
        f"NL_{signal_name}_{idx}",
    )

    plans = []
    for line in result.splitlines():
        if line.strip().startswith("Plan:"):
            plans.append(line.split(":", 1)[-1].strip())

    return plans

def _process_signal_nlp(
    *,
    signal_name: str,
    spec_text: str,
    kg: Optional[Dict],
    valid_signals: Set[str],
    rtl_knowledge,
    llm_agent,
    context_summarizer,
    context_generators,
    objdir: Path,
) -> tuple[str, List[str]]:

    # ---- Cache check
    cached = load_cached_nl_plans(objdir, signal_name)
    if cached is not None:
        print(f"[CACHE HIT] {signal_name}")
        return signal_name, cached

    print(f"[RUNNING] {signal_name}")

    # ---- New builder per thread (important)
    prompt_builder = DynamicPromptBuilder(
        context_generators=context_generators,
        pruning_config=FLAGS.dynamic_prompt_settings["pruning"],
        llm_agent=llm_agent,
        context_summarizer=context_summarizer,
    )

    base_prompt = ""

    dynamic_contexts = prompt_builder.build_prompt(
        query=signal_name,
        base_prompt=base_prompt,
        signal_name=signal_name,
        enable_context_enhancement=FLAGS.enable_context_enhancement,
    )

    all_plans: List[str] = []

    # Context-level threading
    max_ctx_workers = min(
        getattr(FLAGS, "context_workers", 4),
        len(dynamic_contexts),
    )

    if max_ctx_workers <= 1:
        # Serial fallback
        for idx, ctx in enumerate(dynamic_contexts):
            all_plans.extend(
                _generate_plans_for_context(
                    ctx=ctx,
                    idx=idx,
                    signal_name=signal_name,
                    valid_signals=valid_signals,
                    llm_agent=llm_agent,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=max_ctx_workers) as ex:
            futures = [
                ex.submit(
                    _generate_plans_for_context,
                    ctx=ctx,
                    idx=idx,
                    signal_name=signal_name,
                    valid_signals=valid_signals,
                    llm_agent=llm_agent,
                )
                for idx, ctx in enumerate(dynamic_contexts)
            ]

            for f in as_completed(futures):
                all_plans.extend(f.result())

    # Deduplicate (order-preserving)
    seen = set()
    unique_plans = []
    for plan in all_plans:
        key = " ".join(plan.lower().split())
        if key not in seen:
            seen.add(key)
            unique_plans.append(plan)

    save_cached_nl_plans(objdir, signal_name, unique_plans)
    print(
        f"[DONE] {signal_name}: {len(unique_plans)} unique plans "
        f"(from {len(all_plans)} total)"
    )

    return signal_name, unique_plans

def generate_dynamic_nl_plans_threaded(
    spec_text: str,
    kg: Optional[Dict],
    llm_agent,
    valid_signals: Optional[Set[str]],
    rtl_knowledge,
    context_summarizer,
    objdir: Path,
) -> Dict[str, List[str]]:
    """
    Generate natural language test plans using dynamic context synthesis.
    Threaded, cached, and resumable.
    """

    assert valid_signals, "valid_signals must not be empty"

    context_generators = create_context_generators(
        spec_text,
        kg,
        valid_signals,
        rtl_knowledge,
    )

    results: Dict[str, List[str]] = {}

    signals = sorted(valid_signals)[: FLAGS.max_num_signals_process]
    max_signal_workers = min(
        getattr(FLAGS, "signal_workers", 4),
        len(signals),
    )

    with ThreadPoolExecutor(max_workers=max_signal_workers) as ex:
        futures = [
            ex.submit(
                _process_signal_nlp,
                signal_name=signal,
                spec_text=spec_text,
                kg=kg,
                valid_signals=valid_signals,
                rtl_knowledge=rtl_knowledge,
                llm_agent=llm_agent,
                context_summarizer=context_summarizer,
                context_generators=context_generators,
                objdir=objdir,
            )
            for signal in signals
        ]

        for f in as_completed(futures):
            signal_name, plans = f.result()
            results[signal_name] = plans

    return results
    
def generate_static_nl_plans(
    spec_text: str, kg: Optional[Dict], llm_agent, valid_signals: Optional[Set[str]]
) -> Dict[str, List[str]]:
    nl_gen_prompt = construct_static_nl_prompt(spec_text, kg, valid_signals)

    try:
        result = llm_inference(llm_agent, nl_gen_prompt, "NL_Plans")

        # Parse the result into a dictionary
        nl_plans = parse_nl_plans(result)
        return nl_plans
    except Exception as e:
        print(f"Error generating NL description: {str(e)}")
        raise


def generate_svas(
    spec_text: str,
    nl_plans: Dict[str, List[str]],
    kg: Optional[Dict],
    llm_agent,
    valid_signals: Optional[Set[str]],
    rtl_knowledge,
    context_summarizer,
) -> List[str]:
    """
    Generate SVAs using LLM based on the design specification, natural language test plans,
    and optionally a Knowledge Graph, ensuring only valid signal names are used if provided.

    Args:
        spec_text (str): The design specification text.
        nl_plans (Dict[str, List[str]]): Dictionary mapping signal names to lists of natural language test plans.
        kg (Optional[Dict]): The processed Knowledge Graph, if available.
        llm_agent: The language model agent.
        valid_signals (Optional[Set[str]]): Set of valid signal names, if using valid signals.

    Returns:
        List[str]: A list of generated SVAs.
    """
    if FLAGS.prompt_builder == 'dynamic':
        return generate_dynamic_svas(
            spec_text,
            nl_plans,
            kg,
            llm_agent,
            valid_signals,
            rtl_knowledge,
            context_summarizer,
        )
    if FLAGS.prompt_builder == 'dynamic_threaded':
        return generate_dynamic_svas_threaded(
            spec_text,
            nl_plans,
            kg,
            llm_agent,
            valid_signals,
            rtl_knowledge,
            context_summarizer,
        )
    elif FLAGS.prompt_builder == 'static':
        return generate_static_svas(spec_text, nl_plans, kg, llm_agent, valid_signals)
    else:
        raise NotImplementedError("Unsupported prompt builder type")


def generate_dynamic_svas(
    spec_text: str,
    nl_plans: Dict[str, List[str]],
    kg: Optional[Dict],
    llm_agent,
    valid_signals: Optional[Set[str]],
    rtl_knowledge,
    context_summarizer,
) -> List[str]:
    """
    Generate SVAs using dynamic context synthesis with support for multiple contexts per signal.
    """
    # Create context generators using factory function
    context_generators = create_context_generators(
        spec_text, kg, valid_signals, rtl_knowledge
    )

    # Initialize the prompt builder with the context_summarizer
    prompt_builder = DynamicPromptBuilder(
        context_generators=context_generators,
        pruning_config=FLAGS.dynamic_prompt_settings['pruning'],
        llm_agent=llm_agent,
        context_summarizer=context_summarizer,  # Pass the summarizer here
    )

    all_svas = []
    for i, (signal_name, plans) in enumerate(nl_plans.items()):
        if i >= FLAGS.max_num_signals_process:
            print(
                f'Reached max signals limit ({FLAGS.max_num_signals_process}), stopping.'
            )
            break

        if len(plans) == 0:
            print(f'Empty NL plans for signal {signal_name}')
            continue

        print(f'Processing signal {i+1}/{len(nl_plans)}: {signal_name}')

        # Get dynamic contexts with enhancement integrated if enabled
        dynamic_context_list = prompt_builder.build_prompt(
            query=signal_name,
            base_prompt="Generate SystemVerilog Assertions based on the following information:",
            signal_name=signal_name,
            enable_context_enhancement=FLAGS.enable_context_enhancement,  # Pass the enhancement flag
        )

        print(
            f'Generated {len(dynamic_context_list)} dynamic contexts for signal {signal_name}'
        )

        # Get SVA examples once (reused for each context)
        sva_examples = get_sva_icl_examples()

        signal_svas = []

        # Determine if we should distribute plans or use all plans for each context
        distribute_plans = len(plans) > 10 and len(dynamic_context_list) > 1

        if distribute_plans:
            # Prepare distributed plans
            plans_per_context = [[] for _ in range(len(dynamic_context_list))]
            for j, plan in enumerate(plans):
                context_idx = j % len(dynamic_context_list)
                plans_per_context[context_idx].append((j, plan))

            print(
                f'Distributing {len(plans)} plans across {len(dynamic_context_list)} contexts'
            )
        else:
            # Use all plans for each context
            plans_text = "\n".join(
                f"Plan {j+1}: {plan}" for j, plan in enumerate(plans)
            )

        # Process each dynamic context
        for context_idx, dynamic_context in enumerate(dynamic_context_list):
            # Context enhancement is now integrated within the prompt builder
            # No need to call add_enhanced_context here anymore

            try:
                # Determine which plans to use for this context
                if distribute_plans:
                    context_plans = plans_per_context[context_idx]
                    if not context_plans:
                        print(f'No plans assigned to context {context_idx+1}, skipping')
                        continue

                    # Create subset of plans text for this context
                    current_plans_text = "\n".join(
                        f"Plan {j+1}: {plan}" for j, plan in context_plans
                    )
                    print(
                        f'Processing context {context_idx+1} with {len(context_plans)} plans'
                    )
                else:
                    current_plans_text = plans_text
                    print(
                        f'Processing context {context_idx+1} with all {len(plans)} plans'
                    )

                # Create the full prompt for this context
                full_prompt = (
                    f"{dynamic_context}\n\n"
                    f"Natural Language Test Plans for signal '{signal_name}':\n{current_plans_text}\n\n"
                    f"{sva_examples}\n\n"
                    "Generate one SVA for each of the provided natural language test plans. "
                    "Enclose each SVA in triple backticks (```) and prefix it with 'SVA:'."
                )
                result = llm_inference(
                    llm_agent, full_prompt, f"SVAs_{signal_name}_{context_idx+1}"
                )

                context_svas = extract_svas_from_block(result)
                print(
                    f'Generated {len(context_svas)} SVAs from context {context_idx+1} for signal {signal_name}'
                )
                signal_svas.extend(context_svas)

            except Exception as e:
                print(
                    f"Error generating SVAs for signal {signal_name} context {context_idx+1}: {str(e)}"
                )
                print(f"Continuing with other contexts for this signal")
                continue

        # Deduplicate SVAs
        unique_svas = []
        sva_set = set()
        for sva in signal_svas:
            # Use a simplified version of the SVA for deduplication
            # Remove comments, extra spaces and convert to lowercase
            simplified_sva = ' '.join(
                line
                for line in sva.lower().split('\n')
                if not line.strip().startswith('//')
            ).strip()

            if simplified_sva not in sva_set:
                sva_set.add(simplified_sva)
                unique_svas.append(sva)

        print(
            f'Generated {len(unique_svas)} unique SVAs for signal {signal_name} from {len(signal_svas)} total SVAs'
        )
        all_svas.extend(unique_svas)

    return all_svas

def _generate_svas_for_context(
    *,
    dynamic_context: str,
    context_idx: int,
    signal_name: str,
    plans: List[str],
    distribute_plans: bool,
    plans_per_context: Optional[List[List[tuple[int, str]]]],
    sva_examples: str,
    llm_agent,
) -> List[str]:

    if distribute_plans:
        context_plans = plans_per_context[context_idx]
        if not context_plans:
            return []

        plans_text = "\n".join(
            f"Plan {j+1}: {plan}" for j, plan in context_plans
        )
    else:
        plans_text = "\n".join(
            f"Plan {j+1}: {plan}" for j, plan in enumerate(plans)
        )

    full_prompt = (
        f"{dynamic_context}\n\n"
        f"Natural Language Test Plans for signal '{signal_name}':\n"
        f"{plans_text}\n\n"
        f"{sva_examples}\n\n"
        "Generate one SystemVerilog Assertion (SVA) for each test plan.\n"
        "Enclose each SVA in triple backticks (```) and prefix it with 'SVA:'."
    )

    result = llm_inference(
        llm_agent,
        full_prompt,
        f"SVAs_{signal_name}_{context_idx}",
    )

    return extract_svas_from_block(result)

def _process_signal_svas(
    *,
    signal_name: str,
    plans: List[str],
    spec_text: str,
    kg: Optional[Dict],
    valid_signals: Set[str],
    rtl_knowledge,
    llm_agent,
    context_summarizer,
    context_generators,
    objdir: Path,
) -> List[str]:

    # ---- Cache check
    cached = load_cached_svas(objdir, signal_name)
    if cached is not None:
        print(f"[SVA CACHE HIT] {signal_name}")
        return cached

    print(f"[SVA RUNNING] {signal_name}")

    if not plans:
        print(f"No NL plans for signal {signal_name}, skipping")
        return []

    prompt_builder = DynamicPromptBuilder(
        context_generators=context_generators,
        pruning_config=FLAGS.dynamic_prompt_settings["pruning"],
        llm_agent=llm_agent,
        context_summarizer=context_summarizer,
    )

    dynamic_contexts = prompt_builder.build_prompt(
        query=signal_name,
        base_prompt="Generate SystemVerilog Assertions based on the following information:",
        signal_name=signal_name,
        enable_context_enhancement=FLAGS.enable_context_enhancement,
    )

    sva_examples = get_sva_icl_examples()

    distribute_plans = len(plans) > 10 and len(dynamic_contexts) > 1

    plans_per_context = None
    if distribute_plans:
        plans_per_context = [[] for _ in range(len(dynamic_contexts))]
        for j, plan in enumerate(plans):
            plans_per_context[j % len(dynamic_contexts)].append((j, plan))

    signal_svas: List[str] = []

    max_ctx_workers = min(
        getattr(FLAGS, "context_workers", 4),
        len(dynamic_contexts),
    )

    with ThreadPoolExecutor(max_workers=max_ctx_workers) as ex:
        futures = [
            ex.submit(
                _generate_svas_for_context,
                dynamic_context=ctx,
                context_idx=idx,
                signal_name=signal_name,
                plans=plans,
                distribute_plans=distribute_plans,
                plans_per_context=plans_per_context,
                sva_examples=sva_examples,
                llm_agent=llm_agent,
            )
            for idx, ctx in enumerate(dynamic_contexts)
        ]

        for f in as_completed(futures):
            signal_svas.extend(f.result())

    # ---- Deduplicate SVAs (order-preserving)
    seen = set()
    unique_svas = []
    for sva in signal_svas:
        simplified = " ".join(
            line
            for line in sva.lower().splitlines()
            if not line.strip().startswith("//")
        ).strip()

        if simplified not in seen:
            seen.add(simplified)
            unique_svas.append(sva)

    save_cached_svas(objdir, signal_name, unique_svas)

    print(
        f"[SVA DONE] {signal_name}: {len(unique_svas)} unique SVAs "
        f"(from {len(signal_svas)} total)"
    )

    return unique_svas

def generate_dynamic_svas_threaded(
    spec_text: str,
    nl_plans: Dict[str, List[str]],
    kg: Optional[Dict],
    llm_agent,
    valid_signals: Optional[Set[str]],
    rtl_knowledge,
    context_summarizer,
    objdir: Path,
) -> List[str]:
    """
    Generate SVAs using dynamic context synthesis.
    Threaded, cached, and resumable.
    """

    context_generators = create_context_generators(
        spec_text,
        kg,
        valid_signals,
        rtl_knowledge,
    )

    all_svas: List[str] = []

    signals = list(nl_plans.keys())[: FLAGS.max_num_signals_process]

    max_signal_workers = min(
        getattr(FLAGS, "signal_workers", 4),
        len(signals),
    )

    with ThreadPoolExecutor(max_workers=max_signal_workers) as ex:
        futures = {
            ex.submit(
                _process_signal_svas,
                signal_name=signal,
                plans=nl_plans[signal],
                spec_text=spec_text,
                kg=kg,
                valid_signals=valid_signals,
                rtl_knowledge=rtl_knowledge,
                llm_agent=llm_agent,
                context_summarizer=context_summarizer,
                context_generators=context_generators,
                objdir=objdir,
            ): signal
            for signal in signals
        }

        for f in as_completed(futures):
            all_svas.extend(f.result())

    return all_svas

def generate_static_svas(
    spec_text: str,
    nl_plans: Dict[str, List[str]],
    kg: Optional[Dict],
    llm_agent,
    valid_signals: Optional[Set[str]],
) -> List[str]:
    """
    Generate SVAs using LLM based on the design specification, natural language test plans,
    and optionally a Knowledge Graph, ensuring only valid signal names are used if provided.

    Args:
        spec_text (str): The design specification text.
        nl_plans (Dict[str, List[str]]): Dictionary mapping signal names to lists of natural language test plans.
        kg (Optional[Dict]): The processed Knowledge Graph, if available.
        llm_agent: The language model agent.
        valid_signals (Optional[Set[str]]): Set of valid signal names, if using valid signals.

    Returns:
        List[str]: A list of generated SVAs.
    """
    sva_gen_prompt = construct_static_sva_prompt(spec_text, nl_plans, kg, valid_signals)

    # try:

    result = llm_inference(llm_agent, sva_gen_prompt, "SVAs")

    # Use extract_svas_from_block to extract SVAs
    svas = extract_svas_from_block(result)

    if not svas:
        print(
            "Warning: No valid SVAs were generated. Please check the output and adjust the prompt if necessary."
        )
    else:
        print(f"Generated {len(svas)} SVAs.")

    return svas

    # except Exception as e:
    #     print(f"Error generating SVAs: {str(e)}")
    #     return []


def get_sva_icl_examples():
    return """
    Examples:
    SVA:
    ```
    @(posedge PCLK) ((PWDATA >= 230) && (PWDATA <= 255)) |-> (PWDATA >= 205) && (PWDATA <= 255);
    ```
    NL: that when PWDATA is within the range of 230 to 255, in the next cycle PWDATA will remain within the range of 205 to 255. Use the signals 'PCLK' for the clock edge and 'PWDATA' for the data being checked.

    SVA:
    ```
    @(posedge PCLK) (PRESETn) |-> (PWDATA >= 0) && (PWDATA <= 45);
    ```
    NL: that the input data is within the valid range when not in reset. Use the signals 'PRESETn', 'PCLK', and 'PWDATA'.
    """


def construct_static_nl_prompt(
    spec_text: str, kg: Optional[Dict], valid_signals: Optional[Set[str]]
) -> str:
    nl_gen_prompt = f"""
    Given the following design specification{' and Knowledge Graph' if kg else ''}, generate natural language test plans:

    {spec_text}

    """

    if valid_signals:
        nl_gen_prompt += f"""
    CRITICAL - Valid Signal Names (USE ONLY THESE SIGNALS):
    {', '.join(sorted(valid_signals))}

    WARNING: It is ABSOLUTELY ESSENTIAL that you ONLY use signals from the above list in your test plans. 
    DO NOT introduce or use ANY signals that are not in this list. Any test plan using undefined signals will be considered invalid.

    """

    if kg:
        nl_gen_prompt += f"""
    Knowledge Graph:
    {json.dumps(kg, indent=2)}

    """

    nl_gen_prompt += """
    Use the following examples as a guide for the format and style of the test plans:

    1. that when PWDATA is within the range of 230 to 255, in the next cycle PWDATA will remain within the range of 205 to 255. Use the signals 'PCLK' for the clock edge and 'PWDATA' for the data being checked.
    2. that the input data is within the valid range when not in reset. Use the signals 'PRESETn', 'PCLK', and 'PWDATA'.
    3. that if the input data 'PWDATA' is within the range of 138 to 153 inclusive, then in the subsequent cycles, 'PWDATA' must continue to be within the range of 98 to 153 inclusive. Use the signals 'PWDATA' and 'PCLK'.
    4. that the input data PWDATA has a value between 83 and 165, inclusive, 3 clock cycles after the reset signal PRESETn becomes deasserted. Use the signals 'PRESETn', 'PCLK', and 'PWDATA'.
    5. that the input data signal 'PWDATA' is within the range 0 to 45 inclusive, starting from four clock cycles after the reset signal 'PRESETn' becomes deasserted. Use the signals 'PRESETn', 'PCLK', and 'PWDATA'.

    Generate diverse test plans based on the given specification"""

    if kg:
        nl_gen_prompt += " and Knowledge Graph"

    nl_gen_prompt += "."

    nl_gen_prompt += """

    FINAL REMINDER:
    - You MUST ONLY use signals from the 'Valid Signal Names' list provided above.
    - DO NOT introduce or use any signals that are not in this list.
    - Any test plan using undefined signals will be rejected.
    - Double-check each test plan to ensure it ONLY uses valid signals.

    For each test plan, start with the signal name followed by a colon, then the test plan. For example:
    PWDATA: that when PWDATA is within the range of 230 to 255, in the next cycle PWDATA will remain within the range of 205 to 255.
    """

    return nl_gen_prompt


def construct_static_sva_prompt(
    spec_text: str,
    nl_plans: Dict[str, List[str]],
    kg: Optional[Dict],
    valid_signals: Optional[Set[str]],
) -> str:
    sva_gen_prompt = f"""
    Given the following design specification, natural language test plans{', and Knowledge Graph' if kg else ''}, generate SVAs (System Verilog Assertions):

    {spec_text}

    Test Plans:
    """

    for signal, plans in nl_plans.items():
        sva_gen_prompt += f"\n{signal}:\n"
        for i, plan in enumerate(plans, 1):
            sva_gen_prompt += f"  {i}. {plan}\n"

    if valid_signals:
        sva_gen_prompt += f"""
    Valid Signal Names:
    {', '.join(sorted(valid_signals))}

    """

    if kg:
        sva_gen_prompt += f"""
    Knowledge Graph:
    {json.dumps(kg, indent=2)}

    """

    sva_gen_prompt += """
    Generate one SVA for each of the provided natural language test plans. 
    Enclose each SVA in triple backticks (```) and prefix it with 'SVA:'. 
    Each SVA should be in the following format:
    
    SVA:
    ```
    @(posedge PCLK) <condition> |-> <consequence>;
    ```

    Use the following examples as a guide:

    SVA:
    ```
    @(posedge PCLK) ((PWDATA >= 230) && (PWDATA <= 255)) |-> (PWDATA >= 205) && (PWDATA <= 255);
    ```
    NL: that when PWDATA is within the range of 230 to 255, in the next cycle PWDATA will remain within the range of 205 to 255. Use the signals 'PCLK' for the clock edge and 'PWDATA' for the data being checked.

    SVA:
    ```
    @(posedge PCLK) (PRESETn) |-> (PWDATA >= 0) && (PWDATA <= 45);
    ```
    NL: that the input data is within the valid range when not in reset. Use the signals 'PRESETn', 'PCLK', and 'PWDATA'.

    Ensure that each SVA is a complete and valid System Verilog assertion.
    """

    if valid_signals:
        sva_gen_prompt += "IMPORTANT: Only use the signal names provided in the 'Valid Signal Names' list above. Do not introduce any new signal names."

    return sva_gen_prompt


def parse_nl_plans(result: str) -> Dict[str, List[str]]:
    nl_plans = {}
    current_signal = None

    for line in result.split('\n'):
        line = line.strip()
        if not line:
            continue

        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                signal, plan = parts
                signal = signal.strip()
                plan = plan.strip()

                if signal not in nl_plans:
                    nl_plans[signal] = []

                nl_plans[signal].append(plan)
                current_signal = signal
        elif current_signal:
            # If there's no colon but we have a current signal, assume it's a continuation of the previous plan
            nl_plans[current_signal][-1] += " " + line

    return nl_plans


def write_svas_to_file(svas: List[str], design_dir: str = None, out_dir: str = None) -> Tuple[List[str], Set[str]]:
    """
    Main orchestration function for the SVA pipeline.
    - If no SVAs are provided, return the valid signal names only.
    - If SVAs are provided, generate a checker module and bind file.

    Args:
        svas (List[str]): List of assertion properties (strings).
        design_dir (str): Path to the directory containing design/SVA files.
        out_dir (str): Output directory for generated files. Defaults to "./_out".

    Returns:
        Tuple[List[str], Set[str]]: List of paths to generated SVA files,
                                    and a set of valid signal names from the design.
    """
    module_interface, valid_signals = None, set()
    chosen_top = None
    design_dir = design_dir or FLAGS.design_dir
    signal_hierarchy = {}

    # Step 1: Try to find an existing property_goldmine.sva
    module_interface, valid_signals = find_existing_sva(design_dir)
    if not module_interface:
        print("Finding existing SVAs to write to")
        module_interface, valid_signals, chosen_top, signal_hierarchy = extract_signals_from_rtl(design_dir)
    else:
        chosen_top = re.search(r'module\s+(\w+)', module_interface).group(1)

    # Step 2: Behavior depending on input
    if not svas:   # no SVAs → return only signals
        return [], valid_signals, signal_hierarchy
    else:          # SVAs exist → generate checker+bind
        file_path = generate_checker_and_bind(svas, module_interface, valid_signals, chosen_top, FLAGS.sva_out_dir)
        return [file_path], valid_signals, signal_hierarchy


def generate_tcl_scripts(sva_file_paths: List[str]) -> List[str]:
    """
    Generate TCL scripts for JasperGold, one for each SVA file.

    Args:
        sva_file_paths (List[str]): Paths to the generated SVA files.

    Returns:
        List[str]: Paths to the generated TCL scripts.
    """
    design_dir = FLAGS.design_dir
    if not os.path.exists(design_dir):
        raise Exception(f"Design directory {design_dir} does not exist")

    # Find the original TCL file
    original_tcl_path = find_original_tcl_file(design_dir)
    if not original_tcl_path:
        raise Exception("Could not find the original TCL file")

    # Read the original TCL content
    with open(original_tcl_path, 'r') as f:
        original_tcl_content = f.read()

    tcl_file_paths = []
    for i, sva_file_path in enumerate(sva_file_paths):
        # Modify the TCL content
        modified_tcl_content = modify_tcl_content(original_tcl_content, sva_file_path)

        tcl_file_path = os.path.join(
            saver.logdir, "tcl_scripts", f"FPV_{os.path.basename(design_dir)}_{i}.tcl"
        )
        os.makedirs(os.path.dirname(tcl_file_path), exist_ok=True)
        with open(tcl_file_path, "w") as f:
            f.write(modified_tcl_content)
        tcl_file_paths.append(tcl_file_path)

    return tcl_file_paths


def modify_tcl_content(original_content: str, new_sva_path: str) -> str:
    """
    Modify the TCL content to use the new SVA file.

    Args:
        original_content (str): Original TCL file content.
        new_sva_path (str): Path to the new SVA file.

    Returns:
        str: Modified TCL content.
    """
    # Replace the property_goldmine.sva file path
    modified_content = re.sub(
        r'(\$\{RTL_PATH\}/bindings\.sva\s*\\\s*)\$\{RTL_PATH\}/property_goldmine\.sva',
        f'\\1{new_sva_path}',
        original_content,
    )

    return modified_content


def run_jaspergold(tcl_file_paths: List[str]) -> List[str]:
    """
    Run JasperGold using the generated TCL scripts for each SVA.

    Args:
        tcl_file_paths (List[str]): Paths to the TCL scripts.

    Returns:
        List[str]: List of paths to JasperGold report files.
    """
    jasper_reports = []
    for i, tcl_file_path in tqdm(
        enumerate(tcl_file_paths),
        total=len(tcl_file_paths),
    ):
        # Create a unique project directory
        project_dir = os.path.join(saver.logdir, 'jgproject', f"jgproject_{i}")
        os.makedirs(project_dir, exist_ok=True)

        jasper_command = f"/<path>/<to>/jg -batch -proj {project_dir} -tcl {tcl_file_path}"

        try:
            result = subprocess.run(
                jasper_command,
                shell=True,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=FLAGS.design_dir,
            )

            report = result.stdout

            print(f"JasperGold for SVA {i} exited with code: {result.returncode}")

            # if result.returncode != 0:
            #     print(f"Warning: JasperGold for SVA {i} returned non-zero exit status.")
            # print(f"Command output:\n{report}")

        except Exception as e:
            print(f"Error running JasperGold for SVA {i}: {str(e)}")
            report = f"Error: {str(e)}\n"

        report_file_path = os.path.join(
            saver.logdir, "jasper_reports", f"jasper_report_{i}.txt"
        )
        os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
        with open(report_file_path, "w") as f:
            f.write(report)
        jasper_reports.append(report_file_path)

        # Optional: Clean up the project directory
        # shutil.rmtree(project_dir)

    return jasper_reports


def analyze_results(
    pdf_stats: dict,
    nl_plans: Dict[str, List[str]],
    svas: List[str],
    jasper_reports: List[str],
    coverage_report: str,
):
    """
    Analyze and print statistics about the generated plans, SVAs, JasperGold results, and coverage report.

    Args:
        pdf_stats (dict): Statistics about the input PDF file.
        nl_plans (Dict[str, List[str]]): Dictionary of generated natural language test plans.
        svas (List[str]): List of generated SVAs.
        jasper_reports (List[str]): List of paths to JasperGold report files.
        coverage_report (str): Coverage report from JasperGold.
    """
    # Generate and print the detailed SVA report
    detailed_report_path = generate_detailed_sva_report(svas, jasper_reports)

    # General statistics
    print("\nGeneral Statistics:")
    print("PDF Statistics:")
    print(f"  Number of pages: {pdf_stats['num_pages']}")
    print(f"  Number of tokens: {pdf_stats['num_tokens']}")
    print(f"  File size: {pdf_stats['file_size']} bytes")

    print("\nNatural Language Test Plans:")
    if nl_plans:
        total_plans = sum(len(plans) for plans in nl_plans.values())
        print(f"  Total number of plans generated: {total_plans}")
        print(f"  Number of signals with plans: {len(nl_plans)}")
        avg_plans_per_signal = total_plans / len(nl_plans) if nl_plans else 0
        print(f"  Average plans per signal: {avg_plans_per_signal:.2f}")

        all_plans = [plan for plans in nl_plans.values() for plan in plans]
        avg_plan_length = (
            sum(len(plan.split()) for plan in all_plans) / len(all_plans)
            if all_plans
            else 0
        )
        print(f"  Average plan length: {avg_plan_length:.2f} words")
    else:
        print(f'nl_plans={nl_plans}')

    print("\nSVAs:")
    print(f"  Number of SVAs generated: {len(svas)}")
    avg_sva_length = sum(len(sva.split()) for sva in svas) / len(svas) if svas else 0
    print(f"  Average SVA length: {avg_sva_length:.2f} words")

    print("\nJasperGold Results Summary:")
    with open(detailed_report_path, 'r') as f:
        df = pd.read_csv(f)

    proven_count = sum(df['Proof Status'] == 'proven')
    cex_count = sum(df['Proof Status'] == 'cex')
    inconclusive_count = sum(df['Proof Status'] == 'inconclusive')
    error_count = sum(df['Proof Status'] == 'error')
    syntax_correct_count = len(svas) - error_count

    print(f"  Total SVAs evaluated: {len(jasper_reports)}")
    print(f"  Proven: {proven_count}")
    print(f"  Counterexample found: {cex_count}")
    print(f"  Inconclusive: {inconclusive_count}")
    print(f"  Errors: {error_count}")
    print(f"  Syntax-correct SVAs: {syntax_correct_count}")

    success_rate = proven_count / len(jasper_reports) if jasper_reports else 0
    print(f"  Success rate: {success_rate:.2%}")

    print("\nDetailed results saved in:")
    print(f"  SVA files: {os.path.join(saver.logdir, 'tbs')}")
    print(f"  Jasper reports: {os.path.join(saver.logdir, 'jasper_reports')}")
    print(f"  Detailed SVA report: {detailed_report_path}")

    print("\nCoverage Report:")
    # coverage_lines = coverage_report.split('\n')
    # try:
    #     start_index = coverage_lines.index("COVERAGE REPORT")
    #     for line in coverage_lines[start_index:]:
    #         print(line)
    # except ValueError:
    #     print("Coverage report format is unexpected. Raw output:")
    #     print(coverage_report)

    # Extract critical coverage metrics
    coverage_metrics = calculate_coverage_metric(coverage_report)

    # Calculate final metrics
    total_assertions = len(svas)
    syntax_correct_assertions = syntax_correct_count
    proven_assertions = proven_count
    syntax_correction_rate = (
        syntax_correct_assertions / total_assertions if total_assertions else 0
    )
    proven_rate = proven_assertions / total_assertions if total_assertions else 0

    # Print final results in tab-separated format
    print("\nFinal Results (Tab-separated for easy copying to spreadsheets):")
    metric_names = [
        "# Assertions",
        "# Syntax Correct Assertions",
        "# Proven Assertions",
        "Syntax Correction Rate",
        "Pass/Proven Rate",
        "Stimuli Statement Coverage",
        "Stimuli Branch Coverage",
        "Stimuli Functional Coverage",
        "Stimuli Toggle Coverage",
        "Stimuli Expression Coverage",
        "COI Statement Coverage",
        "COI Branch Coverage",
        "COI Functional Coverage",
        "COI Toggle Coverage",
        "COI Expression Coverage",
    ]
    print("\t".join(metric_names))

    metric_values = [
        f"{total_assertions}",
        f"{syntax_correct_assertions}",
        f"{proven_assertions}",
        f"{syntax_correction_rate:.4f}",
        f"{proven_rate:.4f}",
        f"{coverage_metrics.get('coverage_stimuli_statement', 0):.4f}",
        f"{coverage_metrics.get('coverage_stimuli_branch', 0):.4f}",
        f"{coverage_metrics.get('coverage_stimuli_functional', 0):.4f}",
        f"{coverage_metrics.get('coverage_stimuli_toggle', 0):.4f}",
        f"{coverage_metrics.get('coverage_stimuli_expression', 0):.4f}",
        f"{coverage_metrics.get('coverage_coi_statement', 0):.4f}",
        f"{coverage_metrics.get('coverage_coi_branch', 0):.4f}",
        f"{coverage_metrics.get('coverage_coi_functional', 0):.4f}",
        f"{coverage_metrics.get('coverage_coi_toggle', 0):.4f}",
        f"{coverage_metrics.get('coverage_coi_expression', 0):.4f}",
    ]
    print("\t".join(metric_values))


def calculate_coverage_metric(jasper_out_str):
    coverage_dict = {
        "stimuli_statement": 0.0,
        "stimuli_branch": 0.0,
        "stimuli_functional": 0.0,
        "stimuli_toggle": 0.0,
        "stimuli_expression": 0.0,
        "coi_statement": 0.0,
        "coi_branch": 0.0,
        "coi_functional": 0.0,
        "coi_toggle": 0.0,
        "coi_expression": 0.0,
    }

    # Extract coverage metrics
    coverage_matches = re.findall(r"(\w+)\|(\w+)\|(\d+\.\d+)", jasper_out_str)
    key_map = {
        ("coi", "statement"): "coi_statement",
        ("coi", "branch"): "coi_branch",
        ("coi", "functional"): "coi_functional",
        ("coi", "toggle"): "coi_toggle",
        ("coi", "expression"): "coi_expression",
        ("stimuli", "statement"): "stimuli_statement",
        ("stimuli", "branch"): "stimuli_branch",
        ("stimuli", "functional"): "stimuli_functional",
        ("stimuli", "toggle"): "stimuli_toggle",
        ("stimuli", "expression"): "stimuli_expression",
    }

    for category, model, value in coverage_matches:
        key = key_map.get((category, model))
        if key:
            coverage_dict[key] = float(value)

    # Initialize metric
    metric = {
        "syntax": 1.0,
        "functionality": 0.0,
        **{f"coverage_{k}": v for k, v in coverage_dict.items()},
    }

    # Check for syntax errors in the output
    if re.search(r"ERROR \(VERI-", jasper_out_str, re.IGNORECASE):
        metric["syntax"] = 0.0
        metric["functionality"] = 0.0
    # Search for proof results in the output

    if re.search(r"syntax error", jasper_out_str, re.IGNORECASE):
        metric["syntax"] = 0.0
        metric["functionality"] = 0.0
    # Search for proof results in the output
    proof_result_match = re.findall(r"\bproofs:[^\n]*", jasper_out_str)
    # coverage_result_match = re.findall(r"\bcoverage:[^\n]*", jasper_out_str)
    # print(f"Proof_result_match: {proof_result_match}")

    if not proof_result_match:
        metric["functionality"] = 0.0
        return metric

    proof_result_list = proof_result_match[-1].split(":")[-1].strip().split()
    # if coverage_result_match:
    # Proceed only if there's a match in coverage_result_match
    # coverage_result_list = coverage_result_match[-1].split(":")[-1].strip().split()

    # Count number of "proven" assertions
    if proof_result_list.count("proven") != 0:
        metric["functionality"] = 1.0

    # if FLAGS.both_cover_and_assertion:
    #   if coverage_result_match:
    #     if coverage_result_list.count("covered") == 0:
    #       metric["functionality"] = 0.0
    #   else:
    #     metric["functionality"] = 0.0

    return metric


def generate_detailed_sva_report(svas: List[str], jasper_reports: List[str]) -> str:
    """
    Generate a detailed report for each SVA, including syntax correctness, FPV status, and error messages.

    Args:
        svas (List[str]): List of generated SVAs.
        jasper_reports (List[str]): List of paths to JasperGold report files.

    Returns:
        str: Path to the generated CSV file containing the detailed report.
    """
    sva_details = []
    syntax_correct_count = 0
    for i, (sva, report_path) in enumerate(zip(svas, jasper_reports)):
        with open(report_path, 'r') as f:
            report_content = f.read()

        proof_status = extract_proof_status(report_content)

        if proof_status != "error":
            syntax_correct_count += 1

        error_message = ""
        if proof_status == "error":
            error_message = extract_short_error_message(
                extract_error_message(report_content)
            )

        sva_details.append(
            {"SVA ID": i, "Proof Status": proof_status, "Error Message": error_message}
        )

    # Create a DataFrame and save it as a CSV
    df = pd.DataFrame(sva_details)
    csv_path = os.path.join(saver.logdir, "sva_details.csv")
    df.to_csv(csv_path, index=False)

    # Print the table
    print("\nDetailed SVA Results:")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

    print(f"\nSyntax-correct SVAs: {syntax_correct_count} out of {len(svas)}")
    print(f"\nDetailed SVA results saved to: {csv_path}")

    return csv_path


def extract_error_message(report_content: str) -> str:
    """
    Extract the first error message from the JasperGold report.

    Args:
        report_content (str): Content of the JasperGold report.

    Returns:
        str: The first error message found, or "Unknown error" if none found.
    """
    error_lines = [line for line in report_content.split('\n') if "ERROR" in line]
    return error_lines[0] if error_lines else "Unknown error"


def log_llm_interaction(prompt: str, response: str, interaction_type: str):
    """
    Log the prompt and response from LLM interactions to a file.

    Args:
        prompt (str): The prompt sent to the LLM.
        response (str): The response received from the LLM.
        interaction_type (str): Type of interaction (e.g., 'NL_Plans', 'SVAs').
    """
    log_file_path = os.path.join(saver.logdir, 'llm_interactions.txt')
    with open(log_file_path, 'a') as f:
        f.write(f"\n\n{'=' * 50}\n")
        f.write(f"Interaction Type: {interaction_type}\n")
        f.write(f"{'=' * 50}\n")
        f.write("Prompt:\n")
        f.write(prompt)
        f.write("\n\nResponse:\n")
        f.write(response)
        f.write("\n\n")


def extract_short_error_message(full_error: str) -> str:
    """
    Extract a short version of the error message.

    Args:
        full_error (str): The full error message.

    Returns:
        str: A shortened version of the error message.
    """
    # Look for the main error description, typically after the last colon
    parts = full_error.split(':')
    if len(parts) > 1:
        return parts[-1].strip()
    return full_error

def find_existing_sva(design_dir: str) -> Tuple[str, Set[str]]:
    """
    Look for an existing property_goldmine.sva file in the design directory.

    Args:
        design_dir (str): Path to the directory containing design and SVA files.

    Returns:
        Tuple[str, Set[str]]: The reconstructed module interface string and a set of valid signal names.
                             If no file is found, returns (None, empty set).
    """
    sva_path = FLAGS.sva_file_path or os.path.join(design_dir, "property_goldmine.sva")
    if not os.path.exists(sva_path):
        return None, set()
    with open(sva_path, "r") as f:
        content = f.read()

    module_match = re.search(r'module\s+(\w+)\s*\((.*?)\);', content, re.DOTALL)
    if not module_match:
        raise ValueError("Could not find module declaration in property_goldmine.sva")

    module_name = module_match.group(1)
    module_interface = f"module {module_name}({module_match.group(2)});"
    valid_signals = extract_signal_names(content)
    return module_interface, valid_signals

def extract_signals_from_rtl(design_dir: str) -> Tuple[str, Set[str], str]:
    """
    New wrapper that keeps original behavior and adds internal/hierarchical extraction
    when cfg.include_internal_signals = True.
    """

    # Use your accurate old parser
    module_interface, top_ports, chosen_top = extract_top_rtl_ports(design_dir)

    # EXACT old behavior unless flag is enabled
    if not FLAGS.include_internal_signals:
        return module_interface, top_ports, chosen_top, {}

    # Extract internal signals (PATCH 2)
    internal_signals = extract_internal_signals_with_pyverilog(design_dir)

    # Extract hierarchical signals (PATCH 3)
    hierarchical_signals = extract_hierarchical_signals(
        design_dir,
        top_module=chosen_top,
        max_depth=FLAGS.hierarichal_signal_depth,
        include_ports=False
    )

    # Create segregated hierarchy
    signal_hierarchy = {"top_ports":top_ports, "internal_signals":internal_signals, "hierarchical_signals":hierarchical_signals}

    if FLAGS.prune_signals:
      # Step 1 — Run basic filtering (PATCH 4)
      internal_signals = filter_signal_set(internal_signals)
      hierarchical_signals = filter_signal_set(hierarchical_signals)

      # Step 2 — Remove propagation of ports inside hierarchy
      hierarchical_signals = prune_hierarchical_signals(hierarchical_signals, top_ports)

      # Step 3 — Remove hierarchical shadows of internal nets
      hierarchical_signals = remove_child_port_shadows(hierarchical_signals, internal_signals)

    # Merge
    valid_signals = set()
    valid_signals |= top_ports
    valid_signals |= internal_signals
    valid_signals |= hierarchical_signals

    # Filter (PATCH 4)
    valid_signals = filter_signal_set(valid_signals)

    return module_interface, valid_signals, chosen_top, signal_hierarchy

def extract_top_rtl_ports(design_dir: str):
    sv_files, v_files = [], []

    for f in os.listdir(design_dir):
        if f.endswith((".sv", ".svh")):
            sv_files.append(os.path.join(design_dir, f))
        elif f.endswith((".v", ".vh")):
            v_files.append(os.path.join(design_dir, f))

    if sv_files:
        return _extract_top_rtl_ports_pyslang(design_dir)

    if v_files:
        return _extract_top_rtl_ports_old(design_dir)

    raise FileNotFoundError("No RTL files found")

def _extract_top_rtl_ports_pyslang(design_dir) -> Tuple[str, Set[str], str]:
    """
    Parse SystemVerilog RTL files using pyslang to extract the top module and its ports.

    Args:
        design_dir (str): Path to the directory containing RTL files.

    Returns:
        Tuple[str, Set[str], str]: The reconstructed module interface string,
                                   a set of valid signal names, and the detected top module name.
    """
    # 1. Match the old glob pattern (including headers)
    rtl_files = []
    for ext in ["*.v", "*.sv", "*.vh", "*.svh"]:
        rtl_files.extend(glob.glob(os.path.join(design_dir, ext)))
    
    if not rtl_files:
        raise FileNotFoundError(f"No RTL files found in {design_dir}")

    compilation = pyslang.Compilation()
    for f in rtl_files:
        # Using fromFile captures the syntax tree for each
        compilation.addSyntaxTree(pyslang.SyntaxTree.fromFile(f))

    top_instances = compilation.getRoot().topInstances
    if not top_instances:
        diag = compilation.getAllDiagnostics()
        raise RuntimeError(f"No module definitions found in RTL. \n{diag}")
    
    # IMPROVEMENT: Try to find a module name that looks like the directory name, 
    # otherwise take the first one. This avoids grabbing 'tb_top' by accident.
    chosen_top = top_instances[0]
    dir_name = os.path.basename(os.path.normpath(design_dir))
    for inst in top_instances:
        if inst.definition.name.lower() in dir_name.lower():
            chosen_top = inst
            break
            
    top_module_name = chosen_top.definition.name
    
    port_names = []
    port_decl_lines = []
    valid_signals = set()

    for port in chosen_top.body.portList:
        p_name = port.name
        direction = port.direction.name.lower()
        
        # Resolve type and handle errors
        p_type_obj = port.type
        if p_type_obj.isError:
             p_type = "logic" # Fallback to standard logic if type is missing
        else:
             p_type = str(p_type_obj)
        
        port_names.append(p_name)
        valid_signals.add(p_name)
        # Construct line: "input logic [7:0] my_signal"
        port_decl_lines.append(f"    {direction} {p_type} {p_name}")

    # Construct the ANSI-style interface
    module_interface = f"module {top_module_name} (\n" + ",\n".join(port_decl_lines) + "\n);"

    return module_interface, valid_signals, top_module_name

def _extract_top_rtl_ports_old(design_dir: str) -> Tuple[str, Set[str], str]:
    """
    Parse RTL files using PyVerilog to extract the top module and its ports.

    Args:
        design_dir (str): Path to the directory containing RTL files.

    Returns:
        Tuple[str, Set[str], str]: The reconstructed module interface string,
                                   a set of valid signal names, and the detected top module name.
    """
    rtl_files = []
    for pat in ["*.v", "*.sv", "*.vh", "*.svh"]:
        rtl_files.extend(glob.glob(os.path.join(design_dir, pat)))
    if not rtl_files:
        raise FileNotFoundError(f"No RTL files found in {design_dir}")

    ast, _ = parse(rtl_files)
    module_defs, instantiated = {}, set()

    def _walk(node):
        if isinstance(node, ModuleDef):
            module_defs[node.name] = node
            for item in node.items or []:
                if isinstance(item, InstanceList) and hasattr(item, "module"):
                    instantiated.add(item.module)
        for c in getattr(node, "children", lambda: [])():
            _walk(c)
    _walk(ast)

    if not module_defs:
        raise RuntimeError("No module definitions found in RTL.")

    candidates = set(module_defs.keys()) - instantiated
    chosen_top = sorted(candidates)[0] if candidates else sorted(module_defs.keys())[0]
    top_def = module_defs[chosen_top]

    port_names, port_decl_lines = [], []

    def _fmt_width(width):
        if width is None:
            return ""
        try:
            msb = getattr(width.msb, "value", None) or getattr(width.msb, "name", None) or str(width.msb)
            lsb = getattr(width.lsb, "value", None) or getattr(width.lsb, "name", None) or str(width.lsb)
            return f"[{msb}:{lsb}]"
        except Exception:
            return ""

    def _decl_from_io(io):
        dir_node = io.first
        direction = dir_node.__class__.__name__.lower()
        width = _fmt_width(getattr(dir_node, "width", None))
        dtype = "logic"
        name = getattr(io, "second", None) or getattr(dir_node, "name", None)
        if hasattr(name, "name"):
            name = name.name
        elif not isinstance(name, str):
            name = str(name)
        width_sp = (width + " ") if width else ""
        return direction, name, f"{direction} {dtype} {width_sp}{name}"

    if top_def.portlist:
        for item in top_def.portlist.ports:
            if isinstance(item, Ioport):
                _, pname, decl = _decl_from_io(item)
                port_names.append(pname)
                port_decl_lines.append(decl + ";")
            else:
                pname = getattr(item, "name", None)
                if hasattr(pname, "name"):
                    pname = pname.name
                if not isinstance(pname, str):
                    pname = str(pname)
                port_names.append(pname)
                port_decl_lines.append(f"input logic {pname}; // direction unknown")

    valid_signals = set(port_names)
    module_interface = f"module {chosen_top}({', '.join(port_names)});"
    return module_interface, valid_signals, chosen_top

def extract_internal_signals_with_pyverilog(design_dir: str) -> Set[str]:
    """
    Extract internal (non-port) RTL signals using PyVerilog.
    This includes wires, regs, logic, integers, and internal declarations.

    Args:
        design_dir (str): Path to directory containing Verilog RTL files.

    Returns:
        Set[str]: internal signal names found inside all modules.
    """
    import glob
    import os

    # Collect all .v and .sv files
    rtl_files = glob.glob(os.path.join(design_dir, "*.v")) + \
                glob.glob(os.path.join(design_dir, "*.sv"))

    if not rtl_files:
        print("[WARN] No RTL files found for internal extraction")
        return set()

    try:
        ast, _ = parse(rtl_files)
    except Exception as e:
        print("[ERROR] PyVerilog failed to parse RTL:", e)
        return set()

    internal_signals = set()

    # Traverse AST for module definitions and internal declarations
    for child in ast.description.definitions:
        if isinstance(child, ModuleDef):

            # Traverse declarations inside each module
            for item in child.items:
                if isinstance(item, Decl):
                    for decl in item.list:
                        # Wires
                        if isinstance(decl, Wire):
                            internal_signals.add(decl.name)
                        # Regs
                        elif isinstance(decl, Reg):
                            internal_signals.add(decl.name)
                        # In some designs PyVerilog stores logic/reg differently
                        elif hasattr(decl, "name"):
                            internal_signals.add(decl.name)

    return internal_signals

def build_module_map(ast):
    """
    Build a map of module_name -> ModuleDef node for quick lookup.
    """
    module_map = {}
    for child in ast.description.definitions:
        if isinstance(child, ModuleDef):
            module_map[child.name] = child
    return module_map


def _collect_decls_from_module(module_node):
    """
    Collect declaration (internal) names and port names from a ModuleDef node.
    Returns (ports_set, internals_set)
    """
    ports = set()
    internals = set()

    # Ports: module_node.portlist may hold port names as Identifier objects
    try:
        # module_node.portlist is a Portlist object with .ports (list of Port)
        if hasattr(module_node, "portlist") and module_node.portlist:
            for p in getattr(module_node.portlist, "ports", []):
                # Port may have .name or .children[0] as Identifier
                try:
                    if hasattr(p, "name") and p.name:
                        ports.add(p.name)
                    elif hasattr(p, "children") and p.children:
                        # fallback
                        first = p.children[0]
                        if hasattr(first, "name"):
                            ports.add(first.name)
                except Exception:
                    pass
    except Exception:
        pass

    # Internal declarations
    for item in getattr(module_node, "items", []):
        if isinstance(item, Decl):
            for decl in getattr(item, "list", []):
                # Many declaration node types have .names or .name attribute
                try:
                    # Wire, Reg
                    if hasattr(decl, "name") and decl.name:
                        # single-name node
                        internals.add(decl.name)
                    else:
                        # some nodes expose .list of Declr
                        for possible in getattr(decl, "list", []):
                            if hasattr(possible, "name"):
                                internals.add(possible.name)
                except Exception:
                    # best-effort
                    try:
                        if hasattr(decl, "names"):
                            for nm in decl.names:
                                if hasattr(nm, "name"):
                                    internals.add(nm.name)
                    except Exception:
                        continue
    return ports, internals


def _find_instances_in_module(module_node):
    """
    Return list of Instance objects found inside the module node (flat).
    """
    instances = []
    for item in getattr(module_node, "items", []):
        if isinstance(item, InstanceList):
            for inst in getattr(item, "instances", []):
                instances.append(inst)
    return instances


def extract_hierarchical_signals(design_dir: str, top_module: str = None, max_depth: int = 3, include_ports: bool = False) -> Set[str]:
    """
    Extract hierarchical signals from design by traversing module instances starting
    from the top module (if provided) or the first module in the AST.

    Returns a set of hierarchical signal names like: top.uart_rx.bit_cnt

    Args:
        design_dir: path to RTL files (same as other extractors)
        top_module: name of the top-level module (optional). If None, pick first module.
        max_depth: maximum hierarchy depth to traverse (prevents explosion)
        include_ports: if True, include each module's ports as well as internals

    Returns:
        Set[str]: hierarchical signal names
    """
    import glob, os
    rtl_files = glob.glob(os.path.join(design_dir, "*.v")) + glob.glob(os.path.join(design_dir, "*.sv"))
    if not rtl_files:
        print("[WARN] No RTL files found for hierarchical extraction")
        return set()

    try:
        ast, _ = parse(rtl_files)
    except Exception as e:
        print("[ERROR] PyVerilog parse failed for hierarchical extraction:", e)
        return set()

    module_map = build_module_map(ast)
    if not module_map:
        return set()

    # choose top
    if top_module is None:
        # Heuristic: prefer module with same name as design_dir basename or first module
        guessed = os.path.basename(os.path.normpath(design_dir))
        if guessed in module_map:
            top_module = guessed
        else:
            # fallback to first module in map
            top_module = next(iter(module_map.keys()))

    # BFS/DFS traversal
    hierarchical_signals = set()
    visited = set()

    def traverse(module_name: str, inst_path: str, depth: int):
        """
        Recursive traversal. inst_path is the hierarchical prefix (e.g., top.u1.u2).
        """
        if depth < 0:
            return
        if module_name not in module_map:
            return

        unique_node_key = f"{module_name}::{inst_path}"
        if unique_node_key in visited:
            return
        visited.add(unique_node_key)

        module_node = module_map[module_name]
        ports, internals = _collect_decls_from_module(module_node)

        # include internals
        for sig in internals:
            hierarchical_signals.add(f"{inst_path}.{sig}" if inst_path else f"{module_name}.{sig}")

        # optionally include ports
        if include_ports:
            for p in ports:
                hierarchical_signals.add(f"{inst_path}.{p}" if inst_path else f"{module_name}.{p}")

        # find instances and traverse
        instances = _find_instances_in_module(module_node)
        for inst in instances:
            try:
                # Instance has .module in some pyverilog versions or .module in InstanceList item
                child_module_type = getattr(inst, "module", None) or getattr(inst, "module", None)
                child_inst_name = getattr(inst, "name", None) or getattr(inst, "instname", None)
                if child_module_type is None:
                    # try to extract from inst.module if it's an Identifier
                    if hasattr(inst, "module") and hasattr(inst.module, "name"):
                        child_module_type = inst.module.name
                # instance names sometimes are strings; if not, skip
                if child_inst_name is None:
                    child_inst_name = getattr(inst, "name", None) or getattr(inst, "instname", None)
                if child_inst_name is None or child_module_type is None:
                    continue
                # build new path
                new_path = f"{inst_path}.{child_inst_name}" if inst_path else f"{child_inst_name}"
                traverse(child_module_type, new_path, depth - 1)
            except Exception:
                # best-effort: skip instance on error
                continue

    traverse(top_module, top_module, max_depth)
    return hierarchical_signals

def prune_hierarchical_signals(hier_set: Set[str], top_ports: Set[str]) -> Set[str]:
    """
    Remove redundant hierarchical signals such as:
        - uart2bus_top.clock  (duplicate of top-level port 'clock')
        - uart2bus_top.uart1.clock (propagated clock)
        - any hierarchical signal whose final basename is a top-level port
    """
    pruned = set()

    for sig in hier_set:
        base = sig.split('.')[-1]

        # Remove hierarchical duplicates of ports
        if base in top_ports:
            continue

        # Remove hierarchical clock/reset propagation
        if base.lower() in ("clock", "clk", "reset", "rst", "rst_n"):
            continue

        pruned.add(sig)

    return pruned


def remove_child_port_shadows(hier_set: Set[str], internal_set: Set[str]) -> Set[str]:
    """
    Remove hierarchical signals that only shadow internal signals.
    Example:
      - uart2bus_top.uart1.uart_tx_1.bit_count
      - bit_count   <-- internal signal already exists

    Keep only the shorter (non-hierarchical) version.
    """
    pruned = set()
    internal_basenames = {s.split('.')[-1] for s in internal_set}

    for sig in hier_set:
        base = sig.split('.')[-1]
        if base in internal_basenames:
            continue  # reduce duplication
        pruned.add(sig)

    return pruned

def filter_signal_name(name: str) -> bool:
    """
    Return True if the signal name should be kept, False if it should be filtered out.
    This removes tool-generated names, temporary nets, synthetic identifiers,
    and other noise that confuses the LLM.
    """

    # Remove empty or None
    if not name or not isinstance(name, str):
        return False

    # Strip hierarchical prefixes for pattern matching
    base = name.split('.')[-1]

    # 1. Remove auto-generated synthesis names (common patterns)
    autogenerated_patterns = [
        r'^_+',              # _tmp, __net, ___abc123
        r'^n\d+$',           # n1234
        r'^tmp\d+$',         # tmp12
        r'^genblk\d+',       # genblk1, genblk3_
        r'^_zz_.*',          # _zz_123 (chisel/verilator)
        r'^_\d+$',           # _42
        r'^unnamed.*',       # unnamed blocks
    ]
    for pattern in autogenerated_patterns:
        if re.match(pattern, base):
            return False

    # 2. Remove number-only nets
    if re.match(r'^\d+$', base):
        return False

    # 3. Remove Verilog parameters/constants
    const_patterns = [
        r'^[A-Z0-9_]+$',  # ALL CAPS → usually parameters
        r'^D_.*',         # parameter naming (D_BAUD_LIMIT)
    ]
    for pattern in const_patterns:
        if re.match(pattern, base):
            return False

    # 4. Remove numeric-index hierarchical expansions
    if '[' in name or ']' in name:
        # we do NOT remove bits like signal[3], but we remove genblk[3].foo
        if "genblk" in name:
            return False

    # 5. Remove clock/reset variants IF user only wants top-level clocks
    # (optional — keep for now)
    ignore_list = [
        "rst_n", "resetn", "reset", "clk", "clock"
    ]
    if base.lower() in ignore_list:
        # Keep them for now; comment this line to filter them
        pass

    # Otherwise keep this signal
    return True



def filter_signal_set(signals: Set[str]) -> Set[str]:
    """
    Apply filter_signal_name() to an entire set. Removes duplicates automatically.
    """
    return {sig.split(".")[-1] for sig in signals if filter_signal_name(sig)}


def generate_checker_and_bind(
    svas: List[str], 
    module_interface: str, 
    valid_signals: Set[str], 
    chosen_top: str, 
    out_dir: str
) -> str:
    """
    Generate a checker module and bind file for the given SVAs.

    Args:
        svas (List[str]): List of assertion properties (strings).
        module_interface (str): The reconstructed module interface declaration.
        valid_signals (Set[str]): Set of valid signal names extracted from the top module.
        chosen_top (str): Name of the detected top module.
        out_dir (str): Directory where the generated SVA file will be written.

    Returns:
        str: Path to the generated checker and bind .sva file.
    """
    identifiers = set()
    for sva in svas:
        tokens = re.findall(r"\b[a-zA-Z_]\w*\b", sva)
        identifiers.update(tokens)

    keywords = {"posedge","negedge","disable","iff","property","assert","endproperty",
                "logic","begin","end","if","else","inside","not","or","and"}
    identifiers -= keywords

    missing_signals = identifiers - valid_signals
    checker_name = f"{chosen_top}_checker"
    checker_ports = list(valid_signals) + sorted(missing_signals)
    checker_port_list = ", ".join(checker_ports)

    checker_content = f"{module_interface}\n"
    for sig in sorted(missing_signals):
        checker_content += f"  input logic {sig}; // internal (bind)\n"
    checker_content += "\n"
    for i, sva in enumerate(svas):
        checker_content += f"  property a{i};\n{sva}\n  endproperty\n"
        checker_content += f"  assert_a{i}: assert property(a{i});\n\n"
    checker_content += "endmodule\n\n"

    bind_lines = [f".{p}({p})" for p in checker_ports]
    bind_block = f"bind {chosen_top} {checker_name} checker_inst (\n  " + ",\n  ".join(bind_lines) + "\n);\n"

    os.makedirs(out_dir, exist_ok=True)
    sva_file_path = os.path.join(out_dir, f"{chosen_top}_checker_bind.sva")
    with open(sva_file_path, "w") as f:
        f.write(checker_content)
        f.write(bind_block)
    return sva_file_path

def extract_signal_names(module_interface: str) -> Set[str]:
    """
    Extract signal names from the module interface.

    Args:
        module_interface (str): The module interface declaration.

    Returns:
        Set[str]: A set of signal names found in the interface.
    """
    # Regular expression to match signal declarations
    signal_pattern = (
        r'\b(?:input|output|inout)\s+(?:reg|wire)?\s*(?:\[[^\]]+\])?\s*(\w+)'
    )

    # Find all matches
    matches = re.findall(signal_pattern, module_interface)

    # Extract signal names from matches
    signal_names = set(matches)

    return signal_names

