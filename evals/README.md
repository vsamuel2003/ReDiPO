# unified_evals

A self-contained, plug-and-play evaluation suite for four benchmarks:
- **MTBench** — multi-turn instruction following (GPT-4 judge)
- **AlpacaEval** — instruction following win-rate (gpt-5.4-2026-03-05 annotator)
- **Novelty-bench** — response diversity / novelty (Skywork reward model)
- **HarmBench** — safety / jailbreak resistance (HarmBench Llama-2-13b classifier, transformers-only, no vLLM)

All four share a common model-loading and prompt-building layer that supports:
- **Base models** — no chat template; Qwen models get raw instructions, others get `Question:/Answer:` scaffolding
- **Instruct models** — `tokenizer.apply_chat_template()` with no injected system prompt
- **LoRA models** (trained from instruct) — same as instruct, with PEFT adapter loaded on top

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # for HarmBench hash-check behaviors
```

Set your API keys before running MTBench or AlpacaEval:
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...  # only if using Claude as judge
```

## Quick Start

```bash
# Run a single eval
./mtbench/run_mtbench.sh meta-llama/Llama-3-8B-Instruct --model-type instruct
./alpaca_eval_suite/run_alpaca.sh meta-llama/Llama-3-8B-Instruct --model-type instruct
./novelty_bench/run_novelty.sh meta-llama/Llama-3-8B-Instruct --model-type instruct
./harmbench/run_harmbench.sh meta-llama/Llama-3-8B-Instruct --model-type instruct \
    --test-cases-path /path/to/test_cases.json

# Run all four evals
./run_all.sh meta-llama/Llama-3-8B-Instruct --model-type instruct \
    --test-cases-path /path/to/test_cases.json
```

## LoRA Models

Pass `--model-type lora --lora /path/to/adapter`. The `<model_path>` positional
argument should be the **base model** the adapter was trained from.

```bash
./run_all.sh meta-llama/Llama-3-8B-Instruct \
    --model-type lora \
    --lora /path/to/my_lora_adapter \
    --test-cases-path /path/to/test_cases.json
```

The LoRA adapter directory must contain `adapter_config.json` and
`adapter_model.safetensors` (standard PEFT save format).

## Base Models

```bash
./run_all.sh meta-llama/Llama-3-8B \
    --model-type base \
    --test-cases-path /path/to/test_cases.json
```

- Qwen base models receive the raw instruction text unchanged.
- All other base models receive `Question: {q}\nAnswer:` (single-turn) or
  alternating Q/A blocks (multi-turn for MTBench).
- No system prompt is injected for any model type.

## AlpacaEval Annotator

The default annotator is `weighted_alpaca_eval_gpt5_4`, which uses `gpt-5.4-2026-03-05`
and is self-contained under `alpaca_eval_suite/configs/`. It requires no changes to the
vendored `alpaca_eval/` directory and works with the pip-installed `alpaca-eval` package.

To use a different annotator:
- **By short name** — place a `configs.yaml` under `alpaca_eval_suite/configs/<name>/` and
  pass `--annotator <name>`. The config is resolved to that directory automatically.
- **By path** — pass any string containing `/` or ending in `.yaml` and it is used as-is
  (absolute or relative).

The first annotator API call in each run is logged to `<output-dir>/first_annotation.json`
for debugging (request payload + raw response with logprobs). Use `--first-call-log` to
override the path.

HarmBench classifier inference uses `transformers.pipeline` only — there is no vLLM
dependency anywhere in this harness.

## Resume

All four suites support mid-run resume:

| Suite | Resume key |
|---|---|
| MTBench answers | `question_id` |
| MTBench judgments | `(question_id, model, judge_template, turn)` |
| AlpacaEval generation | `instruction` text |
| AlpacaEval annotation | built-in alpaca_eval annotation cache |
| HarmBench generation | `(behavior_id, test_case)` |
| HarmBench evaluation | `behavior_id` (fully-labelled behaviors skipped) |
| Novelty-bench all stages | `id` |

Just re-run the same command and previously completed items are skipped automatically.

## Directory Structure

```
unified_evals/
├── requirements.txt
├── run_all.sh                      # Run all 4 evals
├── common/
│   ├── model_loader.py             # Unified HF + PEFT loader
│   ├── prompt_builder.py           # Chat template / base-model formatter
│   └── generate.py                 # Batched generation utilities
├── mtbench/
│   ├── data/                       # MTBench questions, judge prompts, ref answers
│   ├── gen_model_answer.py
│   ├── gen_judgment.py
│   ├── show_result.py
│   ├── common.py                   # MTBench-specific utilities (no fastchat deps)
│   └── run_mtbench.sh
├── alpaca_eval_suite/
│   ├── configs/                    # Self-contained annotator configs (no vendored alpaca_eval needed)
│   │   ├── weighted_alpaca_eval_gpt5_4/configs.yaml   # default: gpt-5.4-2026-03-05
│   │   └── alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt  # shared prompt template
│   ├── generate_outputs.py         # HF inference on tatsu-lab/alpaca_eval (checkpointed)
│   ├── run_evaluation.py           # Calls alpaca_eval annotator (logs first call)
│   └── run_alpaca.sh
├── novelty_bench/
│   ├── data/                       # curated.jsonl, wildchat-1k.jsonl
│   ├── inference.py                # HF inference (replaces TransformersService)
│   ├── partition.py                # DeBERTa equivalence partitioning
│   ├── score.py                    # Skywork reward scoring
│   ├── summarize.py                # Final metrics
│   └── run_novelty.sh
└── harmbench/
    ├── data/behavior_datasets/     # HarmBench behavior CSVs
    ├── generate_completions.py     # HF inference
    ├── evaluate_completions.py     # HarmBench classifier (HF, no vLLM)
    ├── eval_utils.py               # Classifier prompts + scoring utilities
    └── run_harmbench.sh
```
