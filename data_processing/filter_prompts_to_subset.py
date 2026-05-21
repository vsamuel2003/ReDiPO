"""Restrict a DiversityTuning HF dataset (output of Script 0) to only the prompts
present in our subset JSONL.  Saves to a new on-disk path so the original is untouched.
"""
import json
from argparse import ArgumentParser
from datasets import load_from_disk

parser = ArgumentParser()
parser.add_argument("--input_dataset",  required=True, help="Path to HF dataset on disk (Script 0 output)")
parser.add_argument("--subset_jsonl",   required=True, help="Path to our subset JSONL (field: 'prompt')")
parser.add_argument("--output_dataset", required=True, help="Where to write the filtered dataset")
args = parser.parse_args()

subset_prompts = set()
with open(args.subset_jsonl) as fh:
    for line in fh:
        row = json.loads(line)
        subset_prompts.add(row["prompt"].strip())

print(f"Loaded {len(subset_prompts)} unique prompts from {args.subset_jsonl}")

ds = load_from_disk(args.input_dataset)

for key in ds:
    orig = len(ds[key])
    ds[key] = ds[key].filter(
        lambda ex: f"{ex['post_title']}\n{ex['post_text']}".strip() in subset_prompts
    )
    print(f"  {key}: {len(ds[key])} / {orig} rows kept")

kept  = sum(len(ds[k]) for k in ds)
total = sum(1 for _ in subset_prompts)  # for pct display
print(f"Total rows in filtered dataset: {kept}")

ds.save_to_disk(args.output_dataset)
print(f"Saved to {args.output_dataset}")
