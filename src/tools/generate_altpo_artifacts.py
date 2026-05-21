#!/usr/bin/env python3
"""Faithful AltPO alternate-answer generation for DUET/RWKU.

This is a dataset-generalized version of AltPO's original generate.py.
It does not read DualCF artifacts and does not call build_altpo_artifact.py.

It writes JSONL rows with both:
  - sub_answer: original AltPO field
  - alternate: repo-compatible alias used by DPO/alternate datasets
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


# Llama-2 AltPO prompt.
INST_QAS_INSTR = (
    "Now write another version of the answer with some alternate plausible facts "
    "that change answer details.\n"
)
INST_QAS_TEMPLATE_QUERY = (
    "[INST] Question: {question}\nAnswer:{answer}\n"
    + INST_QAS_INSTR
    + " [/INST]"
    + " Alternate Answer :"
)

# Llama-3 AltPO prompt. Use this for Llama-3.1-8B-Instruct too.
INST_QAS_LLAMA3_INSTR = (
    "Now pretend you are making things up. Write another answer to the question "
    "that is of a different template than the given answer and changes all facts "
    "from what are introduced in the given answer (changed answers must be "
    "plausible while being inconsistent with given answer). Ensure that your "
    "alternate answer is a plausible response to the question and doesn't change "
    "any details mentioned in question and only introduces changes to all the "
    "facts introduced answer."
)
INST_QAS_LLAMA3_TEMPLATE_QUERY = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful AI assistant<|eot_id|>\n"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Question: {question}\nAnswer: {answer}\n"
    + INST_QAS_LLAMA3_INSTR
    + "<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|> \n\n"
    "Alternate Answer :"
)

DEFAULT_UNTIL = ["Question:", "Question: ", "Q: ", "Q:"]
HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface")


def str_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value)
    if value == "" or value.lower() in {"none", "null"}:
        return None
    return value


def get_prompt_template(name: str) -> str:
    if name == "llama3_altpo":
        return INST_QAS_LLAMA3_TEMPLATE_QUERY
    if name == "llama2_altpo":
        return INST_QAS_TEMPLATE_QUERY
    raise ValueError(f"Unsupported prompt template: {name}")


def custom_format(prompt: str, example: Dict[str, Any]) -> str:
    for key, value in example.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return {key: [row.get(key) for row in batch] for key in batch[0].keys()}


def tok_batch_encode(
    strings: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    padding_side: str = "left",
    left_truncate_len: Optional[int] = None,
    truncation: bool = False,
):
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    encoding = tokenizer(
        strings,
        truncation=truncation,
        padding="longest",
        return_tensors="pt",
    )
    if left_truncate_len:
        encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
        encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
    tokenizer.padding_side = old_padding_side
    return encoding["input_ids"], encoding["attention_mask"]


def tok_decode(tokens: Iterable[int], tokenizer: transformers.PreTrainedTokenizer) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=True)


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Stop once a decoded stop sequence appears in all batch generations."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]
        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]
        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
        for index, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[index] = self.sequence in lookback_tokens_batch[index]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            MultiTokenEOSCriteria(seq, tokenizer, initial_decoder_input_length, batch_size)
            for seq in stop_sequences
        ]
    )


def load_model_and_tokenizer(args: argparse.Namespace):
    model_subfolder = str_or_none(args.model_subfolder)
    tokenizer_subfolder = str_or_none(args.tokenizer_subfolder)

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.torch_dtype]

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "cache_dir": HF_HOME,
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "none":
        model_kwargs["attn_implementation"] = args.attn_implementation
    if model_subfolder is not None:
        model_kwargs["subfolder"] = model_subfolder

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    tokenizer_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "cache_dir": HF_HOME,
    }
    if tokenizer_subfolder is not None:
        tokenizer_kwargs["subfolder"] = tokenizer_subfolder
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, **tokenizer_kwargs)

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Unable to set pad token for tokenizer")

    if args.device_map.lower() == "none":
        model = model.to(args.device)
    model.eval()
    return model, tokenizer


def load_source_dataset(args: argparse.Namespace):
    dataset_kwargs: Dict[str, Any] = {"path": args.dataset_path, "split": args.split}
    dataset_name = str_or_none(args.dataset_name)
    data_files = str_or_none(args.data_files)
    if dataset_name is not None:
        dataset_kwargs["name"] = dataset_name
    if data_files is not None:
        dataset_kwargs["data_files"] = data_files

    dataset = load_dataset(**dataset_kwargs)

    if args.max_examples and args.max_examples > 0:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    def add_source_row_id(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        example = dict(example)
        example["__source_row_id"] = idx
        return example

    return dataset.map(add_source_row_id, with_indices=True)


def build_prompts(
    batch: Dict[str, List[Any]],
    prompt: str,
    question_key: str,
    answer_key: str,
) -> List[str]:
    outputs: List[str] = []
    for index in range(len(batch[question_key])):
        outputs.append(
            custom_format(
                prompt,
                {
                    "question": batch[question_key][index],
                    "answer": batch[answer_key][index],
                },
            )
        )
    return outputs


def clean_stop_terms(text: str, stop_terms: List[str]) -> str:
    for term in stop_terms:
        if term:
            text = text.split(term)[0]
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate faithful AltPO sub_answer artifacts.")

    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--data-files", default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument("--question-key", required=True)
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--index-key", default="index")

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-subfolder", default=None)
    parser.add_argument("--tokenizer-subfolder", default=None)
    parser.add_argument(
        "--prompt-template",
        choices=["llama3_altpo", "llama2_altpo"],
        default="llama3_altpo",
    )

    parser.add_argument("--output-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0)

    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--until", nargs="*", default=DEFAULT_UNTIL)
    parser.add_argument("--left-truncate-len", type=int, default=None)
    parser.add_argument("--padding-side", default="left")
    parser.add_argument("--truncation", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")

    # Compatibility only. These fields let the existing alternate-metadata
    # dataset configs read the artifact. DPO/AltPO does not use them.
    parser.add_argument(
        "--write-compat-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--preserve-extra", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args)
    dataset = load_source_dataset(args)
    prompt = get_prompt_template(args.prompt_template)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
    }
    generation_kwargs = {
        key: value for key, value in generation_kwargs.items() if value is not None
    }

    rows_written = 0
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    with output_path.open("w", encoding="utf-8") as output_handle:
        for batch in tqdm(data_loader, desc="altpo-generate"):
            inputs = build_prompts(batch, prompt, args.question_key, args.answer_key)
            input_ids, attention_mask = tok_batch_encode(
                inputs,
                tokenizer,
                padding_side=args.padding_side,
                left_truncate_len=args.left_truncate_len,
                truncation=args.truncation,
            )
            batch_size = input_ids.shape[0]

            for repeat in range(args.repeats):
                stopping_criteria = stop_sequences_criteria(
                    tokenizer, args.until, input_ids.shape[1], batch_size
                )
                with torch.no_grad():
                    output = model.generate(
                        input_ids=input_ids.to(args.device),
                        attention_mask=attention_mask.to(args.device),
                        stopping_criteria=stopping_criteria,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        **generation_kwargs,
                    )

                out_toks_list = output.tolist()
                for batch_index, continuation_tokens in enumerate(out_toks_list):
                    continuation_tokens = continuation_tokens[input_ids.shape[1] :]
                    sub_answer = clean_stop_terms(
                        tok_decode(continuation_tokens, tokenizer), args.until
                    )

                    question = str(batch[args.question_key][batch_index])
                    answer = str(batch[args.answer_key][batch_index])
                    source_row_id = int(batch["__source_row_id"][batch_index])
                    source_index_raw = batch.get(args.index_key, [None] * batch_size)[
                        batch_index
                    ]
                    try:
                        source_index = int(source_index_raw)
                    except Exception:
                        source_index = source_row_id

                    unique_index = source_index * args.repeats + repeat

                    if args.preserve_extra:
                        row = {
                            key: batch[key][batch_index]
                            for key in batch.keys()
                            if not key.startswith("__")
                            and key not in {args.question_key, args.answer_key}
                        }
                    else:
                        row = {}

                    row.update(
                        {
                            "index": unique_index,
                            "source_index": source_index,
                            "altpo_repeat": repeat,
                            "question": question,
                            args.question_key: question,
                            "answer": answer,
                            "sub_answer": sub_answer,
                            "alternate": sub_answer,
                            "altpo_seed": args.seed,
                            "altpo_prompt_template": args.prompt_template,
                            "altpo_generation_model": args.model_path,
                            "altpo_generation_model_subfolder": str_or_none(
                                args.model_subfolder
                            ),
                        }
                    )

                    if args.write_compat_metadata:
                        row.setdefault("difficulty_score", 0.0)
                        row.setdefault("attribution_score", 0.0)
                        row.setdefault("rarity_score", 0.0)

                    json.dump(row, output_handle, ensure_ascii=False)
                    output_handle.write("\n")
                    rows_written += 1

    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary = {
        "output_path": str(output_path),
        "rows_written": rows_written,
        "source_rows": len(dataset),
        "repeats": args.repeats,
        "seed": args.seed,
        "prompt_template": args.prompt_template,
        "model_path": args.model_path,
        "model_subfolder": str_or_none(args.model_subfolder),
        "dataset_path": args.dataset_path,
        "dataset_name": str_or_none(args.dataset_name),
        "split": args.split,
        "question_key": args.question_key,
        "answer_key": args.answer_key,
        "generation_kwargs": generation_kwargs,
        "until": args.until,
    }
    with summary_path.open("w", encoding="utf-8") as summary_handle:
        json.dump(summary, summary_handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
