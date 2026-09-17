# -*- coding: utf-8 -*-
"""Generate the BERT-base statement tensor used by the Table A16 experiment."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import BertModel, BertTokenizer


STATEMENT_COLUMNS = (
    "self_statement",
    "neighbor_statement",
    "together_statement",
)


REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT = (
    REPO_ROOT / "implement" / "CANCER_go_features3.csv"
    if (REPO_ROOT / "implement" / "CANCER_go_features3.csv").exists()
    else REPO_ROOT / "data" / "CPDB" / "CANCER_go_features3.csv"
)
_DEFAULT_OUTPUT = REPO_ROOT / "data" / "CPDB" / "PAN-CANCER_statement_features_bert_base.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode BIMODriver statement features with BERT-base-cased."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="CSV containing self/neighbor/together statement columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Destination for the N x 2304 statement tensor.",
    )
    parser.add_argument(
        "--model",
        default="./bert/google-bert_bert-base-uncased",
        help="Hugging Face model id or a local BERT-base-cased directory.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def load_statements(csv_path):
    frame = pd.read_csv(csv_path, encoding="utf-8")
    missing = [column for column in STATEMENT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing statement columns: {missing}")

    statements = frame.loc[:, STATEMENT_COLUMNS].fillna("").astype(str)
    return statements.to_numpy().reshape(-1).tolist(), len(frame)


def encode_statements(texts, model, tokenizer, device, batch_size, max_length):
    encoded_batches = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            output = model(**tokenized)
        encoded_batches.append(output.last_hidden_state[:, 0, :].cpu())
        completed = min(start + batch_size, len(texts))
        print(f"Encoded {completed}/{len(texts)} statements")
    return torch.cat(encoded_batches, dim=0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device}")

    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertModel.from_pretrained(args.model).to(device)
    model.eval()

    texts, row_count = load_statements(args.input)
    embeddings = encode_statements(
        texts,
        model,
        tokenizer,
        device,
        args.batch_size,
        args.max_length,
    )

    # CSV rows are flattened in self/neighbor/together order. Restoring that
    # order produces the N x 2304 tensor consumed by main_A16_bert_base.py.
    statement_tensor = embeddings.reshape(row_count, len(STATEMENT_COLUMNS), 768)
    statement_tensor = statement_tensor.reshape(row_count, 2304).contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(statement_tensor, args.output)
    print(f"Saved {tuple(statement_tensor.shape)} tensor to {args.output}")


if __name__ == "__main__":
    main()
