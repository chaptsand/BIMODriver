"""
Extract biological text embeddings using BioBERT (dmis-lab/biobert-base-cased-v1.2).

This script encodes either:
1. Multi-scale context statements (self, neighbor, collective interaction) into
   statement feature tensors (e.g., PAN-CANCER_statement_features.pt).
2. Gene Ontology functional descriptions (BP, MF, CC) into GO feature tensors
   (e.g., PAN-CANCER_go_features.pt).
"""

import os
import sys
import argparse
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_bert_embeddings(texts, model, tokenizer, device, max_length=512):
    """Batch-extract [CLS] token embeddings from BioBERT."""
    clean_texts = [str(t).strip() if pd.notna(t) and str(t).strip() else "none" for t in texts]
    inputs = tokenizer(
        clean_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Return [CLS] token embedding
    return outputs.last_hidden_state[:, 0, :].cpu()


def process_statements(csv_path, output_path, model_name_or_path, device, batch_size=32):
    """Encode multi-scale statements: self, neighbor, and together statements."""
    print(f"Loading statements from: {csv_path}")
    data = pd.read_csv(csv_path)

    statement_fields = ["self_statement", "neighbor_statement", "together_statement"]
    for field in statement_fields:
        if field not in data.columns:
            raise KeyError(f"Expected column '{field}' in {csv_path}")

    print(f"Loading BioBERT model: {model_name_or_path}...")
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    model = BertModel.from_pretrained(model_name_or_path).to(device)
    model.eval()

    num_samples = len(data)
    print(f"Encoding statements for {num_samples} genes...")

    all_tensors = []
    for i in range(num_samples):
        row = data.iloc[i]
        texts = [str(row[field]) if pd.notna(row[field]) else "" for field in statement_fields]
        embs = get_bert_embeddings(texts, model, tokenizer, device, max_length=512)
        # embs shape: (3, 768) -> concatenate to (2304,)
        combined = torch.cat([embs[0], embs[1], embs[2]], dim=0)
        all_tensors.append(combined)

        if (i + 1) % 1000 == 0 or (i + 1) == num_samples:
            print(f"  Processed {i + 1}/{num_samples} genes")

    final_tensor = torch.stack(all_tensors)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(final_tensor, output_path)
    print(f"Successfully saved statement embeddings to: {output_path} (shape: {final_tensor.shape})")


def process_go_features(csv_path, output_path, model_name_or_path, device):
    """Encode GO functional descriptions (BP, MF, CC)."""
    import torch_geometric

    print(f"Loading GO descriptions from: {csv_path}")
    data = pd.read_csv(csv_path)

    bp_features = ["BP_Feature1", "BP_Feature2", "BP_Feature3"]
    mf_features = ["MF_Feature1", "MF_Feature2", "MF_Feature3"]
    cc_features = ["CC_Feature1", "CC_Feature2", "CC_Feature3"]

    print(f"Loading BioBERT model: {model_name_or_path}...")
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    model = BertModel.from_pretrained(model_name_or_path).to(device)
    model.eval()

    weights = torch.linspace(1.0, 0.5, steps=3)  # [1.0, 0.75, 0.5]

    def weighted_mean(embeddings, w):
        weighted = embeddings * w.view(-1, 1)
        return weighted.sum(dim=0) / w.sum()

    num_samples = len(data)
    print(f"Encoding GO features for {num_samples} genes...")

    results = []
    for idx, row in data.iterrows():
        bp_texts = [row.get(f, "none") for f in bp_features]
        bp_embs = get_bert_embeddings(bp_texts, model, tokenizer, device)
        bp_mean = weighted_mean(bp_embs, weights)

        mf_texts = [row.get(f, "none") for f in mf_features]
        mf_embs = get_bert_embeddings(mf_texts, model, tokenizer, device)
        mf_mean = weighted_mean(mf_embs, weights)

        cc_texts = [row.get(f, "none") for f in cc_features]
        cc_embs = get_bert_embeddings(cc_texts, model, tokenizer, device)
        cc_mean = weighted_mean(cc_embs, weights)

        all_mean = (bp_mean + mf_mean + cc_mean) / 3.0
        combined = torch.cat([bp_mean, mf_mean, cc_mean, all_mean])
        results.append(combined)

        if (idx + 1) % 1000 == 0 or (idx + 1) == num_samples:
            print(f"  Processed {idx + 1}/{num_samples} genes")

    final_data = torch.stack(results)
    dataframe = torch_geometric.data.Data()
    dataframe['BP'] = final_data[:, 0:768]
    dataframe['MF'] = final_data[:, 768:1536]
    dataframe['CC'] = final_data[:, 1536:2304]
    dataframe['All'] = final_data[:, 2304:3072]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(dataframe, output_path)
    print(f"Successfully saved GO embeddings to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="BioBERT Feature Extraction for BIMODriver")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["statement", "go"],
        default="statement",
        help="Feature extraction target: 'statement' (multi-scale context) or 'go' (Gene Ontology aspects)"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Path to input CSV containing generated descriptions"
    )
    parser.add_argument(
        "--output_pt",
        type=str,
        default=None,
        help="Output path for the serialized PyTorch tensor"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="dmis-lab/biobert-base-cased-v1.2",
        help="Hugging Face model identifier or local checkpoint directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device ('cuda' or 'cpu')"
    )

    args = parser.parse_args()

    if args.mode == "statement":
        input_csv = args.input_csv or os.path.join(BASE_DIR, "LLM", "gemma2_PAN-CANCER_cpdb_new_neiber2.csv")
        output_pt = args.output_pt or os.path.join(BASE_DIR, "data", "CPDB", "PAN-CANCER_statement_features.pt")
        process_statements(input_csv, output_pt, args.model_name_or_path, args.device)
    else:
        input_csv = args.input_csv or os.path.join(BASE_DIR, "LLM", "PAN-CANCER_go_features.csv")
        output_pt = args.output_pt or os.path.join(BASE_DIR, "data", "CPDB", "PAN-CANCER_go_features.pt")
        process_go_features(input_csv, output_pt, args.model_name_or_path, args.device)


if __name__ == "__main__":
    main()
