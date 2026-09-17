# Biological Knowledge Generation via LLMs and Semantic Embeddings

This directory contains the pipeline for generating biological knowledge-guided textual descriptions and extracting gene semantic representations:

1. **`LLM-go-part.py`**:
   Prompts the local LLM (e.g., Gemma-2 via Ollama) to generate functional gene descriptions across three Gene Ontology aspects: Biological Process (BP), Cellular Component (CC), and Molecular Function (MF).

2. **`LLM-satment.py`**:
   Generates multi-scale context statements (self, local network neighbors, and collective interactions) for each gene using prompt templates and network topology.

3. **`extract_embeddings.py`**:
   Encodes the generated biological text into dense semantic representations using BioBERT (`dmis-lab/biobert-base-cased-v1.2`), producing precomputed feature tensors (e.g., `PAN-CANCER_statement_features.pt` and `PAN-CANCER_go_features.pt`).

4. **Reference Gene & Prompt Files**:
   - `contxt_prompt.txt`, `go_prompt.txt`: Domain knowledge-guided prompts.
   - `cpdb_node_names.txt`, `string_node_names.txt`: Ordered gene symbol lists mapped to the CPDB and STRING networks.
   - `796true.txt`, `2187false.txt`: Benchmark reference driver and non-driver gene lists.
   - `PAN-CANCER_go_features.csv`: Processed Gene Ontology annotation features for pan-cancer genes.
