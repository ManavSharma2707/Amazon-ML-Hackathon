# Business Entity Resolution — code package

Amazon ML Challenge 2026, Business Entity Resolution. This folder is the
exact content of the submission zip's `code/` directory: all source lives
under `src/` (configs, scripts and notebooks included).

## Environment

- Python 3.10+ (developed on 3.13).
- Kaggle: P100 or 2x T4 (16 GB each), fp16 only, SDPA attention (no bf16, no flash-attn 2).
- `pip install -r requirements.txt`

## Base models (downloaded once, internet ON, then run fully offline)

- `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0)
- `Qwen/Qwen3-4B` (Apache-2.0)
- `Qwen/Qwen3-Reranker-0.6B` (Apache-2.0, fallback)
- LightGBM (MIT)

No external data, APIs, geocoders or lookups are used anywhere in this
pipeline; every inference notebook runs with internet OFF, loading weights
from a local folder or Kaggle dataset.

## Running

```bash
# from code/business_entity_resolution/
python -m src.metrics                       # unit-test the official scorer
pytest -q src/tests/

# training (fills in as modules land)
bash src/scripts/run_train.sh

# inference: produces output/matching_results.tsv + output/candidate_pairs.tsv
bash src/scripts/run_predict.sh

# pre-flight checks (from code/business_entity_resolution/)
python -m src.check_outputs --out-dir ../../output --test-dir ../../student_resource/dataset/test

# official validator (from student_resource/)
python3 utils/validate_submission.py \
  --matching ../output/matching_results.tsv \
  --candidate ../output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Expected runtimes

Filled in as notebooks are run (see `progress.md`, not part of this zip).

## Licences

| Component | Licence |
|---|---|
| Qwen3-Embedding-0.6B, Qwen3-4B, Qwen3-Reranker-0.6B | Apache-2.0 |
| LightGBM | MIT |
| rapidfuzz, faiss-cpu, bitsandbytes | MIT |
| pandas, numpy, scipy, scikit-learn | BSD |
| sentence-transformers, transformers, peft, accelerate | Apache-2.0 |
| anyascii | ISC |

No GPL dependencies (e.g. `unidecode`) are used.
