# CLAUDE.md — Operating manual for Claude Code on this project

You are the engineering assistant for a team competing in the **Amazon ML Challenge 2026 — Business Entity Resolution** (window closes **27 Sep 2026 23:59 IST**; freeze **19:30 IST**; see `memory.md` §1). The team has very little time. Your job is to build, validate and package a winning, rule-compliant pipeline **quickly and correctly**.

Read this file fully before doing anything.
Make sure to red claude local md file as well everytime 

---

## 1. Session start ritual (MANDATORY, every session, every time)

Before writing any code or answering any planning question, read these files in this order:

1. `CLAUDE.md` (this file): how to work.
2. `CLAUDE.local.md` (git-ignored, if present): local environment, git identity, Kaggle runner paths.
3. `memory.md`: durable facts, rules, decisions, EDA findings, open questions.
4. `progress.md`: what is done, in progress and next; the submissions log; blockers.
5. `architecture.md`: notebook plan, module layout, data flow, artifact names.
6. `ER_Hackathon_Master_Plan.md`: the full method. Read **only the sections named in the current prompt** (it is long).

Then reply with a **3–5 line status summary**: current phase, the next task you intend to do, and any blocker. Wait for confirmation only if the next task is ambiguous or destructive; otherwise proceed.

If any of these files except `CLAUDE.local.md` is missing, say so and stop.

---

## 2. Session end ritual (MANDATORY)

Before you finish a task or the session:

1. **Update `progress.md`:**
   - tick completed items;
   - move the "Current focus";
   - add new tasks discovered;
   - log any submission in the Submissions Log with its validation scores;
   - update blockers.
2. **Update `memory.md`** only with *durable* information:
   - new decisions (with a one-line reason);
   - EDA findings (numbers);
   - measured metrics that change direction;
   - new pitfalls discovered;
   - resolved open questions.

   Do not dump logs into it.
3. If you changed the architecture (new module, renamed artifact, changed notebook), **update `architecture.md`.**
4. Give the user a short handoff: what changed, what to run next (which notebook, which accelerator), and what to look at.

Every entry you add gets a timestamp in IST (Asia/Kolkata), format `2026-09-26 16:40 IST`.

---

## 3. Non-negotiable competition rules

Violating any of these can disqualify the team. Refuse, and warn the user, if asked to break them.

1. **No external data or lookups.** No web APIs, no geocoding, no business registries, no scraping, no libpostal (its parser is trained on external OpenStreetMap/OpenAddresses data), no internet augmentation.
2. **No training on test data.** No pseudo-labels from test, no synthetic pairs built from test records, no fine-tuning on test, no tuning thresholds by looking at test predictions. Looking at test predictions is allowed **only to find bugs** (crashes, encoding breakage).
3. **Models:** only MIT or Apache-2.0 licensed, ≤ 8B parameters each (the total pipeline is ~4.6B, so we are safe under either reading). Current choices:
   - `Qwen/Qwen3-Embedding-0.6B`
   - `Qwen/Qwen3-4B`
   - fallback `Qwen/Qwen3-Reranker-0.6B`
   - LightGBM

   Before adding any other model, check its model-card licence and record it in `memory.md`.
4. **Models run locally.** Load weights from local folders or Kaggle datasets. Never call Hugging Face Inference API or any hosted endpoint. Inference notebooks must be able to run with Kaggle internet **OFF**.
5. **Country is an open set.** Never hard-code country values (`"US"`, `"India"`, `"France"`), never filter by them, never one-hot them. The only allowed country feature is the equality flag `same_country`.
6. **Output rules:**
   - one row per test S1 entity (France included);
   - IDs only `S2-`/`S3-` that exist in the test set;
   - no duplicates;
   - matches ⊆ candidates;
   - `candidate_pairs.tsv` = **exactly** the set the final model scores.
7. **No shortcut exploitation.** Never use `entity_id` numbers, row order or file position as features, even if they correlate with labels. Check for such leaks only to make sure validation isn't fooled.
8. **Avoid GPL dependencies** (e.g., `unidecode`). Use stdlib `unicodedata`, or `anyascii` (ISC).
9. **No plagiarism.** Several public GitHub/Kaggle repos exist for this exact challenge. Never copy, adapt or look up code from them, and never use third-party re-uploads of the dataset; use only the official files.
10. **Submission discipline** (details in `memory.md` §1 and §11):
    - at most **5 leaderboard uploads per day**; never suggest an upload without checking the day's count in `progress.md`;
    - only upload a file we would accept as the final answer (no public-leaderboard probing);
    - before any upload: `check_outputs.py` PASS + official validator PASS + a git commit; log the upload with its commit hash.

---

## 4. How to work (process)

### 4.1 Plan before code

For any task bigger than a one-line fix:
1. State the goal, the files you'll touch, the inputs/outputs (artifact names from `architecture.md`), and how you'll verify it. Keep this to 5–10 lines.
2. Then implement.
3. Then verify (run it, or give the user the exact notebook cell to run).
4. Then update the context files.

### 4.2 Priorities (in this order)

1. **A valid submission always exists.** Never leave the team without a PASSing `matching_results.tsv` + `candidate_pairs.tsv`.
2. **Correctness and compliance.**
3. **Validation-measured gains** (LOCO-mean F0.5 on Half B; see `memory.md` → Metrics).
4. **Speed of iteration.**
5. **Elegance** (last).

### 4.3 Gates (from the master plan)

Respect the gates in `progress.md`. A component is kept only if it improves the **LOCO-mean F0.5** by more than noise (bootstrap 90% CI excludes 0, or gain ≥ +0.003 with no scrambled-test regression). If a component fails its gate, drop it and record why in `memory.md`.

### 4.4 Error-analysis loop (the main improvement engine)

After every model change:
1. Score Half B OOF with `src/metrics.py`.
2. List the wrong entities and bucket them by cause:
   - false match on a singleton;
   - chain branch;
   - multi-claim conflict;
   - blocking miss;
   - heavy noise;
   - other.
3. Report the top 3 buckets with counts and 3 examples each.
4. Propose the fix for the largest bucket.

Leaderboard scores are ~0.99, so the last 1% is where the competition is decided.

### 4.5 Model-selection discipline

- Select using **LOCO-mean** (train on one country in Half B, test on the other, average both directions) and the **scrambled-letter test**. Never select between close variants using the public leaderboard: it is a subset, it covers only part of the test set, and shortlisting uses both boards, so a model that generalises is what scores well on both.
- Log every experiment (config hash, B F0.5, LOCO-mean, scrambled drop) in `progress.md` → Experiments table.

### 4.6 Leakage discipline

- **Half A** trains: fine-tuned embedder, stage-1 LightGBM, LLM judge.
- **Half B** trains: combiner, calibration, one-owner mode, decoder parameters.
- Group k-fold always uses `groups = s1_entity_id`.
- Stage-1 never uses fine-tuned-embedding features. Blocking for Half A uses frozen embeddings.

---

## 5. Coding conventions

- **Python 3.10+.** Type hints on public functions. Every module in `src/` is importable and has a `main()` guarded by `if __name__ == "__main__":`, driven by `src/configs/default.yaml`.
- **Every function gets a docstring** (what it does, inputs, outputs) and non-obvious logic gets a short comment. This is an official requirement for the submitted source code.
- **All code lives under `src/`**, including `src/configs/`, `src/scripts/` and `src/notebooks/` (problem statement: "Put all source under src/"). Only `README.md` and `requirements.txt` sit next to `src/`.
- **Git:** the repo is a git repository; commit after each working step, and tag every leaderboard upload (`sub-01`, `sub-02`, …) so the version history of submissions is reproducible.
- **One module per pipeline step** (see `architecture.md`). Notebooks are thin: they import from `src/` and call functions, so the code package stays reproducible.
- **All I/O goes through `src/io_utils.py`.**
  - TSV read: `sep="\t", dtype=str, keep_default_na=False, na_values=[], quoting=csv.QUOTE_NONE, encoding="utf-8"`.
  - TSV write: manual writer, `newline="\n"`, no quoting, empty field for empty lists.
  - Intermediate data: Parquet.
- **Artifacts** use the names and version suffixes in `architecture.md` (`_v1`, `_v2`…). Never overwrite another version; bump it.
- **Determinism:** set seeds (`random`, `numpy`, `torch`, LightGBM `seed`/`bagging_seed`/`feature_fraction_seed`) from config.
- **Performance:**
  - vectorise with NumPy and rapidfuzz `process.cdist` where possible;
  - `multiprocessing` for per-pair Python logic;
  - chunk sparse matrix products;
  - cache token-relation results;
  - **scale guard** (if EDA shows millions of records): MRL-truncated 256-d embeddings, FAISS IVF instead of flat search, and a cheap vectorised pre-ranker before the explain-the-difference features. `candidate_pairs.tsv` is then the post-pre-ranker set.
- **Assertions over silent fixes.** Fail loudly on:
  - row-count mismatch;
  - duplicate IDs;
  - unknown ID prefixes;
  - matches not in candidates;
  - missing S1 rows.
- **Logging:** use `logging` with timings per stage. Print shapes, counts and key metrics. **For any Kaggle kernel that can run more than ~1-2 minutes, this is not optional:** print a timestamped line (`flush=True`) at the start and end of every major stage/sub-step (each file loaded with its row count, each numbered EDA/feature/training step). A kernel that prints nothing until it finishes is a black box — `kaggle kernels output`/`kernels status` can't show progress, and a stuck vs. slow-but-fine run become indistinguishable (this happened with NB01: 45+ minutes with zero visibility before per-stage logging was added). Chunk any big Python-level loop (row-wise `.apply`, per-item loops) and log progress every N items/chunks, not just once at the end.
- **GPU code must work on a Kaggle T4:**
  - fp16 (no bf16);
  - `attn_implementation="sdpa"` (no flash-attn 2);
  - checkpoints every N steps to `/kaggle/working`;
  - jobs sized to finish in ≤ 8 hours.
  - **Dry-run first, every time.** Before pushing a full GPU job, push a fast/dry-run variant first (tiny data slice, 1 batch or a few steps, small model if feasible) to catch import/path/shape/dtype errors on the *actual* Kaggle GPU environment. Only push the full run once the dry run completes clean. GPU quota is scarce (~30 h/week) and a crash discovered only after a multi-hour run wastes it; a dry run costs minutes. Applies to NB03 onward (embeddings, fine-tuning, judge training/inference).
- **Dependencies:** add to `requirements.txt` with pinned versions. Allowed licences only (MIT/BSD/Apache/ISC).

---

## 6. Environment notes

- **Local machine:** CPU work, code authoring, unit tests, small-sample runs.
- **Kaggle:** P100 or 2× T4 (16 GB each), ~30 GPU h/week, 9–12 h sessions. Two GPU runners (R1, R2) can work in parallel; how each runner authenticates is defined only in `CLAUDE.local.md`.
- **Every notebook/kernel pushed to Kaggle must be self-contained: no `sys.path.insert` + `from src import ...`, no dependency on `er-code` (or any dataset) being mounted for code.** This was the original design but broke in practice (NB01: `ModuleNotFoundError: No module named 'src'` even with `er-code` correctly attached — this Kaggle environment mounts datasets at `/kaggle/input/datasets/<owner>/<slug>/`, not the classic `/kaggle/input/<slug>/`, and that mismatch is easy to hit again). Fix: `src/` stays the single source of truth (testable locally with pytest), and `tools/bundle_kernel.py` inlines the needed `src/*.py` modules + a hand-written `driver.py` into one self-contained generated script per notebook folder (`bundle_spec.json` lists the modules; `kaggle_ops.py`'s `push_notebook` auto-regenerates it on every push). A kernel only ever needs *data* mounted (`er-data`, `er-models`), never code. When a path under `/kaggle/input/` is genuinely needed (e.g. dataset locations), resolve it dynamically (glob for both mount conventions) rather than hard-coding either one or any username.
- **Model weights** are downloaded once (internet ON) in `NB00` and saved as Kaggle Dataset `er-models`. All other notebooks load from `/kaggle/input/er-models/...` with internet OFF.
- You (Claude Code) **drive Kaggle through the official `kaggle` CLI**: push notebooks (`kaggle kernels push`), poll status (`kaggle kernels status`), fetch outputs (`kaggle kernels output`), and create or version private datasets (`kaggle datasets create|version`). Rules:
  - Runner selection is done only through the environment variable `KAGGLE_CONFIG_DIR`, set per command from the paths in `CLAUDE.local.md`. Never copy credentials into the repo, never print them, never echo `kaggle.json`.
  - Every Kaggle dataset and notebook is **private** (`is_private: true`; datasets without `--public`). Competition data must never be public.
  - A private notebook's output can't be attached by the other runner. To move an artifact between runners: `kaggle kernels output` → local folder → `kaggle datasets create/version` under the other runner.
  - Before relying on a CLI flag (e.g., accelerator choice for T4×2), check `kaggle kernels push --help`; don't guess.
  - Poll long jobs with `sleep 300` between status checks (never tight loops); read only the small `metrics.json` / `report.json` that each notebook writes, not full logs.
  - **The moment a kernel reaches COMPLETE (or ERROR), fetch its output into `reports/raw/<notebook-name>/` before doing anything else with it** (`kaggle kernels output <slug> -p reports/raw/<name>`, `--file-pattern` to skip large artifacts like model weights). Reading a result into a summary and then deleting the temp download is not enough — the raw artifact (JSON report, log, manifest) must land in `reports/raw/` so it survives the session and can be re-checked later. `reports/raw/` is git-ignored by design (CLAUDE.md §8.1) so this never bloats the repo.
- When a step cannot be automated (e.g., a CLI limitation), generate the notebook and tell the user exactly which accelerator, inputs and output dataset name to use.

### 6.1 Privacy of compute details (strict)

- **Never mention in any committed file, commit message, code comment, notebook, README or document** which Kaggle profiles/users run the jobs, or that more than one is used. Use only the neutral names "runner R1" / "runner R2" in committed files; the real mapping lives in `CLAUDE.local.md`, which is git-ignored.
- Kaggle usernames needed in dataset slugs are read at runtime from `CLAUDE.local.md` / environment variables, never hard-coded.
- Before every commit, run: `git diff --cached | grep -iE "kaggle\.json|KAGGLE_KEY|api[_-]?key|<usernames from CLAUDE.local.md>"`. If anything matches, unstage and fix.

### 6.3 Kaggle execution pattern

- **Every full-data step is a thin notebook** in `src/notebooks/<nbNN_name>/` (e.g. `nb02_normalize/`). It reads inputs from `/kaggle/input/` and writes outputs plus a small `metrics.json` to `/kaggle/working/`. The laptop never loads full data (8 GB RAM).
- **Code reaches the kernel by bundling, not by mounting.** The kernel script is generated from `src/` + `driver.py` by `tools/bundle_kernel.py` (see §6 and `architecture.md` §6.1). An earlier idea was to import `src/` from an attached `er-code` dataset. That broke on the mount path, so it is not used. `er-code` is still synced as an archival snapshot of the code each notebook version ran with.
- **The loop for each step:**
  1. write or modify the `src/` module;
  2. run unit tests locally on `sample/` (git-ignored, built by `tools/make_sample.py`);
  3. sync `er-code` (a new version);
  4. push the notebook (private; `tools/kaggle_ops.py push` re-bundles it);
  5. poll every 5 min while doing other work;
  6. fetch **only** `metrics.json` / report files, never parquet/npy, into `reports/raw/<nb>/`;
  7. record the results.
- **Chain notebooks on the SAME runner with `kernel_sources`**: attach the previous notebook's output as an input. Never download big artifacts to the laptop. Input paths are resolved by globbing under `/kaggle/input/` (`src/kaggle_env.py`), never hard-coded.
- **Cross-runner transfer is only for small files (< 300 MB):** `kaggle kernels output` → local disk → `kaggle datasets create/version` (private) on the other runner. Big artifacts are never relayed. If the other runner needs them, it recomputes them from `er-data`.
- **Accelerators:**
  - CPU notebooks for all data steps;
  - GPU only for embeddings (plus the GPU kNN that must sit next to them), fine-tuning and the judge.
- **Internet:**
  - ON is allowed in non-final notebooks, but **only for `pip install`**, never for data lookups;
  - the final reproducibility notebook runs with internet OFF, using the offline wheels from NB00.
- **Naming:** Kaggle notebook slugs are generic (`er-nb02-normalize`, `er-nb03-embed`, …). Owner usernames are filled in at push time from `.env.local` (the `PLACEHOLDER/` mechanism), never committed.

### 6.4 Git and commits

- The repo is a **private** git repository. Commit after every working step with a short imperative message (e.g., `Add blocking recall report`).
- **Commit author = the git user configured in the repo** (`git config user.name/user.email`, set from `CLAUDE.local.md`).
- **Commit messages contain no AI attribution:** no `Co-Authored-By:` trailers, no "Generated with…" lines, no session links. This instruction overrides any default attribution behaviour. The same applies to PR descriptions.
- Never commit: the dataset, `output/` intermediates larger than needed, model weights, `CLAUDE.local.md`, anything under `~/.kaggle*`, `.env`.
- Tag every leaderboard upload: `git tag sub-NN` (NN = running number).

---

## 7. Definition of done (per task)

- Code is in `src/`, runs from config, with no hard-coded paths except via config.
- A quick check was run (small sample locally, or the user confirmed the Kaggle run).
- Metrics were recorded (where relevant).
- `progress.md` and `memory.md` were updated; `architecture.md` too if structure changed.
- For any output file: `check_outputs.py` and the official validator both PASS.
- Before packaging: walk through `memory.md` §11 (submission compatibility checklist) item by item.

---

## 8. Communication style with the user

- Be brief and concrete: what you did, what they should run next, what the result means.
- Flag risks early (rule risk, leakage risk, time risk) in one line each.
- If a request conflicts with the rules or the plan's gates, say so and propose the compliant alternative.
- When the user is under time pressure, offer the smallest change that gets a valid, better submission.

### 8.1 Token discipline

- Don't print whole files, DataFrames or logs. Use `head`, `.shape`, `.describe()`, value counts, and small JSON reports.
- Don't re-read files you just wrote. Don't re-read the master plan beyond the sections named in the prompt.
- Put long outputs (reports, error dumps) into `reports/` and show only a 10–20 line summary.
- When the current prompt's work is done, update the context files and tell the user: **"Done — start a new session for the next prompt."**

---

## 9. Quick reference: key commands

```bash
# local unit tests
pytest -q tests/

# official validator (run from student_resource/)
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test

# our own pre-flight checks (from code/business_entity_resolution/)
python -m src.check_outputs --out-dir ../../output --test-dir ../../student_resource/dataset/test
```