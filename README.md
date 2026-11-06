# quality-contamination-eval

An [Inspect AI](https://inspect.aisi.org.uk/) task and analysis script that measure how much of a model's QuALITY score survives without the story, and whether what survives depends on the original answer letters or the original answer options. The design is in [context/project-overview.md](context/project-overview.md).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"     # or: pip install -e ".[dev]"
source .venv/Scripts/activate                 # Windows Git Bash; Linux/macOS: .venv/bin/activate

mkdir -p data
curl -L -o data/QuALITY.v1.0.1.htmlstripped.dev \
  https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev
```

The editable install matters: the task file imports `quality_probe`, and Inspect runs it from inside that folder.

Each provider reads its own key: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `<PROVIDER>_API_KEY` and `<PROVIDER>_BASE_URL` for `openai-api/<provider>/<model>`.

## Running

```bash
# One condition, smoke-sized
inspect eval quality_probe/task.py@quality -T story=false \
  --model anthropic/claude-haiku-5-5 --limit 20

# Context check plus the full matrix. Re-run the same command to resume.
python study.py --models anthropic/claude-haiku-5-5,openai/gpt-6-luna,ollama/llama3.1 \
  --log-dir logs/study-1

# Report from logs only (no network): report.md, summary.json, results.csv, borrowed_options.csv
python analysis.py logs/study-1

# Browse every prompt, completion and score
inspect view --log-dir logs/study-1
```

If a run is interrupted, `analysis.py` refuses the directory until the same command has been re-run to completion. The re-run reuses completed samples, then removes the logs the interrupted run left behind.

`--limit` takes the first N questions, which are mostly from one or two stories. For a quick estimate across stories, use `--stories 30 --per-story 5`, which picks random stories and questions from the seed. Both are marked `[partial]`.

**Letters-only conditions.** `--letters-only` runs three extra conditions with the option text hidden, so there is nothing to eliminate: `full_letters_only`, `no_story_letters_only` and `full_letters_reordered` (letters listed in a random order). Point it at an existing `--log-dir`, with the same `--stories`, `--per-story` or `--limit`, to add them to that run's report. The report compares accuracy with a letter-habit baseline: what the model's letter preferences would score by luck.

```bash
python study.py --models anthropic/claude-haiku-5-5 --stories 30 --per-story 5 --letters-only --log-dir logs/haiku-30x5
python analysis.py logs/haiku-30x5
```

**Probes.** `--probes` runs options only (no story, no question), a prompt asking directly for the remembered letter, and verbatim recall, which asks for a removed wrong option word for word and compares it with a decoy option from the same story.

**Paraphrased options.** `paraphrase.py` generates reworded options with a model and saves them beside the dataset, then `--paraphrased` runs `no_story_paraphrased` and `full_paraphrased`. Review the reworded options before trusting a result.

```bash
python paraphrase.py --model anthropic/claude-haiku-5-5 --stories 30 --per-story 5 --log-dir logs/paraphrase-generation
python study.py --models anthropic/claude-haiku-5-5 --stories 30 --per-story 5 --paraphrased --log-dir logs/haiku-30x5
```

**Control set.** Write new questions in the same JSONL format as QuALITY, about the same stories (same `article_id` and `article`), and run the main matrix on them with `--data` and a separate `--log-dir`. Then compare the two runs. The difference-of-differences it reports is the main result:

```bash
python study.py --models anthropic/claude-haiku-5-5 --data data/control.jsonl --log-dir logs/control
python analysis.py logs/control
python compare.py logs/haiku-full logs/control
```

The rules for reading all of this are drafted in [context/decision-rules.md](context/decision-rules.md).

Use a separate `--log-dir` for each `--limit`, because a limited run and a full run in one directory count as two logs for the same condition, and `analysis.py` refuses that.

Pass per-model arguments with `-M <model>@key=value`:

```bash
python study.py --models vllm/org/model --log-dir logs/study-1 \
  -M vllm/org/model@max_model_len=16384 -M vllm/org/model@reasoning_parser=deepseek_r1
```

**Local servers.** Every story must fit in the context window. Ollama's OpenAI-compatible endpoint ignores per-request context settings, so start the server with `OLLAMA_CONTEXT_LENGTH=16384`. The context check stops the study for any model whose server cut the prompt. After you fix the server, re-run the same command and the failed check is run again.

**Re-scoring.** After changing the parser, re-score each log with no API calls, then re-run the analysis. Run this from the repo root. `--overwrite` stops Inspect prompting and keeps one log per condition, and `--scorer` is required because Inspect can't find a packaged scorer from the name it records in the log:

```bash
inspect score <log>.eval --action overwrite --overwrite --scorer quality_probe/scorer.py@quality_scorer
python analysis.py logs/study-1
```

## Tests

```bash
pytest                                      # everything, offline, no API keys (about 18 minutes)
pytest --ignore=tests/test_end_to_end.py    # skips the full-dataset fake studies (under a minute)
```

The tests fail immediately if the dataset is missing, rather than skipping.

## Where this differs from the overview

Each of these came from checking the overview against Inspect 0.3.287's behaviour.

| Overview | This repo | Why |
|---|---|---|
| Reference regex uses `\W` | `[^\w\n]` plus `_` | `\W` crosses newlines, so `ANSWER:\nA` would parse. It also misses `_ANSWER: A_` emphasis. |
| Regex letter class `[a-e]` | `[a-z]`, then checked against the offered letters | So `ANSWER: A` followed by `ANSWER: Z` is a `parse_fail`, not a silent fallback to `A`. |
| `eval_set(tasks, model=models, …)` | Each task carries its model through `task_with(…, model=…)` | A task's `max_tokens` overrides a model's, so per-model overrides need per-model tasks. This adds a `max_tokens` argument to both `@task`s. |
| Context-check logs and study logs share a directory | Same, with `log_dir_allow_dirty=True` | `eval_set` otherwise rejects logs from tasks outside the current set. |
| `inspect score --action overwrite` | Also needs `--overwrite` and `--scorer` | Without `--overwrite` Inspect prompts, and its default writes a second log file. Without `--scorer` it can't resolve the scorer. |
| "Re-run the same command to resume" | Same, and `study.py` then removes the interrupted run's `started` logs | Inspect keeps them, and `analysis.py` must refuse any log that isn't `success`, so a resumed directory could never be reported. |
| `.env` isn't mentioned | `python study.py` loads `.env` | Inspect reads it only through its own CLI. |
| `truncated_output` is `stop_reason == "max_tokens"` | Also `model_length` | Inspect uses `model_length` when the context window runs out. |
| Inspect's `system_message()` is implied | A plain insert of the system message | `system_message()` applies a template using every metadata field. |
| Not specified | `max_tokens` 4096 for Haiku, Sonnet and Opus 5.5 | They think by default, and thinking counts against `max_tokens`. A direct `inspect eval` still uses 512 unless you pass `-T max_tokens=4096`. |

This repo also adds to the overview: the letters-only conditions, the probes, paraphrased options, `compare.py`, and `--stories` with `--per-story`, which are marked `[partial]` like `--limit`.

`analysis.py` also adds checks: two logs for one model and condition, mixed seeds, and mixed dataset files each stop the report. Fake runs are labelled `[fake:…]` in the report.

A model with only a failed context check and no study logs is named as excluded, and the other models are still reported. A model with study logs and no passing check stops the whole report.
