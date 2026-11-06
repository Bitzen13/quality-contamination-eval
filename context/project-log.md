# Project log

A record of what was built, run, found and decided on 2026-11-05. For what happens next, see [next-steps.md](next-steps.md).

## How the goal changed

| Stage | The question |
|---|---|
| Start | How much of a model's QuALITY score survives without the story, and does what survives depend on the original letters or options? |
| Midway | Has the model seen the answers? |
| End | **Did training on this material raise the model's score on this benchmark?** |

The study in the overview (S1 to S3) hasn't been run. Everything below is the build, plus small runs that are not final results.

## What was built

**From the overview:**
- `quality_probe/`: data loader and guardrails, the shuffle and swap conditions, prompts, scorer, tasks and test fakes
- `study.py`: context check, then the condition matrix through Inspect's `eval_set`
- `analysis.py`: checks, metrics, story-level bootstrap and reports

**Added beyond the overview:**
- `--stories` and `--per-story`: random stories and questions from a fixed seed, because `--limit` only takes questions in story order
- `--letters-only`: `full_letters_only`, `no_story_letters_only` and `full_letters_reordered`, with a letter-habit baseline
- `--probes`: `no_story_options_only`, `no_story_letters_recall`, `full_letters_recall` and verbatim recall
- `--paraphrased` and `paraphrase.py`: reworded options (built and tested with fakes, never run on a real model)
- `compare.py`: QuALITY vs control set (tested with fakes, and needs rework for the new question)
- `context/decision-rules.md`: draft rules for reading the results
- `context/next-steps.md`: the to-do list

**Built, then removed at your request:** a results web page (`results.py`, `results.bat`). `inspect view --log-dir logs` shows every run instead.

## Tests

All 208 tests passed on the last full run, which takes about 20 minutes. They run offline with no API keys. After that run, only report text, docs and test names changed, and the recall tests passed again.

## Runs

| Run | Model | Questions | Conditions | Calls | Purpose |
|---|---|---|---|---|---|
| `smoke-haiku` | Haiku 5.5 | First 20 (2 stories) | Main 5 | About 100 | Pipeline check |
| `smoke-resume` | Haiku 5.5 | First 30 (2 stories) | Main 5 | About 150 | Interrupt-and-resume check |
| `smoke-cli` | Haiku 5.5 | First 20 (2 stories) | `no_story` | 20 | Direct `inspect eval` command |
| `haiku-30x5` | Haiku 5.5 | 150 (30 stories × 5) | Main 5, letters-only, probes | About 1,800 | **The only run to draw on** |
| Undersized check (temp folder) | `dolphin-llama3:8b` on Ollama at 2048 tokens | Context check only | 1 | 1 | Check that a cut-off server fails |
| `smoke-local` | `dolphin-llama3:8b` | None finished | None | None | Stopped at your request |

That's about 2,070 Haiku calls and roughly 60 cents, by my estimate. Your Anthropic console has the exact cost.

## Results (`haiku-30x5`)

These are preliminary: one model, one seed, 150 of 2,086 questions. Intervals are 95%, from a story-level bootstrap. There were no non-answers.

| Condition | Accuracy | Hard questions |
|---|---|---|
| `full` | 0.900 (0.833–0.953) | 0.903 (0.815–0.974) |
| `full_shuffled` | 0.907 (0.853–0.953) | 0.889 (0.808–0.959) |
| `no_story` | 0.487 (0.407–0.567) | 0.403 (0.310–0.500) |
| `no_story_shuffled` | 0.467 (0.380–0.553) | 0.361 (0.257–0.467) |
| `no_story_swapped` | 0.780 (0.707–0.847) | 0.764 (0.658–0.860) |
| `no_story_options_only` | 0.400 (0.333–0.467) | 0.292 (0.200–0.386) |
| `full_letters_only` | 0.240 (0.173–0.307) | 0.194 (0.106–0.288) |
| `no_story_letters_only` | 0.280 (0.213–0.347) | 0.250 (0.167–0.333) |
| `full_letters_reordered` | 0.293 (0.213–0.380) | 0.236 (0.152–0.322) |
| `no_story_letters_recall` | 0.273 (0.187–0.360) | 0.208 (0.122–0.307) |
| `full_letters_recall` | 0.253 (0.200–0.307) | 0.236 (0.141–0.338) |

| Comparison | Difference |
|---|---|
| Gap (`full` − `no_story`) | +0.413 (0.313 to 0.513) |
| Letter probe (`no_story` − `no_story_shuffled`) | +0.020 (−0.033 to 0.073) |
| Option probe (`no_story` − `no_story_swapped`) | −0.293 (−0.393 to −0.193) |
| Shuffle control (`full` − `full_shuffled`) | −0.007 (−0.040 to 0.027) |
| Letters only: story − no story | −0.040 (−0.113 to 0.033) |
| Letters only: listed − reordered | −0.053 (−0.120 to 0.013) |
| Letters only: recall prompt − guess prompt | −0.007 (−0.113 to 0.100) |
| Question clues (`no_story` − `no_story_options_only`) | +0.087 (0.000 to 0.167) |

| Hidden-answer condition | Expected from letter habits | Above that |
|---|---|---|
| `full_letters_only` | 0.239 | +0.001 (−0.053 to 0.055) |
| `no_story_letters_only` | 0.247 | +0.033 (−0.014 to 0.082) |
| `full_letters_reordered` | 0.249 | +0.044 (−0.012 to 0.104) |
| `no_story_letters_recall` | 0.263 | +0.011 (−0.069 to 0.092) |
| `full_letters_recall` | 0.258 | −0.004 (−0.058 to 0.052) |

| Verbatim recall (110 options of 5+ words) | Removed option | Decoy | Difference |
|---|---|---|---|
| Exact match | 0.009 | 0.000 | +0.009 (0.000 to 0.029) |
| Near match | 0.036 | 0.000 | +0.036 (0.009 to 0.073) |
| Mean similarity | 0.390 | 0.289 | +0.101 (0.070 to 0.134) |

## What was learned

- **Reading matters a lot.** The score falls from 0.90 to 0.49 without the story.
- **Most of the unexplained no-story score needs no question.** About 15 of its 24 points above chance remain with only the answer options. The likely reason is answers that give themselves away, but memorized option text would look the same.
- **No sign of a memorized answer key on this sample.** Small effects aren't ruled out.
- **With answers hidden, Haiku mostly picks a favourite letter.** With the guess prompt that's A, even when the letters are reordered.
- **Early smoke numbers misled.** The first 20 questions scored 0.90 without the story, mostly because 18 of them came from one easy story (16 of 18 right, against 7 of 12 on the next story).
- **More stories narrow the intervals more than more questions per story,** because questions from one story succeed or fail together.
- **Repeat runs differ slightly.** 0–2 of the first 20 answers changed between runs, because Anthropic no longer accepts a temperature setting.
- **The swap condition doesn't work as designed.** Borrowed answers are often off-topic, so the right answer stands out and accuracy rises.
- **Verbatim recall needs the control set.** It beat the decoy, but the closest matches could be worked out from the prompt, and on average 40% of a hidden option's words were already visible.
- **A pre-QuALITY model is a poor control,** because it also differs in ability. The same model on new questions is better.
- **A with-story gap on the control set can't separate inflation from a harder test** without human scores.

## Bugs found and fixed

- **Default data path:** Inspect changes directory when loading a task file, so the overview's `inspect eval` command couldn't find the dataset.
- **`.env` not loaded:** `python study.py` never read it, because Inspect only does so through its own CLI.
- **Reports blocked after a resume:** an interrupted run left `started` logs behind, so `analysis.py` refused to report. `study.py` now removes them once the task has succeeded.
- **`inspect score`:** it also needs `--overwrite` and `--scorer quality_probe/scorer.py@quality_scorer`. An entry point I'd added didn't help, so I removed it.
- **`[partial]` mark:** runs using `--stories` weren't marked until `analysis.py` was fixed to detect them.
- **Answer parsing:** the overview's reference regex crossed newlines and missed `_` emphasis.
- **Per-model `max_tokens`:** a task's setting overrides a model's, so tasks now carry their own model.

The README lists every place the repo differs from the overview.

## Mistakes I made and corrected

- **Scrambled answer key:** I said it would measure letter habits, but it scores about 25% regardless. I replaced it with an arithmetic baseline.
- **Verbatim recall:** I called it the closest thing to a smoking gun and called its decoy a no-memory baseline. The decoy is too weak, and my draft rule would have given a false positive. I corrected the report and the rule.
- **The story and hidden answers:** I first said the story wouldn't help, and you pointed out it could help the model recognize the item.
- **Pre-QuALITY model as a control:** I suggested it, and you pointed out that ability differences confound it.
- **With-story comparison:** my draft rules treated a difference as a sign of mismatched tests, when under your question it's the main thing to measure. This isn't fixed yet (see next-steps).
- **Full-run cost:** I first repeated the overview's estimate, but Haiku's thinking makes output longer, so the real figure is higher.
- **PowerShell:** I gave Git Bash commands before realizing you were in PowerShell.
- **Test run flag:** a flag I added caused 7 false errors in one test run. A later clean run passed everything.

## Decisions you made

- Skipped the local model smoke run.
- Removed the results web page.
- Ran 5 questions from each of 30 stories instead of the full dataset, then ran the letters-only conditions and the probes on the same questions.
- You'll build the control set.
- Changed the question to whether training raised the score.

## Acceptance criteria from the overview

| Criterion | Status |
|---|---|
| Offline tests pass, including end-to-end fakes | Met |
| Smoke run on Anthropic, report generated | Met |
| Smoke run on one local model | **Not met, skipped** |
| Context check fails an undersized local server | Met, live |
| `analysis.py` refuses without a passing check | Met |
| Resumed eval set regenerates no completed sample | Met, live: no flushed sample regenerated |
| Parser change plus `inspect score` updates the report with no API calls | Met with fake logs. Real logs re-scored with the existing parser gave an identical report |
| `analysis.py` runs with networking disabled | Met, using fake logs |

## State of your machine

- **Git:** nothing is committed. All code is uncommitted on `main`. `.env`, `data/` and `logs/` are gitignored.
- **Dataset:** in `data/`, downloaded from `nyu-mll/quality`. The README has the command to download it again.
- **Python:** the environment is `.venv` (Python 3.12).
- **Claude Code memory:** I saved a short note that you're not a researcher and use PowerShell.
- **Ollama:** the desktop app is running, probably started by my `ollama list`, and has an update downloaded but not installed. Leftover runner PID 8876 may still be there using almost no memory: `Stop-Process -Id 8876` ends it.
