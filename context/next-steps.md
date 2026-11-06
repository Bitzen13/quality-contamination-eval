# Next steps

Last updated 2026-11-05.

## The question we're now answering

**Did training on this material raise the model's score on this benchmark?** "Has it seen the answers?" is only part of that. We can't retrain a model without QuALITY, so the closest test is the same model on a test it can't have seen: the control set.

## Where things stand

All numbers are Haiku 5.5 on 150 questions from 30 stories, so they're preliminary. They're in `logs/haiku-30x5/report.md`.

| What the model gets | Score | What it suggests |
|---|---|---|
| Story, question and answers | 90% | The normal score |
| Question and answers | 49% | 24 points above guessing, unexplained |
| Answers only | 40% | About 15 of those points need no question, probably answers that give themselves away |
| Answers hidden, letters only | 24–29% | About what its letter habits score by luck, so no sign of a memorized answer key |
| Asked to recite a removed answer | 1 of 110 exact | No convincing sign of memory: the closest matches could be worked out from the prompt |

The full dataset hasn't been run. All 208 tests passed on the last full run.

## What you need to do

1. **Build the control set.** Decide how many stories and questions after looking at the size of the intervals. A few dozen stories is a reasonable start. It has three parts:
   - **New questions on QuALITY's stories:** shows whether the benchmark's own questions and answers help the model.
   - **New questions on new stories** published after the model's training cutoff: shows whether knowing the stories helps too.
   - **Human scores:** a few people take your questions with the stories, which shows whether a score gap is inflation or just a harder test.
2. **Match QuALITY's style.** Right and wrong answers should be written the same way as QuALITY's. Answers-only should score about the same on both tests. If it doesn't, the two tests aren't comparable.
3. **Keep it private.** Anything published can end up in future training data.
4. **Use the same file format as QuALITY** (JSONL), with `article_id`, `article` and `questions`. See `data/QuALITY.v1.0.1.htmlstripped.dev`.

## What Claude needs to do

1. **Rework the main result around the with-story comparison.** Update `compare.py` and `decision-rules.md` to:
   - make QuALITY vs control, with the story, the headline result
   - stop treating a with-story difference as just a check that the tests match
   - add the human scores to adjust for difficulty
2. **Look at how QuALITY's human accuracy was measured,** so the human scores can be matched to it.
3. **Split the control comparison in two:** same stories, and new stories.

`compare.py` currently treats a with-story difference as a sign the tests aren't matched. That's out of date until step 1 is done.

## Decisions for you

- **Thresholds in `decision-rules.md`:** these are my draft. Set them before the full run, and don't change them after seeing the results.
- **Full run on Haiku:** about 25,000 calls, roughly $3.50–$8, likely a few hours. It can run before or after the control set. Commands are below.
- **Paraphrased options:** built but never run on a real model, and now less central. Roughly $1.50–$3 on the full dataset. Try it on the 30-story sample first.
- **Commit:** nothing is committed. All the code is uncommitted on `main`, so ask Claude to commit it to a branch soon.
- **Slower tests:** the full test suite takes about 20 minutes. Claude offered to make a plain `pytest` skip the slowest tests.

## Commands

Run these in PowerShell from the repo root, with `(.venv)` showing.

```powershell
# See every run so far
inspect view --log-dir logs

# Full run on Haiku
python study.py --models anthropic/claude-haiku-5-5 --log-dir logs/haiku-full
python study.py --models anthropic/claude-haiku-5-5 --letters-only --log-dir logs/haiku-full
python study.py --models anthropic/claude-haiku-5-5 --probes --log-dir logs/haiku-full
python analysis.py logs/haiku-full

# Once the control set exists
python study.py --models anthropic/claude-haiku-5-5 --data data/control.jsonl --log-dir logs/control
python analysis.py logs/control
python compare.py logs/haiku-full logs/control
```

## Known limits

- **`no_story_swapped` isn't evidence.** Borrowed answers are often off-topic, so the right answer stands out.
- **Verbatim recall needs the control set.** Beating its decoy baseline doesn't show memory.
- **The local model smoke run was skipped.** A correctly sized local server hasn't passed the context check, so test that before adding a local model.
- **Some effects can't be measured here.** General reading skill that benchmark-style data might have improved would lift both tests equally.
