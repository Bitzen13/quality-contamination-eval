# Decision rules (draft)

Fix these before the full run, and don't change them after seeing its results. All 30-story results so far are exploratory: they shaped this design, so they don't count as evidence for it.

## What counts as a result

Every rule uses the full dev set, all 115 stories, and the story-level 95% interval. "Above zero" means the whole interval is above zero.

| Question | Measure | Evidence of exposure if |
|---|---|---|
| Is the benchmark's advantage specific to its own questions? | `compare.py`: difference of differences, with the with-story difference inside ±0.03 | Above zero, by at least 0.05 |
| Can the model reproduce the text? | Verbatim recall: near match on QuALITY minus near match on the control set, plus a read of the closest matches | Above zero, by at least 0.02, with matches that can't be worked out from the options shown |
| Does it know the answer key? | `no_story_letters_recall` and `full_letters_recall`: accuracy above letter habits | Above zero, by at least 0.03 |
| Does it rely on exact wording? | `no_story` − `no_story_paraphrased`, with the paraphrase control inside ±0.03 | Above zero, by at least 0.05 |

The thresholds are starting points, chosen by judgement. Change them now if a smaller effect would matter to you.

## How to read them together

- **Exposure shown:** verbatim recall or the difference of differences clears its rule. Report which one, and report the size.
- **Exposure suggested:** only the letter or wording probes clear their rules. Those have other possible explanations.
- **No evidence found:** nothing clears. Say what the intervals rule out. Don't say the model is uncontaminated, because a model can retain meaning without wording.

## What not to read into them

- **Conditions that aren't evidence:** `no_story_swapped` (its options are often off-topic) and `no_story_options_only` (it describes the score, and doesn't test for exposure).
- **Verbatim recall against the decoy:** the 30-story run beat the decoy, but its closest matches could be worked out from the options shown. Only the control-set comparison counts.
- **Multiple tests:** four questions means a higher chance that one clears by luck. A single marginal result should be repeated with a new seed before it's reported.
- **Scale:** even clear exposure doesn't show how much it inflates a with-story score. Use the gap and the control set for scale.
