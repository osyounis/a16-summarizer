# ROUGE: base vs fine-tuned (DialogSum test)

Qwen2.5-1.5B-Instruct, stock vs QLoRA fine-tune (r=16, a=32) merged to fp16.
500 dialogues, multi-reference ROUGE (max f-measure over 2/3 human refs per dialogue).

## Headline (raw model output)

Exactly what each model emits, no post-processing. This is the number for the README.

| Metric | Base | Fine-tuned | Δ |
|---|---:|---:|---:|
| rouge1 | 0.3706 | 0.5581 | +0.1875 |
| rouge2 | 0.1498 | 0.3101 | +0.1602 |
| rougeL | 0.2889 | 0.4808 | +0.1919 |

- **rouge1** Δ = +0.1875, 95% CI [+0.1740, +0.2017], p(tuned not better) = 0.0000
- **rouge2** Δ = +0.1602, 95% CI [+0.1446, +0.1766], p(tuned not better) = 0.0000
- **rougeL** Δ = +0.1919, 95% CI [+0.1772, +0.2067], p(tuned not better) = 0.0000

Paired bootstrap over dialogues, 10,000 resamples, seed 0.

## Diagnostics

| | Base | Fine-tuned |
|---|---:|---:|
| preamble rate | 0.000 | 0.000 |
| mean output tokens (refs: 27.8) | 68.3 | 33.6 |
| median output tokens | 69.0 | 30.5 |
| truncation rate (hit 96 cap) | 0.170 | 0.002 |
| empty output rate | 0.000 | 0.000 |

The human reference summaries average 27.8 tokens. The base model's output is not
malformed; it writes correct third-person `#Person1#`-style summaries. It's just much
longer and more detailed than DialogSum's house style. The decomposition below looks at
what that costs it.

The non-zero base truncation rate raises a fair question: is the `max_new_tokens=96` cap
handicapping the base model? I re-ran with `--max-new-tokens 192` to check, and it isn't.
Doubling the cap drives base truncation to roughly zero and leaves its ROUGE slightly
lower (it simply writes more, so precision falls further), while the fine-tune's scores
are bit-identical because it never reaches the cap. The delta is not a truncation
artifact.

## Precision / recall decomposition

The headline alone can't tell you whether the fine-tune learned summarization or just
brevity. A verbose model that captures the content will score high recall and low
precision.

| Metric | Model | Precision | Recall | F |
|---|---|---:|---:|---:|
| rouge1 | base | 0.2767 | 0.6162 | 0.3706 |
| rouge1 | tuned | 0.5402 | 0.6042 | 0.5581 |
| rouge2 | base | 0.1093 | 0.2688 | 0.1498 |
| rouge2 | tuned | 0.3022 | 0.3365 | 0.3101 |
| rougeL | base | 0.2138 | 0.4939 | 0.2889 |
| rougeL | tuned | 0.4643 | 0.5221 | 0.4808 |

On ROUGE-1 the base model's recall is 0.616 vs the fine-tune's 0.604, while its precision
is 0.277 vs 0.540. The base model recovers about as much reference content as the
fine-tune does, and loses almost entirely on precision. In other words, the gain here is
dominated by length and style calibration rather than better content selection. That is a
real and useful result for a summarizer (matching the target register is the job), but it
shouldn't be reported as "the base model can't summarize". It can; it just won't stop.

### Preamble check

Regex, applied at most once to the first line, identically to both models:

```
^(sure|certainly|of course|okay|here('s| is| are)|the following)\b.*:\s*$
```

It matches nothing (0.000 on both models), so this control is inert and the stripped
scores are identical to the headline. I'm noting it rather than quietly dropping it: the
confound I expected, stock Qwen opening with "Sure! Here's a summary:" and getting
punished for formatting rather than comprehension, simply doesn't occur. The base model's
disadvantage is length, not preamble, which is why the decomposition above is the
analysis that matters.

## Run config

- examples: 500
- refs per dialogue: {2: 10, 3: 490}
- decoding (both models, identical): `{"max_new_tokens": 96, "do_sample": false, "num_beams": 1, "repetition_penalty": 1.0, "pad_token_id": 151643, "eos_token_id": [151645, 151643], "transformers_version": "5.13.1"}`
- stemmer: `use_stemmer=True` | aggregator: own mean over per-example scores
- batch size: 16
- base: 0.0s
- tuned: 0.0s
- transformers 5.13.1, torch 2.13.0+cu130

Note on comparability: `use_stemmer=True` follows the ROUGE-1.5.5 `-m` convention most
summarization papers use (HF `evaluate` defaults it to False). I'm stating the setting
rather than claiming to match the DialogSum paper's exact configuration. The base-vs-tuned
delta is the claim; the absolute values are context.
