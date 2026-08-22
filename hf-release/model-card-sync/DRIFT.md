# Model card sync — drift report

Live Hub cards fetched **2026-08-21** from
`https://huggingface.co/DantheMan124/<repo>/raw/main/README.md` (read-only HTTP
GET, no auth, no write endpoint touched) and diffed against the canonical copies
in `training/`.

| Repo | Live card | Canonical | Verdict |
|---|---:|---:|---|
| `DantheMan124/deberta-hallucination-judge` | 3,328 B / 80 lines | 7,991 B / 167 lines | **Severe drift — publishes a known defect as undisclosed** |
| `DantheMan124/deberta-preference-reward` | 13,567 B / 286 lines | 13,980 B / 290 lines | Minor — one artifact note missing, plus a table bug to fix in transit |

Staged replacements are the two `*.README.md` files here. The `*.diff` files are
the reviewable form of the same change.

---

## 1. `deberta-hallucination-judge` — the serious one

The live card is a **pre-2026-08-13 snapshot**. Everything learned since is
missing from the public page while sitting in the repository.

### 1a. The encoding defect is not disclosed publicly

The live card's Usage section still presents this, with no warning:

```python
text = "Q: What is the capital of France? C: France is in Europe. A: Paris"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
```

On any real RAG context that snippet **silently deletes the answer** — the span
being classified — because `truncation=True` cuts from the right and the answer
is on the right. Measured on the 200-example sample: **fully deleted on 50.5% of
examples, partially on a further 15.5%.**

A visitor copying that snippet today gets a classifier scoring an input it never
saw, with nothing on the page telling them so. This is the single highest-value
fix in this entire release, and it is a correctness disclosure, not polish.

### 1b. Three dead links to a repository that does not exist

The live card links three times to `github.com/DantheMan124/evalforge`. The
repository is `github.com/Larmstrong1127/evalforge` (confirmed against the git
remote). All three links 404 today:

- the EvalForge project link in the intro,
- the "training code and full writeup" link,
- the benchmark link in Recommended use.

The canonical card already has these right; they were fixed locally and never
republished.

### 1c. Whole sections missing from the live card

| Section | Live | Canonical |
|---|---|---|
| Intended use | **absent** | present |
| Training data | present | present |
| Evaluation | present | present |
| Limitations | **absent** | present |
| License | **absent** | present |
| Usage | present (unsafe snippet) | present (with the budgeting fix) |
| Labels | present | present |

The absent **Limitations** section is where the 512-token defect, the
`0.444 → 0.603` ROC-AUC re-measurement, the "still below the all-faithful
baseline" conclusion, and the ECE-0.40 calibration collapse all live. The absent
**License** section means the card body never states MIT (the YAML metadata
does, so this is a completeness gap rather than a licensing one).

### 1d. Staged addition

Beyond restoring the canonical text, the staged card adds one link: the
diagnostic sample is now published, so the measurement is inspectable rather
than merely cited.

```
  RAGTruth benchmark sample (2026-08-09,
  `training/scripts/diagnose_ragtruth_agreement.py`).
+ The exact 200-example sample, with the per-example truncation
+ accounting under both encodings, is published as
+ `DantheMan124/ragtruth-diagnostic-200`.
```

---

## 2. `deberta-preference-reward` — one addition and one bug fixed in transit

### 2a. The missing artifact note (the requested change)

The live card lacks the paragraph recording that all three accuracy rows are
committed as `reward_results.json` with the command that produced each, and that
the 0.6009 baseline was re-measured on CPU on 2026-08-17 after an audit found it
existed only in prose.

### 2b. A markdown table bug in the canonical copy, corrected here

The canonical `training/MODEL_CARD_preference_reward.md` inserts that note
**inside** the evaluation table:

```markdown
| **This model** — UltraFeedback `test_prefs` | 184M | 1,987 | **0.7026** |

_All three rows are committed as reward_results.json..._
| Human OOD probe (EvalForge rating room) | 184M | 15 | 0.4000 |
```

The blank line terminates the table, so the **Human OOD probe row renders as
loose body text rather than a table row** — and that row is the honest
counterweight to the 0.7026 headline. The row most easily read past is the one
currently falling out of the table.

The staged card closes the table first, then adds the note:

```markdown
| **This model** — UltraFeedback `test_prefs` | 184M | 1,987 | **0.7026** |
| Human OOD probe (EvalForge rating room) | 184M | 15 | 0.4000 |

_All three rows are committed as reward_results.json..._
```

**This fix should be back-ported to `training/MODEL_CARD_preference_reward.md`.**
It is not staged as a repo edit here because this directory's remit is Hub cards;
flagged so it does not get lost.

### 2c. Staged addition

A pointer to the published eval split, with the reason it ships as a script
rather than as rows stated in one sentence rather than left for the reader to
discover on the dataset page.

---

## What was NOT changed

- **No metric was edited.** Every number in both staged cards is the number
  already in the canonical copy; this sync moves text, not measurements.
- **No YAML metadata changed** on either card beyond what the canonical copies
  already carry — same `license: mit`, same `base_model`, same tags.
- **The hallucination judge's weights are untouched.** The encoding fix was
  always inference-time; nothing here implies a retrain.
- **Nothing was uploaded.** These are files on disk. See `../PUBLISH.md`.
