# PUBLISH.md — the owner's runbook

**Nothing in `hf-release/` has been published.** No repo was created, no file
uploaded, no Space pushed, no collection made. `hf auth` state was not read,
written, or refreshed. Every command below is yours to run.

Everything staged here has been built and tested locally against the real
checkpoints on CPU. What has *not* happened is any write to the Hub.

Run the steps in order. Step 1 is a gate, not a formality.

---

## Prerequisites

```bash
cd /path/to/evalforge

# The CLI. `hf` replaced `huggingface-cli`; both are in huggingface_hub >= 0.34.
pip install -U "huggingface_hub[cli]>=0.34"
hf version

# Who am I about to publish as? Confirm this BEFORE anything else.
hf auth whoami
# Expect: DantheMan124
```

If `whoami` shows a different account or errors, stop and fix that first — every
command below hardcodes the `DantheMan124` namespace.

You need a token with **write** scope. If `whoami` fails:

```bash
hf auth login          # paste a write token from huggingface.co/settings/tokens
```

Never paste a token into a file in this repository, a shell one-liner that gets
saved to history, or a command you share.

---

## Step 1 — Leak scan (GATE: do not skip, do not proceed on a failure)

```bash
cd /path/to/evalforge
python hf-release/leak_scan.py
```

Expected:

```
ADVISORY -- third-party corpus content (redistributed, not authored here):
  datasets/ragtruth-diagnostic-200/data/diagnostic-200.jsonl:185: [private-email] ...
  datasets/ragtruth-diagnostic-200/data/diagnostic-200.jsonl:196: [workplace-reference] ...

scanned 22 files under hf-release/ (21 rules, 17 enforced on corpus data)
LEAK SCAN CLEAN (2 corpus advisory/advisories to acknowledge)
```

**The two advisories are expected and already adjudicated** — they are verbatim
business contact details inside upstream RAGTruth source text, documented under
"Contact details present in the source text" in that dataset's card. Read them
once and move on.

**Any line under `FAILURES:` stops the publish.** A Hub repo keeps its git
history, so a secret pushed once is a secret *rotated*, not a secret deleted.

Also run the tests, since a broken regeneration script is its own kind of leak —
of credibility:

```bash
python -m pytest hf-release/tests/test_split_determinism.py -q
cd hf-release/space/reward-serve-demo && python -m pytest test_scoring.py -q && cd -
# expect 25 passed and 14 passed
```

And confirm the split still reproduces against live upstream data before you
publish a card that claims it does:

```bash
cd hf-release/datasets/ultrafeedback-eval-split
python regenerate_split.py --verify --out /tmp/uf.jsonl && rm -f /tmp/uf.jsonl
# VERIFY OK: 1987 pairs, sha256 9c392052da72518e...
cd ../ragtruth-diagnostic-200 && python regenerate_sample.py --verify && cd -
# VERIFY OK: 200 rows, sha256 db98a0226b28783d...
```

---

## Step 2 — Create the empty repos

Creating them first (rather than letting upload auto-create) means a typo in a
namespace fails here, loudly, instead of silently creating a repo under the
wrong name that you then have to find and delete.

```bash
hf repo create DantheMan124/ultrafeedback-eval-split --repo-type dataset --private
hf repo create DantheMan124/ragtruth-diagnostic-200  --repo-type dataset --private
hf repo create DantheMan124/reward-serve-demo        --repo-type space \
    --space_sdk gradio --private
```

**Everything is created `--private` on purpose.** You look at each one as a
stranger would, then flip it public in Step 7. Publishing is the step that is
hard to undo; staging privately is free.

The two model repos already exist and are public — they are updated in place in
Step 5, not created.

---

## Step 3 — Publish the UltraFeedback eval split (script, no rows)

```bash
cd /path/to/evalforge
hf upload DantheMan124/ultrafeedback-eval-split \
    hf-release/datasets/ultrafeedback-eval-split . \
    --repo-type dataset \
    --commit-message "Regeneration script, manifest, and dataset card for the 1,987-pair eval split"
```

Publishes 4 files: `README.md`, `regenerate_split.py`, `manifest.json`,
`requirements.txt`.

**Verify no rows went up** — this is the license position, and a stray data file
silently reverses it:

```bash
hf repo files DantheMan124/ultrafeedback-eval-split --repo-type dataset
# There must be NO .jsonl, .parquet, or .arrow in that listing.
```

---

## Step 4 — Publish the RAGTruth diagnostic sample (rows included)

```bash
hf upload DantheMan124/ragtruth-diagnostic-200 \
    hf-release/datasets/ragtruth-diagnostic-200 . \
    --repo-type dataset \
    --commit-message "200-example RAGTruth diagnostic sample with both encodings"
```

Publishes `README.md`, `LICENSE`, `manifest.json`, `regenerate_sample.py`,
`requirements.txt`, and `data/diagnostic-200.jsonl` (724 KB, 200 rows).

`LICENSE` carries the upstream MIT notice and the "Copyright (c) 2023 Particle
Media" line. **That file is the thing that makes redistributing these rows
lawful — do not drop it.**

---

## Step 5 — Update the two model cards

Read `hf-release/model-card-sync/DRIFT.md` first. The hallucination judge's live
card currently ships a usage snippet that silently deletes the answer on real
RAG inputs and three dead links; this step is a correctness fix, not polish.

```bash
hf upload DantheMan124/deberta-hallucination-judge \
    hf-release/model-card-sync/deberta-hallucination-judge.README.md README.md \
    --commit-message "Restore full model card: encoding defect disclosure, limitations, correct repo links"

hf upload DantheMan124/deberta-preference-reward \
    hf-release/model-card-sync/deberta-preference-reward.README.md README.md \
    --commit-message "Note the committed results artifact and link the published eval split"
```

The reward card links `reward_results.json` as a repo-relative file, so that
artifact has to exist alongside it:

```bash
hf upload DantheMan124/deberta-preference-reward \
    training/reward_results.json reward_results.json \
    --commit-message "Commit the measured accuracy artifact next to the card that cites it"

hf upload DantheMan124/deberta-preference-reward \
    training/rewardbench2_results.json rewardbench2_results.json \
    --commit-message "Commit the RewardBench 2 protocol and per-domain scores"
```

Then open both cards in a browser and confirm the tables render — in particular
that the reward card's **Human OOD probe row is inside the table**, which is the
bug fixed in transit (see DRIFT.md §2b).

---

## Step 6 — Push the Space

```bash
hf upload DantheMan124/reward-serve-demo \
    hf-release/space/reward-serve-demo . \
    --repo-type space \
    --commit-message "Pairwise preference scoring demo with the honest framing in the UI"
```

Publishes `app.py`, `scoring.py`, `README.md`, `requirements.txt`,
`test_scoring.py`, `smoke_test.py`, `smoke_proof.json`.

The Space builds on the free CPU tier. It will install torch (a few minutes on
first build) and then start; the model loads lazily on the **first request**, so
expect the first visitor to wait ~10s and everyone after to get ~0.5s.

Watch the build:

```bash
hf repo info DantheMan124/reward-serve-demo --repo-type space
```

Or the **Logs** tab on the Space page. If the build fails, that is usually the
`sdk_version` in `README.md` frontmatter (`6.25.0`) not matching a version
Spaces offers — bump it to the nearest available and re-upload just the README.

Confirm it actually serves before making it public: open the Space, paste the
"What causes seasons?" example, and check you get `margin = +1.2231` and
`P(A) = 0.740`. Those are the numbers from `smoke_proof.json` and from the model
card's worked example; if they match, the Space is serving the right weights.

---

## Step 7 — Go public, then build the collection

```bash
hf repo settings DantheMan124/ultrafeedback-eval-split --repo-type dataset --private false
hf repo settings DantheMan124/ragtruth-diagnostic-200  --repo-type dataset --private false
hf repo settings DantheMan124/reward-serve-demo        --repo-type space   --private false
```

Then follow `hf-release/collection.md`. The collection is last because every
item has to exist and be public before it can be added.

---

## Rollback — read this before Step 7, not after

```bash
hf repo delete DantheMan124/ultrafeedback-eval-split --repo-type dataset
hf repo delete DantheMan124/ragtruth-diagnostic-200  --repo-type dataset
hf repo delete DantheMan124/reward-serve-demo        --repo-type space
```

Model cards are not deleted — they are reverted, since the repos predate this
release and hold the weights:

```bash
# Find the commit before your card update, then restore that README:
hf repo commits DantheMan124/deberta-hallucination-judge
hf download DantheMan124/deberta-hallucination-judge README.md \
    --revision <previous-sha> --local-dir ./rollback
hf upload DantheMan124/deberta-hallucination-judge ./rollback/README.md README.md \
    --commit-message "Revert card to previous revision"
```

**What rollback does not undo.** Delete removes the repo from the Hub. It does
not:

- **Reach cached forks and clones.** Anyone who cloned, ran
  `load_dataset`, or pulled the Space has a copy you cannot recall.
- **Purge git history from a repo you keep.** Reverting a card adds a new
  commit; the old revision stays reachable by SHA. This is why Step 1 is a
  gate — for a leaked secret, delete-and-recreate is not a fix, rotation is.
- **Clear search-engine and dataset-aggregator indexes**, which crawl the Hub
  and keep snapshots on their own schedule.
- **Un-see anything.** A public repo that existed for an hour may have been read.

The practical consequence: **the private-first flow in Steps 2–6 is the real
rollback mechanism.** Step 7 is the irreversible one. Everything before it is
cheap to undo; nothing after it fully is.

---

## What each command publishes, at a glance

| Step | Command | Creates / writes | Public after |
|---|---|---|---|
| 1 | `leak_scan.py` | nothing (read-only gate) | — |
| 2 | `hf repo create` ×3 | 3 empty private repos | no |
| 3 | `hf upload` split | 4 files, **no data rows** | no |
| 4 | `hf upload` ragtruth | 6 files incl. 200 rows | no |
| 5 | `hf upload` ×4 | 2 model READMEs + 2 JSON artifacts | **yes — repos already public** |
| 6 | `hf upload` space | 7 files, triggers a build | no |
| 7 | `hf repo settings` ×3 | visibility flip | **yes** |

**Step 5 is the one that goes live immediately**, because those two model repos
are already public. If you want a dry run of the whole thing, do Steps 2–4 and 6
first, look at everything privately, and leave Step 5 until you are committed.
