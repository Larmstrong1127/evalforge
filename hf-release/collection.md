# HF Collection plan

## Title

**Small Judges, Measured Honestly**

Alternates, if that reads too much like a manifesto:
`EvalForge: Local Eval Judges` · `DeBERTa Judges + Their Evaluation Splits`

## Description

One paragraph, and it should be the honest one — the collection's value is that
the boundaries are marked, not that the numbers are high.

> Two 184M DeBERTa judges built for EvalForge, published with the splits and the
> diagnostics that show where they stop working. The preference reward model
> reaches 0.7026 pairwise accuracy in-distribution — beating a 435M public
> baseline that transfers onto the same split at 0.6009 — and sits at the random
> floor out of distribution, averaging 25.3 on RewardBench 2 against a 25.0
> baseline. The hallucination judge scores 0.9937 F1 on held-out HaluEval and
> cannot beat "predict faithful for everything" on real RAG failures. Both
> in-distribution numbers are real; neither model generalizes, and each dataset
> here exists so you can check that claim rather than take it. Includes the
> encoding defect that made one of those measurements wrong for months, and the
> re-measurement that corrected it.

## Ordered items

Order matters: the Space is first because it is the only item a visitor can use
without reading anything, and its UI carries the framing. Models before datasets;
each dataset sits directly under the model it measures.

| # | Item | Type | Why here |
|---|---|---|---|
| 1 | `DantheMan124/reward-serve-demo` | Space | The one clickable thing. Try the model, read the floor next to the input box. |
| 2 | `DantheMan124/deberta-preference-reward` | Model | 184M Bradley-Terry reward model. 0.7026 ID / 25.3 RB2. |
| 3 | `DantheMan124/ultrafeedback-eval-split` | Dataset | The exact 1,987 pairs behind item 2's number. Script + digest, not rows. |
| 4 | `DantheMan124/deberta-hallucination-judge` | Model | 184M faithfulness classifier. 0.9937 ID F1 / no usable RAGTruth operating point. |
| 5 | `DantheMan124/ragtruth-diagnostic-200` | Dataset | The 200 examples that found item 4's encoding defect, both encodings. |

Five items: 1 Space, 2 models, 2 datasets.

## Creating it

The Collections API cannot create the items themselves — every repo in the list
must exist first. **Run `PUBLISH.md` steps 1–6 before any of this.**

### Option A — `huggingface_hub` (preferred; reproducible and re-runnable)

```bash
python -c "import huggingface_hub; print(huggingface_hub.__version__)"   # need >= 0.21
```

```python
from huggingface_hub import add_collection_item, create_collection

DESCRIPTION = (
    "Two 184M DeBERTa judges built for EvalForge, published with the splits and "
    "the diagnostics that show where they stop working. The preference reward "
    "model reaches 0.7026 pairwise accuracy in-distribution -- beating a 435M "
    "public baseline that transfers onto the same split at 0.6009 -- and sits at "
    "the random floor out of distribution, averaging 25.3 on RewardBench 2 "
    "against a 25.0 baseline. The hallucination judge scores 0.9937 F1 on "
    "held-out HaluEval and cannot beat \"predict faithful for everything\" on "
    "real RAG failures. Both in-distribution numbers are real; neither model "
    "generalizes, and each dataset here exists so you can check that claim "
    "rather than take it. Includes the encoding defect that made one of those "
    "measurements wrong for months, and the re-measurement that corrected it."
)

collection = create_collection(
    title="Small Judges, Measured Honestly",
    description=DESCRIPTION,
    namespace="DantheMan124",
    private=True,     # flip to public via update_collection_metadata once reviewed
    exists_ok=True,
)
print(collection.slug)   # e.g. DantheMan124/small-judges-measured-honestly-<hash>

ITEMS = [
    ("DantheMan124/reward-serve-demo", "space"),
    ("DantheMan124/deberta-preference-reward", "model"),
    ("DantheMan124/ultrafeedback-eval-split", "dataset"),
    ("DantheMan124/deberta-hallucination-judge", "model"),
    ("DantheMan124/ragtruth-diagnostic-200", "dataset"),
]

for item_id, item_type in ITEMS:
    add_collection_item(
        collection.slug,
        item_id=item_id,
        item_type=item_type,
        exists_ok=True,
    )
    print("added", item_id)
```

`private=True` is deliberate: create it hidden, look at it as a stranger would,
then publish.

```python
from huggingface_hub import update_collection_metadata

update_collection_metadata(collection.slug, private=False)
```

**Per-item notes** are worth the extra call — they are the one-line framing a
visitor sees without leaving the collection page:

```python
from huggingface_hub import update_collection_item

NOTES = {
    "DantheMan124/reward-serve-demo":
        "Score two responses to the same prompt. Ranking only -- a single score means nothing.",
    "DantheMan124/deberta-preference-reward":
        "0.7026 pairwise in-distribution, 25.3 on RewardBench 2 against a 25.0 random floor.",
    "DantheMan124/ultrafeedback-eval-split":
        "The exact 1,987 pairs. Regeneration script + SHA-256, not redistributed rows.",
    "DantheMan124/deberta-hallucination-judge":
        "0.9937 F1 in-distribution; no usable operating point on real RAG failures.",
    "DantheMan124/ragtruth-diagnostic-200":
        "The 200 examples where right-truncation deleted the answer on 50.5% of inputs.",
}
# item_object_id comes from the collection object, not the repo id:
collection = get_collection(collection.slug)
for item in collection.items:
    if item.item_id in NOTES:
        update_collection_item(collection.slug, item.item_object_id, note=NOTES[item.item_id])
```

Ordering: `add_collection_item` appends, so adding in the table's order yields
the table's order. To fix afterwards, `update_collection_item(..., position=N)`.

### Option B — the web UI

1. huggingface.co → profile menu → **Collections** → **New collection**.
2. Title and description from above. Set **Private** while assembling.
3. **+ Add item** for each of the five, in table order, pasting the full repo id
   and picking the right type (Space / Model / Dataset).
4. Add the per-item note under each entry.
5. Drag to confirm the order, then switch to **Public**.

## Checks before making it public

- [ ] All five repos exist and load for a logged-out visitor.
- [ ] The Space is **running**, not build-failed or sleeping on first paint.
- [ ] Both model cards are the staged versions — in particular the
      hallucination judge no longer shows the unsafe truncation snippet
      (see `model-card-sync/DRIFT.md`).
- [ ] Dataset card links resolve in both directions (model ↔ dataset).
- [ ] `ultrafeedback-eval-split` contains **no data rows** — script and manifest
      only. This is the license position; a stray jsonl silently reverses it.
- [ ] The description's numbers match the cards: 0.7026 / 0.6009 / 25.3 / 25.0 /
      0.9937.
