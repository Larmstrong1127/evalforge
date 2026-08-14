# Novelty Assessment — prior-art research

**Researched:** 2026-08-13 (findings dated; this space moves monthly — re-check anything older than ~90 days before an interview)
**Scope:** six candidate-novel claims across `github.com/Larmstrong1127` projects.
**Method:** web + GitHub search, multiple phrasings per item; product docs, arXiv, ComfyUI ecosystem, MLOps docs, dental-vendor marketing. Absence of evidence is stated as such — "I could not find" never becomes "does not exist."

---

## Classification table

| # | Item | Classification |
|---|------|---------------|
| A | asset-provenance (content-addressed registry + DAG + USD export + C2PA read + browser) | **RARE COMBINATION** — with one plausibly novel edge (USD export of generation lineage) |
| B | The 97% metadata-recoverability finding + self-audit methodology | **NOVEL FINDING** (data novelty, on his own corpus); methodology is **RARE COMBINATION** |
| C | comfy-workflow-pack | **RARE COMBINATION** — closest prior art is comfy-pack (BentoML); real but narrow delta |
| D1 | EvalForge CI eval gate | **WELL-TRODDEN** |
| D2 | 184M reward model beating a 435M baseline on one harness | **RARE COMBINATION** (the honest-caveat framing is the value, not the result) |
| D3 | Published self-invalidating defect audit | **RARE COMBINATION** — rare in practice, but named prior art exists in the literature |
| E | DentaVision positioning (scan-of-plan, no PMS) | **RARE COMBINATION** — category is well-trodden; the *deliberate non-integration* is the uncommon part |
| F | Shorts Factory as a category | **WELL-TRODDEN** — do not claim novelty |

---

## A. asset-provenance

**Closest prior art found:**

- **Numonic** — commercial ComfyUI asset intelligence; explicitly indexes both PNG metadata chunks (rendered workflow + prompt API graph), every node/model/LoRA/param, and claims to "track lineage across multi-step workflows (img2img, upscale, inpaint) without losing the chain." <https://www.numonic.ai/blog/comfyui-png-metadata-chunks-workflow-parameters> — **this is the single closest thing to the lineage-of-generated-media half.** It also states plainly that ComfyUI itself does not record derivation relationships, so lineage must be reconstructed by heuristics.
- **Weights & Biases Artifacts** — content-addressed storage, artifact lineage as an explicit DAG, traversable UI. <https://docs.wandb.ai/models/artifacts/explore-and-traverse-an-artifact-graph>. Mature, ML-training-oriented, not media-artist-oriented.
- **Machine Genome** (paxlabs-inc) — SHA-256 content-addressed identity, typed parent edges, verifiable lineage graph for models/agents/datasets/artifacts. <https://github.com/paxlabs-inc/machine-genome>
- **art-provenance-vault** — SHA-256 CAS (`vault/assets/<sha256>`) + git-DAG hash chain for art assets. <https://github.com/0thernes/art-provenance-vault>
- **C2PA / Content Credentials** — the mature standard; `c2patool`, viewers, 6,000+ member orgs as of Jan 2026. Reading manifests is commodity. See also `abrignoni/AI_Provenance_Scanner` (exiftool + c2pa AI-generation tag detection). <https://github.com/abrignoni/AI_Provenance_Scanner>
- **ComfyUI browsers/extractors** — `ComfyUI-Majoor-AssetsManager`, `ComfyUI-Metadata-Extractor`, comfyui-metadata.com. All read metadata; none build a derivation DAG.
- **OpenUSD side** — `assetInfo` is the *documented, intended* place for asset-management annotation (name, version, identifier), `customData` for schema-free extras; Animal Logic's asset resolver mutates URI state for versioning. <https://openusd.org/release/glossary.html>, <https://animallogic.com/wp-content/uploads/2023/04/USD-at-Scale.pdf>. So the *mechanism* he used is exactly the sanctioned one — not invented.

**Precise deltas:**
- W&B has content-addressed artifacts and a lineage DAG but is scoped to ML runs, has no C2PA reading, and no USD export.
- Numonic reconstructs lineage but is closed/commercial, ComfyUI-specific, and — as of this research — advertises no USD export and no C2PA quantification.
- C2PA tooling asserts per-file claims; it has no cross-corpus derivation graph and no notion of "this asset has seven parents."
- Machine Genome models *model/agent* lineage, not rendered media files, and does not export to a DCC format.
- USD `assetInfo`/`customData` provenance conventions are documented, and studios annotate assets — but **I could not find a published tool or talk that exports a generation-lineage graph (derivation edges as USD relationships, generation params in customData) onto a USD stage.** Searched: "export provenance lineage OpenUSD customData assetInfo generated asset", "USD exporter provenance metadata customData AI generated image lineage relationships prim", "USD asset resolver provenance SIGGRAPH Animal Logic Netflix Eyeline generative AI lineage", "C2PA content credentials OpenUSD pipeline provenance VFX studio talk". This is the strongest edge in the whole portfolio.

**Caveat he must hold:** a large studio may well do this internally and never publish. Absence of public evidence ≠ absence.

**Interview sentence:**
> "Every ingredient here exists — W&B has content-addressed artifact lineage, C2PA is a mature standard, ComfyUI asset browsers read generation metadata. What I couldn't find prior art for when I built it was the last mile: exporting a generation-lineage graph onto a USD stage, with the derivation edges as USD relationships and the generation params in customData, so a pipeline TD can traverse provenance with `usdview` instead of my tool. If a studio has done that internally I'd expect it — I just couldn't find it published."

---

## B. The 97% finding + "audit your own pipeline's provenance debt"

**Closest prior art found:**

- Numonic's marketing states the same *qualitative* fact (ComfyUI doesn't record derivation), but publishes no measured recoverability rate on a real corpus.
- `arxiv.org/pdf/2404.14378` "Pipeline Provenance for Analysis, Evaluation, Trust or Reproducibility" — argues for capturing pipeline provenance; prescriptive, not a retrospective audit of an existing pipeline's losses.
- Data-provenance auditing of *training corpora* (Data Provenance Initiative) is a well-established genre — but it audits ingested third-party data, not one's own output pipeline.
- I could not find a named, articulated methodology for "re-ingest your own pipeline's output and measure what fraction of generation metadata is recoverable." Searched: `"provenance debt"`, `"metadata recoverability"`, "audit own generation pipeline re-ingest measure methodology".

**Precise delta:** the number (0 of 1,791 assets with recoverable model/checkpoint; 50/1,791 = 2.8% carrying C2PA, publish-paths only) is a measurement on his own system — nobody else can have published it, and nobody else's number would be it. The genuinely interesting structural claim is the **disjointness**: C2PA covers only publish paths, internal lineage covers only pre-publish, so the two provenance systems cover non-overlapping ground and neither alone answers "what produced this." That framing is the part a senior person will find sharp.

**Interview sentence:**
> "The 97% is a finding about my own pipeline, not a claim about the field — I re-ingested 1,791 of its own outputs and could not recover model or checkpoint for essentially any of them, while C2PA manifests covered 2.8%, all on publish paths. The point I'd defend is the structural one: C2PA and internal lineage cover disjoint parts of the pipeline, so having both still leaves a gap. I didn't find that measurement methodology written up anywhere, but I'd frame it as an audit I ran, not a methodology I invented."

**Do not say:** "no one measures provenance loss." Data-lineage auditing is a whole discipline.

---

## C. comfy-workflow-pack

**Closest prior art found — this is a crowded lane:**

- **comfy-pack (BentoML)** — `.cpack.zip` artifact locking Python package versions, ComfyUI + custom-node revisions, and **model hashes**; typed input nodes (ImageInput/StringInput/IntInput/FileInput/AnyInput) declaring which params are user-configurable; unpack reconstructs the environment. <https://www.bentoml.com/blog/comfy-pack-serving-comfyui-workflows-as-apis> — **closest overall.** Its own docs list versioning as roadmap, and describe no load-time target validation, no wired-link protection, and no provenance emission.
- **ViewComfy** (open source) — "select the parameters you want to expose," app builder with grouping, previews, output-type control; serverless API. <https://github.com/ViewComfy/ViewComfy>. Parameter *exposure* is solved here; a declared, versioned, validated schema file is not the artifact — the config lives in their app builder/service.
- **ComfyDeploy**, **SaltAI**, **InvokeAI workflows**, **comfyui-workflow-templates** (ships a manifest + loader for templates) — all touch adjacent ground.
- **ComfyUI-Manager** — solves node-pack installation, not parameter surfaces.

**Precise delta (state exactly these, nothing broader):**
1. comfy-pack locks the *environment*; his pack declares the *editable surface* — per-input types, ranges, approved choices — as a checked-in `pack.json` next to the API graph.
2. Validation is at **pack load**, not at artist run — authoring errors surface to the TD, not to the artist mid-shot. I found no equivalent in comfy-pack or ViewComfy docs.
3. **Refusing to overwrite a wired input with a literal** — this is the single most specific thing he built. I found no other tool that names this failure mode (silent graph corruption when a parameter binding lands on an input that already has an incoming link).
4. Provenance-record emission after execution, closing the loop with item A. Neither comfy-pack nor ViewComfy emits one.

**Also honest:** his README documents exactly one live execution (ComfyUI 0.32.0, 2026-08-11, one image) with batching/most samplers/img2img/remote/auth explicitly unverified. He should lead with that honesty, not hide it — but he must therefore never describe this as production-proven.

**Interview sentence:**
> "The closest prior art is BentoML's comfy-pack, which locks the environment — package versions, node revisions, model hashes — and has typed input nodes. Mine locks a different thing: a declared artist-editable surface with types and ranges, validated when the pack loads rather than when the artist runs it, and it refuses to overwrite an input that already has a wire, which is the failure mode I actually hit. I couldn't find another tool that names that one. It's one verified live execution so far — the mechanism works, it isn't battle-tested."

---

## D. EvalForge

### D1 — CI gate on eval-score regression: **WELL-TRODDEN**

Braintrust ships a GitHub Action that runs the suite per PR, posts per-case regression diffs, and blocks merges below threshold. Promptfoo ships a native GitHub Action plus CLI for GitLab/Jenkins and documents failing the build under a score threshold. <https://www.braintrust.dev/articles/best-ai-evals-tools-cicd-2025>

**Delta worth stating (small but real):** his gate runs against a **pinned local model**, so a red build means his platform regressed, not that a vendor silently rotated a model — and the baseline is deliberately conservative (0.375 vs llama3.2) as an infra tripwire, not a quality score. That's a defensible design choice, not a novelty.

> **Say:** "Eval-in-CI is a solved category — Braintrust and promptfoo both ship actions that block merges on score regression. The only thing I'd argue is mine's design choice: pinning a local model so the gate measures my platform, not the vendor's weekend deploy."
> **Never say:** "I built eval gating into CI before it was a thing."

### D2 — 184M reward model vs 435M baseline: **RARE COMBINATION**

Small reward models are an active research line — TinyRM (arXiv 2507.09973) gets 400M-parameter MLMs rivalling models 175× larger on RewardBench. RewardBench itself is the standard harness. So "small reward model competitive with bigger one" is a published research result, not his discovery.

**Delta:** he trained one, published it, and benchmarked it head-to-head on the *same* harness against a named public baseline — **and stated the in-distribution caveat himself** (0.7026 vs 0.6009 on UltraFeedback test, explicitly attributed to training-distribution difference rather than superiority). Most portfolio projects report the win and omit the caveat.

> "Small reward models punching above their weight is a known result — TinyRM is the reference point. What I'd point to isn't the number, it's that I ran both on the same harness and wrote the in-distribution caveat into the README myself: 0.70 vs 0.60 on UltraFeedback test is my model being in-distribution, not my model being better."

### D3 — Published self-invalidating audit: **RARE COMBINATION**

Named prior art exists: "Auditing the Audit: Five Failure Modes in Benchmark-Validity Audits" (arXiv 2607.02586) documents exactly this pattern — a ToxiGen top-k truncation defect and a TruthfulQA ordering defect that had already produced publishable numbers — and argues audits should publish a **self-audit chronology**, not just cleaned numbers. Stanford Report (Dec 2025) covers benchmark bug-hunting generally. So the *practice* has an academic name and advocates.

**Delta:** it is rare in a *portfolio project*, and his instance is concrete: eval scripts defaulted to 1024-token sequences against a 512-token checkpoint; 39% of held-out pairs exceed 512 on one side; scoring off-regime in production; fixed by deriving sequence length from checkpoint config.

> "Publishing the defect isn't novel — there's literature arguing audits should publish a self-audit chronology. It's just rare to do it on your own portfolio project. Mine was a 512/1024 sequence-length mismatch that put 39% of held-out pairs off-regime; the fix was deriving the length from the checkpoint config so it can't drift again."

---

## E. DentaVision

**Closest prior art:** the category is dense. Smilepass ships "AI-Powered Digital Treatment Plans" with patient-friendly explanations, multi-language, email/text delivery — and **integrates with Dentrix, Open Dental, Curve, ClearDent, MaxiDent** (<https://smilepass.com/dental-treatment-plans/>, <https://smilepass.com/integrations/>). Overjet, Denti.AI, and thedentalapp.com cover AI treatment planning and patient-facing explanation. Coach Barrow publishes a manual ChatGPT workflow for the same job. Weave/RevenueWell/Modento/Dental Intelligence own patient engagement and case acceptance.

**Precise delta:** every incumbent's moat *is* the PMS integration — that's their sales cycle and their lock-in. His positioning inverts it: work from a scan/photo of the plan the practice already printed, so onboarding is zero-IT and the buyer can be a single hygienist rather than an office manager plus a PMS vendor. **I could not find a product marketed on deliberately not integrating.** That is a go-to-market argument, not a technical novelty, and he should present it as such.

> "The category is well served — Smilepass, Overjet, the whole patient-engagement stack. Every one of them sells on PMS integration, which is also their onboarding cost. I built the opposite bet: work from a scan of the plan the practice already produced, so there's no integration at all. I'm not claiming a new capability — I'm claiming a distribution wedge, and I'd want to test whether it's real."

---

## F. Shorts Factory

**WELL-TRODDEN. Confirmed.** Faceless-channel automation is a mature industry with an open-source long tail: `sasharun/awesome-faceless` catalogues 80+ tools; `SamurAIGPT/AI-Youtube-Shorts-Generator`, `SaarD00/AI-Youtube-Shorts-Generator`, `sumanreddy89/flow-youtube-faceless`, `gyoridavid/short-video-maker` on Docker Hub, plus an entire `youtube-shorts-generator` GitHub topic and countless n8n templates. Script→TTS→FFmpeg→upload is the standard shape.

**What is still legitimately his:** operating it for real, on one consumer GPU, across many channels, and — more interesting to an engineer — the *operational* findings from doing so: VRAM headroom cliffs, orphaned duplicate uploads falsifying per-video analytics, thresholds set and never validated against outcomes. Sell the operations, never the category.

> **Say:** "Faceless-channel automation is an industry — there are dozens of open-source pipelines and I wouldn't claim the category. What I'd talk about is what breaks when you actually run one continuously on a single 24GB GPU: VRAM headroom, duplicate uploads silently falsifying analytics, thresholds nobody ever validated against outcomes."
> **Never say:** anything implying he invented autonomous video generation.

---

## Overall verdict

**Single most defensible novelty claim:**
**Exporting a generation-lineage graph to a USD stage** — derivation edges as USD relationships, generation parameters in `customData` — so provenance is traversable in a standard VFX pipeline instead of a bespoke tool. Every neighbouring piece is mature (W&B lineage, C2PA, USD `assetInfo` conventions, ComfyUI metadata readers), which is exactly what makes the joint credible rather than grandiose. Phrase it as *"I couldn't find prior art for this when I built it,"* never as *"no one has done this."*

**Runner-up, and arguably the better interview material:** the 1,791-asset self-audit — because it is a *measurement*, and measurements about your own system cannot be prior-arted. The disjointness argument (C2PA covers publish paths, internal lineage covers pre-publish, together they still leave a gap) is the sentence most likely to make a senior interviewer lean in.

**Biggest overclaim risk:**
**Shorts Factory as a category** — claiming autonomous multi-channel video automation is novel would be instantly falsified by any interviewer who has seen a single n8n faceless-channel template, and it would retroactively discredit the USD claim, which is the one worth defending. **Second-biggest:** describing the EvalForge CI eval gate as novel — Braintrust and promptfoo both ship exactly that, and an ML-platform interviewer will know it cold. **Third:** claiming comfy-workflow-pack is production-proven; the README itself says one live execution.

**Standing rule for all six:** lead with the closest prior art by name, then state the delta. Naming Numonic, comfy-pack, Braintrust, and Smilepass unprompted converts every one of these from a novelty claim into a demonstration of landscape awareness — which is worth more to a senior interviewer than the novelty would have been.
