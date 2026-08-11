# Sequence CNN Explainer: Interpretation Roadmap and Claude Handoff

## Decision

Add **Basset** as the next model, followed by **Basenji/Basenji2**.

Basset is the most informative contrast to ChromBPNet. ChromBPNet preserves fine positional resolution and expands context with dilated residual convolutions. Basset instead uses ordinary convolution, max pooling, and dense layers to turn local motif evidence into one accessibility prediction for each of 164 cell types. It therefore provides a clean answer to: “How can an ordinary CNN combine distant motifs without dilation?”

Basenji should follow because it adds a genuinely long sequence (about 131 kb), pooling into 128-bp bins, dilated context aggregation, quantitative profile prediction, and thousands of output tracks.

Primary references:

- [Basset paper](https://genome.cshlp.org/content/26/7/990) and [original repository with a downloadable pretrained model](https://github.com/davek44/Basset)
- [Basenji paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5932613/) and [official repository](https://github.com/calico/basenji)

## Interpretation methods still missing

The current explainer is strongest as a **forward-pass microscope**: it shows weights, activations, tensor statistics, channel mixing, receptive-field geometry, head calculations, and CKA. Computer-vision research suggests adding three other perspectives: backward tracing, dataset-level semantics, and causal intervention.

### Priority 1 — add before or during Basset

1. **Theoretical versus effective receptive field**
   - The theoretical receptive field marks every input base that could affect a unit.
   - The effective receptive field displays the gradient magnitude from one chosen internal/output unit back to every input position.
   - Show both on the same coordinate axis. This directly tests whether ChromBPNet really uses the full 1,041/1,115-bp reach.
   - Reference: [Luo et al., NeurIPS 2016](https://papers.nips.cc/paper_files/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html).

2. **Dataset activation atlas for intermediate channels**
   - For each channel, collect top-activating windows across many held-out sequences.
   - Cluster the windows before making logos. One channel may have several “facets,” so a single averaged logo can erase real alternatives.
   - Display: channel → activation histogram → clustered sequence examples → one logo per cluster.
   - This generalizes the planned stem activation motif to residual channels.
   - Reference: [Feature Visualization](https://distill.pub/2017/feature-visualization/), especially its warning that one optimized example may show only one facet.

3. **Occlusion and in-silico mutagenesis**
   - Mutate or mask short sequence windows and recompute the selected output.
   - Display the change in profile shape and count separately.
   - Use deletion/insertion and dinucleotide- or GC-matched controls; do not call raw activation causal.
   - Reference: [Zeiler and Fergus, 2014](https://arxiv.org/abs/1311.2901). Basset’s original software also includes saturated mutagenesis.

4. **Channel ablation and pair-interaction tests**
   - Zero one channel at a selected layer, then recompute both heads.
   - For shortlisted pairs, calculate interaction:
     `Δ(A+B) − Δ(A) − Δ(B)`.
   - Positive or negative interaction is evidence of model-level cooperation or competition, not automatically biological cooperation.

### Priority 2 — add to the Model Audit

5. **Linear probes across layers**
   - Freeze the CNN and train simple linear models on each layer to test what becomes decodable: motif presence, accessibility class, count, cell type, or assay.
   - This asks what information is available at a layer, not how the original head uses it.
   - Reference: [Alain and Bengio, 2016](https://arxiv.org/abs/1610.01644).

6. **Concept dissection / TCAV-style tests**
   - Define biological concept sets such as motif family, promoter, enhancer, GC-rich sequence, repeat class, or spacing pattern.
   - Measure which channels or activation-space directions associate with each concept.
   - Keep motif-family matches as hypotheses, not TF-identity claims.
   - References: [Network Dissection](https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html) and [TCAV](https://arxiv.org/abs/1711.11279).

7. **Co-activation and redundancy maps**
   - Calculate channel correlation or mutual information over a held-out corpus, not one locus.
   - Cluster channels and compare those clusters with residual-kernel connectivity and ablation effects.
   - This can reveal redundant channel families and candidate feature coalitions.

### Priority 3 — research experiments, not main-page defaults

8. **Activation maximization for deeper channels**
   - Optimize a probabilistic or discrete DNA sequence to activate a selected channel or output.
   - Generate diverse solutions and compare them with real top-activating windows.
   - Clearly mark optimized sequences as synthetic; unconstrained optimization can create off-distribution artifacts.
   - Reference: [Feature Visualization](https://distill.pub/2017/feature-visualization/).

9. **Representation inversion**
   - Starting from an intermediate tensor, optimize a sequence that reconstructs that representation.
   - This asks what sequence information remains recoverable after pooling or deep residual processing.
   - Reference: [Mahendran and Vedaldi, CVPR 2015](https://www.robots.ox.ac.uk/~vedaldi/research/old/visualization/visualization.html).

10. **Output-targeted localization**
    - Grad-CAM is useful in vision, but for one-hot DNA its coarse channel-weighted map should be compared with nucleotide-resolution gradients, Integrated Gradients, DeepLIFT, or contribution scores.
    - Keep this in the generated-attribution section, not among raw activations.
    - Reference: [Grad-CAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).

## Multi-model implementation sequence

### Milestone A — finish shared analysis primitives

- Implement effective receptive-field overlays.
- Implement dataset activation reservoirs and clustered channel facets.
- Implement single-channel ablation and sequence-window occlusion.
- Add an evidence label to every view: weight, activation, gradient/attribution, intervention, or biological validation.
- Make all model-specific tensor adapters produce a common layer manifest.

Acceptance criteria:

- Every plotted value has a formula, sample definition, coordinate system, and checkpoint provenance.
- The same sequence coordinate remains aligned across raw sequence, activation, attribution, and intervention tracks.
- Synthetic examples and single-locus observations cannot be mistaken for corpus results.

### Milestone B — Basset

- Obtain the original pretrained Basset checkpoint and a documented test sequence; do not retrain.
- Convert or wrap the original Torch7 model without changing numerical predictions.
- Verify every exported layer against the original model on fixed test inputs.
- Visualize:
  - 600 × 4 one-hot input;
  - three convolution/ReLU/max-pooling stages;
  - the loss of fine positional resolution after each pool;
  - dense layers combining features from all remaining positions;
  - 164 cell-type accessibility probabilities.
- Add a comparison panel: ChromBPNet dilation versus Basset pooling + dense communication.

### Milestone C — Basenji/Basenji2

- Confirm the exact published checkpoint, species, targets, input length, bin size, and output tracks before downloading large files.
- Start with one target track and a compact set of intermediate summaries.
- Visualize the hierarchy: bases → pooled 128-bp bins → dilated long-range context → quantitative output tracks.
- Compare theoretical and effective receptive fields and retain identical evidence labels.

### Milestone D — freeze the communication release

Before handing the folder to Claude, create:

- `MODEL_MANIFEST.md`: exact checkpoint, source URL, license, input/output definitions, tensor shapes, and numerical tests for every model.
- `CLAIM_LEDGER.md`: each intended claim, its evidence level, supporting visualization/data, and caveats.
- `VISUAL_ASSET_INDEX.md`: page/section, screenshot or export instructions, caption, and provenance.
- `GLOSSARY.md`: channel, filter, kernel, activation, attribution, intervention, receptive field, profile, count, and pooling.
- A tagged release or commit hash that Claude must treat as read-only source evidence.

## Claude production workflow

The workflow has explicit gates. Claude must not jump from reading the folder directly to producing a video.

1. **Repository audit** — Claude reads the visualizer, runs it locally, maps every claim to code/data, and reports contradictions or missing evidence.
2. **Research report** — after resolving blockers, Claude writes a rigorous but readable report.
3. **Narrative script** — Claude writes a scene-by-scene YouTube script for viewers who understand image CNN intuition but are new to genomic sequences.
4. **Slide deck** — Claude creates Google Slides if a Slides integration is available; otherwise it creates a `.pptx` plus a Slides-ready asset folder and speaker notes.
5. **Human audit gate** — stop. The user reviews and approves the slides and narration. No video is generated before explicit approval.
6. **Video production** — after approval, Claude generates narration, slide timing, captions, and restrained animations, then renders a draft video if the necessary TTS/video tools are available.

## Prompt to send to Claude Code

```text
You are working inside the Sequence CNN Explainer repository. Treat the checked-out commit as the evidence source for this project.

Goal
Turn the visualizer into a rigorous, readable research communication package: a technical report, a YouTube narrative, an auditable slide deck, and—only after my explicit approval—a narrated video.

Audience
The audience understands the intuition of the image-based CNN Explainer (filters slide over images, early layers detect local patterns, later layers combine them), but does not yet have a strong mental model for one-hot DNA sequences, genomic tensors, dilated convolution, pooling, or profile prediction.

Non-negotiable scientific rules
1. Distinguish weights, activations, gradients/attributions, interventions, model predictions, and biological validation. Never use “importance” without saying which quantity was measured.
2. Distinguish a single-locus observation from a corpus-level result.
3. Treat motif/database matches as candidate motif families, not proof of TF identity.
4. Do not invent citations, model dimensions, checkpoint provenance, numerical results, or biological conclusions.
5. Verify important claims against the code, exported artifacts, numerical tests, and primary papers.
6. Preserve sequence coordinates and explain any cropping, pooling, binning, or receptive-field transformation.
7. Prefer an intuitive explanation first, then the precise mathematical explanation.

Phase 1 — read and audit; do not rewrite the application yet
- Read README.md, docs/, app/, scripts/, tests/, model manifests, claim ledger, and visual asset index.
- Run the visualizer and inspect the main explainer, dilation trace, and model-audit pages for every included model.
- Read the referenced CNN Explainer paper/site and the primary paper for every included sequence model.
- Produce `deliverables/01_REPOSITORY_AUDIT.md` containing:
  - architecture and tensor-shape map for each model;
  - inventory of visualizations and what each actually measures;
  - claim-to-evidence table;
  - contradictions, unclear labels, unsupported conclusions, and missing provenance;
  - recommendations ranked as blocking, important, or optional.
- Stop and ask me to resolve blocking scientific ambiguities before proceeding.

Phase 2 — research-level report, optimized for readability
- Write `deliverables/02_TECHNICAL_REPORT.md`.
- Suggested structure:
  1. Executive summary
  2. From image pixels to one-hot sequence matrices
  3. Shared CNN operations
  4. One model card and architecture walkthrough per sequence model
  5. How each architecture combines local motifs over distance
  6. What kernels, intermediate tensors, effective receptive fields, and interventions reveal
  7. Cross-model comparison
  8. Biological hypotheses versus established evidence
  9. Limitations and next experiments
  10. Methods, provenance, and references
- Use primary citations and attach a source to every nontrivial external claim.
- Label observations as descriptive, mechanistic/model intervention, or biological evidence.

Phase 3 — YouTube narrative
- Write `deliverables/03_YOUTUBE_SCRIPT.md` as a scene-by-scene table with:
  - scene number and estimated duration;
  - visual shown from the explainer;
  - narration;
  - on-screen text;
  - animation/cursor action;
  - claim and evidence source.
- Begin from the familiar image-CNN mental model, then translate pixel channels into A/C/G/T channels.
- Build one continuous story: local motif detection → intermediate feature channels → communication across distance → output head.
- Explicitly compare at least two architectures, such as ChromBPNet dilation versus Basset pooling/dense aggregation.
- Avoid reading equations aloud unless the equation materially improves intuition.

Phase 4 — slide deck
- First write `deliverables/04_SLIDE_SPEC.md` with one row per slide: purpose, visual, headline, narration link, source, and transition.
- Reuse visualizer assets or create faithful exports; do not redraw tensors with dimensions that differ from the real checkpoint.
- If direct Google Slides access is available, create the deck there. Otherwise create `deliverables/sequence-cnn-explainer.pptx`, `deliverables/slides-assets/`, and speaker notes so I can import it into Google Slides.
- Make the deck understandable when presented, not necessarily as a standalone paper. Keep detailed evidence in notes and the report.
- Stop after the deck and ask for my audit. Do not create audio or video yet.

Phase 5 — only after I explicitly approve the audited deck and script
- Ask me to choose or approve narration voice, speaking pace, aspect ratio, resolution, caption style, and animation intensity.
- Generate narration audio, slide timings, subtitles/captions, and restrained animations.
- Produce a low-resolution review draft first, then a final 1080p video after approval.
- Preserve an edit decision list so narration or slide corrections do not require rebuilding everything.

Required final inventory
- 01_REPOSITORY_AUDIT.md
- 02_TECHNICAL_REPORT.md
- 03_YOUTUBE_SCRIPT.md
- 04_SLIDE_SPEC.md
- Google Slides URL or PPTX fallback with assets and speaker notes
- After approval only: narration files, captions, edit decision list, review video, and final video

Start with Phase 1 only. Report what you inspected, what you could run, and any blockers. Do not silently fill gaps with assumptions.
```

## Success criterion

The final material should let a reader say, in precise language:

> A stem filter detects local sequence patterns. Later operations combine those learned features across positions and channels. Different architectures use dilation, pooling, dense connections, or global dense layers to do this. A bright activation shows a response; attribution traces a prediction; intervention tests what changes the model; biological experiments establish what happens in cells.
