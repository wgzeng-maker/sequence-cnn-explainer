"use client";

import { useMemo, useState, type CSSProperties } from "react";
import Link from "next/link";
import auditArtifact from "../data/model-audit-summary.json";
import { CHANNEL_ORDER_LABELS, type ChannelOrder } from "../model-analysis";
import styles from "./page.module.css";

type LayerName = "stem" | "res1" | "res2" | "res3" | "res4" | "res5" | "res6" | "res7" | "res8";
type RegistryRow = { id_zero_based: number; label: string; checkpoint_id: string; stem_occupancy: number; final_rms: number; profile_influence: number; count_influence: number };
type LayerMetric = { shape: number[]; exact_zero_fraction: number; tolerance_zero_fraction: number; value_quantiles: number[]; dynamic_range: number; median_active_channels_per_position: number; positive_run_count: number };
type Checkpoint = {
  checkpoint: { experiment: string; biosample: string; assay: string; fold: number; model: string };
  evidence_scope: { activation_sample_count: number; activation_corpus: string; warning: string; model_weights: string };
  channel_registry: RegistryRow[];
  channel_orders: Record<ChannelOrder, number[]>;
  layers: Record<LayerName, LayerMetric>;
  layer_similarity: { method: string; labels: LayerName[]; values: number[][] };
  kernels: {
    stem: Record<string, number | number[] | number[][]>;
    residual: Array<{ block: number; dilation: number; tap_energy_fraction: number[]; left_right_tap_cosine: number; diagonal_energy_fraction: number; effective_input_channels_quantiles: number[]; stable_rank: number; effective_rank: number }>;
    heads: { profile_position_energy: number[]; profile_position_center_of_mass: number; profile_effective_input_channels: number; count_profile_absolute_weight_correlation: number };
  };
  activation_motifs: { status: string; reason: string; target_site_count_per_filter: number; planned_corpus: string; selection_rule: string };
};
type Artifact = { schema_version: string; evidence_levels: Array<{ id: string; label: string; definition: string }>; checkpoints: Record<string, Checkpoint>; planned_corpus_audit: Record<string, string> };

const audit = auditArtifact as unknown as Artifact;
const LAYERS: LayerName[] = ["stem", "res1", "res2", "res3", "res4", "res5", "res6", "res7", "res8"];
const ORDER_KEYS = Object.keys(CHANNEL_ORDER_LABELS) as ChannelOrder[];
const css = (values: Record<string, string | number>) => values as CSSProperties;
const percent = (value: number) => `${(value * 100).toFixed(2)}%`;
const format = (value: number) => value === 0 ? "0" : Math.abs(value) < .001 ? value.toExponential(2) : value.toFixed(4);

function MetricBars({ values, labels }: { values: number[]; labels: string[] }) {
  const maximum = Math.max(...values.map(Math.abs), 1e-12);
  return <div className={styles.metricBars}>{values.map((value, index) => <div key={labels[index]}><span>{labels[index]}</span><i style={css({ "--bar": Math.abs(value) / maximum })} /><b>{format(value)}</b></div>)}</div>;
}

function SimilarityMatrix({ checkpoint }: { checkpoint: Checkpoint }) {
  return <div className={styles.similarityPanel}>
    <div className={styles.panelHeading}><div><small>LAYER-SIMILARITY MATRIX</small><h2>Which stages retain similar position-level representations?</h2></div><span>{checkpoint.layer_similarity.method}</span></div>
    <div className={styles.similarityGrid} style={css({ "--size": checkpoint.layer_similarity.labels.length + 1 })}>
      <i />{checkpoint.layer_similarity.labels.map(label => <b key={`top-${label}`}>{label}</b>)}
      {checkpoint.layer_similarity.values.flatMap((row, y) => [<b key={`left-${y}`}>{checkpoint.layer_similarity.labels[y]}</b>, ...row.map((value, x) => <i key={`${x}-${y}`} style={{ background: `color-mix(in srgb, var(--blue) ${Math.round(value * 100)}%, #f7f4ec)` }} title={`${checkpoint.layer_similarity.labels[y]} × ${checkpoint.layer_similarity.labels[x]}: ${value.toFixed(3)}`}>{value.toFixed(2)}</i>)])}
    </div>
    <p>CKA compares whole representations after aligning positions. It does not claim that Channel 17 in one layer is the same feature as Channel 17 in another.</p>
  </div>;
}

export default function ModelAuditPage() {
  const [preset, setPreset] = useState("k562-peak");
  const [orderName, setOrderName] = useState<ChannelOrder>("stem_occupancy");
  const [layer, setLayer] = useState<LayerName>("stem");
  const [evidence, setEvidence] = useState("descriptive");
  const checkpoint = audit.checkpoints[preset];
  const order = checkpoint.channel_orders[orderName];
  const rows = useMemo(() => order.slice(0, 80).map((channel, rank) => ({ ...checkpoint.channel_registry[channel], rank: rank + 1 })), [checkpoint, order]);
  const residual = checkpoint.kernels.residual;
  const layerValues = LAYERS.map(name => checkpoint.layers[name]);

  return <main className={styles.audit}>
    <header className={styles.topbar}><Link href="/">← Main explainer</Link><b>MODEL AUDIT</b><Link href="/dilation-trace">Dilation evolution ↗</Link></header>
    <section className={styles.hero}><p>CHROMBPNET · FOLD-0 WORKSPACE</p><h1>Separate what the tensors show from what the model uses—and what biology supports.</h1><span>This page is intentionally denser than the explainer. Every panel states its evidence scope, checkpoint, and sample size.</span></section>

    <section className={styles.controls}>
      <label><span>Checkpoint</span><select value={preset} onChange={event => setPreset(event.target.value)}><option value="k562-peak">K562 · DNase · fold 0</option><option value="gm21515">GM21515 · ATAC · fold 0</option></select></label>
      <label><span>Layer</span><select value={layer} onChange={event => setLayer(event.target.value as LayerName)}>{LAYERS.map(name => <option key={name}>{name}</option>)}</select></label>
      <label><span>Global channel order</span><select value={orderName} onChange={event => setOrderName(event.target.value as ChannelOrder)}>{ORDER_KEYS.map(key => <option value={key} key={key}>{CHANNEL_ORDER_LABELS[key]}</option>)}</select></label>
    </section>

    <section className={styles.scope}>
      <div><small>CHECKPOINT</small><b>{checkpoint.checkpoint.biosample} · {checkpoint.checkpoint.assay}</b><span>{checkpoint.checkpoint.experiment} · fold {checkpoint.checkpoint.fold} · complete stored weights</span></div>
      <div><small>ACTIVATION SAMPLE</small><b>{checkpoint.evidence_scope.activation_sample_count} exact locus</b><span>{checkpoint.evidence_scope.activation_corpus}</span></div>
      <div className={styles.warning}><small>INTERPRETATION LIMIT</small><b>Descriptive pilot</b><span>{checkpoint.evidence_scope.warning}</span></div>
    </section>

    <section className={styles.section}>
      <div className={styles.panelHeading}><div><small>EVIDENCE LADDER</small><h2>Three different strengths of claim</h2></div><span>Do not collapse these into one “important” label.</span></div>
      <div className={styles.evidenceTabs}>{audit.evidence_levels.map(item => <button key={item.id} className={evidence === item.id ? styles.active : ""} onClick={() => setEvidence(item.id)}><b>{item.label}</b><span>{item.definition}</span></button>)}</div>
      <div className={styles.evidenceDetail}>{evidence === "descriptive" ? <><b>Available now</b><p>Stored weights, exact forward activations, sparsity, spectra, channel occupancy, receptive-field geometry, and representation similarity.</p></> : evidence === "mechanism" ? <><b>Next generated analysis</b><p>Activation×weight contributions and single-channel ablations first; pair ablations only for shortlisted pairs. A large activation is not automatically important.</p></> : <><b>Requires external biological validation</b><p>TF-MoDISco contribution motifs, Tomtom/JASPAR family hypotheses, then deletion, matched-background insertion, multiplicity, spacing, and orientation experiments.</p></>}</div>
    </section>

    <section className={styles.section}>
      <div className={styles.panelHeading}><div><small>LAYER TABLE</small><h2>How the observed representation changes through the backbone</h2></div><span>exact full tensor · one displayed locus</span></div>
      <div className={styles.layerTable}><span>layer</span><span>shape</span><span>exact zeros</span><span>median active / position</span><span>dynamic range</span><span>positive runs</span>{layerValues.map((metric, index) => <div className={layer === LAYERS[index] ? styles.selectedRow : ""} key={LAYERS[index]} onClick={() => setLayer(LAYERS[index])}><b>{LAYERS[index]}</b><span>{metric.shape[0]} × {metric.shape[1].toLocaleString()}</span><span>{percent(metric.exact_zero_fraction)}</span><span>{metric.median_active_channels_per_position.toFixed(0)} / 512</span><span>{format(metric.dynamic_range)}</span><span>{metric.positive_run_count.toLocaleString()}</span></div>)}</div>
      <div className={styles.selectedMetrics}><div><small>SELECTED LAYER</small><h3>{layer} · {checkpoint.layers[layer].shape.join(" × ")}</h3><p>Value quantiles: {checkpoint.layers[layer].value_quantiles.map(format).join(" · ")}</p></div><MetricBars values={checkpoint.layers[layer].value_quantiles} labels={["min", "p01", "p25", "p50", "p75", "p99", "max"]} /></div>
    </section>

    <section className={styles.section}>
      <div className={styles.panelHeading}><div><small>PERMANENT CHANNEL REGISTRY</small><h2>Sort the display without losing channel identity</h2></div><span>showing first 80 of 512</span></div>
      <p className={styles.explainer}>The immutable ID is zero-based internally and shown as Channel 1–512. Display rank can change; the ID and checkpoint provenance cannot. Equal IDs across independently trained checkpoints do not imply correspondence.</p>
      <div className={styles.channelTable}><span>rank</span><span>immutable ID</span><span>checkpoint</span><span>stem occupancy</span><span>final RMS</span><span>profile influence</span><span>count influence</span>{rows.map(row => <div key={row.id_zero_based}><b>#{row.rank}</b><strong>Channel {row.id_zero_based + 1}</strong><small>{row.checkpoint_id}</small><span>{percent(row.stem_occupancy)}</span><span>{format(row.final_rms)}</span><span>{format(row.profile_influence)}</span><span>{format(row.count_influence)}</span></div>)}</div>
      <p className={styles.caveat}>“Influence” here is a single-locus empirical activation×head-weight proxy. It is useful for ordering the display, not a causal or biological conclusion.</p>
    </section>

    <section className={styles.section}>
      <div className={styles.panelHeading}><div><small>KERNEL DIAGNOSTICS</small><h2>Do dilated kernels really mix channels?</h2></div><span>complete 3 × 512 × 512 kernels</span></div>
      <div className={styles.residualCards}>{residual.map(block => <article key={block.block}><header><b>Block {block.block}</b><span>d={block.dilation}</span></header><MetricBars values={block.tap_energy_fraction} labels={["left tap", "center", "right tap"]} /><dl><dt>diagonal energy</dt><dd>{percent(block.diagonal_energy_fraction)}</dd><dt>effective input channels · median</dt><dd>{block.effective_input_channels_quantiles[3].toFixed(1)}</dd><dt>left/right cosine</dt><dd>{block.left_right_tap_cosine.toFixed(3)}</dd><dt>stable / effective rank</dt><dd>{block.stable_rank.toFixed(1)} / {block.effective_rank.toFixed(1)}</dd></dl></article>)}</div>
      <p className={styles.takeaway}><b>Reading this:</b> low diagonal energy means a residual output channel is not merely reading the same-numbered input channel. Each tap mixes many learned feature channels; dilation controls how far apart the three mixed feature vectors are.</p>
    </section>

    <SimilarityMatrix checkpoint={checkpoint} />

    <section className={styles.section}>
      <div className={styles.panelHeading}><div><small>HEAD DIAGNOSTICS</small><h2>How the final feature tensor is read</h2></div><span>profile and count remain separate readers</span></div>
      <div className={styles.headGrid}><article><small>PROFILE KERNEL</small><b>75 × 512 × 1</b><span>effective input channels: {checkpoint.kernels.heads.profile_effective_input_channels.toFixed(1)}</span><span>positional energy center: {checkpoint.kernels.heads.profile_position_center_of_mass.toFixed(2)} / 74</span><MetricBars values={checkpoint.kernels.heads.profile_position_energy} labels={checkpoint.kernels.heads.profile_position_energy.map((_, index) => `${index + 1}`)} /></article><article><small>COUNT / PROFILE AGREEMENT</small><b>{checkpoint.kernels.heads.count_profile_absolute_weight_correlation.toFixed(3)}</b><span>correlation between absolute count weights and profile channel-energy</span><p>A weak correlation is not a contradiction: the two heads ask different questions—total amount versus spatial distribution.</p></article></div>
    </section>

    <section className={styles.section}>
      <div className={styles.panelHeading}><div><small>ACTIVATION MOTIFS</small><h2>Corpus-derived logos are deliberately not fabricated</h2></div><span>status: {checkpoint.activation_motifs.status.replace("_", " ")}</span></div>
      <div className={styles.emptyState}><b>Weight logos are available in the main explainer.</b><p>{checkpoint.activation_motifs.reason}</p><dl><dt>planned corpus</dt><dd>{checkpoint.activation_motifs.planned_corpus}</dd><dt>reservoir</dt><dd>up to {checkpoint.activation_motifs.target_site_count_per_filter} sites per filter</dd><dt>selection</dt><dd>{checkpoint.activation_motifs.selection_rule}</dd></dl></div>
    </section>

    <section className={styles.section}>
      <div className={styles.panelHeading}><div><small>BIOLOGICAL BRIDGE</small><h2>From model patterns to testable motif hypotheses</h2></div><span>future offline analyses · clearly labeled</span></div>
      <ol className={styles.protocol}><li><b>Activation logo</b><span>Collect high-activating stem 21-mers. This asks what sequences trigger a first-layer filter.</span></li><li><b>Attribution motifs</b><span>Generate profile-head and count-head contribution scores separately, then run TF-MoDISco.</span></li><li><b>Database hypothesis</b><span>Compare discovered motifs and reverse complements to JASPAR CORE with Tomtom; report q-value, alignment, seqlets, and family ambiguity.</span></li><li><b>Perturbation</b><span>Test deletion, matched-background insertion, multiplicity, spacing, and orientation. A JASPAR match is not proof of TF identity.</span></li></ol>
      <div className={styles.links}><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC6941814/">Koo & Eddy · motif representations ↗</a><a href="https://proceedings.mlr.press/v97/kornblith19a.html">CKA ↗</a><a href="https://academic.oup.com/nar/article/52/D1/D174/7420101">JASPAR 2024 ↗</a><a href="https://meme-suite.org/meme/doc/tomtom.html">Tomtom ↗</a></div>
    </section>

    <aside className={styles.footerNote}><b>Planned population audit</b><span>{audit.planned_corpus_audit.fast_pass}</span><span>{audit.planned_corpus_audit.motif_pass}</span><span>{audit.planned_corpus_audit.storage}</span><span>{audit.planned_corpus_audit.replication}</span></aside>
    <footer>Audit artifact schema {audit.schema_version} · full intermediate activations remain outside the deployed JSON.</footer>
  </main>;
}
