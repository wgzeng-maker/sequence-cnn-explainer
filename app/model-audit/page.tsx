"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import Link from "next/link";
import auditArtifact from "../data/model-audit-summary.json";
import { CHANNEL_ORDER_LABELS, type ChannelOrder } from "../model-analysis";
import { loadFloat32Tensor } from "../tensor-loader";
import styles from "./page.module.css";

type LayerName = "stem" | "res1" | "res2" | "res3" | "res4" | "res5" | "res6" | "res7" | "res8";
type RegistryRow = { id_zero_based: number; label: string; checkpoint_id: string; stem_occupancy: number; final_rms: number; profile_influence: number; count_influence: number; final_mean_activation: number; count_weight: number; count_contribution: number };
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
    heads: { profile_position_energy: number[]; profile_position_center_of_mass: number; profile_effective_input_channels: number; count_profile_absolute_weight_correlation: number; count_weights: number[]; count_bias: number; logcount: number; predicted_total_count: number };
  };
  activation_motifs: { status: string; reason: string; target_site_count_per_filter: number; planned_corpus: string; selection_rule: string };
};
type Artifact = { schema_version: string; evidence_levels: Array<{ id: string; label: string; definition: string }>; checkpoints: Record<string, Checkpoint>; planned_corpus_audit: Record<string, string> };

const audit = auditArtifact as unknown as Artifact;
const LAYERS: LayerName[] = ["stem", "res1", "res2", "res3", "res4", "res5", "res6", "res7", "res8"];
const ORDER_KEYS = Object.keys(CHANNEL_ORDER_LABELS) as ChannelOrder[];
const PROFILE_KERNEL_URLS: Record<string, string> = {
  "k562-peak": "/data/tensors/k562_peak/profile_kernel.f32.gz",
  gm21515: "/data/tensors/gm21515/profile_kernel.f32.gz",
};
const css = (values: Record<string, string | number>) => values as CSSProperties;
const percent = (value: number) => `${(value * 100).toFixed(2)}%`;
const format = (value: number) => value === 0 ? "0" : Math.abs(value) < .001 ? value.toExponential(2) : value.toFixed(4);

function QuantileChart({ values }: { values: number[] }) {
  const labels = ["min", "p01", "p25", "p50", "p75", "p99", "max"];
  const maximum = Math.max(...values.map(Math.abs), 1e-12);
  return <div className={styles.quantileWrap}>
    <div className={styles.quantileChart}>{values.map((value, index) => <div key={labels[index]}><span>{labels[index]}</span><i style={css({ "--bar": Math.sqrt(Math.abs(value) / maximum) })} /><b>{format(value)}</b></div>)}</div>
    <small>Seven cut-points summarize every tensor entry. Bar height uses a square-root display scale so p99 remains visible; printed values are exact.</small>
  </div>;
}

function TapEnergyChart({ values }: { values: number[] }) {
  const labels = ["left", "center", "right"];
  return <div className={styles.tapEnergy}>
    <small className={styles.tapTotal}>one block = 100% weight energy</small>
    <div className={styles.tapStack} role="img" aria-label={labels.map((label, index) => `${label} tap ${(values[index] * 100).toFixed(2)} percent`).join(", ")}>
      {values.map((value, index) => <i key={labels[index]} title={`${labels[index]} tap: ${(value * 100).toFixed(2)}%`} style={{ width: `${value * 100}%` }} />)}
    </div>
    <div className={styles.tapLegend}>{values.map((value, index) => <div key={labels[index]}><i /><span>{labels[index]} tap</span><b>{(value * 100).toFixed(2)}%</b></div>)}</div>
  </div>;
}

function ProfileEnergyChart({ values }: { values: number[] }) {
  const maximum = Math.max(...values, 1e-12);
  return <div className={styles.profileEnergy}>
    <div>{values.map((value, index) => <i key={index} title={`kernel position ${index}: energy ${format(value)}`} style={{ height: `${Math.max(2, value / maximum * 92)}%` }} />)}</div>
    <footer><span>position 0</span><b>energy = sum of squared weights over 512 channels</b><span>position 74</span></footer>
  </div>;
}

function profileWeightRgb(value: number, scale: number) {
  const background = [247, 244, 236];
  const target = value >= 0 ? [232, 100, 75] : [50, 136, 163];
  const level = Math.min(Math.abs(value) / Math.max(scale, 1e-12), 1);
  return background.map((start, index) => Math.round(start + (target[index] - start) * level));
}

function ProfileKernelHeatmap({ preset, order }: { preset: string; order: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [weights, setWeights] = useState<Float32Array | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [gain, setGain] = useState(1.8);
  const [cursor, setCursor] = useState({ displayRow: 255, position: 37 });

  useEffect(() => {
    const controller = new AbortController();
    setWeights(null);
    setError(null);
    loadFloat32Tensor(PROFILE_KERNEL_URLS[preset], 512 * 75, controller.signal).then(setWeights).catch(reason => {
      if (reason?.name !== "AbortError") setError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => controller.abort();
  }, [preset]);

  const robustMaximum = useMemo(() => {
    if (!weights) return 1;
    const absolute = Array.from(weights, Math.abs).sort((a, b) => a - b);
    return absolute[Math.floor(absolute.length * .995)] || absolute.at(-1) || 1;
  }, [weights]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !weights) return;
    canvas.width = 75;
    canvas.height = 512;
    const context = canvas.getContext("2d");
    if (!context) return;
    const image = context.createImageData(75, 512);
    for (let displayRow = 0; displayRow < 512; displayRow += 1) {
      const channel = order[displayRow];
      for (let position = 0; position < 75; position += 1) {
        const value = weights[channel * 75 + position];
        const [red, green, blue] = profileWeightRgb(value, robustMaximum / gain);
        const pixel = (displayRow * 75 + position) * 4;
        image.data[pixel] = red;
        image.data[pixel + 1] = green;
        image.data[pixel + 2] = blue;
        image.data[pixel + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [weights, order, robustMaximum, gain]);

  const moveCursor = (clientX: number, clientY: number, target: HTMLCanvasElement) => {
    const bounds = target.getBoundingClientRect();
    setCursor({
      position: Math.max(0, Math.min(74, Math.floor((clientX - bounds.left) / bounds.width * 75))),
      displayRow: Math.max(0, Math.min(511, Math.floor((clientY - bounds.top) / bounds.height * 512))),
    });
  };
  const channel = order[cursor.displayRow];
  const selectedWeight = weights?.[channel * 75 + cursor.position];

  return <div className={styles.profileKernelHeatmap}>
    <div className={styles.profileKernelYAxis}><span>rank 1 · Ch {order[0] + 1}</span><b>512 learned feature channels</b><span>rank 512 · Ch {order[511] + 1}</span></div>
    <div className={styles.profileKernelCanvas}>
      {error ? <p>{error}</p> : !weights ? <p>Loading exact profile weights…</p> : null}
      <canvas ref={canvasRef} role="img" aria-label="Signed profile-head kernel heatmap with 512 channels and 75 relative positions" onPointerMove={event => moveCursor(event.clientX, event.clientY, event.currentTarget)} />
      <i style={{ left: `${cursor.position / 75 * 100}%`, top: `${cursor.displayRow / 512 * 100}%`, width: `${100 / 75}%`, height: `${100 / 512}%` }} />
    </div>
    <div className={styles.profileKernelXAxis}><span>relative position 1</span><b>75 kernel positions</b><span>relative position 75</span></div>
    <div className={styles.profileKernelControls}>
      <label><span>Weight contrast · {gain.toFixed(1)}×</span><input type="range" min=".5" max="5" step=".1" value={gain} onChange={event => setGain(Number(event.target.value))} /></label>
      <div><span>Selected cell</span><b>rank {cursor.displayRow + 1} · Channel {channel + 1} · relative position {cursor.position + 1}</b><strong>{selectedWeight === undefined ? "—" : selectedWeight.toFixed(5)}</strong></div>
      <div className={styles.profileKernelLegend}><i /><span>negative weight</span><i /><span>zero</span><i /><span>positive weight</span></div>
    </div>
  </div>;
}

function SignedStrip({ values, order, label }: { values: number[]; order: number[]; label: string }) {
  const maximum = Math.max(...values.map(Math.abs), 1e-12);
  return <div className={styles.signedStrip} aria-label={label}><div>{order.map(channel => {
    const value = values[channel];
    return <i key={channel} title={`Channel ${channel + 1}: ${format(value)}`} className={value >= 0 ? styles.stripPositive : styles.stripNegative} style={css({ "--magnitude": Math.abs(value) / maximum, "--side": value >= 0 ? -1 : 1 })} />;
  })}</div><footer><span>display rank 1</span><b>zero</b><span>display rank 512</span></footer></div>;
}

function CorrelationRuler({ value }: { value: number }) {
  const magnitude = Math.abs(value);
  const strength = magnitude < .1 ? "very weak" : magnitude < .3 ? "weak" : magnitude < .5 ? "moderate" : "strong";
  const direction = value < 0 ? "negative" : value > 0 ? "positive" : "zero";
  const displayed = value.toFixed(3).replace("-", "−");
  return <div className={styles.correlationRuler}><div><span>−1</span><i /><span>0</span><i /><span>+1</span><b style={{ left: `${(value + 1) * 50}%` }} title={displayed} /></div><p><strong>{displayed}</strong> is a <b>{strength} {direction}</b> channel-level association; the shared variance is {(value * value * 100).toFixed(1)}%. This is a descriptive weight statistic for one checkpoint. Confirm its direction and size across independently trained checkpoints and folds before generalizing.</p></div>;
}

function SimilarityMatrix({ checkpoint }: { checkpoint: Checkpoint }) {
  return <div className={styles.similarityPanel}>
    <div className={styles.panelHeading}><div><small>LAYER-SIMILARITY MATRIX</small><h2>Which stages retain similar position-level representations?</h2></div><span>{checkpoint.layer_similarity.method}</span></div>
    <div className={styles.similarityGrid} style={css({ "--size": checkpoint.layer_similarity.labels.length + 1 })}>
      <i />{checkpoint.layer_similarity.labels.map(label => <b key={`top-${label}`}>{label}</b>)}
      {checkpoint.layer_similarity.values.flatMap((row, y) => [<b key={`left-${y}`}>{checkpoint.layer_similarity.labels[y]}</b>, ...row.map((value, x) => <i key={`${x}-${y}`} style={{ background: `color-mix(in srgb, var(--blue) ${Math.round(value * 100)}%, #f7f4ec)` }} title={`${checkpoint.layer_similarity.labels[y]} × ${checkpoint.layer_similarity.labels[x]}: ${value.toFixed(3)}`}>{value.toFixed(2)}</i>)])}
    </div>
    <p>This pilot CKA compares 128 aligned positions and each layer’s 64 highest-variance channels, as stated in the method label—not every tensor entry. It does not claim that Channel 17 in one layer is the same feature as Channel 17 in another.</p>
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
      <label><span>Checkpoint</span><select data-testid="checkpoint-select" aria-label="Checkpoint" value={preset} onChange={event => setPreset(event.target.value)}><option value="k562-peak">K562 · DNase · fold 0</option><option value="gm21515">GM21515 · ATAC · fold 0</option></select></label>
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
      <div className={styles.layerTable}><span>layer</span><span>shape</span><span>exact zeros</span><span>median active / position</span><span>dynamic range</span><span>positive runs</span>{layerValues.map((metric, index) => <div className={layer === LAYERS[index] ? styles.selectedRow : ""} key={LAYERS[index]}><button aria-pressed={layer === LAYERS[index]} onClick={() => setLayer(LAYERS[index])}>{LAYERS[index]}</button><span>{metric.shape[0]} × {metric.shape[1].toLocaleString()}</span><span>{percent(metric.exact_zero_fraction)}</span><span>{metric.median_active_channels_per_position.toFixed(0)} / 512</span><span>{format(metric.dynamic_range)}</span><span>{metric.positive_run_count.toLocaleString()}</span></div>)}</div>
      <div className={styles.metricDefinitions}><p><b>Median active / position</b><span>At each position, count channels with activation &gt; 10⁻⁷; then take the median across positions.</span></p><p><b>Positive runs</b><span>Total contiguous above-threshold stretches, counted separately within every channel. One long line is one run; one isolated cell is also one run.</span></p></div>
      <div className={styles.selectedMetrics}><div><small>SELECTED LAYER</small><h3>{layer} · {checkpoint.layers[layer].shape.join(" × ")}</h3><p>This is a distribution summary of all {checkpoint.layers[layer].shape[0].toLocaleString()} × {checkpoint.layers[layer].shape[1].toLocaleString()} entries—not seven selected channels or positions. ReLU sparsity is why several percentiles can all equal zero.</p></div><QuantileChart values={checkpoint.layers[layer].value_quantiles} /></div>
    </section>

    <section className={styles.section}>
      <div className={styles.panelHeading}><div><small>PERMANENT CHANNEL REGISTRY</small><h2>Sort the display without losing channel identity</h2></div><span>showing first 80 of 512</span></div>
      <p className={styles.explainer}>The immutable ID is zero-based internally and shown as Channel 1–512. Display rank can change; the ID and checkpoint provenance cannot. Equal IDs across independently trained checkpoints do not imply correspondence.</p>
      <div className={styles.channelTable}><span>rank</span><span>immutable ID</span><span>checkpoint</span><span>stem occupancy</span><span>final RMS</span><span>profile influence</span><span>count influence</span>{rows.map(row => <div key={row.id_zero_based}><b>#{row.rank}</b><strong>Channel {row.id_zero_based + 1}</strong><small>{row.checkpoint_id}</small><span>{percent(row.stem_occupancy)}</span><span>{format(row.final_rms)}</span><span>{format(row.profile_influence)}</span><span>{format(row.count_influence)}</span></div>)}</div>
      <p className={styles.caveat}>“Influence” here is a single-locus empirical activation×head-weight proxy. It is useful for ordering the display, not a causal or biological conclusion.</p>
    </section>

    <section className={styles.section}>
      <div className={styles.panelHeading}><div><small>KERNEL DIAGNOSTICS</small><h2>Do dilated kernels really mix channels?</h2></div><span>complete 3 × 512 × 512 kernels</span></div>
      <div className={styles.residualCards}>{residual.map(block => <article key={block.block}><header><b>Block {block.block}</b><span>d={block.dilation}</span></header><TapEnergyChart values={block.tap_energy_fraction} /><dl><dt>diagonal energy</dt><dd>{percent(block.diagonal_energy_fraction)}</dd><dt>effective input channels · median</dt><dd>{block.effective_input_channels_quantiles[3].toFixed(1)}</dd><dt>left/right cosine</dt><dd>{block.left_right_tap_cosine.toFixed(3)}</dd><dt>stable / effective rank</dt><dd>{block.stable_rank.toFixed(1)} / {block.effective_rank.toFixed(1)}</dd></dl></article>)}</div>
      <p className={styles.takeaway}><b>Surprising result:</b> in every block, left, center, and right taps each carry close to one-third of the kernel’s total squared-weight energy. Every bar uses the full 0–100% width with no magnification, so the nearly equal thirds are the visual message. The printed percentages preserve the small differences. Weight energy measures squared weight magnitude—not activation or causal importance.</p>
    </section>

    <SimilarityMatrix checkpoint={checkpoint} />

    <section className={styles.section}>
      <div className={styles.panelHeading}><div><small>HEAD DIAGNOSTICS</small><h2>How the final feature tensor is read</h2></div><span>profile and count remain separate readers</span></div>
      <div className={styles.headGrid}><article><small>PROFILE KERNEL</small><b>75 × 512 × 1</b><span>Every pixel below is one signed weight connecting a final feature channel at one relative position to one profile logit. The selected global channel order is retained vertically.</span><div className={styles.profileDiagnosticGrid}><ProfileKernelHeatmap preset={preset} order={order} /><div className={styles.profileEnergySummary}><b>Collapsed positional energy</b><span>At each relative position, square and sum all 512 channel weights. This summary hides sign and channel identity; the heatmap preserves both.</span><span>effective input channels: {checkpoint.kernels.heads.profile_effective_input_channels.toFixed(1)}</span><span>energy center: {checkpoint.kernels.heads.profile_position_center_of_mass.toFixed(2)} / 74</span><ProfileEnergyChart values={checkpoint.kernels.heads.profile_position_energy} /></div></div></article><article><small>COUNT–PROFILE CHANNEL-WEIGHT CORRELATION</small><CorrelationRuler value={checkpoint.kernels.heads.count_profile_absolute_weight_correlation} /><span>Pearson correlation across 512 channels: |count dense weight| versus √(profile weight energy). This is not prediction agreement, accuracy, or biological agreement.</span></article></div>
      <div className={styles.countDense}><div className={styles.countFlow}><span><small>FINAL TENSOR</small><b>512 × 1,074</b></span><i>mean each channel<br />across positions →</i><span><small>POOLED FEATURES</small><b>512 numbers</b></span><i>× 512 dense weights<br />and sum →</i><span><small>LOG-COUNT</small><b>{checkpoint.kernels.heads.logcount.toFixed(3)}</b></span><i>exp(x) − 1 →</i><span className={styles.countResult}><small>PREDICTED TOTAL</small><b>{checkpoint.kernels.heads.predicted_total_count.toFixed(1)}</b></span></div>
        <div className={styles.denseChart}><header><b>Learned dense weights</b><span>coral raises log-count · blue lowers it</span></header><SignedStrip values={checkpoint.channel_registry.map(row => row.count_weight)} order={order} label="Count dense weights by channel" /></div>
        <div className={styles.denseChart}><header><b>This locus: pooled activation × weight</b><span>same global channel order · signed contribution before bias</span></header><SignedStrip values={checkpoint.channel_registry.map(row => row.count_contribution)} order={order} label="Count contributions by channel" /></div>
        <p className={styles.denseEquation}>Σ 512 channel contributions <b>{(checkpoint.kernels.heads.logcount - checkpoint.kernels.heads.count_bias).toFixed(3)}</b> + dense bias <b>{checkpoint.kernels.heads.count_bias.toFixed(3)}</b> = log-count <b>{checkpoint.kernels.heads.logcount.toFixed(3)}</b>.</p>
      </div>
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
