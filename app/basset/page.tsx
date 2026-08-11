"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import artifact from "../data/basset-demo.json";
import { loadFloat32Tensor } from "../tensor-loader";
import styles from "./page.module.css";

const BASES = ["A", "C", "G", "T"] as const;
const BASE_COLORS = ["#e96b54", "#4f97b2", "#e5b33f", "#72a17e"];

type Asset = { url: string; channels: number; positions: number; values: number; minimum: number; maximum: number };
type Layer = (typeof artifact.layers)[number];
type StemFilter = (typeof artifact.stem_filters)[number];

const css = (values: Record<string, string | number>) => values as CSSProperties;
const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));

function activationColor(level: number) {
  const t = Math.sqrt(clamp(level, 0, 1));
  const stops = [[249, 246, 238], [239, 198, 73], [231, 100, 72], [91, 34, 64]];
  const scaled = t * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const fraction = scaled - index;
  const color = stops[index].map((value, channel) => Math.round(value + (stops[index + 1][channel] - value) * fraction));
  return color;
}

function signedColor(value: number, maximum: number) {
  const strength = Math.sqrt(clamp(Math.abs(value) / Math.max(maximum, 1e-12), 0, 1));
  const paper = [249, 246, 238];
  const target = value >= 0 ? [45, 136, 166] : [231, 95, 72];
  return `rgb(${paper.map((start, index) => Math.round(start + (target[index] - start) * strength)).join(" ")})`;
}

function useTensor(asset: Asset) {
  const [loaded, setLoaded] = useState<{ url: string; values: Float32Array } | null>(null);
  const [failure, setFailure] = useState<{ url: string; message: string } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    loadFloat32Tensor(asset.url, asset.values, controller.signal)
      .then(values => setLoaded({ url: asset.url, values }))
      .catch(reason => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setFailure({ url: asset.url, message: reason instanceof Error ? reason.message : "Tensor could not be loaded" });
      });
    return () => controller.abort();
  }, [asset.url, asset.values]);
  return { values: loaded?.url === asset.url ? loaded.values : null, error: failure?.url === asset.url ? failure.message : null };
}

function SectionHeading({ number, eyebrow, title, children }: { number: string; eyebrow: string; title: string; children: React.ReactNode }) {
  return <div className={styles.sectionHeading}>
    <span>{number}</span>
    <div><small>{eyebrow}</small><h2>{title}</h2><p>{children}</p></div>
  </div>;
}

function SequenceRibbon({ sequence, start, width, onSelect }: { sequence: string; start: number; width: number; onSelect: (position: number) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const move = (clientX: number) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    onSelect(Math.round(clamp((clientX - rect.left) / rect.width, 0, 1) * (sequence.length - width)));
  };
  return <div>
    <div ref={ref} className={styles.sequenceRibbon} onClick={event => move(event.clientX)} onKeyDown={event => {
      if (event.key === "ArrowLeft") onSelect(clamp(start - 1, 0, sequence.length - width));
      if (event.key === "ArrowRight") onSelect(clamp(start + 1, 0, sequence.length - width));
    }} role="button" tabIndex={0} aria-label="Move the 19-base convolution window">
      {Array.from(sequence).map((base, index) => <i key={index} style={{ background: BASE_COLORS[BASES.indexOf(base as typeof BASES[number])] }} />)}
      <b style={{ left: `${start / sequence.length * 100}%`, width: `${width / sequence.length * 100}%` }}><span>{start + 1}–{start + width}</span></b>
    </div>
    <div className={styles.axis}><span>base 1</span><span>click to move the kernel</span><span>base 600</span></div>
  </div>;
}

function KernelMatrix({ weights, sequence, start }: { weights: number[][]; sequence: string; start: number }) {
  const maximum = Math.max(...weights.flat().map(Math.abs));
  return <div className={styles.kernelMatrix}>
    {BASES.map((base, row) => <div key={base}><b>{base}</b><span>{weights[row].map((value, position) => <i key={position} style={{ background: signedColor(value, maximum) }} title={`${base}, kernel position ${position + 1}: ${value.toFixed(5)}`} />)}</span></div>)}
    <div className={styles.kernelLetters}><b>input</b><span>{Array.from(sequence.slice(start, start + 19)).map((base, index) => <i key={index} style={{ background: BASE_COLORS[BASES.indexOf(base as typeof BASES[number])] }}>{base}</i>)}</span></div>
  </div>;
}

function SignalTrack({ values, selected, color = "blue" }: { values: number[]; selected: number; color?: "blue" | "gold" }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);
    const maximum = Math.max(...values, 1e-12);
    context.fillStyle = color === "gold" ? "#dcae35" : "#3288a3";
    values.forEach((value, index) => {
      const x0 = index / values.length * width;
      const x1 = (index + 1) / values.length * width;
      const barHeight = value / maximum * (height - 4);
      context.fillRect(x0, height - barHeight, Math.max(1, x1 - x0), barHeight);
    });
    const x = (selected + 0.5) / values.length * width;
    context.fillStyle = "#e8644b";
    context.fillRect(Math.max(0, x - dpr), 0, 2 * dpr, height);
  }, [values, selected, color]);
  return <canvas ref={ref} className={styles.signalTrack} />;
}

function TensorCanvas({ values, asset, contrast, selected, onSelect }: { values: Float32Array | null; asset: Asset; contrast: number; selected: number; onSelect: (position: number) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !values) return;
    canvas.width = asset.positions;
    canvas.height = asset.channels;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;
    const image = context.createImageData(asset.positions, asset.channels);
    const maximum = Math.max(asset.maximum, 1e-12);
    for (let index = 0; index < values.length; index += 1) {
      const color = activationColor(values[index] / maximum * contrast);
      image.data[index * 4] = color[0]; image.data[index * 4 + 1] = color[1]; image.data[index * 4 + 2] = color[2]; image.data[index * 4 + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, [values, asset, contrast]);
  return <div className={styles.tensorShell} role="button" tabIndex={0} aria-label="Move the tensor zoom window" onKeyDown={event => {
    if (event.key === "ArrowLeft") onSelect(clamp(selected - 1, 0, asset.positions - 1));
    if (event.key === "ArrowRight") onSelect(clamp(selected + 1, 0, asset.positions - 1));
  }} onClick={event => {
    const rect = event.currentTarget.getBoundingClientRect();
    onSelect(Math.round(clamp((event.clientX - rect.left) / rect.width, 0, 1) * (asset.positions - 1)));
  }}>
    <canvas ref={ref} />
    {!values && <span>loading exact tensor…</span>}
    <i style={{ left: `${(selected + 0.5) / asset.positions * 100}%` }} />
  </div>;
}

function ZoomCanvas({ values, layer, contrast, center }: { values: Float32Array | null; layer: Layer; contrast: number; center: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const windowSize = Math.min(40, layer.asset.positions);
  const start = clamp(Math.round(center - windowSize / 2), 0, layer.asset.positions - windowSize);
  const channels = useMemo(() => {
    if (!values) return [] as number[];
    const scores = Array.from({ length: layer.asset.channels }, (_, channel) => {
      let maximum = 0;
      for (let position = start; position < start + windowSize; position += 1) maximum = Math.max(maximum, values[channel * layer.asset.positions + position]);
      return { channel, maximum };
    });
    return scores.sort((a, b) => b.maximum - a.maximum).slice(0, 18).map(item => item.channel);
  }, [values, layer.asset.channels, layer.asset.positions, start, windowSize]);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !values || channels.length === 0) return;
    canvas.width = windowSize; canvas.height = channels.length;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;
    const image = context.createImageData(windowSize, channels.length);
    const maximum = Math.max(layer.asset.maximum, 1e-12);
    channels.forEach((channel, row) => {
      for (let offset = 0; offset < windowSize; offset += 1) {
        const value = values[channel * layer.asset.positions + start + offset];
        const color = activationColor(value / maximum * contrast);
        const index = (row * windowSize + offset) * 4;
        image.data[index] = color[0]; image.data[index + 1] = color[1]; image.data[index + 2] = color[2]; image.data[index + 3] = 255;
      }
    });
    context.putImageData(image, 0, 0);
  }, [values, channels, contrast, layer.asset.maximum, layer.asset.positions, start, windowSize]);
  const inputStart = start * layer.center_spacing_bp + 1;
  const inputEnd = (start + windowSize - 1) * layer.center_spacing_bp + layer.receptive_field_bp;
  return <div className={styles.zoomCard}>
    <div><b>Magnifying glass</b><span>{channels.length} most active channels in this window</span></div>
    <canvas ref={ref} />
    <div className={styles.axis}><span>tensor position {start + 1}</span><span>input bases {inputStart}–{inputEnd}</span><span>tensor position {start + windowSize}</span></div>
    <p>Rows retain their immutable channel IDs: {channels.map(channel => `C${channel + 1}`).join(", ")}.</p>
  </div>;
}

function ContributionCanvas({ values, rows, columns, selectedRows }: { values: number[][]; rows: number; columns: number; selectedRows?: number[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = columns; canvas.height = rows;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;
    const flat = values.flat();
    const maximum = Math.max(...flat.map(Math.abs), 1e-12);
    const image = context.createImageData(columns, rows);
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const value = values[row]?.[column] ?? 0;
        const colorText = signedColor(value, maximum).match(/\d+/g)?.map(Number) ?? [249, 246, 238];
        const index = (row * columns + column) * 4;
        image.data[index] = colorText[0]; image.data[index + 1] = colorText[1]; image.data[index + 2] = colorText[2]; image.data[index + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [values, rows, columns]);
  return <div className={styles.contributionShell}><canvas ref={ref} />{selectedRows?.map(row => <i key={row} style={{ top: `${row / rows * 100}%`, height: `${100 / rows}%` }} />)}</div>;
}

function ArchitectureMap() {
  const widths = artifact.architecture.map(stage => stage.shape.length > 1 ? stage.shape.at(-1) ?? 1 : 600);
  return <div className={styles.architectureMap}>
    {artifact.architecture.map((stage, index) => <div key={stage.stage}>
      <article className={stage.shape.length === 1 ? styles.nonSpatial : ""} style={css({ "--stage-width": `${Math.max(3, widths[index] / 600 * 100)}%` })}>
        <small>{String(index + 1).padStart(2, "0")}</small><h3>{stage.stage}</h3><b>{stage.shape.join(" × ")}</b><span>{stage.operation}</span>
        {stage.note && <em>{stage.note}</em>}
      </article>
      {index < artifact.architecture.length - 1 && <i>↓</i>}
    </div>)}
  </div>;
}

export default function BassetPage() {
  const [filterIndex, setFilterIndex] = useState(0);
  const filter = artifact.stem_filters[filterIndex] as StemFilter;
  const [kernelPosition, setKernelPosition] = useState(filter.peak_position_zero_based);
  const [layerId, setLayerId] = useState("conv1_relu");
  const layer = artifact.layers.find(item => item.id === layerId) ?? artifact.layers[0];
  const [contrast, setContrast] = useState(1.8);
  const [tensorPosition, setTensorPosition] = useState(Math.floor(artifact.layers[0].asset.positions / 2));
  const { values, error } = useTensor(layer.asset as Asset);

  const selectedWeights = useMemo(() => Array.from({ length: 19 }, (_, offset) => {
    const base = artifact.sequence[kernelPosition + offset];
    return filter.weights_bases_by_positions[BASES.indexOf(base as typeof BASES[number])][offset];
  }), [filter, kernelPosition]);
  const dotProduct = selectedWeights.reduce((sum, value) => sum + value, 0);
  const afterRawBias = dotProduct + filter.raw_bias;
  const afterBatchNorm = (afterRawBias - filter.batch_norm.running_mean) * filter.batch_norm.running_inverse_std * filter.batch_norm.gamma + filter.batch_norm.beta;
  const afterRelu = Math.max(0, afterBatchNorm);
  const poolIndex = Math.floor(kernelPosition / 3);
  const poolStart = poolIndex * 3;
  const poolValues = filter.activation_track.slice(poolStart, poolStart + 3);
  const poolWinner = poolValues.indexOf(Math.max(...poolValues));

  const mixing = artifact.conv2_mixing_example;
  const dense = artifact.dense_readout_example;
  const topMixRows = mixing.top_input_channels.map(item => item.channel_zero_based);

  return <main className={styles.page}>
    <header className={styles.topbar}><Link href="/">← ChromBPNet explainer</Link><b>BASSET EXPLAINER</b><nav><a href="#architecture">Model map</a><a href="#stem">Stem + pool</a><a href="#mixing">Feature mixing</a><a href="#dense">Dense readout</a><Link href="/model-audit">ChromBPNet audit ↗</Link></nav></header>

    <section className={styles.hero}>
      <div><small>MODEL 2 · PUBLISHED CHECKPOINT</small><h1>Shrink the sequence, mix the motifs, classify 164 cell types.</h1><p>Basset offers a different answer to long-range communication. It repeatedly detects patterns and max-pools nearby positions, then lets a dense layer read all <b>200 × 10</b> remaining features at once.</p></div>
      <aside><small>NUMERICAL GATE PASSED</small><b>NumPy ↔ TensorFlow</b><strong>{artifact.verification.maximum_absolute_error.toExponential(2)}</strong><span>maximum absolute disagreement across all exported states</span><em>Official Torch7 checkpoint · SHA-256 {artifact.provenance.checkpoint_sha256.slice(0, 12)}…</em></aside>
    </section>

    <div className={styles.thesis}><b>ChromBPNet:</b> preserve a long spatial grid and communicate with dilation <i>versus</i> <b>Basset:</b> progressively compress positions, then communicate globally through dense layers.</div>

    <section id="architecture" className={styles.section}>
      <SectionHeading number="1" eyebrow="THE COMPLETE CHECKPOINT GRAPH" title="Every layer and every real tensor dimension">The graph below comes from the stored modules—not from a diagram or a reimplementation. Through Pool3, each blue bar is proportional to the remaining position count; gold means the spatial grid has been flattened.</SectionHeading>
      <ArchitectureMap />
      <div className={styles.modelTruth}><b>Critical checkpoint finding</b><span>The convolutions are <em>valid</em> (no padding), so their widths shrink before pooling. The newer repository code can construct padded models, but this published checkpoint stores <code>padW = 0</code>.</span></div>
    </section>

    <section id="stem" className={styles.section}>
      <SectionHeading number="2" eyebrow="LOCAL FEATURE DETECTION" title="A 4 × 19 filter scans 600 DNA bases">Move the filter just as in CNN Explainer. The selected 19 bases choose one weight per column; those 19 numbers are summed, followed by the stored convolution bias, batch normalization, and ReLU.</SectionHeading>
      <div className={styles.controls}><label><span>Stem filter</span><select value={filterIndex} onChange={event => { const next = Number(event.target.value); setFilterIndex(next); setKernelPosition(artifact.stem_filters[next].peak_position_zero_based); }}>{artifact.stem_filters.map((item, index) => <option key={item.filter_id_zero_based} value={index}>Channel {item.filter_id_zero_based + 1} · peak {item.peak_activation.toFixed(2)}</option>)}</select></label><label><span>Kernel start · input base {kernelPosition + 1}</span><input type="range" min="0" max="581" value={kernelPosition} onChange={event => setKernelPosition(Number(event.target.value))} /></label></div>
      <SequenceRibbon sequence={artifact.sequence} start={kernelPosition} width={19} onSelect={setKernelPosition} />
      <div className={styles.stemGrid}>
        <article><div className={styles.cardTitle}><small>LEARNED QUESTION</small><h3>Filter Channel {filter.filter_id_zero_based + 1} · 4 × 19</h3><span>blue positive · coral negative</span></div><KernelMatrix weights={filter.weights_bases_by_positions} sequence={artifact.sequence} start={kernelPosition} /></article>
        <article><div className={styles.cardTitle}><small>ONE OUTPUT CELL</small><h3>Convolution at position {kernelPosition + 1}</h3><span>cross-correlation: no reversal</span></div><div className={styles.productRow}>{selectedWeights.map((value, index) => <i key={index} style={{ background: signedColor(value, Math.max(...selectedWeights.map(Math.abs))) }}>{value.toFixed(2)}</i>)}</div><div className={styles.equation}><span><small>sum products</small><b>{dotProduct.toFixed(3)}</b></span><i>+</i><span><small>conv bias</small><b>{filter.raw_bias.toFixed(3)}</b></span><i>→</i><span><small>batch norm</small><b>{afterBatchNorm.toFixed(3)}</b></span><i>→</i><strong><small>ReLU</small>{afterRelu.toFixed(3)}</strong></div></article>
      </div>
      <div className={styles.trackPair}>
        <article><div className={styles.cardTitle}><small>AFTER RELU</small><h3>Channel {filter.filter_id_zero_based + 1} · 582 positions</h3><span>one response for every valid 19-base window</span></div><SignalTrack values={filter.activation_track} selected={kernelPosition} /><div className={styles.axis}><span>window 1</span><span>red = current kernel location</span><span>window 582</span></div></article>
        <article><div className={styles.cardTitle}><small>AFTER MAX POOL</small><h3>Same channel · 194 positions</h3><span>every 3 neighboring responses become one</span></div><SignalTrack values={filter.pooled_track} selected={poolIndex} color="gold" /><div className={styles.axis}><span>pool 1</span><span>red = current pool</span><span>pool 194</span></div></article>
      </div>
      <div className={styles.poolMicroscope}><div><small>MAX-POOL MICROSCOPE</small><h3>Three nearby motif scores compete</h3><p>Pool {poolIndex + 1} summarizes convolution positions {poolStart + 1}–{poolStart + 3}, together covering input bases {poolStart + 1}–{poolStart + 21}.</p></div><div className={styles.poolValues}>{poolValues.map((value, index) => <span key={index} className={index === poolWinner ? styles.winner : ""}><small>conv {poolStart + index + 1}</small><b>{value.toFixed(3)}</b>{index === poolWinner && <em>kept</em>}</span>)}<i>→</i><strong>{Math.max(...poolValues).toFixed(3)}</strong></div></div>
    </section>

    <section className={styles.section}>
      <SectionHeading number="3" eyebrow="TOP DOWN, THEN ZOOM" title="Watch the full tensors shrink">Start from the entire channel × position heatmap. Select a layer, adjust contrast, click a location, and then inspect the strongest rows in a local magnifying glass.</SectionHeading>
      <div className={styles.tensorControls}><div>{artifact.layers.map(item => <button key={item.id} className={item.id === layer.id ? styles.active : ""} onClick={() => { setLayerId(item.id); setTensorPosition(Math.floor(item.asset.positions / 2)); }}><b>{item.id.replace("_relu", "")}</b><span>{item.asset.channels} × {item.asset.positions}</span></button>)}</div><label><span>contrast · {contrast.toFixed(1)}×</span><input type="range" min="0.5" max="6" step="0.1" value={contrast} onChange={event => setContrast(Number(event.target.value))} /></label></div>
      <div className={styles.tensorFacts}><b>{layer.asset.channels} channels × {layer.asset.positions} positions</b><span>{(layer.stats.exact_zero_fraction * 100).toFixed(1)}% exact zeros</span><span>receptive field {layer.receptive_field_bp} bp</span><span>centers {layer.center_spacing_bp} bp apart</span></div>
      {error && <div className={styles.error}>{error}</div>}
      <TensorCanvas values={values} asset={layer.asset as Asset} contrast={contrast} selected={tensorPosition} onSelect={setTensorPosition} />
      <div className={styles.axis}><span>position 1</span><span>all entries shown · normalized only for color</span><span>position {layer.asset.positions}</span></div>
      <ZoomCanvas values={values} layer={layer} contrast={contrast} center={tensorPosition} />
    </section>

    <section id="mixing" className={styles.section}>
      <SectionHeading number="4" eyebrow="HOW LOCAL FEATURES COMMUNICATE" title="A second-layer filter reads 300 channels × 11 nearby positions">This is not one one-dimensional vector acting independently on each channel. One Conv2 output cell combines <b>3,300 learned inputs</b>: 300 Conv1 feature channels across 11 pooled locations.</SectionHeading>
      <div className={styles.mixingLayout}>
        <article><div className={styles.cardTitle}><small>ELEMENT PRODUCTS</small><h3>Conv2 Channel {mixing.output_channel_zero_based + 1} · output position {mixing.output_position_zero_based + 1}</h3><span>rows = Conv1 channels · columns = 11 pooled locations</span></div><ContributionCanvas values={mixing.contributions_input_channels_by_taps} rows={300} columns={11} selectedRows={topMixRows} /><div className={styles.axis}><span>Conv1 Channel 1</span><span>coral negative · blue positive</span><span>Conv1 Channel 300</span></div></article>
        <aside><small>THE 12 LARGEST CHANNEL CONTRIBUTIONS</small><div className={styles.contributionList}>{mixing.top_input_channels.map(item => <div key={item.channel_zero_based}><b>Channel {item.channel_zero_based + 1}</b><i style={{ width: `${Math.abs(item.signed_contribution) / Math.max(...mixing.top_input_channels.map(value => Math.abs(value.signed_contribution))) * 100}%`, background: item.signed_contribution >= 0 ? "#3288a3" : "#e8644b" }} /><span>{item.signed_contribution.toFixed(3)}</span></div>)}</div><p>These signed values already include both the incoming activation and the learned Conv2 weight. They are contributions to this one output cell, not attribution scores for the final prediction.</p></aside>
      </div>
      <div className={styles.equationWide}><span><small>3,300 products summed</small><b>{mixing.sum_products.toFixed(3)}</b></span><i>+</i><span><small>raw bias</small><b>{mixing.raw_bias.toFixed(3)}</b></span><i>→</i><span><small>batch norm</small><b>{mixing.after_batch_norm.toFixed(3)}</b></span><i>→</i><strong><small>ReLU</small>{mixing.after_relu.toFixed(3)}</strong></div>
      <div className={styles.takeaway}><b>Biological intuition:</b> if different Conv1 channels detect motif-like patterns, Conv2 can learn combinations of those channels within a 51 bp receptive field. Max pooling makes the combination somewhat tolerant to small shifts.</div>
    </section>

    <section className={styles.section}>
      <SectionHeading number="5" eyebrow="RECEPTIVE FIELD GROWTH" title="Pooling trades exact position for broader context">The network keeps fewer coordinates, but each surviving coordinate summarizes a wider interval. After Pool3, ten locations cover the full 600 bp input.</SectionHeading>
      <div className={styles.receptiveMap}>{[
        ["input", 600, 1, 1], ["conv1", 582, 19, 1], ["pool1", 194, 21, 3], ["conv2", 184, 51, 3], ["pool2", 46, 60, 12], ["conv3", 40, 132, 12], ["pool3", 10, 168, 48],
      ].map(([name, positions, field, jump]) => <article key={String(name)}><div><b>{name}</b><span>{positions} positions</span></div><i style={{ width: `${Number(positions) / 600 * 100}%` }} /><p>each cell sees {field} bp · centers {jump} bp apart</p></article>)}</div>
      <div className={styles.coverageDiagram}>{Array.from({ length: 10 }, (_, index) => <span key={index} style={{ left: `${index * 48 / 600 * 100}%`, width: `${168 / 600 * 100}%` }}><b>{index + 1}</b></span>)}<i>600 bp input</i></div>
    </section>

    <section id="dense" className={styles.section}>
      <SectionHeading number="6" eyebrow="GLOBAL COMMUNICATION" title="The dense layer reads all 200 × 10 cells at once">This is where any remaining location can directly influence the same hidden unit. Flattening preserves every Channel × Position identity; it does not average them.</SectionHeading>
      <div className={styles.denseLayout}>
        <article><div className={styles.cardTitle}><small>ONE DENSE-1 UNIT</small><h3>Unit {dense.unit_zero_based + 1} · 200 × 10 contributions</h3><span>rows = Pool3 channels · columns = the ten broad locations</span></div><ContributionCanvas values={dense.contributions_channels_by_positions} rows={200} columns={10} /><div className={styles.axis}><span>Channel 1</span><span>each column covers 168 bp; centers 48 bp apart</span><span>Channel 200</span></div></article>
        <aside><small>WHAT “FULLY CONNECTED” MEANS HERE</small><b>2,000 → 1</b><p>One dense unit owns 2,000 separate weights. It can give different importance to the same feature channel at the left, center, or right of the 600 bp sequence.</p><dl><dt>sum products</dt><dd>{dense.sum_products.toFixed(3)}</dd><dt>raw bias</dt><dd>{dense.raw_bias.toFixed(3)}</dd><dt>after batch norm</dt><dd>{dense.after_batch_norm.toFixed(3)}</dd><dt>after ReLU</dt><dd>{dense.after_relu.toFixed(3)}</dd></dl></aside>
      </div>
      <div className={styles.flowSentence}><b>200 × 10</b><i>flatten</i><b>2,000</b><i>dense + BN + ReLU</i><b>1,000</b><i>dense + BN + ReLU</i><b>1,000</b><i>164 sigmoid readers</i><b>164 probabilities</b></div>
      <div className={styles.outputs}><div><small>OFFICIAL HOXA TUTORIAL SEQUENCE</small><h3>Highest predicted accessibility probabilities</h3><p>These are model outputs for one sequence, not experimental measurements.</p></div><div className={styles.outputBars}>{artifact.outputs.top_predictions.slice(0, 12).map(item => <div key={item.target_index_zero_based} className={item.label === "K562" ? styles.k562 : ""}><b>{item.label}</b><i style={{ width: `${item.probability * 100}%` }} /><span>{item.probability.toFixed(3)}</span></div>)}</div><aside><small>K562</small><strong>{artifact.outputs.k562_probability.toFixed(3)}</strong><span>target #{artifact.outputs.k562_index_zero_based + 1}</span></aside></div>
    </section>

    <section className={styles.compare}>
      <SectionHeading number="7" eyebrow="CROSS-MODEL MENTAL MODEL" title="Two valid ways to combine distant sequence features">Neither architecture is “the CNN way.” They make different compromises about spatial resolution, output task, and how information travels.</SectionHeading>
      <div className={styles.compareGrid}><article><small>CHROMBPNET</small><h3>Preserve + dilate</h3><b>2,114 bp → 1,074 spatial positions</b><p>Eight residual dilated convolutions let feature vectors communicate over increasing distances while retaining a base-resolution profile head.</p><Link href="/">Open ChromBPNet explainer →</Link></article><article><small>BASSET</small><h3>Compress + connect</h3><b>600 bp → 10 positions → dense</b><p>Three convolution/pooling stages build broader features; dense layers then mix every remaining channel and location to classify accessibility in 164 cell types.</p><a href="#architecture">Review Basset map ↑</a></article></div>
    </section>

    <section className={styles.caveats}><b>Scientific boundary</b><p>This page uses one official tutorial sequence to explain a verified checkpoint computation. Strongest-channel examples are selected for visibility. They do not establish motif identity, causal biology, or importance to a final output; those require corpus-level activation motifs, attribution, and perturbation experiments.</p></section>
    <footer><b>Sequence CNN Explainer · Basset checkpoint adapter</b><span>Official model format: Torch7 · browser artifact: compact float32 activations · full checkpoint is never deployed</span></footer>
  </main>;
}
