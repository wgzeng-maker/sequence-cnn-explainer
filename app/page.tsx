"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { loadFloat32Tensor } from "./tensor-loader";
import { CHANNEL_ORDER_LABELS, centeredStemWeights, displayRank, informationContent, reverseComplementBaseRows, reverseComplementSequence, type ActivationMotif, type ChannelOrder, type KernelOrientation, type KernelViewMode } from "./model-analysis";

// Use the common checkpoint schema here. K562 carries additional teaching-only
// residual examples, but those are consumed by the dedicated dilation page.
type Demo = typeof import("./data/gm21515-activations.json");
type ResidualExample = { selection_rule: string; output_position_zero_based: number; input_aligned_coordinate_one_based?: number; output_channel_zero_based: number; weights_input_channels_by_taps: number[][]; bias: number; transformed_after_relu: number; shortcut_value: number; block_output: number };
type Tensor = Demo["tensors"]["stem"];
type TensorKey = "stem" | "res1" | "res2" | "res3" | "res4" | "res5" | "res6" | "res7" | "res8";
type LayerKey = TensorKey | "profile";
type Transfer = "linear" | "sqrt" | "log";
type InspectorView = "tensor" | "channel" | "max" | "mean" | "min";
type ComputationStage = "conv" | "bias" | "relu" | "output";

const BASES = ["A", "C", "G", "T"] as const;
const BASE_COLORS = ["#e96b54", "#4f97b2", "#e5b33f", "#72a17e"];
type AuditCheckpoints = Record<string, { channel_orders: Record<ChannelOrder, number[]>; activation_motifs: { status: string; reason: string; target_site_count_per_filter: number; planned_corpus: string; selection_rule: string; motifs: ActivationMotif[] } }>;
const PRESET_URLS: Record<string, string> = { k562: "/data/demos/k562.json.gz", gm21515: "/data/demos/gm21515.json.gz", synthetic: "/data/demos/synthetic.json.gz" };
const CHANNEL_ORDER_KEYS = Object.keys(CHANNEL_ORDER_LABELS) as ChannelOrder[];
const DILATIONS = [2, 4, 8, 16, 32, 64, 128, 256];
const LENGTHS = [2114, 2094, 2090, 2082, 2066, 2034, 1970, 1842, 1586, 1074];
const RECEPTIVE_FIELDS = [21, 25, 33, 49, 81, 145, 273, 529, 1041];
const MAX_TENSOR_LENGTH = 2094;

const css = (values: Record<string, string | number>) => values as CSSProperties;
const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));
const offsetForLength = (length: number) => (2114 - length) / 2;
const inputCoordinate = (local: number, length: number) => Math.round(offsetForLength(length) + local + 1);

function useJsonAsset<T>(url: string) {
  const [loaded, setLoaded] = useState<{ url: string; value: T } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setError(null);
    fetch(url, { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error(`Data request failed (${response.status}) for ${url}`);
      let buffer = await response.arrayBuffer();
      const bytes = new Uint8Array(buffer);
      if (bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b) {
        if (typeof DecompressionStream === "undefined") throw new Error("This browser cannot decode compressed checkpoint data.");
        buffer = await new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
      }
      return JSON.parse(new TextDecoder().decode(buffer)) as T;
    }).then(value => setLoaded({ url, value })).catch(reason => {
      if (reason instanceof DOMException && reason.name === "AbortError") return;
      setError(reason instanceof Error ? reason.message : "The data could not be loaded.");
    });
    return () => controller.abort();
  }, [url]);
  return { value: loaded?.url === url ? loaded.value : null, error };
}

function tensorFor(demo: Demo, key: TensorKey): Tensor {
  return demo.tensors[key] as Tensor;
}

function inspectorTensorFor(demo: Demo, layer: TensorKey, computation: ComputationStage): Tensor {
  const key = computation === "output" || (layer === "stem" && computation === "relu")
    ? layer
    : `${layer}_${computation}`;
  return demo.tensors[key as keyof Demo["tensors"]] as Tensor;
}

function transformValue(value: number, maximum: number, transfer: Transfer, gain: number) {
  const ratio = clamp(value / Math.max(maximum, 1e-12), 0, 1);
  const mapped = transfer === "linear" ? ratio : transfer === "sqrt" ? Math.sqrt(ratio) : Math.log1p(ratio * 100) / Math.log(101);
  return clamp(mapped * gain, 0, 1);
}

function heatColor(level: number) {
  if (level <= 0) return "rgb(247 244 236)";
  const stops = [
    [247, 244, 236],
    [244, 209, 94],
    [235, 105, 73],
    [109, 38, 67],
  ];
  const position = clamp(level, 0, 1) * (stops.length - 1);
  const start = Math.min(stops.length - 2, Math.floor(position));
  const t = position - start;
  const color = stops[start].map((value, index) => Math.round(value + (stops[start + 1][index] - value) * t));
  return `rgb(${color.join(" ")})`;
}

function signedColor(value: number, maximum: number) {
  const strength = Math.sqrt(clamp(Math.abs(value) / Math.max(maximum, 1e-12), 0, 1));
  const target = value >= 0 ? [225, 91, 69] : [44, 129, 158];
  const background = [247, 244, 236];
  const color = background.map((base, index) => Math.round(base + (target[index] - base) * strength));
  return `rgb(${color.join(" ")})`;
}

function signedRgb(value: number, maximum: number) {
  const strength = Math.sqrt(clamp(Math.abs(value) / Math.max(maximum, 1e-12), 0, 1));
  const target = value >= 0 ? [225, 91, 69] : [44, 129, 158];
  const background = [247, 244, 236];
  return background.map((base, index) => Math.round(base + (target[index] - base) * strength));
}

function rgbCssToHex(color: string) {
  const channels = color.match(/\d+/g)?.map(Number) ?? [0, 0, 0];
  return `#${channels.slice(0, 3).map(value => value.toString(16).padStart(2, "0")).join("")}`;
}

function useTensorData(tensor: Tensor | null) {
  const [loaded, setLoaded] = useState<{ url: string; values: Float32Array } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!tensor) return;
    const controller = new AbortController();
    setError(null);
    const heatmap = tensor.full_heatmap as typeof tensor.full_heatmap & {
      display_transform?: "add_channel_bias" | "add_channel_bias_relu";
      channel_bias?: number[];
    };
    loadFloat32Tensor(heatmap.url, heatmap.height_channels * heatmap.width_positions, controller.signal)
      .then(source => {
        if (!heatmap.display_transform || !heatmap.channel_bias) return source;
        const values = new Float32Array(source.length);
        for (let channel = 0; channel < heatmap.height_channels; channel += 1) {
          const bias = heatmap.channel_bias[channel];
          const offset = channel * heatmap.width_positions;
          for (let position = 0; position < heatmap.width_positions; position += 1) {
            const biased = source[offset + position] + bias;
            values[offset + position] = heatmap.display_transform === "add_channel_bias_relu" ? Math.max(0, biased) : biased;
          }
        }
        return values;
      })
      .then(values => setLoaded({ url: heatmap.url, values }))
      .catch(reason => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "The tensor data could not be loaded.");
      });
    return () => controller.abort();
  }, [tensor]);
  return { values: loaded && loaded.url === tensor?.full_heatmap.url ? loaded.values : null, error };
}

function useFloat32Asset(url: string, expectedValues: number) {
  const [loaded, setLoaded] = useState<{ url: string; values: Float32Array } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setError(null);
    loadFloat32Tensor(url, expectedValues, controller.signal)
      .then(values => setLoaded({ url, values }))
      .catch(reason => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "The data could not be loaded.");
      });
    return () => controller.abort();
  }, [url, expectedValues]);
  return { values: loaded?.url === url ? loaded.values : null, error };
}

function SectionHeading({ step, eyebrow, title, children }: { step: string; eyebrow: string; title: string; children: React.ReactNode }) {
  return <div className="section-heading" data-feedback-id={`${step} · ${title}`}>
    <span className="step-number">{step}</span>
    <div><small>{eyebrow}</small><h2>{title}</h2><p>{children}</p></div>
  </div>;
}

function FlowArrow({ label }: { label?: string }) {
  return <div className="flow-arrow" aria-hidden="true"><i /><span>{label ?? "next"}</span></div>;
}

function BaseLetters({ sequence, positions }: { sequence: string; positions: number[] }) {
  return <div className="base-letters" style={css({ "--columns": positions.length })}>
    {positions.map(position => {
      const base = sequence[position] ?? "N";
      return <i key={position} style={{ background: BASE_COLORS[BASES.indexOf(base as typeof BASES[number])] ?? "#ddd" }}>{base}</i>;
    })}
  </div>;
}

function SequenceOverview({ sequence, start, onStart }: { sequence: string; start: number; onStart: (value: number) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  // Position the left edge in the same base grid as the colored ribbon. Using
  // sequence.length - 21 would push the final 21-base box beyond the ribbon.
  const markerLeft = start / sequence.length * 100;
  const select = (clientX: number) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    onStart(Math.round(clamp((clientX - rect.left) / rect.width, 0, 1) * (sequence.length - 21)));
  };
  return <div className="sequence-overview">
    <div className="overview-label"><b>All 2,114 input bases</b><span>one thin column per base · no quiet-region compression</span></div>
    <div ref={ref} className="sequence-ribbon" role="button" tabIndex={0} aria-label="Move the selected 21-base input window" onClick={event => select(event.clientX)} onKeyDown={event => {
      if (event.key === "ArrowLeft") onStart(clamp(start - 21, 0, sequence.length - 21));
      if (event.key === "ArrowRight") onStart(clamp(start + 21, 0, sequence.length - 21));
    }}>
      {Array.from(sequence).map((base, index) => <i key={index} style={{ background: BASE_COLORS[BASES.indexOf(base as typeof BASES[number])] }} />)}
      <b style={{ left: `${markerLeft}%` }}><span>selected 21 bases</span></b>
    </div>
    <div className="axis"><span>input 1</span><span>click to move the 21-base window</span><span>input 2,114</span></div>
  </div>;
}

function OneHotZoom({ sequence, start }: { sequence: string; start: number }) {
  const positions = Array.from({ length: 21 }, (_, index) => start + index);
  return <div className="onehot-card" data-feedback-id="One-hot matrix zoom">
    <div className="mini-heading"><div><small>EXACT INPUT WINDOW</small><h3>4 rows × 21 base positions</h3></div><span>one colored 1 in every column</span></div>
    <div className="onehot-grid">
      {BASES.map((base, row) => <div className="onehot-row" key={base}>
        <b>{base}</b><div style={css({ "--columns": 21 })}>{positions.map(position => <i key={position} className={sequence[position] === base ? "on" : ""} style={sequence[position] === base ? { background: BASE_COLORS[row] } : undefined}>{sequence[position] === base ? "1" : "0"}</i>)}</div>
      </div>)}
    </div>
    <BaseLetters sequence={sequence} positions={positions} />
    <div className="axis"><span>{start + 1}</span><span>input-relative coordinates</span><span>{start + 21}</span></div>
  </div>;
}

function WeightMatrix({ weights, gain }: { weights: number[][]; gain: number }) {
  const maximum = Math.max(...weights.flat().map(Math.abs), 1e-12);
  return <div className="weight-matrix">
    {BASES.map((base, row) => <div key={base}><b>{base}</b><span style={css({ "--columns": 21 })}>{weights[row].map((value, index) => <i key={index} style={{ background: signedColor(value, maximum), filter: `saturate(${gain})` }} title={`${base}, kernel position ${index + 1}: ${value.toFixed(4)}`} />)}</span></div>)}
  </div>;
}

function SignedWeightLogo({ weights, gain }: { weights: number[][]; gain: number }) {
  const maximum = Math.max(...weights.flat().map(Math.abs), 1e-12);
  return <div className="signed-logo" aria-label="Signed centered stem-filter weight logo">
    {Array.from({ length: 21 }, (_, position) => {
      const positive = BASES.map((base, row) => ({ base, row, value: Math.max(0, weights[row][position]) })).filter(item => item.value > 0).sort((a, b) => a.value - b.value);
      const negative = BASES.map((base, row) => ({ base, row, value: Math.min(0, weights[row][position]) })).filter(item => item.value < 0).sort((a, b) => Math.abs(a.value) - Math.abs(b.value));
      return <div className="logo-column" key={position} title={`kernel position ${position + 1}`}>
        <span className="logo-positive">{positive.map(item => <i key={item.base} style={{ color: BASE_COLORS[item.row], height: `${Math.max(6, Math.abs(item.value) / maximum * 78 * gain)}px`, fontSize: `${Math.max(8, Math.abs(item.value) / maximum * 68 * gain)}px` }}>{item.base}</i>)}</span>
        <b>{position + 1}</b>
        <span className="logo-negative">{negative.map(item => <i key={item.base} style={{ color: BASE_COLORS[item.row], height: `${Math.max(6, Math.abs(item.value) / maximum * 78 * gain)}px`, fontSize: `${Math.max(8, Math.abs(item.value) / maximum * 68 * gain)}px` }}>{item.base}</i>)}</span>
      </div>;
    })}
  </div>;
}

function ActivationLogo({ motif, orientation }: { motif: ActivationMotif; orientation: KernelOrientation }) {
  const original = motif.pfm_positions_by_bases;
  const pfm = orientation === "model" ? original : [...original].reverse().map(column => [column[3], column[2], column[1], column[0]]);
  const information = informationContent(pfm, motif.background_frequencies);
  return <div>
    <div className="activation-logo" aria-label={`Activation motif for Channel ${motif.filter_id_zero_based + 1}`}>
      {pfm.map((column, position) => <div key={position}>{BASES.map((base, row) => ({ base, row, height: column[row] * information[position] })).sort((a, b) => a.height - b.height).map(item => <i key={item.base} style={{ color: BASE_COLORS[item.row], height: `${Math.max(2, item.height / 2 * 145)}px`, fontSize: `${Math.max(7, item.height / 2 * 120)}px` }}>{item.base}</i>)}<b>{position + 1}</b></div>)}
    </div>
    <p className="logo-explanation"><b>{motif.site_count} aligned 21-mers</b> · {motif.corpus_id} · {motif.activation_selection_rule}. Letter height is frequency × information content (0–2 bits).</p>
  </div>;
}

function ActivationMotifState({ status }: { status: { status: string; reason: string; target_site_count_per_filter: number; planned_corpus: string; selection_rule: string } }) {
  return <div className="activation-motif-state">
    <b>Corpus-derived activation motif not generated yet</b>
    <p>{status.reason}</p>
    <dl><dt>planned corpus</dt><dd>{status.planned_corpus}</dd><dt>site reservoir</dt><dd>up to {status.target_site_count_per_filter} non-overlapping 21-mers per filter</dd><dt>selection rule</dt><dd>{status.selection_rule}</dd></dl>
    <a href="/model-audit">Open the audit workspace and evidence plan →</a>
  </div>;
}

function KernelBankCanvas({ values, selected, onSelect, channelOrder }: { values: Float32Array | null; selected: number; onSelect: (filter: number) => void; channelOrder: number[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !values) return;
    canvas.width = 21; canvas.height = 512;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;
    const image = context.createImageData(21, 512);
    let maximum = 1e-12;
    values.forEach(value => { maximum = Math.max(maximum, Math.abs(value)); });
    for (let displayRow = 0; displayRow < 512; displayRow += 1) {
      const filter = channelOrder[displayRow];
      for (let position = 0; position < 21; position += 1) {
        let strongest = 0;
        for (let base = 0; base < 4; base += 1) {
          const value = values[filter * 84 + base * 21 + position];
          if (Math.abs(value) > Math.abs(strongest)) strongest = value;
        }
        const color = signedColor(strongest, maximum).match(/\d+/g)!.map(Number);
        const target = (displayRow * 21 + position) * 4;
        image.data[target] = color[0]; image.data[target + 1] = color[1]; image.data[target + 2] = color[2]; image.data[target + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [values, channelOrder]);
  const select = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const displayRow = clamp(Math.floor((event.clientY - rect.top) / rect.height * 512), 0, 511);
    onSelect(channelOrder[displayRow]);
  };
  return <div className="kernel-bank-canvas" role="button" tabIndex={0} aria-label="Select one of 512 stem filters" onClick={select} onKeyDown={event => { if (event.key === "Home") onSelect(channelOrder[0]); if (event.key === "End") onSelect(channelOrder[511]); }}>
    {!values && <span className="loading">loading all stem kernels…</span>}
    <canvas ref={ref} />
    <i style={{ top: `${Math.max(0, channelOrder.indexOf(selected)) / 512 * 100}%` }} />
  </div>;
}

function StemStory({ demo, start, setStart, gain, channelOrder, orderName, motifStatus }: { demo: Demo; start: number; setStart: (value: number) => void; gain: number; channelOrder: number[]; orderName: ChannelOrder; motifStatus: { status: string; reason: string; target_site_count_per_filter: number; planned_corpus: string; selection_rule: string; motifs: ActivationMotif[] } }) {
  const [filterNumber, setFilterNumber] = useState(demo.filter_demos[0].filter_zero_based);
  const [kernelView, setKernelView] = useState<KernelViewMode>("heatmap");
  const [orientation, setOrientation] = useState<KernelOrientation>("model");
  const bank = demo.stem_kernel_bank;
  const { values: kernelValues, error: kernelError } = useFloat32Asset(bank.url, bank.filter_count * bank.base_count * bank.position_count);
  const { values: stemValues } = useTensorData(demo.tensors.stem);
  const fallback = demo.filter_demos.find(item => item.filter_zero_based === filterNumber);
  const weights = BASES.map((_, base) => Array.from({ length: 21 }, (_, position) => kernelValues?.[filterNumber * 84 + base * 21 + position] ?? fallback?.weights_base_rows_by_positions[base][position] ?? 0));
  const filter = {
    filter_zero_based: filterNumber,
    filter_human_label: `Filter ${filterNumber + 1}`,
    weights_base_rows_by_positions: weights,
    bias: bank.biases[filterNumber],
    maximum_activation: bank.maximum_activations[filterNumber],
    peak_position_zero_based: bank.peak_positions_zero_based[filterNumber],
  };
  const sequence = demo.input.sequence;
  const modelWindow = sequence.slice(start, start + 21);
  const orientedWeights = orientation === "model" ? filter.weights_base_rows_by_positions : reverseComplementBaseRows(filter.weights_base_rows_by_positions);
  const orientedWindow = orientation === "model" ? modelWindow : reverseComplementSequence(modelWindow);
  const window = orientedWindow.split("");
  const weighted = window.map((base, index) => orientedWeights[BASES.indexOf(base as typeof BASES[number])][index]);
  const dotProduct = weighted.reduce((sum, value) => sum + value, 0);
  const beforeRelu = dotProduct + filter.bias;
  const output = Math.max(0, beforeRelu);
  const maximum = Math.max(...orientedWeights.flat().map(Math.abs), 1e-12);
  const centered = centeredStemWeights(orientedWeights, filter.bias);
  const activationMotif = motifStatus.motifs?.find(motif => motif.filter_id_zero_based === filterNumber);
  const track = stemValues ? Array.from(stemValues.subarray(filterNumber * 2094, (filterNumber + 1) * 2094)) : [];
  const zoomStart = clamp(start - 20, 0, 2094 - 61);
  const zoomPositions = Array.from({ length: 61 }, (_, index) => zoomStart + index);
  const trackMax = Math.max(...track, 1e-12);

  return <section className="story-section" id="stem">
    <SectionHeading step="2" eyebrow="LOCAL FEATURE DETECTION" title="A stem filter asks the same 21-base question everywhere">
      One learned kernel slides one base at a time. The calculation below produces exactly one cell in the <b>512 × 2,094</b> stem tensor.
    </SectionHeading>
    <div className="filter-picker" aria-label="Choose a real stem filter">
      <div className="choice-row">{demo.filter_demos.map(item => <button key={item.filter_zero_based} className={filterNumber === item.filter_zero_based ? "active" : ""} onClick={() => { setFilterNumber(item.filter_zero_based); setStart(item.peak_position_zero_based); }}><b>{item.filter_human_label}</b><small>peak {item.maximum_activation.toFixed(2)}</small></button>)}</div>
      <label><span>Or enter any filter</span><input type="number" min="1" max="512" value={filterNumber + 1} onChange={event => { const next = clamp(Number(event.target.value || 1) - 1, 0, 511); setFilterNumber(next); setStart(bank.peak_positions_zero_based[next]); }} /></label>
    </div>
    <details className="kernel-bank" data-feedback-id="All 512 stem filters heatmap">
      <summary>Open all 512 stem filters as one 512 × 21 heatmap</summary>
      <p>Each row is one filter. Each cell summarizes one kernel position using the strongest A/C/G/T weight there; coral is positive and blue is negative. Click a row to inspect its full 4 × 21 kernel below.</p>
      {kernelError && <div className="tensor-error"><b>Kernel data did not load</b><span>{kernelError}</span></div>}
      <KernelBankCanvas values={kernelValues} selected={filterNumber} channelOrder={channelOrder} onSelect={next => { setFilterNumber(next); setStart(bank.peak_positions_zero_based[next]); }} />
      <div className="axis"><span>rank 1 · Channel {channelOrder[0] + 1}</span><span>{CHANNEL_ORDER_LABELS[orderName]} · immutable IDs retained</span><span>rank 512 · Channel {channelOrder[511] + 1}</span></div>
    </details>
    <div className="kernel-view-controls">
      <div><span>Kernel presentation</span><div className="segmented">{(["heatmap", "signed_logo", "activation_logo"] as KernelViewMode[]).map(mode => <button key={mode} className={kernelView === mode ? "active" : ""} onClick={() => setKernelView(mode)}>{mode === "heatmap" ? "Heatmap" : mode === "signed_logo" ? "Weight logo" : "Activation motif"}</button>)}</div></div>
      <div><span>Orientation</span><div className="segmented"><button className={orientation === "model" ? "active" : ""} onClick={() => setOrientation("model")}>Model orientation</button><button className={orientation === "reverse_complement" ? "active" : ""} onClick={() => setOrientation("reverse_complement")}>Reverse complement</button></div></div>
      <p><b>Immutable identity:</b> display rank {displayRank(channelOrder, filterNumber)} · Channel {filterNumber + 1} · {demo.provenance.experiment} fold 0</p>
    </div>
    <div className="stem-stage">
      <div className="kernel-card" data-feedback-id="Actual 21 by 4 stem kernel">
        <div className="mini-heading"><div><small>THE LEARNED QUESTION</small><h3>{filter.filter_human_label} kernel · 4 × 21</h3></div><span>{orientation === "model" ? "stored left-to-right" : "positions reversed · bases complemented"}</span></div>
        {kernelView === "heatmap" && <WeightMatrix weights={orientedWeights} gain={gain} />}
        {kernelView === "signed_logo" && <><SignedWeightLogo weights={centered.centered} gain={gain} /><p className="logo-explanation">Each position’s A/C/G/T mean was removed. The exact equivalent bias is <b>{centered.adjustedBias.toFixed(5)}</b> instead of {filter.bias.toFixed(5)}. Positive letters are above zero; negative letters are below it.</p></>}
        {kernelView === "activation_logo" && (activationMotif ? <ActivationLogo motif={activationMotif} orientation={orientation} /> : <ActivationMotifState status={motifStatus} />)}
      </div>
      <div className="slide-symbol"><b>slides</b><span>↓</span></div>
      <div className="kernel-card selected-window">
        <div className="mini-heading"><div><small>THE 21 BASES UNDER IT</small><h3>Input {start + 1}–{start + 21}</h3></div><span>move the slider</span></div>
        <div className="sequence-cells" style={css({ "--columns": 21 })}>{window.map((base, index) => <i key={index} style={{ background: BASE_COLORS[BASES.indexOf(base as typeof BASES[number])] }}>{base}</i>)}</div>
        <div className="selected-weights" style={css({ "--columns": 21 })}>{weighted.map((value, index) => <i key={index} style={{ background: signedColor(value, maximum) }}>{value.toFixed(2)}</i>)}</div>
      </div>
    </div>
    <div className="orientation-convention" data-feedback-id="Stem-kernel orientation convention"><b>TensorFlow Conv1D convention · cross-correlation, not a spatially reversed mathematical convolution</b><code>output[o,f] = bias[f] + Σposition Σbase input[o+position,base] × kernel[position,base,f]</code><p>Sequence positions run left to right exactly as drawn. A/C/G/T is the feature-channel axis, so it is never rotated. Reverse complement reverses positions and exchanges A↔T and C↔G.</p><a href="https://www.tensorflow.org/api_docs/python/tf/nn/conv1d">TensorFlow Conv1D documentation ↗</a></div>
    <label className="range-control"><b>Slide the stem filter</b><input type="range" min="0" max="2093" value={start} onChange={event => setStart(Number(event.target.value))} /><span>output position {start + 1}</span></label>
    <div className="calculation-flow" data-feedback-id="Stem calculation">
      <span><small>dot product</small><b>{dotProduct.toFixed(3)}</b><em>21 selected weights added</em></span><i>+</i>
      <span><small>bias</small><b>{filter.bias.toFixed(3)}</b></span><i>→</i>
      <span><small>before ReLU</small><b>{beforeRelu.toFixed(3)}</b></span><i>→</i>
      <span className="result"><small>after ReLU</small><b>{output.toFixed(3)}</b><em>one tensor cell</em></span>
    </div>
    <div className="track-card">
      <div className="mini-heading"><div><small>ONE OUTPUT ROW</small><h3>{filter.filter_human_label} across 61 nearby positions</h3></div><span>same kernel, new 21-base window at each step</span></div>
      <div className="signal-bars" style={css({ "--columns": 61 })}>{zoomPositions.map(position => <i key={position} className={position === start ? "selected" : ""} style={{ height: `${3 + (track[position] ?? 0) / trackMax * 82}px` }} title={`output ${position + 1}: ${(track[position] ?? 0).toFixed(4)}`} />)}</div>
      <BaseLetters sequence={sequence} positions={zoomPositions.map(position => position + 10)} />
      <div className="axis"><span>output {zoomStart + 1}</span><span>input base below = center of each 21-base window</span><span>output {zoomStart + 61}</span></div>
    </div>
    <p className="takeaway"><b>Mental picture:</b> each of 512 stem filters marks where its learned local pattern is present. Stacking their 512 output rows creates a sparse feature map.</p>
  </section>;
}

function ArchitectureMap({ selectedBlock, onBlock }: { selectedBlock: number; onBlock: (value: number) => void }) {
  return <section className="architecture" id="architecture">
    <SectionHeading step="0" eyebrow="MODEL MAP" title="The complete checkpoint, read from top to bottom">
      Shapes are <b>channels × positions</b>. Every residual block uses the same two-path pattern, while dilation changes the positions read by its transform path.
    </SectionHeading>
    <div className="architecture-stack">
      <article className="architecture-node input-node"><small>INPUT</small><h3>One-hot DNA</h3><b>4 × 2,114</b></article>
      <FlowArrow label="valid Conv1D · k=21" />
      <article className="architecture-node stem-node"><small>STEM</small><h3>512 local-feature detectors</h3><b>512 × 2,094</b><span>convolution → bias → ReLU</span></article>
      <FlowArrow label="eight context-building blocks" />
      <div className="residual-stack-map">
        {DILATIONS.map((dilation, index) => {
          const block = index + 1;
          const inputLength = LENGTHS[index + 1];
          const outputLength = LENGTHS[index + 2];
          return <button key={block} className={selectedBlock === block ? "selected" : ""} onClick={() => onBlock(block)} data-feedback-id={`Residual block ${block} architecture`}>
            <div className="block-title"><span>BLOCK {block}</span><b>dilation {dilation}</b><em>RF {RECEPTIVE_FIELDS[index + 1].toLocaleString()} bp</em></div>
            <div className="block-input">512 × {inputLength.toLocaleString()}</div>
            <div className="block-paths">
              <span><small>TRANSFORM PATH</small><b>Conv k=3, d={dilation}</b><i>bias → ReLU</i></span>
              <span><small>SHORTCUT PATH</small><b>crop {dilation} left + {dilation} right</b><i>preserve center features</i></span>
            </div>
            <div className="block-merge"><i>+</i><b>512 × {outputLength.toLocaleString()}</b></div>
          </button>;
        })}
      </div>
      <FlowArrow label="two parallel readers" />
      <div className="head-map">
        <article className="architecture-node"><small>PROFILE HEAD</small><h3>Where?</h3><b>1,000 positions</b><span>Conv1D k=75 → softmax</span></article>
        <article className="architecture-node"><small>COUNT HEAD</small><h3>How much?</h3><b>1 scalar</b><span>global mean → dense</span></article>
      </div>
      <p className="parallel-note">The heads branch from the same final tensor. Neither head feeds into the other.</p>
    </div>
  </section>;
}

function TensorCanvas({ tensor, values, mode, start, width, transfer, gain, signed = false, displayMaximum, marker, onMarker, selectionStart, selectionWidth, channelOrder }: {
  tensor: Tensor; values: Float32Array | null; mode: "full" | "zoom"; start: number; width: number; transfer: Transfer; gain: number; signed?: boolean; displayMaximum?: number; marker?: number; onMarker?: (value: number) => void; selectionStart?: number; selectionWidth?: number; channelOrder: number[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fullWidth = tensor.full_heatmap.width_positions;
  const height = tensor.full_heatmap.height_channels;
  const shownWidth = mode === "full" ? fullWidth : Math.min(width, fullWidth);
  const safeStart = mode === "full" ? 0 : clamp(start, 0, fullWidth - shownWidth);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !values) return;
    canvas.width = shownWidth;
    canvas.height = height;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;
    const image = context.createImageData(shownWidth, height);
    for (let displayRow = 0; displayRow < height; displayRow += 1) {
      const channel = channelOrder[displayRow];
      for (let column = 0; column < shownWidth; column += 1) {
        const sourceColumn = safeStart + column;
        const value = values[channel * fullWidth + sourceColumn];
        const maximum = displayMaximum ?? (signed ? Math.max(Math.abs(tensor.min), Math.abs(tensor.max), 1e-12) : tensor.max);
        const level = transformValue(Math.abs(value), maximum, transfer, gain);
        const color = signed
          ? [247, 244, 236].map((base, index) => {
              const target = value >= 0 ? [225, 91, 69] : [44, 129, 158];
              return Math.round(base + (target[index] - base) * level);
            })
          : heatColor(level).match(/\d+/g)!.map(Number);
        const target = (displayRow * shownWidth + column) * 4;
        image.data[target] = color[0]; image.data[target + 1] = color[1]; image.data[target + 2] = color[2]; image.data[target + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [values, shownWidth, height, safeStart, fullWidth, tensor.min, tensor.max, transfer, gain, signed, displayMaximum, channelOrder]);
  const click = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!onMarker) return;
    const rect = event.currentTarget.getBoundingClientRect();
    onMarker(Math.round(safeStart + clamp((event.clientX - rect.left) / rect.width, 0, 1) * (shownWidth - 1)));
  };
  const markerLeft = marker === undefined ? null : clamp((marker - safeStart) / Math.max(1, shownWidth - 1) * 100, 0, 100);
  const selectionLeft = selectionStart === undefined ? null : clamp((selectionStart - safeStart) / shownWidth * 100, 0, 100);
  const selectionPercent = selectionWidth === undefined ? null : clamp(selectionWidth / shownWidth * 100, 0, 100);
  return <div className={`tensor-canvas ${mode}`}>
    <div className="channel-axis"><span>rank 1 · Ch {channelOrder[0] + 1}</span><b>512 display rows</b><span>rank 512 · Ch {channelOrder[511] + 1}</span></div>
    <div className="canvas-shell" role={onMarker ? "button" : undefined} tabIndex={onMarker ? 0 : undefined} aria-label={onMarker ? "Move tensor zoom marker" : undefined} onClick={click} onKeyDown={event => {
      if (!onMarker || marker === undefined) return;
      if (event.key === "ArrowLeft") onMarker(clamp(marker - 1, safeStart, safeStart + shownWidth - 1));
      if (event.key === "ArrowRight") onMarker(clamp(marker + 1, safeStart, safeStart + shownWidth - 1));
    }}>{!values && <span className="loading">loading raw tensor…</span>}<canvas ref={canvasRef} />{markerLeft !== null && <i className="position-marker" style={{ left: `${markerLeft}%` }} />}{selectionLeft !== null && selectionPercent !== null && <i className="tensor-window-marker" style={{ left: `${selectionLeft}%`, width: `${selectionPercent}%` }} />}</div>
    <div className="axis"><span>{safeStart + 1}</span><span>{mode === "full" ? "every location shown · float32 values" : `${shownWidth} positions × all 512 channels`}</span><span>{safeStart + shownWidth}</span></div>
  </div>;
}

function SignalCanvas({ track, start, width, marker, signed, onMarker, label, colorGain = 1 }: {
  track: number[]; start: number; width: number; marker: number; signed: boolean; onMarker?: (value: number) => void; label: string; colorGain?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const shownWidth = Math.min(width, track.length);
  const safeStart = clamp(start, 0, Math.max(0, track.length - shownWidth));
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || track.length === 0) return;
    canvas.width = shownWidth;
    canvas.height = 150;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;
    context.fillStyle = "#f7f4ec";
    context.fillRect(0, 0, canvas.width, canvas.height);
    const visible = track.slice(safeStart, safeStart + shownWidth);
    const min = signed ? Math.min(...visible, 0) : 0;
    const max = Math.max(...visible, 1e-12);
    const absoluteMax = Math.max(...visible.map(Math.abs), 1e-12);
    const range = Math.max(max - min, 1e-12);
    const zeroY = Math.round((max / range) * (canvas.height - 1));
    context.fillStyle = "#d1cbc0";
    context.fillRect(0, zeroY, canvas.width, 1);
    visible.forEach((value, index) => {
      const valueY = Math.round(((max - value) / range) * (canvas.height - 1));
      const strength = clamp(Math.sqrt(Math.abs(value) / absoluteMax) * colorGain, 0, 1);
      const background = [247, 244, 236];
      const target = value < 0 ? [44, 129, 158] : [225, 91, 69];
      const color = background.map((base, channel) => Math.round(base + (target[channel] - base) * strength));
      context.fillStyle = `rgb(${color.join(" ")})`;
      context.fillRect(index, Math.min(valueY, zeroY), 1, Math.max(1, Math.abs(zeroY - valueY)));
    });
  }, [track, safeStart, shownWidth, signed, colorGain]);
  const markerLeft = clamp((marker - safeStart) / Math.max(1, shownWidth - 1) * 100, 0, 100);
  const select = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!onMarker) return;
    const rect = event.currentTarget.getBoundingClientRect();
    onMarker(Math.round(safeStart + clamp((event.clientX - rect.left) / rect.width, 0, 1) * (shownWidth - 1)));
  };
  return <div className={`signal-canvas-shell ${onMarker ? "interactive" : ""}`} onClick={select} onKeyDown={event => { if (!onMarker) return; if (event.key === "ArrowLeft") onMarker(clamp(marker - 1, safeStart, safeStart + shownWidth - 1)); if (event.key === "ArrowRight") onMarker(clamp(marker + 1, safeStart, safeStart + shownWidth - 1)); }} role={onMarker ? "button" : undefined} tabIndex={onMarker ? 0 : undefined} aria-label={label}>
    <canvas ref={ref} />
    <i className="position-marker" style={{ left: `${markerLeft}%` }} />
  </div>;
}

function VectorCanvas({ values, signed = false }: { values: number[]; signed?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = 1; canvas.height = values.length;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;
    const maximum = Math.max(...values.map(Math.abs), 1e-12);
    const image = context.createImageData(1, values.length);
    values.forEach((value, index) => {
      const color = (signed ? signedColor(value, maximum) : heatColor(Math.sqrt(clamp(value / maximum, 0, 1)))).match(/\d+/g)!.map(Number);
      const target = index * 4;
      image.data[target] = color[0]; image.data[target + 1] = color[1]; image.data[target + 2] = color[2]; image.data[target + 3] = 255;
    });
    context.putImageData(image, 0, 0);
  }, [values, signed]);
  return <canvas ref={ref} className="vector-canvas" />;
}

function ResidualStory({ demo, block, setBlock, channelOrder }: { demo: Demo; block: number; setBlock: (value: number) => void; channelOrder: number[] }) {
  const kernel = demo.residual_kernel_demos[block - 1];
  const dilation = DILATIONS[block - 1];
  const inputKey = (block === 1 ? "stem" : `res${block - 1}`) as TensorKey;
  const inputTensor = tensorFor(demo, inputKey);
  const { values: inputValues } = useTensorData(inputTensor);
  const inputLength = LENGTHS[block];
  const outputLength = LENGTHS[block + 1];
  const availableModes = (kernel as typeof kernel & { example_modes?: Record<"balanced" | "correction" | "output", ResidualExample> }).example_modes;
  const [exampleMode, setExampleMode] = useState<"balanced" | "correction" | "output">("balanced");
  const selectedExample = availableModes?.[exampleMode];
  const [position, setPosition] = useState(selectedExample?.output_position_zero_based ?? kernel.trace.output_position_zero_based);
  const outputChannel = selectedExample?.output_channel_zero_based ?? kernel.output_channel_zero_based;
  const exampleWeights = selectedExample?.weights_input_channels_by_taps ?? kernel.weights_input_channels_by_taps;
  const exampleBias = selectedExample?.bias ?? kernel.bias;

  const calculation = useMemo(() => {
    if (!inputValues) return null;
    const tapPositions = [position, position + dilation, position + 2 * dilation];
    const features = tapPositions.map(tap => Array.from({ length: 512 }, (_, channel) => inputValues[channel * inputLength + tap]));
    const weights = [0, 1, 2].map(tap => exampleWeights.map(row => row[tap]));
    const products = features.map((values, tap) => values.map((value, channel) => value * weights[tap][channel]));
    const tapSums = products.map(values => values.reduce((sum, value) => sum + value, 0));
    const convolution = tapSums.reduce((sum, value) => sum + value, 0);
    const beforeRelu = convolution + exampleBias;
    const transformed = Math.max(0, beforeRelu);
    const shortcut = features[1][outputChannel];
    const output = transformed + shortcut;
    const contributors = products.flatMap((values, tap) => values.map((value, channel) => ({ tap, channel, value, feature: features[tap][channel], weight: weights[tap][channel] }))).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 6);
    return { tapPositions, features, weights, products, tapSums, convolution, beforeRelu, transformed, shortcut, output, contributors };
  }, [inputValues, inputLength, position, dilation, exampleWeights, exampleBias, outputChannel]);

  const globalOutput = inputCoordinate(position, outputLength);
  const atSelectedExample = selectedExample?.output_position_zero_based === position;
  const tapSpan = 2 * dilation + 1;

  return <section className="story-section residual-story" id="residual">
    <SectionHeading step="4" eyebrow="CONTEXT BUILDING" title="A residual block combines features across channels and distance">
      The three taps are still a sliding convolution. Dilation changes only their spacing. At every tap, the kernel reads <b>all 512 feature channels</b>—not one channel in isolation.
    </SectionHeading>
    <div className="choice-row block-choices">{DILATIONS.map((value, index) => <button key={value} className={block === index + 1 ? "active" : ""} onClick={() => setBlock(index + 1)}><b>Block {index + 1}</b><small>d={value} · RF {RECEPTIVE_FIELDS[index + 1]}</small></button>)}</div>
    <div className="residual-example-controls"><label><span>Teaching example</span><select value={availableModes ? exampleMode : "legacy"} disabled={!availableModes} onChange={event => { const mode = event.target.value as "balanced" | "correction" | "output"; setExampleMode(mode); setPosition(availableModes![mode].output_position_zero_based); }}><option value="balanced">Balanced merge · both paths active</option><option value="correction">Strongest learned correction</option><option value="output">Strongest final activation</option>{!availableModes && <option value="legacy">Stored checkpoint example</option>}</select></label><p>{selectedExample ? atSelectedExample ? `${selectedExample.selection_rule}. Channel ${selectedExample.output_channel_zero_based + 1}, input-aligned coordinate ${selectedExample.input_aligned_coordinate_one_based}.` : `You moved away from the selected ${exampleMode} example to input coordinate ${globalOutput}. This live cell was not selected by that rule, so either path may now be zero.` : `This preset predates the multi-example export. The exact stored cell currently shows correction ${calculation?.transformed.toFixed(4) ?? "…"} + shortcut ${calculation?.shortcut.toFixed(4) ?? "…"}; it was selected for a high block output, not a balanced merge.`}</p>{selectedExample && !atSelectedExample && <button onClick={() => setPosition(selectedExample.output_position_zero_based)}>Return to selected example</button>}</div>
    <div className="dilation-explainer" data-feedback-id="Dilated three-tap slider">
      <div className="mini-heading"><div><small>SAME SLIDING IDEA</small><h3>Three one-position taps span {tapSpan.toLocaleString()} positions</h3></div><span>kernel width = 3 · dilation = {dilation}</span></div>
      <div className="tap-ruler">
        {[0, 1, 2].map((tap) => <div className="tap-position" key={tap}><b>tap {tap + 1}</b><span>input-tensor position {(calculation?.tapPositions[tap] ?? position + tap * dilation) + 1}</span><em>one 512-value vector</em></div>)}
        <div className="gap gap-one">skip {dilation - 1} positions</div><div className="gap gap-two">skip {dilation - 1} positions</div>
      </div>
      <label className="range-control"><b>Slide this dilated kernel</b><input type="range" min="0" max={outputLength - 1} value={position} onChange={event => setPosition(Number(event.target.value))} /><span>output {position + 1} / {outputLength.toLocaleString()}</span></label>
      <div className="coordinate-readout"><b>Shared input coordinate: {globalOutput.toLocaleString()}</b><span>tap coordinates: {[0, 1, 2].map(tap => inputCoordinate(position + tap * dilation, inputLength)).join(", ")}</span><small>All coordinates on this page use the original 2,114-base input frame.</small></div>
    </div>

    <div className="tap-microscope" data-feedback-id="Cross-channel residual kernel">
      <div className="mini-heading"><div><small>ONE REAL KERNEL SLICE</small><h3>Producing Channel {outputChannel + 1}</h3></div><span>display rank {displayRank(channelOrder, outputChannel)} · immutable ID {outputChannel}</span></div>
      <p className="selection-caveat">This is a useful high-activation example, not a claim that this channel is the most biologically important.</p>
      <div className="tap-columns">
        {[0, 1, 2].map(tap => <article key={tap}>
          <h4>Tap {tap + 1}</h4><div className="vector-equation">
            <span><VectorCanvas values={channelOrder.map(channel => (calculation?.features[tap] ?? Array(512).fill(0))[channel])} /><small>512 input features · global order</small></span><b>×</b>
            <span><VectorCanvas values={channelOrder.map(channel => (calculation?.weights[tap] ?? exampleWeights.map(row => row[tap]))[channel])} signed /><small>same input-channel order</small></span><b>=</b>
            <span><VectorCanvas values={channelOrder.map(channel => (calculation?.products[tap] ?? Array(512).fill(0))[channel])} signed /><small>same product order</small></span>
          </div><strong>tap sum {calculation?.tapSums[tap].toFixed(4) ?? "…"}</strong>
        </article>)}
      </div>
      <details><summary>Inspect the six largest products in this calculation</summary><div className="contributor-table"><span>tap</span><span>channel</span><span>feature</span><span>weight</span><span>product</span>{calculation?.contributors.map((item, index) => <div className="table-row" key={`${item.tap}-${item.channel}-${index}`}><span>{item.tap + 1}</span><span>{item.channel + 1}</span><span>{item.feature.toFixed(4)}</span><span>{item.weight.toFixed(4)}</span><b>{item.value.toFixed(4)}</b></div>)}</div></details>
    </div>

    <div className="residual-calculation" data-feedback-id="Residual block numeric flow">
      <div className="transform-lane"><small>TRANSFORM PATH</small><span><em>3 tap sums</em><b>{calculation?.tapSums.map(value => value.toFixed(3)).join(" + ") ?? "loading"}</b></span><i>→</i><span><em>convolution sum</em><b>{calculation?.convolution.toFixed(4) ?? "…"}</b></span><i>+</i><span><em>bias</em><b>{exampleBias.toFixed(4)}</b></span><i>→</i><span><em>before ReLU</em><b>{calculation?.beforeRelu.toFixed(4) ?? "…"}</b></span><i>→</i><span><em>after ReLU</em><b>{calculation?.transformed.toFixed(4) ?? "…"}</b></span></div>
      <div className="shortcut-lane"><small>SHORTCUT PATH</small><span><em>same channel at the center tap</em><b>{calculation?.shortcut.toFixed(4) ?? "…"}</b></span><p>The shortcut tensor is cropped by {dilation} positions on the left and {dilation} on the right, so it aligns with the shorter convolution output.</p></div>
      <div className="merge-lane"><span>transformed value</span><b>{calculation?.transformed.toFixed(4) ?? "…"}</b><i>+</i><span>preserved shortcut</span><b>{calculation?.shortcut.toFixed(4) ?? "…"}</b><i>=</i><strong>{calculation?.output.toFixed(4) ?? "…"}</strong><em>one cell in block {block} output · no ReLU after this addition</em></div>
    </div>
    <div className="receptive-field-meter"><span>cumulative receptive field</span>{RECEPTIVE_FIELDS.map((field, index) => <i key={field} className={index === block ? "active" : ""}><b>{field.toLocaleString()}</b><small>{index === 0 ? "stem" : `B${index}`}</small></i>)}<strong>→ profile head: 1,115 bp</strong></div>
    <p className="takeaway"><b>Mental picture:</b> stem filters create local-feature channels. A residual kernel can give different weights to motif-like features in different channels and at three separated positions, allowing learned cooperation or competition while the shortcut preserves earlier features.</p>
  </section>;
}

function MiniBars({ values, signed = false, count = 160 }: { values: number[]; signed?: boolean; count?: number }) {
  const sampled = Array.from({ length: Math.min(count, values.length) }, (_, index) => {
    const start = Math.floor(index * values.length / count);
    const end = Math.max(start + 1, Math.floor((index + 1) * values.length / count));
    const bin = values.slice(start, end);
    return signed ? bin.reduce((strongest, value) => Math.abs(value) > Math.abs(strongest) ? value : strongest, 0) : Math.max(...bin, 0);
  });
  const maximum = Math.max(...sampled.map(Math.abs), 1e-12);
  return <div className={`mini-bars ${signed ? "signed" : ""}`} style={css({ "--columns": sampled.length })}>{sampled.map((value, index) => {
    const height = Math.abs(value) <= 1e-12 ? 1 : 3 + Math.abs(value) / maximum * 72;
    const color = signed ? signedColor(value, maximum) : heatColor(value / maximum);
    return <i key={index} style={{ height: `${height.toFixed(3)}px`, backgroundColor: rgbCssToHex(color) }} />;
  })}</div>;
}

function OutputStory({ demo, channelOrder }: { demo: Demo; channelOrder: number[] }) {
  const logits = demo.tensors.profile_logits.position_max;
  const maximumLogit = Math.max(...logits);
  const shiftedLogits = logits.map(value => value - maximumLogit);
  const minimumShiftedLogit = Math.min(...shiftedLogits);
  const relativeLogitHeights = shiftedLogits.map(value => value - minimumShiftedLogit);
  const exponentials = shiftedLogits.map(Math.exp);
  const probabilities = demo.tensors.profile_probabilities.position_max;
  const expectedCounts = demo.tensors.profile_signal.position_max;
  const pooled = demo.head_demos.count_pooled_features;
  const denseProducts = pooled.map((value, index) => value * demo.head_demos.count_dense_weights[index]);
  const denseSum = denseProducts.reduce((sum, value) => sum + value, 0);
  const profileSum = expectedCounts.reduce((sum, value) => sum + value, 0);
  const profilePeak = expectedCounts.indexOf(Math.max(...expectedCounts));
  const profileZoomStart = clamp(profilePeak - 30, 0, expectedCounts.length - 61);
  const profilePositions = Array.from({ length: 61 }, (_, index) => profileZoomStart + index);

  return <section className="story-section output-story" id="outputs">
    <SectionHeading step="5" eyebrow="TWO OUTPUT HEADS" title="The same final tensor answers “where?” and “how much?” in parallel">
      Both heads read the final <b>512 × 1,074</b> backbone tensor. The profile head never feeds the count head, and the count head never feeds the profile head.
    </SectionHeading>
    <div className="shared-tensor"><small>SHARED BACKBONE OUTPUT</small><b>512 channels × 1,074 positions</b><span>one source branches into two computations</span></div>
    <div className="branch-lines"><i /><i /></div>
    <div className="output-branches">
      <article data-feedback-id="Profile head stages">
        <div className="mini-heading"><div><small>PROFILE HEAD · WHERE?</small><h3>A 1,000-position distribution</h3></div><span>Conv1D kernel 75 × 512 × 1</span></div>
        <div className="head-stage"><b>1. Relative logits for an upward bar chart: logit − minimum logit</b><span>the smallest becomes 0. Subtracting one shared constant changes no softmax probability, and the tallest bar remains the largest logit.</span><MiniBars values={relativeLogitHeights} /></div>
        <FlowArrow label="exponentiate the shifted scores" />
        <div className="head-stage"><b>2. Relative positive scores</b><span>exp(logit − max); the same position is still highest</span><MiniBars values={exponentials} /></div>
        <FlowArrow label="divide every score by their sum" />
        <div className="head-stage"><b>3. Profile probabilities</b><span>nonnegative; all 1,000 values sum to 1</span><MiniBars values={probabilities} /></div>
        <FlowArrow label={`multiply every position by ${demo.outputs.predicted_total_count.toFixed(1)}`} />
        <div className="head-stage final"><b>4. Expected counts per base</b><span>the displayed accessibility profile; sums to {profileSum.toFixed(1)}</span><MiniBars values={expectedCounts} /></div>
        <p className="softmax-note"><b>Softmax preserves ordering:</b> raw logits, shifted logits, exponentials, probabilities, and expected counts all have the same peak position. Softmax sharpens relative differences; it does not invent a new peak.</p>
        <div className="profile-zoom"><b>61-position zoom around the predicted peak</b><div className="signal-bars" style={css({ "--columns": 61 })}>{profilePositions.map(position => <i key={position} style={{ height: `${3 + expectedCounts[position] / Math.max(...expectedCounts) * 82}px` }} />)}</div><BaseLetters sequence={demo.input.sequence} positions={profilePositions.map(position => position + 557)} /><div className="axis"><span>input {profileZoomStart + 558}</span><span>expected cuts per base · peak at input {profilePeak + 558}</span><span>input {profileZoomStart + 618}</span></div></div>
      </article>
      <article data-feedback-id="Count head stages">
        <div className="mini-heading"><div><small>COUNT HEAD · HOW MUCH?</small><h3>One total-count scalar</h3></div><span>global mean → dense</span></div>
        <div className="count-stage"><b>512 × 1,074</b><span>mean each channel across all 1,074 positions · current global order</span><MiniBars values={channelOrder.map(channel => pooled[channel])} /></div>
        <FlowArrow label="one mean per channel" />
        <div className="count-stage"><b>512 pooled features</b><span>multiply by matching dense weights · same immutable IDs</span><MiniBars values={channelOrder.map(channel => denseProducts[channel])} signed /></div>
        <FlowArrow label="add products and bias" />
        <div className="count-equation"><span>dense products <b>{denseSum.toFixed(3)}</b></span><i>+</i><span>bias <b>{demo.head_demos.count_dense_bias.toFixed(3)}</b></span><i>=</i><span>log-count <b>{demo.outputs.logcount.toFixed(3)}</b></span></div>
        <FlowArrow label="exp(log-count) − 1" />
        <div className="total-count"><strong>{demo.outputs.predicted_total_count.toFixed(1)}</strong><span>predicted total cuts</span></div>
      </article>
    </div>
    <div className="output-identity"><b>Final identity</b><span>profile probability at each base</span><i>×</i><span>one predicted total count</span><i>=</i><strong>expected counts at each base</strong></div>
  </section>;
}

function ProfileKernelHeatmap({ weights, bias, channelOrder }: { weights: number[][]; bias: number; channelOrder: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [gain, setGain] = useState(1.8);
  const [cursor, setCursor] = useState({ channel: 255, position: 37 });
  const absolute = useMemo(() => weights.flat().map(Math.abs).sort((a, b) => a - b), [weights]);
  const robustMaximum = absolute[Math.floor(absolute.length * .995)] || absolute.at(-1) || 1e-12;
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = 75; canvas.height = 512;
    const context = canvas.getContext("2d");
    if (!context) return;
    const image = context.createImageData(75, 512);
    for (let displayRow = 0; displayRow < 512; displayRow += 1) {
      const channel = channelOrder[displayRow];
      for (let position = 0; position < 75; position += 1) {
        const [red, green, blue] = signedRgb(weights[channel][position], robustMaximum / gain);
        const index = (displayRow * 75 + position) * 4;
        image.data[index] = red; image.data[index + 1] = green; image.data[index + 2] = blue; image.data[index + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [weights, robustMaximum, gain, channelOrder]);
  const select = (clientX: number, clientY: number, target: HTMLCanvasElement) => {
    const rect = target.getBoundingClientRect();
    setCursor({
      position: clamp(Math.floor((clientX - rect.left) / rect.width * 75), 0, 74),
      channel: clamp(Math.floor((clientY - rect.top) / rect.height * 512), 0, 511),
    });
  };
  return <div className="profile-kernel-card">
    <div className="mini-heading"><div><small>THE LEARNED PROFILE KERNEL</small><h3>75 relative positions × 512 input channels</h3></div><span>displayed as 512 channel rows × 75 position columns</span></div>
    <p>This one signed weight matrix is reused at every output position. Coral weights push a logit upward; blue weights push it downward. A color alone is not a biological motif—the weight acts on a learned final-layer feature.</p>
    <div className="profile-kernel-layout">
      <div className="profile-kernel-axis"><span>rank 1 · Ch {channelOrder[0] + 1}</span><b>512 input channels</b><span>rank 512 · Ch {channelOrder[511] + 1}</span></div>
      <div className="profile-kernel-shell">
        <canvas ref={canvasRef} role="img" aria-label="Profile convolution kernel heatmap, 512 channels by 75 relative positions" onPointerMove={event => select(event.clientX, event.clientY, event.currentTarget)} />
        <i style={{ left: `${cursor.position / 75 * 100}%`, top: `${cursor.channel / 512 * 100}%`, width: `${100 / 75}%`, height: `${100 / 512}%` }} />
      </div>
      <div className="axis"><span>relative 1</span><span>75 kernel positions</span><span>relative 75</span></div>
    </div>
    <div className="profile-kernel-controls">
      <label><b>Weight contrast · {gain.toFixed(1)}×</b><input type="range" min="0.5" max="5" step="0.1" value={gain} onChange={event => setGain(Number(event.target.value))} /></label>
      <div><span>selected weight</span><b>relative {cursor.position + 1} × rank {cursor.channel + 1} · Channel {channelOrder[cursor.channel] + 1}</b><strong>{weights[channelOrder[cursor.channel]][cursor.position].toFixed(5)}</strong></div>
      <div><span>profile bias</span><b>added once after the 38,400 products are summed</b><strong>{bias.toFixed(5)}</strong></div>
    </div>
    <div className="profile-kernel-equation"><span>selected 512 × 75 tensor window</span><i>⊙</i><span>shared 512 × 75 kernel</span><i>Σ</i><span>38,400 products + bias</span><i>→</i><strong>one profile logit</strong></div>
  </div>;
}

function ProfileLayerView({ demo, finalTensor, values, error, channelOrder }: { demo: Demo; finalTensor: Tensor; values: Float32Array | null; error: string | null; channelOrder: number[] }) {
  const expected = demo.tensors.profile_signal.position_max;
  const profilePeak = expected.indexOf(Math.max(...expected));
  const [profilePosition, setProfilePosition] = useState(profilePeak);
  const logits = demo.tensors.profile_logits.position_max;
  const maximumLogit = Math.max(...logits);
  const shiftedLogits = logits.map(value => value - maximumLogit);
  const minimumShiftedLogit = Math.min(...shiftedLogits);
  const relativeLogitHeights = shiftedLogits.map(value => value - minimumShiftedLogit);
  const probabilities = demo.tensors.profile_probabilities.position_max;
  const zoomStart = clamp(profilePosition - 30, 0, 939);
  const zoomPositions = Array.from({ length: 61 }, (_, index) => zoomStart + index);
  const inputPositions = zoomPositions.map(position => position + 557);
  const chooseFromTensor = (tensorPosition: number) => setProfilePosition(clamp(tensorPosition - 37, 0, 999));
  const inputWidth = 1074 / MAX_TENSOR_LENGTH * 100;
  const outputWidth = 1000 / MAX_TENSOR_LENGTH * 100;
  return <div className="profile-layer-view">
    <div className="computation-note"><b>Profile head · valid convolution</b><span>The final 512 × 1,074 tensor is the input. A learned 75-position kernel reads all 512 channels and produces 1,000 logits: 1,074 − 75 + 1 = 1,000.</span></div>
    {error && <div className="tensor-error" role="alert"><b>Tensor data did not load</b><span>{error}</span><button onClick={() => window.location.reload()}>Retry</button></div>}
    <div className="full-view-card profile-before">
      <div className="mini-heading"><div><small>BEFORE · FINAL BACKBONE TENSOR</small><h3>512 channels × 1,074 positions</h3></div><span>the outlined 75-position window produces one selected logit</span></div>
      <div className="whole-length-frame" style={{ width: `${inputWidth}%` }}><TensorCanvas tensor={finalTensor} values={values} mode="full" start={0} width={1074} transfer="sqrt" gain={1.8} marker={undefined} onMarker={chooseFromTensor} selectionStart={profilePosition} selectionWidth={75} channelOrder={channelOrder} /></div>
      <label className="range-control"><b>Move the width-75 profile kernel</b><input type="range" min="0" max="999" value={profilePosition} onChange={event => setProfilePosition(Number(event.target.value))} /><span>output {profilePosition + 1} / 1,000</span></label>
    </div>
    <FlowArrow label="the same learned kernel is placed over this 75-position window" />
    <ProfileKernelHeatmap weights={demo.head_demos.profile_weights_input_channels_by_positions} bias={demo.head_demos.profile_bias} channelOrder={channelOrder} />
    <FlowArrow label="element-wise multiply → sum 38,400 products → add bias → slide one position" />
    <div className="profile-stage-stack" style={{ width: `${outputWidth}%` }}>
      <article><div className="mini-heading"><div><small>AFTER CONVOLUTION · SHIFTED FOR AN UPWARD BAR CHART</small><h3>1 × 1,000 relative logits</h3></div><span>logit − minimum logit; the smallest becomes zero and ordering is unchanged</span></div><SignalCanvas track={relativeLogitHeights} start={0} width={1000} marker={profilePosition} signed={false} onMarker={setProfilePosition} label="Select a relative profile logit" /></article>
      <FlowArrow label="softmax across all 1,000 logits" />
      <article><div className="mini-heading"><div><small>NORMALIZED PROFILE SHAPE</small><h3>1 × 1,000 probabilities</h3></div><span>nonnegative · sums to 1</span></div><SignalCanvas track={probabilities} start={0} width={1000} marker={profilePosition} signed={false} onMarker={setProfilePosition} label="Select a profile probability" /></article>
      <FlowArrow label={`multiply every position by total count ${demo.outputs.predicted_total_count.toFixed(1)}`} />
      <article><div className="mini-heading"><div><small>FINAL PROFILE OUTPUT</small><h3>1 × 1,000 expected counts</h3></div><span>same profile shape · biologically scaled height</span></div><SignalCanvas track={expected} start={0} width={1000} marker={profilePosition} signed={false} onMarker={setProfilePosition} label="Select an expected-count position" /></article>
    </div>
    <FlowArrow label="the selected profile position opens here" />
    <div className="zoom-view-card"><div className="mini-heading"><div><small>PROFILE ZOOM</small><h3>61 of the 1,000 output positions</h3></div><span>base below = center of the width-75 final-tensor window</span></div><SignalCanvas track={expected} start={zoomStart} width={61} marker={profilePosition} signed={false} label="Expected-count profile zoom" /><BaseLetters sequence={demo.input.sequence} positions={inputPositions} /></div>
  </div>;
}

function TensorInspector({ demo, channelOrder, orderName }: { demo: Demo; channelOrder: number[]; orderName: ChannelOrder }) {
  const [layer, setLayer] = useState<LayerKey>("stem");
  const [computation, setComputation] = useState<ComputationStage>("relu");
  const [view, setView] = useState<InspectorView>("tensor");
  const [transfer, setTransfer] = useState<Transfer>("sqrt");
  const [gain, setGain] = useState(1.8);
  const [extremaGain, setExtremaGain] = useState(1.8);
  const [sharedMagnitude, setSharedMagnitude] = useState(false);
  const tensor = layer === "profile" ? tensorFor(demo, "res8") : inspectorTensorFor(demo, layer, computation);
  const { values, error } = useTensorData(tensor);
  const length = tensor.full_heatmap.width_positions;
  const [center, setCenter] = useState(Math.floor(length / 2));
  const [channel, setChannel] = useState(tensor.selected_channels_zero_based[0]);
  const signed = computation === "conv" || computation === "bias";
  const zoomWidth = 61;
  const zoomStart = clamp(center - Math.floor(zoomWidth / 2), 0, length - zoomWidth);
  const positions = Array.from({ length: zoomWidth }, (_, index) => zoomStart + index);
  const inputPositions = positions.map(position => Math.round(offsetForLength(length) + position));
  const wholeWidthPercent = length / MAX_TENSOR_LENGTH * 100;
  const croppedPerSide = (MAX_TENSOR_LENGTH - length) / 2;
  const sharedMaximum = Math.max(...(["stem", "res1", "res2", "res3", "res4", "res5", "res6", "res7", "res8"] as TensorKey[]).map(key => {
    const candidate = tensorFor(demo, key);
    return Math.max(Math.abs(candidate.min), Math.abs(candidate.max));
  }));
  const displayMaximum = sharedMagnitude ? sharedMaximum : undefined;

  const chooseStage = (nextStage: LayerKey) => {
    if (nextStage === "profile") {
      const finalTensor = tensorFor(demo, "res8");
      setLayer("profile"); setComputation("output"); setCenter(Math.floor(finalTensor.full_heatmap.width_positions / 2)); setChannel(finalTensor.selected_channels_zero_based[0]);
      return;
    }
    const nextComputation = nextStage === "stem" ? (computation === "output" ? "relu" : computation) : layer === "stem" ? "output" : computation;
    const nextTensor = inspectorTensorFor(demo, nextStage, nextComputation);
    setLayer(nextStage);
    setComputation(nextComputation);
    setCenter(Math.floor(nextTensor.full_heatmap.width_positions / 2));
    setChannel(nextTensor.selected_channels_zero_based[0]);
  };

  const chooseComputation = (next: ComputationStage) => {
    if (layer === "profile") return;
    const nextTensor = inspectorTensorFor(demo, layer, next);
    setComputation(next);
    setCenter(Math.floor(nextTensor.full_heatmap.width_positions / 2));
  };

  const channelValues = values ? Array.from({ length }, (_, position) => values[channel * length + position]) : [];
  const metric = view === "max" ? tensor.position_max : view === "min" ? tensor.position_min : tensor.position_mean;
  const track = view === "channel" ? channelValues : metric;
  const computationLabel = computation === "conv" ? "After convolution · before bias" : computation === "bias" ? "After adding channel bias · before ReLU" : computation === "relu" ? "After ReLU · transform path" : "After shortcut + ReLU correction · block output";
  const viewTitle = view === "channel" ? `Channel ${channel + 1}` : view === "max" ? "Maximum across 512 channels" : view === "min" ? "Minimum across 512 channels" : "Mean across 512 channels";

  if (layer === "profile") return <section className="inspector" id="tensor-inspector">
    <SectionHeading step="6" eyebrow="SUPPORTING INSPECTION TOOL" title="Inspect the raw 512 × N tensors after learning the computation">Choose any backbone layer, or follow the profile head as it converts the complete final feature tensor into a one-dimensional accessibility profile.</SectionHeading>
    <div className="inspector-controls profile-inspector-controls">
      <label><span>Layer</span><select value={layer} onChange={event => chooseStage(event.target.value as LayerKey)}><option value="stem">Stem · 512 × 2,094</option>{DILATIONS.map((dilation, index) => <option key={dilation} value={`res${index + 1}`}>Residual {index + 1} · 512 × {LENGTHS[index + 2].toLocaleString()}</option>)}<option value="profile">Profile output · 512 × 1,074 → 1 × 1,000</option></select></label>
      <div className="profile-shape-summary"><span>Selected transition</span><b>512 × 1,074 → Conv k=75 → 1 × 1,000</b></div>
    </div>
    <ProfileLayerView demo={demo} finalTensor={tensor} values={values} error={error} channelOrder={channelOrder} />
  </section>;

  return <section className="inspector" id="tensor-inspector">
    <SectionHeading step="6" eyebrow="SUPPORTING INSPECTION TOOL" title="Inspect the raw 512 × N tensors after learning the computation">
      Begin with the complete representation, then use the shared marker to open the same coordinate at higher resolution. The computation control exposes exactly what bias, ReLU, and the residual shortcut change.
    </SectionHeading>
    <div className="inspector-controls">
      <label><span>Layer</span><select value={layer} onChange={event => chooseStage(event.target.value as LayerKey)}><option value="stem">Stem · 512 × 2,094</option>{DILATIONS.map((dilation, index) => <option key={dilation} value={`res${index + 1}`}>Residual {index + 1} · 512 × {LENGTHS[index + 2].toLocaleString()}</option>)}<option value="profile">Profile output · 512 × 1,074 → 1 × 1,000</option></select></label>
      <div className="computation-control"><span>Computation state</span><div className="segmented">{(["conv", "bias", "relu", ...(layer === "stem" ? [] : ["output"])] as ComputationStage[]).map(option => <button key={option} className={computation === option ? "active" : ""} onClick={() => chooseComputation(option)}>{option === "conv" ? "Convolution" : option === "bias" ? "+ Bias" : option === "relu" ? "ReLU" : "+ Shortcut"}</button>)}</div></div>
      <div><span>Presentation</span><div className="segmented">{(["tensor", "channel", "max", "mean", "min"] as InspectorView[]).map(option => <button key={option} className={view === option ? "active" : ""} onClick={() => setView(option)}>{option === "tensor" ? "Raw heatmap" : option === "channel" ? "One channel" : option === "max" ? "Max" : option === "mean" ? "Mean" : "Min"}</button>)}</div></div>
      {view === "tensor" && <label><span>Weak-value transform</span><select value={transfer} onChange={event => setTransfer(event.target.value as Transfer)}><option value="linear">Linear · most literal</option><option value="sqrt">Square root · reveal weak</option><option value="log">Log · reveal more weak</option></select></label>}
      {view === "tensor" && <label><span>Magnitude scale</span><select value={sharedMagnitude ? "shared" : "layer"} onChange={event => setSharedMagnitude(event.target.value === "shared")}><option value="layer">Per layer · reveal structure</option><option value="shared">Shared · compare magnitude</option></select></label>}
      {view === "tensor" && <label><span>Brightness gain · {gain.toFixed(1)}×</span><input type="range" min="0.5" max="5" step="0.1" value={gain} onChange={event => setGain(Number(event.target.value))} /></label>}
      {(view === "max" || view === "min") && <label><span>Extrema color strength · {extremaGain.toFixed(1)}×</span><input type="range" min="0.2" max="5" step="0.1" value={extremaGain} onChange={event => setExtremaGain(Number(event.target.value))} /></label>}
      {view === "channel" && <label><span>Channel number</span><input type="number" min="1" max="512" value={channel + 1} onChange={event => setChannel(clamp(Number(event.target.value || 1) - 1, 0, 511))} /></label>}
    </div>
    <div className="computation-note"><b>{computationLabel}</b><span>{layer === "stem" ? "The stem has no shortcut addition." : computation === "output" ? "This is the tensor passed to the next block." : "This view isolates the residual block’s transform path before its shortcut is added."}</span></div>
    <div className="channel-order-note"><b>{CHANNEL_ORDER_LABELS[orderName]}</b><span>Every tensor row uses the same display permutation. Shortcut rows, residual input/output axes, head weights, and immutable channel IDs stay synchronized.</span></div>
    <div className="truth-note"><b>{(tensor.zero_fraction * 100).toFixed(1)}% exact zeros</b><span>{signed ? "Coral cells are positive; blue cells are negative." : "Blank-looking areas are real zero or weak activations, not missing data."} {view === "tensor" ? sharedMagnitude ? `All backbone layers share ±${sharedMaximum.toFixed(2)} where signed.` : "Color is normalized within this layer; compare patterns, not magnitude." : ""}</span><em>Published checkpoint · raw float32</em></div>
    {error && <div className="tensor-error" role="alert"><b>Tensor data did not load</b><span>{error}</span><button onClick={() => window.location.reload()}>Retry</button></div>}
    {view === "tensor" ? <>
      <div className="full-view-card"><div className="mini-heading"><div><small>WHOLE TENSOR</small><h3>All 512 channels × {length.toLocaleString()} positions</h3></div><span>{croppedPerSide ? `${croppedPerSide.toLocaleString()} positions cropped from each side · ` : "full stem width · "}click to move the zoom</span></div><div className="whole-length-frame" style={{ width: `${wholeWidthPercent}%` }}><TensorCanvas tensor={tensor} values={values} mode="full" start={0} width={length} transfer={transfer} gain={gain} signed={signed} displayMaximum={displayMaximum} marker={center} onMarker={setCenter} channelOrder={channelOrder} /></div></div>
      <FlowArrow label="the outlined position opens here" />
      <div className="zoom-view-card"><div className="mini-heading"><div><small>ZOOM</small><h3>All 512 channels × 61 consecutive positions</h3></div><span>same tensor, larger cells</span></div><TensorCanvas tensor={tensor} values={values} mode="zoom" start={zoomStart} width={zoomWidth} transfer={transfer} gain={gain} signed={signed} displayMaximum={displayMaximum} marker={center} channelOrder={channelOrder} /><BaseLetters sequence={demo.input.sequence} positions={inputPositions} /></div>
    </> : <>
      <div className="full-view-card"><div className="mini-heading"><div><small>WHOLE LENGTH · {viewTitle.toUpperCase()}</small><h3>All {length.toLocaleString()} positions</h3></div><span>{wholeWidthPercent.toFixed(1)}% of stem width · centered crop</span></div><div className="whole-length-frame" style={{ width: `${wholeWidthPercent}%` }}><SignalCanvas track={track} start={0} width={length} marker={center} signed={signed} colorGain={view === "max" || view === "min" ? extremaGain : 1} onMarker={setCenter} label={`Move the ${viewTitle} zoom marker`} /><div className="axis"><span>position 1</span><span>complete representation first</span><span>position {length.toLocaleString()}</span></div></div></div>
      <FlowArrow label="the selected coordinate opens here" />
      <div className="zoom-view-card">
      <div className="mini-heading"><div><small>ZOOM · {viewTitle.toUpperCase()}</small><h3>61 consecutive positions</h3></div><span>same values, with sequence bases</span></div>
      <SignalCanvas track={track} start={zoomStart} width={zoomWidth} marker={center} signed={signed} colorGain={view === "max" || view === "min" ? extremaGain : 1} label={`${viewTitle} zoom`} /><BaseLetters sequence={demo.input.sequence} positions={inputPositions} />
      <label className="range-control"><b>Move the 61-position window</b><input type="range" min="30" max={length - 31} value={center} onChange={event => setCenter(Number(event.target.value))} /><span>input {inputCoordinate(center, length)}</span></label>
      <p className="view-definition">{view === "max" ? "Max retains the largest of 512 channel values at each position." : view === "min" ? "Min retains the most negative of 512 channel values; after ReLU it will be zero or positive." : view === "mean" ? "Mean averages the same 512 channels at every position. A sum has the identical shape, multiplied everywhere by 512." : "A channel is a learned feature dimension. High activation says this feature responded here; it does not by itself establish biological causality."}</p>
      </div>
    </>}
  </section>;
}

type FeedbackAnnotation = { id: number; quote: string; note: string; context: string };
const FEEDBACK_STORAGE_KEY = "sequence-cnn-explainer-feedback-v2";

function FeedbackLayer() {
  const [open, setOpen] = useState(false);
  const [quote, setQuote] = useState("");
  const [context, setContext] = useState("Explainer");
  const [note, setNote] = useState("");
  const [annotations, setAnnotations] = useState<FeedbackAnnotation[]>([]);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const timer = window.setTimeout(() => {
      try { setAnnotations(JSON.parse(localStorage.getItem(FEEDBACK_STORAGE_KEY) ?? "[]")); } catch { setAnnotations([]); }
      setReady(true);
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);
  useEffect(() => { if (ready) localStorage.setItem(FEEDBACK_STORAGE_KEY, JSON.stringify(annotations)); }, [annotations, ready]);
  useEffect(() => {
    if (!open) return;
    const select = () => {
      const selection = window.getSelection();
      const text = selection?.toString().trim();
      if (!text || !selection?.anchorNode) return;
      const element = selection.anchorNode.parentElement?.closest("[data-feedback-id], section");
      if (element?.closest(".feedback-drawer")) return;
      setQuote(text); setContext(element?.getAttribute("data-feedback-id") || element?.id || "Explainer section");
    };
    document.addEventListener("mouseup", select);
    return () => document.removeEventListener("mouseup", select);
  }, [open]);
  const save = () => {
    if (!quote.trim() || !note.trim()) return;
    setAnnotations(items => [...items, { id: Date.now(), quote: quote.trim(), note: note.trim(), context }]); setQuote(""); setNote("");
  };
  const markdown = annotations.map((item, index) => `${index + 1}. **${item.context}**\n\n> ${item.quote.replaceAll("\n", "\n> ")}\n\n${item.note}`).join("\n\n---\n\n");
  const download = () => { const url = URL.createObjectURL(new Blob([`# Sequence CNN Explainer feedback\n\n${markdown}`], { type: "text/markdown" })); const anchor = document.createElement("a"); anchor.href = url; anchor.download = "sequence-cnn-explainer-feedback-v2.md"; anchor.click(); URL.revokeObjectURL(url); };
  return <>
    <button className="feedback-toggle" onClick={() => setOpen(value => !value)} aria-expanded={open}>✎ {open ? "Close annotations" : `Annotate${annotations.length ? ` (${annotations.length})` : ""}`}</button>
    {open && <aside className="feedback-drawer" aria-label="Feedback annotations">
      <div className="drawer-head"><div><small>FEEDBACK MODE</small><h2>Highlight text, then annotate</h2></div><button onClick={() => setOpen(false)} aria-label="Close">×</button></div>
      <div className="drawer-compose"><div><label>Quoted target<textarea value={quote} onChange={event => setQuote(event.target.value)} rows={3} placeholder="Highlight words anywhere above, or paste them here." /></label><small>From: {context}</small></div><label>Your note<textarea value={note} onChange={event => setNote(event.target.value)} rows={3} placeholder="What is confusing, incorrect, or missing?" /></label><button disabled={!quote.trim() || !note.trim()} onClick={save}>Save annotation</button></div>
      <div className="saved-annotations">{annotations.map((item, index) => <article key={item.id}><small>#{index + 1} · {item.context}</small><blockquote>{item.quote}</blockquote><p>{item.note}</p><button onClick={() => setAnnotations(items => items.filter(saved => saved.id !== item.id))}>Delete</button></article>)}{annotations.length > 0 && <div className="export-actions"><button onClick={() => navigator.clipboard.writeText(markdown)}>Copy Markdown</button><button onClick={download}>Download .md</button></div>}</div>
      <p className="drawer-note">The drawer stays at the bottom, so the full page width remains available for highlighting.</p>
    </aside>}
  </>;
}

export default function Home() {
  const [preset, setPreset] = useState("k562");
  const [windowStart, setWindowStart] = useState(0);
  const [block, setBlock] = useState(1);
  const [colorGain, setColorGain] = useState(1);
  const [channelOrderName, setChannelOrderName] = useState<ChannelOrder>("original");
  const { value: demo, error: demoError } = useJsonAsset<Demo>(PRESET_URLS[preset]);
  const { value: auditCheckpoints, error: auditError } = useJsonAsset<{ checkpoints: AuditCheckpoints }>("/data/model-audit-summary.json.gz");
  const auditCheckpoint = auditCheckpoints?.checkpoints[preset === "gm21515" ? "gm21515" : "k562-peak"];
  const empiricalOrderingAvailable = preset !== "synthetic";
  useEffect(() => {
    if (demo) setWindowStart(demo.filter_demos[0].peak_position_zero_based);
  }, [demo]);
  if (!demo || !auditCheckpoint) return <main className="route-loading"><b>Loading verified checkpoint data…</b>{(demoError || auditError) && <p role="alert">{demoError || auditError}</p>}</main>;
  const channelOrder = auditCheckpoint.channel_orders[empiricalOrderingAvailable ? channelOrderName : "original"];
  return <main>
    <a className="video-banner" href="https://youtu.be/-wmg_ehUxfk" target="_blank" rel="noreferrer" aria-label="Watch How CNNs Read DNA on YouTube">
      <span>NEW VIDEO</span><b>Watch: How CNNs Read DNA</b><i>Play on YouTube ↗</i>
    </a>
    <header className="topbar"><a href="#top" className="brand"><b>SEQ</b><span>CNN EXPLAINER</span></a><nav><a href="#architecture">Model map</a><a href="#stem">Stem</a><a href="#residual">Residual blocks</a><a href="#outputs">Outputs</a><a href="#tensor-inspector">Tensors</a><a href="/dilation-trace">Dilation evolution ↗</a><a href="/model-audit">Model audit ↗</a><a href="/basset">Basset model ↗</a></nav><div className="topbar-controls"><label><span>Demo</span><select value={preset} onChange={event => { const nextPreset = event.target.value; setPreset(nextPreset); setWindowStart(0); setBlock(1); if (nextPreset === "synthetic") setChannelOrderName("original"); }}><option value="k562">K562 DNase checkpoint</option><option value="gm21515">GM21515 ATAC checkpoint</option><option value="synthetic">K562 synthetic sequence</option></select></label><label title={empiricalOrderingAvailable ? "One global display order" : "Empirical orders were not computed for the synthetic sequence"}><span>Channel order</span><select disabled={!empiricalOrderingAvailable} value={empiricalOrderingAvailable ? channelOrderName : "original"} onChange={event => setChannelOrderName(event.target.value as ChannelOrder)}>{CHANNEL_ORDER_KEYS.map(key => <option value={key} key={key}>{CHANNEL_ORDER_LABELS[key]}</option>)}</select></label></div></header>
    <div id="top" />
    <section className="hero">
      <div><p>ONE CALCULATION · FOLLOWED TOP TO BOTTOM</p><h1>How a DNA sequence becomes an accessibility prediction</h1><span>Start with bases. Watch a local detector slide. Then see dilated blocks combine learned features across distance and channels.</span></div>
      <aside><small>RUNNING EXAMPLE</small><b>{demo.input.locus_label}</b><span>{demo.provenance.biosample} · {demo.provenance.assay} · {demo.provenance.experiment}</span><em>published bias-corrected task-model fold-0 checkpoint · “no-bias” in the source filename means the separate enzyme-sequence-bias model was factored out; ordinary learned layer biases remain · exact forward-pass activations</em></aside>
    </section>
    <div className="promise-strip"><b>The mental model</b><span>detect local patterns</span><i>↓</i><span>combine features over wider context</span><i>↓</i><span>predict where and how much</span></div>
    <section className="cross-model-frame" aria-label="Shared vocabulary for ChromBPNet and Basset">
      <div><small>TWO CNN DESIGNS · ONE VOCABULARY</small><h2>The output task changes which spatial compromises are acceptable</h2><p>A base-resolution output cannot simply discard position without a recovery mechanism. ChromBPNet preserves a long grid and crosses distance with dilation. Basset predicts sequence-level cell-type labels, so it can compress to ten positions and then mix globally with dense layers.</p></div>
      <dl><div><dt>Channel</dt><dd>one learned feature type</dd></div><div><dt>Receptive field</dt><dd>input bases that can affect one cell</dd></div><div><dt>Crossing distance</dt><dd>ChromBPNet: dilated taps · Basset: pooling then dense</dd></div><div><dt>Output</dt><dd>ChromBPNet: 1,000-position profile + count · Basset: 164 sequence labels</dd></div></dl>
      <a href="/basset">Compare with the verified Basset checkpoint →</a>
    </section>

    <ArchitectureMap selectedBlock={block} onBlock={value => { setBlock(value); document.getElementById("residual")?.scrollIntoView({ behavior: "smooth" }); }} />
    <FlowArrow label="start with one real input window" />
    <section className="story-section input-story" id="input">
      <SectionHeading step="1" eyebrow="SEQUENCE INPUT" title="Each DNA base is one column, not a tiny image">
        The model receives a <b>4 × 2,114</b> one-hot matrix. A, C, G, and T are the four rows; every position contains one 1 and three 0s.
      </SectionHeading>
      <SequenceOverview sequence={demo.input.sequence} start={windowStart} onStart={setWindowStart} />
      <label className="range-control"><b>Move the selected input window</b><input type="range" min="0" max="2093" value={windowStart} onChange={event => setWindowStart(Number(event.target.value))} /><span>{windowStart + 1}–{windowStart + 21}</span></label>
      <OneHotZoom sequence={demo.input.sequence} start={windowStart} />
      <p className="takeaway"><b>Mental picture:</b> sequence position runs left to right. The four base rows are channels, not height in a biological image.</p>
    </section>
    <FlowArrow label="apply 512 different local detectors" />
    <StemStory key={preset} demo={demo} start={windowStart} setStart={setWindowStart} gain={colorGain} channelOrder={channelOrder} orderName={empiricalOrderingAvailable ? channelOrderName : "original"} motifStatus={auditCheckpoint.activation_motifs} />
    <FlowArrow label="stack all 512 output rows" />
    <section className="bridge-section" data-feedback-id="Stem tensor bridge"><div><small>THE FIRST FEATURE TENSOR</small><h2>512 learned feature rows × 2,094 positions</h2><p>Most cells are zero after ReLU. A bright cell means one learned detector responded at one position; it does not yet say why the final prediction changed.</p></div><label><span>Kernel color saturation · {colorGain.toFixed(1)}×</span><input type="range" min="0.5" max="2" step="0.1" value={colorGain} onChange={event => setColorGain(Number(event.target.value))} /></label></section>
    <FlowArrow label="build wider context through eight blocks" />
    <ResidualStory key={`${preset}-${block}`} demo={demo} block={block} setBlock={setBlock} channelOrder={channelOrder} />
    <FlowArrow label="read the final 512 × 1,074 tensor two ways" />
    <OutputStory demo={demo} channelOrder={channelOrder} />
    <FlowArrow label="optional: inspect every tensor value" />
    <TensorInspector demo={demo} channelOrder={channelOrder} orderName={channelOrderName} />
    <section className="attribution-note"><div><small>NOT PART OF THIS FORWARD-PASS DEMO</small><h2>Input attribution requires an additional analysis</h2></div><p>DeepLIFT, SHAP, gradients, or another attribution method must generate base-level attribution scores. Activations alone show what each layer computed, not which input bases caused the final prediction.</p></section>
    <footer><b>Sequence CNN Explainer · structural prototype</b><span>Model values: published {demo.provenance.biosample} ChromBPNet checkpoint · coordinates: input-relative unless explicitly labeled</span></footer>
    <FeedbackLayer />
  </main>;
}
