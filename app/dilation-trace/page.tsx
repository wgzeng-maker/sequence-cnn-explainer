"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import Link from "next/link";
import demo from "../data/k562-peak-activations.json";
import auditArtifact from "../data/model-audit-summary.json";
import { loadFloat32Tensor } from "../tensor-loader";
import { CHANNEL_ORDER_LABELS, type ChannelOrder } from "../model-analysis";
import styles from "./page.module.css";

type TensorKey = "stem" | "res1" | "res2" | "res3" | "res4" | "res5" | "res6" | "res7" | "res8";
type RawTensors = Partial<Record<TensorKey, Float32Array>>;
type Matrix = number[][];

const DILATIONS = [2, 4, 8, 16, 32, 64, 128, 256];
const LENGTHS = [2094, 2090, 2082, 2066, 2034, 1970, 1842, 1586, 1074];
const RECEPTIVE_FIELDS = [21, 25, 33, 49, 81, 145, 273, 529, 1041];
const KEYS: TensorKey[] = ["stem", "res1", "res2", "res3", "res4", "res5", "res6", "res7", "res8"];
const LABELS = ["Stem", "Block 1", "Block 2", "Block 3", "Block 4", "Block 5", "Block 6", "Block 7", "Block 8"];
const CHANNEL_ORDERS = (auditArtifact as unknown as { checkpoints: Record<string, { channel_orders: Record<ChannelOrder, number[]> }> }).checkpoints["k562-peak"].channel_orders;
const CHANNEL_ORDER_KEYS = Object.keys(CHANNEL_ORDER_LABELS) as ChannelOrder[];
const BASE_COLORS: Record<string, string> = { A: "#e96b54", C: "#4f97b2", G: "#e5b33f", T: "#72a17e" };
const css = (values: Record<string, string | number>) => values as CSSProperties;
const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));
const offsetForLength = (length: number) => (2114 - length) / 2;
const globalToLocal = (globalOneBased: number, length: number) => Math.round(globalOneBased - 1 - offsetForLength(length));
const localToGlobal = (localZeroBased: number, length: number) => Math.round(localZeroBased + offsetForLength(length) + 1);

function tensorFor(key: TensorKey) {
  return demo.tensors[key] as typeof demo.tensors.stem;
}

function useRawTensors() {
  const [loaded, setLoaded] = useState<RawTensors>({});
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setError(null);
    Promise.all(KEYS.map(async key => {
      const tensor = tensorFor(key);
      const values = await loadFloat32Tensor(tensor.full_heatmap.url, tensor.full_heatmap.height_channels * tensor.full_heatmap.width_positions, controller.signal);
      return [key, values] as const;
    }))
      .then(entries => setLoaded(Object.fromEntries(entries) as RawTensors))
      .catch(reason => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "The tensor data could not be loaded.");
      });
    return () => controller.abort();
  }, []);
  return { raw: loaded, error };
}

function color(value: number, maximum: number, signed = false) {
  const level = Math.sqrt(clamp(Math.abs(value) / Math.max(maximum, 1e-12), 0, 1));
  const background = [247, 244, 236];
  const target = signed && value < 0 ? [225, 91, 69] : signed ? [44, 129, 158] : level < .55 ? [236, 188, 70] : [121, 38, 68];
  const strength = signed ? level : level < .55 ? level / .55 : (level - .55) / .45;
  const from = signed || level < .55 ? background : [236, 188, 70];
  const rgb = from.map((base, index) => Math.round(base + (target[index] - base) * strength));
  return rgb;
}

function matrixAt(key: TensorKey, raw: Float32Array | undefined, channels: number[], globalCenter: number, width = 61): Matrix {
  if (!raw) return channels.map(() => Array(width).fill(0));
  const length = LENGTHS[KEYS.indexOf(key)];
  const center = globalToLocal(globalCenter, length);
  const start = clamp(center - Math.floor(width / 2), 0, length - width);
  return channels.map(channel => Array.from({ length: width }, (_, index) => raw[channel * length + start + index]));
}

function MatrixCanvas({ matrix, maximum, signed = false, label }: { matrix: Matrix; maximum?: number; signed?: boolean; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const max = maximum ?? Math.max(...matrix.flat().map(Math.abs), 1e-12);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || matrix.length === 0) return;
    const height = matrix.length;
    const width = matrix[0].length;
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;
    const image = context.createImageData(width, height);
    matrix.forEach((row, y) => row.forEach((value, x) => {
      const rgb = color(value, max, signed);
      const target = (y * width + x) * 4;
      image.data[target] = rgb[0]; image.data[target + 1] = rgb[1]; image.data[target + 2] = rgb[2]; image.data[target + 3] = 255;
    }));
    context.putImageData(image, 0, 0);
  }, [matrix, max, signed]);
  return <canvas ref={ref} className={styles.matrixCanvas} aria-label={label} />;
}

function BaseRow({ globalCenter, width = 61 }: { globalCenter: number; width?: number }) {
  const start = globalCenter - Math.floor(width / 2) - 1;
  return <div className={styles.baseRow} style={css({ "--columns": width })}>{Array.from({ length: width }, (_, index) => {
    const base = demo.input.sequence[start + index] ?? "N";
    return <i key={index} style={{ background: BASE_COLORS[base] ?? "#ddd" }}>{base}</i>;
  })}</div>;
}

function StageFilmstrip({ raw, channels, globalCenter, stage, setStage, sharedScale }: {
  raw: RawTensors; channels: number[]; globalCenter: number; stage: number; setStage: (value: number) => void; sharedScale: boolean;
}) {
  const matrices = KEYS.map(key => matrixAt(key, raw[key], channels, globalCenter));
  const maximum = sharedScale ? Math.max(...matrices.flat(2).map(Math.abs), 1e-12) : undefined;
  return <div className={styles.filmstrip}>
    {KEYS.map((key, index) => <button key={key} className={stage === index ? styles.stageSelected : ""} onClick={() => setStage(index)} aria-pressed={stage === index}>
      <span><b>{LABELS[index]}</b><small>{index === 0 ? "k=21" : `d=${DILATIONS[index - 1]}`}</small></span>
      <MatrixCanvas matrix={matrices[index]} maximum={maximum} label={`${LABELS[index]} tensor, 12 tracked channels by 61 positions`} />
      <em>{LENGTHS[index].toLocaleString()} positions · RF {RECEPTIVE_FIELDS[index]} bp</em>
    </button>)}
  </div>;
}

function FullTensorMagnifier({ tensorKey, raw, globalCenter, setGlobalCenter, channelStart, setChannelStart, channelOrder }: {
  tensorKey: TensorKey; raw?: Float32Array; globalCenter: number; setGlobalCenter: (value: number) => void; channelStart: number; setChannelStart: (value: number) => void; channelOrder: number[];
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const length = LENGTHS[KEYS.indexOf(tensorKey)];
  const localCenter = globalToLocal(globalCenter, length);
  const localStart = clamp(localCenter - 30, 0, length - 61);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !raw) return;
    canvas.width = length; canvas.height = 512;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;
    const image = context.createImageData(length, 512);
    let maximum = 1e-12;
    raw.forEach(value => { maximum = Math.max(maximum, Math.abs(value)); });
    for (let displayRow = 0; displayRow < 512; displayRow += 1) for (let position = 0; position < length; position += 1) {
      const channel = channelOrder[displayRow];
      const rgb = color(raw[channel * length + position], maximum);
      const target = (displayRow * length + position) * 4;
      image.data[target] = rgb[0]; image.data[target + 1] = rgb[1]; image.data[target + 2] = rgb[2]; image.data[target + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, [raw, length, channelOrder]);
  const move = (clientX: number, clientY: number, element: HTMLDivElement) => {
    const rect = element.getBoundingClientRect();
    const local = Math.round(clamp((clientX - rect.left) / rect.width, 0, 1) * (length - 1));
    const channel = Math.round(clamp((clientY - rect.top) / rect.height, 0, 1) * 511) - 6;
    setGlobalCenter(clamp(localToGlobal(local, length), 558, 1557));
    setChannelStart(clamp(channel, 0, 500));
  };
  return <div className={styles.magnifierOverview}>
    <div className={styles.magnifierHeading}><div><small>WHOLE SELECTED TENSOR</small><h3>{LABELS[KEYS.indexOf(tensorKey)]} · 512 × {length.toLocaleString()}</h3></div><span>move across the heatmap to choose a 12-channel × 61-position window</span></div>
    <div className={styles.fullTensorShell} role="button" tabIndex={0} aria-label="Move the tensor magnifier across channels and input-aligned positions" onPointerMove={event => move(event.clientX, event.clientY, event.currentTarget)} onPointerDown={event => move(event.clientX, event.clientY, event.currentTarget)} onKeyDown={event => {
      if (event.key === "ArrowLeft") setGlobalCenter(clamp(globalCenter - 1, 558, 1557));
      if (event.key === "ArrowRight") setGlobalCenter(clamp(globalCenter + 1, 558, 1557));
      if (event.key === "ArrowUp") setChannelStart(clamp(channelStart - 1, 0, 500));
      if (event.key === "ArrowDown") setChannelStart(clamp(channelStart + 1, 0, 500));
    }}>
      {!raw && <span>loading full tensor…</span>}<canvas ref={ref} />
      <i className={styles.magnifierSelection} style={{ left: `${localStart / length * 100}%`, width: `${61 / length * 100}%`, top: `${channelStart / 512 * 100}%`, height: `${12 / 512 * 100}%` }}><b /></i>
    </div>
    <div className={styles.magnifierReadout}><span>display ranks {channelStart + 1}–{channelStart + 12}</span><span>immutable IDs: {channelOrder.slice(channelStart, channelStart + 12).map(channel => channel + 1).join(", ")}</span><span>input-aligned positions {globalCenter - 30}–{globalCenter + 30}</span></div>
  </div>;
}

function FeatureReach({ stemRaw, channels, globalCenter, stage }: { stemRaw?: Float32Array; channels: number[]; globalCenter: number; stage: number }) {
  const events = useMemo(() => {
    if (!stemRaw) return [];
    const length = LENGTHS[0];
    const viewStart = globalCenter - 520;
    const viewEnd = globalCenter + 520;
    return channels.map(channel => {
      let bestPosition = globalToLocal(viewStart, length);
      let bestValue = -Infinity;
      const localEnd = globalToLocal(viewEnd, length);
      for (let position = Math.max(0, bestPosition); position <= Math.min(length - 1, localEnd); position += 1) {
        const value = stemRaw[channel * length + position];
        if (value > bestValue) { bestValue = value; bestPosition = position; }
      }
      return { channel, global: localToGlobal(bestPosition, length), value: bestValue };
    });
  }, [stemRaw, channels, globalCenter]);
  const half = (RECEPTIVE_FIELDS[stage] - 1) / 2;
  const bandLeft = (520 - half) / 1041 * 100;
  const bandWidth = RECEPTIVE_FIELDS[stage] / 1041 * 100;
  return <div className={styles.reachMap}>
    <div className={styles.reachHeader}><b>Stem-feature events inside the final 1,041-bp neighborhood</b><span>bright = geometrically reachable by {LABELS[stage]}</span></div>
    <div className={styles.reachLanes}>
      <div className={styles.receptiveBand} style={{ left: `${bandLeft}%`, width: `${bandWidth}%` }}><span>{RECEPTIVE_FIELDS[stage]} bp</span></div>
      {events.map(event => {
        const reachable = Math.abs(event.global - globalCenter) <= half;
        return <div className={styles.reachLane} key={event.channel}><b>Ch {event.channel + 1}</b><i className={reachable ? styles.reachable : ""} style={{ left: `${(event.global - (globalCenter - 520)) / 1041 * 100}%` }} title={`Channel ${event.channel + 1}, input ${event.global}, activation ${event.value.toFixed(3)}`}><span>{event.value.toFixed(2)}</span></i></div>;
      })}
    </div>
    <div className={styles.axis}><span>{globalCenter - 520}</span><span>selected input coordinate {globalCenter}</span><span>{globalCenter + 520}</span></div>
    <p>This map shows possible reach through the stacked kernels. It does not claim that every reachable feature affected the prediction.</p>
  </div>;
}

function BlockChange({ raw, channels, globalCenter, stage }: { raw: RawTensors; channels: number[]; globalCenter: number; stage: number }) {
  if (stage === 0) return <div className={styles.stemOnly}><b>Stem tensor selected</b><span>Choose Block 1–8 to separate its preserved shortcut from its newly learned correction.</span></div>;
  const inputKey = KEYS[stage - 1];
  const outputKey = KEYS[stage];
  const input = raw[inputKey];
  const output = raw[outputKey];
  const inputLength = LENGTHS[stage - 1];
  const outputLength = LENGTHS[stage];
  const dilation = DILATIONS[stage - 1];
  const width = 61;
  const outputCenter = globalToLocal(globalCenter, outputLength);
  const outputStart = clamp(outputCenter - 30, 0, outputLength - width);
  const shortcut = channels.map(channel => Array.from({ length: width }, (_, index) => input ? input[channel * inputLength + outputStart + index + dilation] : 0));
  const after = channels.map(channel => Array.from({ length: width }, (_, index) => output ? output[channel * outputLength + outputStart + index] : 0));
  const correction = after.map((row, y) => row.map((value, x) => Math.max(0, value - shortcut[y][x])));
  const maximum = Math.max(...shortcut.flat(), ...correction.flat(), ...after.flat(), 1e-12);
  const changed = correction.flat().filter(value => value > 1e-7).length;
  return <div className={styles.changeEquation}>
    <article><small>PRESERVED SHORTCUT</small><h3>Earlier features, center-cropped</h3><MatrixCanvas matrix={shortcut} maximum={maximum} label="Shortcut tensor excerpt" /></article>
    <b>+</b>
    <article><small>LEARNED CORRECTION</small><h3>Dilated Conv → bias → ReLU</h3><MatrixCanvas matrix={correction} maximum={maximum} label="New nonnegative correction tensor excerpt" /><em>{changed} of {channels.length * width} shown cells receive a positive correction</em></article>
    <b>=</b>
    <article><small>BLOCK OUTPUT</small><h3>Preserved + newly combined</h3><MatrixCanvas matrix={after} maximum={maximum} label="Residual block output tensor excerpt" /></article>
    <BaseRow globalCenter={globalCenter} />
  </div>;
}

function ExactCommunication({ raw, globalCenter, stage, jumpToPeak }: { raw: RawTensors; globalCenter: number; stage: number; jumpToPeak: () => void }) {
  if (stage === 0) return null;
  const block = demo.residual_kernel_demos[stage - 1];
  const inputKey = KEYS[stage - 1];
  const input = raw[inputKey];
  const inputLength = LENGTHS[stage - 1];
  const outputLength = LENGTHS[stage];
  const dilation = DILATIONS[stage - 1];
  const outputPosition = globalToLocal(globalCenter, outputLength);
  const outputChannel = block.output_channel_zero_based;
  const calculation = useMemo(() => {
    if (!input) return null;
    const taps = [0, 1, 2].map(tap => {
      const position = outputPosition + tap * dilation;
      const contributions = block.weights_input_channels_by_taps.map((weights, channel) => {
        const activation = input[channel * inputLength + position];
        const weight = weights[tap];
        return { channel, activation, weight, product: activation * weight };
      });
      const sum = contributions.reduce((total, item) => total + item.product, 0);
      return { position, sum, top: contributions.sort((a, b) => Math.abs(b.product) - Math.abs(a.product)).slice(0, 4) };
    });
    const convolution = taps.reduce((sum, tap) => sum + tap.sum, 0);
    const beforeRelu = convolution + block.bias;
    const transformed = Math.max(0, beforeRelu);
    const shortcut = input[outputChannel * inputLength + outputPosition + dilation];
    return { taps, convolution, beforeRelu, transformed, shortcut, output: transformed + shortcut };
  }, [input, inputLength, outputPosition, dilation, block, outputChannel]);
  return <div className={styles.communication}>
    <div className={styles.panelHeading}><div><small>ONE EXACT CELL CALCULATION</small><h2>How three feature neighborhoods communicate in {LABELS[stage]}</h2></div><button onClick={jumpToPeak}>Jump to this channel&apos;s strongest example</button></div>
    <p className={styles.scopeNote}>Example output channel {outputChannel + 1} was selected because it reaches a large activation in this block—not because it is known to represent a named biological motif.</p>
    <div className={styles.tapNetwork}>
      {calculation?.taps.map((tap, tapIndex) => <article key={tapIndex}>
        <header><b>Tap {tapIndex + 1}</b><span>input coordinate {localToGlobal(tap.position, inputLength)}</span></header>
        {tap.top.map(item => <div className={styles.contributor} key={item.channel}><b>Ch {item.channel + 1}</b><span>{item.activation.toFixed(3)} activation</span><i className={item.product < 0 ? styles.negative : ""}>{item.weight.toFixed(3)} weight</i><strong>{item.product >= 0 ? "+" : ""}{item.product.toFixed(3)}</strong></div>)}
        <footer>all 512 products sum to <b>{tap.sum.toFixed(4)}</b></footer>
      </article>)}
      {!calculation && <p>Loading real input-feature vectors…</p>}
    </div>
    {calculation && <div className={styles.numericFlow}><span>three tap sums <b>{calculation.convolution.toFixed(4)}</b></span><i>+</i><span>bias <b>{block.bias.toFixed(4)}</b></span><i>→</i><span>ReLU correction <b>{calculation.transformed.toFixed(4)}</b></span><i>+</i><span>shortcut <b>{calculation.shortcut.toFixed(4)}</b></span><i>=</i><strong>{calculation.output.toFixed(4)}</strong></div>}
    <p className={styles.explanation}>Each tap reads all 512 channels. Therefore a positive feature in one channel at the left tap and another feature in a different channel at the right tap can jointly raise—or oppose—the same output cell.</p>
  </div>;
}

function ProfileReader({ raw, globalCenter }: { raw: RawTensors; globalCenter: number }) {
  const finalTensor = raw.res8;
  const profileIndex = globalCenter - 558;
  const weights = demo.head_demos.profile_weights_input_channels_by_positions;
  const calculation = useMemo(() => {
    if (!finalTensor) return null;
    const contributions: { channel: number; kernelPosition: number; activation: number; weight: number; product: number }[] = [];
    for (let kernelPosition = 0; kernelPosition < 75; kernelPosition += 1) {
      for (let channel = 0; channel < 512; channel += 1) {
        const activation = finalTensor[channel * 1074 + profileIndex + kernelPosition];
        const weight = weights[channel][kernelPosition];
        contributions.push({ channel, kernelPosition, activation, weight, product: activation * weight });
      }
    }
    const convolution = contributions.reduce((sum, item) => sum + item.product, 0);
    return { convolution, logit: convolution + demo.head_demos.profile_bias, top: contributions.sort((a, b) => Math.abs(b.product) - Math.abs(a.product)).slice(0, 8) };
  }, [finalTensor, profileIndex, weights]);
  const probability = demo.tensors.profile_probabilities.position_max[profileIndex];
  const expected = demo.tensors.profile_signal.position_max[profileIndex];
  const chartStart = clamp(profileIndex - 50, 0, 899);
  const chartPositions = Array.from({ length: 101 }, (_, index) => chartStart + index);
  const maxExpected = Math.max(...demo.tensors.profile_signal.position_max);
  return <div className={styles.profileReader}>
    <div className={styles.panelHeading}><div><small>FINAL PROFILE READER</small><h2>How the accumulated features become one profile position</h2></div><span>selected input coordinate {globalCenter}</span></div>
    <div className={styles.profileInput}><b>75 final-tensor positions × 512 channels</b><span>38,400 activation × weight products feed this one logit</span><em>input coordinates {globalCenter - 37}–{globalCenter + 37}</em></div>
    <div className={styles.profileContributors}>{calculation?.top.map((item, index) => <div key={`${item.channel}-${item.kernelPosition}`}><small>#{index + 1}</small><b>Ch {item.channel + 1}</b><span>at input {globalCenter - 37 + item.kernelPosition}</span><i>{item.activation.toFixed(3)} × {item.weight.toFixed(3)}</i><strong>{item.product >= 0 ? "+" : ""}{item.product.toFixed(3)}</strong></div>)}</div>
    <div className={styles.profileFlow}><span>calculated logit <b>{calculation?.logit.toFixed(4) ?? "…"}</b></span><i>softmax with 999 other logits</i><span>probability <b>{probability.toExponential(2)}</b></span><i>× total count</i><strong>{expected.toFixed(3)} expected cuts</strong></div>
    <div className={styles.profileChart} style={css({ "--columns": 101 })}>{chartPositions.map(position => <i key={position} className={position === profileIndex ? styles.profileSelected : ""} style={{ height: `${3 + demo.tensors.profile_signal.position_max[position] / maxExpected * 100}px` }} />)}</div>
    <div className={styles.axis}><span>input {chartStart + 558}</span><span>final expected-count profile</span><span>input {chartStart + 658}</span></div>
  </div>;
}

export default function DilationTracePage() {
  const { raw, error } = useRawTensors();
  const [stage, setStage] = useState(1);
  const [globalCenter, setGlobalCenter] = useState(1058);
  const [channelStart, setChannelStart] = useState(clamp(demo.tensors.stem.selected_channels_zero_based[0] - 6, 0, 500));
  const [channelOrderName, setChannelOrderName] = useState<ChannelOrder>("original");
  const [playing, setPlaying] = useState(false);
  const [sharedScale, setSharedScale] = useState(false);
  const channelOrder = CHANNEL_ORDERS[channelOrderName];
  const channels = channelOrder.slice(channelStart, channelStart + 12);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setStage(current => {
      if (current >= 8) { setPlaying(false); return 8; }
      return current + 1;
    }), 1100);
    return () => window.clearInterval(timer);
  }, [playing]);

  const jumpToPeak = () => {
    if (stage === 0) return;
    const block = demo.residual_kernel_demos[stage - 1];
    setGlobalCenter(clamp(localToGlobal(block.trace.output_position_zero_based, LENGTHS[stage]), 558, 1557));
  };

  return <main className={styles.lab}>
    <header className={styles.topbar}><Link href="/">← Full explainer</Link><b>DILATION EVOLUTION LAB</b><span>real K562 checkpoint</span></header>
    <section className={styles.hero}>
      <p>EXPERIMENTAL VISUAL STORY</p>
      <h1>Watch local features become a wider conversation.</h1>
      <span>Keep one input coordinate fixed. Step through the stem and eight dilated residual blocks to see which feature cells are preserved, which corrections are added, and how the final tensor feeds the accessibility profile.</span>
    </section>
    <section className={styles.controls}>
      <div><button onClick={() => { setStage(0); setPlaying(true); }} className={styles.playButton}>{playing ? "Playing…" : "▶ Play all stages"}</button><button onClick={() => setSharedScale(value => !value)}>Color scale: {sharedScale ? "shared across stages" : "normalized per stage"}</button><label><b>Global channel order</b><select value={channelOrderName} onChange={event => { setChannelOrderName(event.target.value as ChannelOrder); setChannelStart(0); }}>{CHANNEL_ORDER_KEYS.map(key => <option value={key} key={key}>{CHANNEL_ORDER_LABELS[key]}</option>)}</select></label></div>
      <label><b>Tracked input coordinate</b><input type="range" min="558" max="1557" value={globalCenter} onChange={event => setGlobalCenter(Number(event.target.value))} /><span>{globalCenter}</span></label>
      <div className={styles.stageButtons}>{LABELS.map((label, index) => <button key={label} aria-pressed={stage === index} className={stage === index ? styles.active : ""} onClick={() => { setPlaying(false); setStage(index); }}><b>{label}</b><span>RF {RECEPTIVE_FIELDS[index]} bp</span></button>)}</div>
    </section>
    {error && <div className={styles.tensorError} role="alert"><b>Tensor data did not load</b><span>{error}</span><button onClick={() => window.location.reload()}>Retry</button></div>}

    <section className={styles.section}>
      <div className={styles.sectionHeading}><span>1</span><div><small>WHOLE TENSOR → MAGNIFIED FILMSTRIP</small><h2>Select one tensor region, then compare that exact region through every stage</h2><p>The large heatmap establishes where the excerpt comes from. The magnifier chooses 12 display-adjacent, permanently identified channels and 61 input-aligned positions; every small panel below uses that same selection.</p></div></div>
      <FullTensorMagnifier tensorKey={KEYS[stage]} raw={raw[KEYS[stage]]} globalCenter={globalCenter} setGlobalCenter={setGlobalCenter} channelStart={channelStart} setChannelStart={setChannelStart} channelOrder={channelOrder} />
      <div className={styles.magnifiedLabel}><b>MAGNIFIED COMPARISON</b><span>ranks {channelStart + 1}–{channelStart + 12} · IDs {channels.map(channel => channel + 1).join(", ")} · input {globalCenter - 30}–{globalCenter + 30}</span></div>
      <StageFilmstrip raw={raw} channels={channels} globalCenter={globalCenter} stage={stage} setStage={setStage} sharedScale={sharedScale} />
    </section>

    <section className={styles.section}>
      <div className={styles.sectionHeading}><span>2</span><div><small>GROWING REACH</small><h2>Dilation lets the selected position receive information from farther features</h2><p>The receptive field grows cumulatively from 21 bases after the stem to 1,041 bases after Block 8.</p></div></div>
      <FeatureReach stemRaw={raw.stem} channels={channels} globalCenter={globalCenter} stage={stage} />
    </section>

    <section className={styles.section}>
      <div className={styles.sectionHeading}><span>3</span><div><small>BEFORE → CHANGE → AFTER</small><h2>See what this residual block actually changes in the tensor</h2><p>The learned correction is computed as block output minus its aligned shortcut. Because the transform ends with ReLU, this correction is nonnegative.</p></div></div>
      <BlockChange raw={raw} channels={channels} globalCenter={globalCenter} stage={stage} />
    </section>

    <section className={styles.section}><ExactCommunication raw={raw} globalCenter={globalCenter} stage={stage} jumpToPeak={jumpToPeak} /></section>

    <section className={styles.section}>
      <div className={styles.sectionHeading}><span>5</span><div><small>FINAL READOUT</small><h2>The profile head reads the accumulated feature tensor</h2><p>At each output position, a width-75 kernel mixes all 512 final channels. Softmax then turns all 1,000 logits into a distribution.</p></div></div>
      <ProfileReader raw={raw} globalCenter={globalCenter} />
    </section>

    <aside className={styles.caveat}><b>Interpretation boundary</b><p>The one-block products above are exact forward contributions to a selected cell. The growing receptive field shows possible information flow. Establishing which original bases caused the final prediction still requires attribution or controlled sequence perturbations.</p></aside>
  </main>;
}
