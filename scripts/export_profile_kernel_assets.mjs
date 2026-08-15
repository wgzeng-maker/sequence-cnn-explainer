#!/usr/bin/env node

import { gzipSync } from "node:zlib";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const presets = [
  ["k562-peak", "k562_peak"],
  ["gm21515", "gm21515"],
];

for (const [sourceName, publicName] of presets) {
  const source = JSON.parse(readFileSync(resolve(root, `app/data/${sourceName}-activations.json`), "utf8"));
  const weights = source.head_demos.profile_weights_input_channels_by_positions;
  if (weights.length !== 512 || weights.some(row => row.length !== 75)) {
    throw new Error(`${sourceName}: expected a 512 channel × 75 position profile kernel`);
  }

  const raw = Buffer.allocUnsafe(512 * 75 * Float32Array.BYTES_PER_ELEMENT);
  let offset = 0;
  for (const channel of weights) {
    for (const value of channel) {
      raw.writeFloatLE(value, offset);
      offset += Float32Array.BYTES_PER_ELEMENT;
    }
  }

  const target = resolve(root, `public/data/tensors/${publicName}/profile_kernel.f32.gz`);
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, gzipSync(raw, { level: 9, mtime: 0 }));
  process.stdout.write(`Wrote ${target}\n`);
}
