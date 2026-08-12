import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import test from "node:test";
import { gunzipSync } from "node:zlib";

const root = new URL("../", import.meta.url);
const lengths = [2094, 2090, 2082, 2066, 2034, 1970, 1842, 1586, 1074];
const tensorNames = ["stem", "res1", "res2", "res3", "res4", "res5", "res6", "res7", "res8"];

async function loadDemo(name) {
  return JSON.parse(await readFile(new URL(`app/data/${name}-activations.json`, root), "utf8"));
}

for (const preset of ["k562-peak", "gm21515", "synthetic"]) {
  test(`${preset} tensors preserve the published checkpoint shapes and raw values`, async () => {
    const demo = await loadDemo(preset);
    assert.equal(demo.input.sequence.length, 2114);
    assert.deepEqual(demo.input.base_order, ["A", "C", "G", "T"]);
    for (let index = 0; index < tensorNames.length; index += 1) {
      const tensor = demo.tensors[tensorNames[index]];
      assert.deepEqual(tensor.shape_batch_positions_channels, [1, lengths[index], 512]);
      assert.equal(tensor.full_heatmap.dtype, "float32_le");
      assert.match(tensor.full_heatmap.value_preservation, /no display quantization/i);
      const file = new URL(`public/data/tensors/${demo.input.preset_id}/${tensorNames[index]}.f32.gz`, root);
      assert.ok((await stat(file)).size > 0);
      assert.equal(gunzipSync(await readFile(file)).byteLength, 512 * lengths[index] * 4);
    }
  });

  test(`${preset} residual traces obey convolution, ReLU, and shortcut identities`, async () => {
    const demo = await loadDemo(preset);
    for (let blockIndex = 0; blockIndex < demo.residual_kernel_demos.length; blockIndex += 1) {
      const block = demo.residual_kernel_demos[blockIndex];
      const trace = block.trace;
      const tapSum = trace.tap_sums.reduce((sum, value) => sum + value, 0);
      assert.ok(Math.abs(tapSum - trace.convolution_sum) < 2e-5);
      assert.ok(Math.abs(trace.convolution_sum + block.bias - trace.before_relu) < 2e-6);
      assert.ok(Math.abs(Math.max(0, trace.before_relu) - trace.transformed_after_relu) < 2e-6);
      assert.ok(Math.abs(trace.transformed_after_relu + trace.shortcut_value - trace.block_output) < 2e-6);
      assert.deepEqual(trace.input_tap_positions_zero_based, [0, 1, 2].map(tap => trace.output_position_zero_based + tap * block.dilation));

      const inputName = blockIndex === 0 ? "stem" : `res${blockIndex}`;
      const inputLength = lengths[blockIndex];
      const raw = gunzipSync(await readFile(new URL(`public/data/tensors/${demo.input.preset_id}/${inputName}.f32.gz`, root)));
      const input = new Float32Array(new Uint8Array(raw).slice().buffer);
      const tapSums = [0, 1, 2].map(tap => block.weights_input_channels_by_taps.reduce((sum, weights, channel) => {
        const position = trace.output_position_zero_based + tap * block.dilation;
        return sum + input[channel * inputLength + position] * weights[tap];
      }, 0));
      const transformed = Math.max(0, tapSums.reduce((sum, value) => sum + value, 0) + block.bias);
      const shortcut = input[block.output_channel_zero_based * inputLength + trace.output_position_zero_based + block.dilation];
      assert.ok(Math.abs(transformed + shortcut - trace.block_output) < 2e-5);
    }
  });

  test(`${preset} output heads retain distinct logits, probabilities, and expected counts`, async () => {
    const demo = await loadDemo(preset);
    const probabilities = demo.tensors.profile_probabilities.position_max;
    const counts = demo.tensors.profile_signal.position_max;
    assert.equal(probabilities.length, 1000);
    assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-4);
    assert.ok(Math.abs(counts.reduce((sum, value) => sum + value, 0) - demo.outputs.predicted_total_count) < 1e-2);
    const dense = demo.head_demos.count_pooled_features.reduce((sum, value, index) => sum + value * demo.head_demos.count_dense_weights[index], 0);
    assert.ok(Math.abs(dense + demo.head_demos.count_dense_bias - demo.outputs.logcount) < 2e-5);

    assert.equal(demo.head_demos.profile_weights_input_channels_by_positions.length, 512);
    assert.equal(demo.head_demos.profile_weights_input_channels_by_positions[0].length, 75);
    const finalRaw = gunzipSync(await readFile(new URL(`public/data/tensors/${demo.input.preset_id}/res8.f32.gz`, root)));
    const finalTensor = new Float32Array(new Uint8Array(finalRaw).slice().buffer);
    for (const profilePosition of [0, 500, 999]) {
      let calculatedLogit = demo.head_demos.profile_bias;
      for (let kernelPosition = 0; kernelPosition < 75; kernelPosition += 1) {
        for (let channel = 0; channel < 512; channel += 1) {
          calculatedLogit += finalTensor[channel * 1074 + profilePosition + kernelPosition]
            * demo.head_demos.profile_weights_input_channels_by_positions[channel][kernelPosition];
        }
      }
      assert.ok(Math.abs(calculatedLogit - demo.tensors.profile_logits.position_max[profilePosition]) < 2e-4);
    }
  });

  test(`${preset} exposes exact convolution, bias, ReLU, and shortcut tensors`, async () => {
    const demo = await loadDemo(preset);
    const readTensor = async name => {
      const tensor = demo.tensors[name];
      const raw = gunzipSync(await readFile(new URL(`public${tensor.full_heatmap.url}`, root)));
      const source = new Float32Array(new Uint8Array(raw).slice().buffer);
      if (!tensor.full_heatmap.display_transform) return source;
      const values = new Float32Array(source.length);
      const width = tensor.full_heatmap.width_positions;
      for (let channel = 0; channel < 512; channel += 1) {
        const bias = tensor.full_heatmap.channel_bias[channel];
        for (let position = 0; position < width; position += 1) {
          const index = channel * width + position;
          const biased = source[index] + bias;
          values[index] = tensor.full_heatmap.display_transform === "add_channel_bias_relu" ? Math.max(0, biased) : biased;
        }
      }
      return values;
    };
    for (const layer of ["stem", "res1", "res4", "res8"]) {
      const outputLength = demo.tensors[layer].full_heatmap.width_positions;
      const convolution = await readTensor(`${layer}_conv`);
      const biased = await readTensor(`${layer}_bias`);
      const relu = layer === "stem" ? await readTensor("stem") : await readTensor(`${layer}_relu`);
      for (const index of [0, 17, Math.floor(convolution.length / 2), convolution.length - 1]) {
        assert.ok(Math.abs(relu[index] - Math.max(0, biased[index])) < 1e-6, `${preset} ${layer} ReLU mismatch`);
      }
      if (layer !== "stem") {
        const output = await readTensor(layer);
        const block = Number(layer.slice(3));
        const inputLayer = block === 1 ? "stem" : `res${block - 1}`;
        const input = await readTensor(inputLayer);
        const inputLength = demo.tensors[inputLayer].full_heatmap.width_positions;
        const dilation = 2 ** block;
        for (const channel of [0, 127, 511]) {
          for (const position of [0, Math.floor(outputLength / 2), outputLength - 1]) {
            const index = channel * outputLength + position;
            const shortcut = input[channel * inputLength + position + dilation];
            assert.ok(Math.abs(output[index] - (relu[index] + shortcut)) < 2e-5, `${preset} ${layer} shortcut mismatch`);
          }
        }
      }
      assert.equal(convolution.length, biased.length);
    }
  });
}

test("the page teaches one connected flow and avoids the discarded GC overview", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.match(page, /ONE CALCULATION · FOLLOWED TOP TO BOTTOM/);
  assert.match(page, /convolution → bias → ReLU/);
  assert.match(page, /no ReLU after this addition/);
  assert.match(page, /profile probabilities/i);
  assert.match(page, /expected counts per base/i);
  assert.match(page, /The heads branch from the same final tensor/);
  assert.match(page, /Open all 512 stem filters as one 512 × 21 heatmap/);
  assert.match(page, /Computation state/);
  assert.match(page, /complete representation first/);
  assert.match(page, /Extrema color strength/);
  assert.match(page, /whole-length-frame/);
  assert.match(page, /Profile output · 512 × 1,074 → 1 × 1,000/);
  assert.match(page, /1,074 − 75 \+ 1 = 1,000/);
  assert.match(page, /THE LEARNED PROFILE KERNEL/);
  assert.match(page, /38,400 products \+ bias/);
  assert.match(page, /Profile convolution kernel heatmap, 512 channels by 75 relative positions/);
  assert.match(page, /Coral cells are positive; blue cells are negative/);
  assert.match(page, /GM21515 ATAC checkpoint/);
  assert.match(page, /nextStage === "stem" \? \(computation === "output" \? "relu" : computation\) : layer === "stem" \? "output"/);
  assert.match(page, /type="number" min="1" max="512"/);
  assert.match(page, /Heatmap/);
  assert.match(page, /Weight logo/);
  assert.match(page, /Activation motif/);
  assert.match(page, /TensorFlow Conv1D convention/);
  assert.match(page, /cross-correlation/);
  assert.match(page, /A↔T and C↔G/);
  assert.match(page, /centeredStemWeights/);
  assert.match(page, /channelOrder\.map/);
  assert.match(page, /Model audit/);
  assert.doesNotMatch(page, /GC fraction/i);
  assert.match(page, /This live cell was not selected by that rule/);
  assert.match(page, /Return to selected example/);
});

test("compressed browser JSON assets exactly reproduce canonical data", async () => {
  const mappings = [
    ["app/data/k562-peak-activations.json", "public/data/demos/k562.json.gz"],
    ["app/data/gm21515-activations.json", "public/data/demos/gm21515.json.gz"],
    ["app/data/synthetic-activations.json", "public/data/demos/synthetic.json.gz"],
    ["app/data/model-audit-summary.json", "public/data/model-audit-summary.json.gz"],
  ];
  for (const [source, compressed] of mappings) {
    const canonical = await readFile(new URL(source, root));
    const browser = gunzipSync(await readFile(new URL(compressed, root)));
    assert.deepEqual(browser, canonical, `${compressed} must be regenerated from ${source}`);
  }
});

test("the dilation evolution demo separates geometric reach from measured contribution", async () => {
  const page = await readFile(new URL("app/dilation-trace/page.tsx", root), "utf8");
  assert.match(page, /possible reach/i);
  assert.match(page, /not proof that every reachable feature affected the prediction/i);
  assert.match(page, /RECEPTIVE_FIELDS\[stage\] - 21/);
  assert.match(page, /Balanced merge · both paths active/);
  assert.match(page, /Shared profile-output-aligned centers/);
  assert.match(page, /You moved from the selected/);
  assert.match(page, /PRESERVED SHORTCUT/);
  assert.match(page, /LEARNED CORRECTION/);
  assert.match(page, /Each tap reads all 512 channels/);
  assert.match(page, /38,400 activation × weight products/);
  assert.match(page, /WHOLE TENSOR → MAGNIFIED FILMSTRIP/);
  assert.match(page, /FullTensorMagnifier/);
  assert.match(page, /12 display-adjacent, permanently identified channels/);
  assert.match(page, /Global channel order/);
});

test("balanced residual examples keep both paths active in the shared profile region", async () => {
  const demo = await loadDemo("k562-peak");
  for (const block of demo.residual_kernel_demos) {
    const balanced = block.example_modes.balanced;
    assert.ok(balanced.input_aligned_coordinate_one_based >= 558 && balanced.input_aligned_coordinate_one_based <= 1557);
    assert.ok(balanced.transformed_after_relu > 0, `block ${block.block} correction should be active`);
    assert.ok(balanced.shortcut_value > 0, `block ${block.block} shortcut should be active`);
    assert.ok(Math.abs(balanced.transformed_after_relu + balanced.shortcut_value - balanced.block_output) < 2e-5);
    assert.equal(balanced.weights_input_channels_by_taps.length, 512);
    assert.equal(balanced.weights_input_channels_by_taps[0].length, 3);
  }
});

test("the audit artifact preserves immutable channels and evidence boundaries", async () => {
  const artifact = JSON.parse(await readFile(new URL("app/data/model-audit-summary.json", root), "utf8"));
  assert.equal(artifact.schema_version, "1.0.0");
  assert.deepEqual(artifact.evidence_levels.map(level => level.id), ["descriptive", "mechanism", "biology"]);
  for (const preset of ["k562-peak", "gm21515"]) {
    const checkpoint = artifact.checkpoints[preset];
    assert.equal(checkpoint.channel_registry.length, 512);
    assert.deepEqual(checkpoint.channel_registry.map(channel => channel.id_zero_based), Array.from({ length: 512 }, (_, index) => index));
    for (const order of Object.values(checkpoint.channel_orders)) {
      assert.equal(order.length, 512);
      assert.deepEqual([...order].sort((a, b) => a - b), Array.from({ length: 512 }, (_, index) => index));
    }
    assert.equal(checkpoint.layer_similarity.values.length, 9);
    assert.equal(checkpoint.activation_motifs.status, "not_generated");
    assert.equal(checkpoint.kernels.heads.count_weights.length, 512);
    assert.ok(Number.isFinite(checkpoint.kernels.heads.count_bias));
    assert.ok(Number.isFinite(checkpoint.kernels.heads.logcount));
    assert.ok(Number.isFinite(checkpoint.kernels.heads.predicted_total_count));
    assert.ok(checkpoint.channel_registry.every(channel => Number.isFinite(channel.count_contribution)));
    for (const block of checkpoint.kernels.residual) {
      assert.ok(Math.abs(block.tap_energy_fraction.reduce((sum, value) => sum + value, 0) - 1) < 1e-9);
    }
  }
  assert.ok(Math.abs(artifact.checkpoints["k562-peak"].layers.stem.exact_zero_fraction - .9863) < .001);
  assert.ok(Math.abs(artifact.checkpoints["k562-peak"].layers.res8.exact_zero_fraction - .8844) < .001);
  assert.ok(Math.abs(artifact.checkpoints.gm21515.layers.stem.exact_zero_fraction - .9813) < .001);
  assert.ok(Math.abs(artifact.checkpoints.gm21515.layers.res8.exact_zero_fraction - .8187) < .001);
});

test("the audit page distinguishes description, mechanism, and biology", async () => {
  const page = await readFile(new URL("app/model-audit/page.tsx", root), "utf8");
  assert.match(page, /Three different strengths of claim/);
  assert.match(page, /Permanent channel registry/i);
  assert.match(page, /Do dilated kernels really mix channels/);
  assert.match(page, /128 aligned positions/);
  assert.match(page, /64 highest-variance channels/);
  assert.match(page, /Corpus-derived logos are deliberately not fabricated/);
  assert.match(page, /TF-MoDISco/);
  assert.match(page, /JASPAR/);
  assert.match(page, /Tomtom/);
  assert.match(page, /Median active \/ position/);
  assert.match(page, /Positive runs/);
  assert.match(page, /COUNT–PROFILE CHANNEL-WEIGHT CORRELATION/);
  assert.match(page, /Learned dense weights/);
  assert.match(page, /This locus: pooled activation × weight/);
  assert.match(page, /Signed profile-head kernel heatmap/);
  assert.match(page, /512 learned feature channels/);
  assert.match(page, /Collapsed positional energy/);
  assert.match(page, /coral raises log-count · blue lowers it/);
  assert.match(page, /one block = 100% weight energy/);
  assert.match(page, /Every bar uses the full 0–100% width with no magnification/);
  assert.doesNotMatch(page, /±2\.5 percentage-point scale/);
});

test("profile-head heatmaps preserve all 512 × 75 signed weights", async () => {
  for (const [preset, publicName] of [["k562-peak", "k562_peak"], ["gm21515", "gm21515"]]) {
    const demo = await loadDemo(preset);
    const compressed = await readFile(new URL(`public/data/tensors/${publicName}/profile_kernel.f32.gz`, root));
    const raw = gunzipSync(compressed);
    assert.equal(raw.byteLength, 512 * 75 * 4);
    assert.ok(Math.abs(raw.readFloatLE(0) - demo.head_demos.profile_weights_input_channels_by_positions[0][0]) < 1e-6);
    assert.ok(Math.abs(raw.readFloatLE(raw.byteLength - 4) - demo.head_demos.profile_weights_input_channels_by_positions[511][74]) < 1e-6);
  }
});

test("the Basset page keeps motif detection, pooling, channel mixing, and dense readout connected", async () => {
  const page = await readFile(new URL("app/basset/page.tsx", root), "utf8");
  assert.match(page, /Shrink the sequence, mix the motifs, classify 164 cell types/);
  assert.match(page, /no padding/i);
  assert.match(page, /MAX-POOL MICROSCOPE/);
  assert.match(page, /complete checkpoint graph/i);
  assert.match(page, /coral positive · blue negative/);
  assert.match(page, /full tensors shrink/i);
  assert.match(page, /300 channels × 11 nearby positions/i);
  assert.match(page, /3,300 learned inputs/);
  assert.match(page, /Two hidden dense layers turn spatial features into shared cell-type evidence/i);
  assert.match(page, /Dense 1 weights/);
  assert.match(page, /Dense 2 weights/);
  assert.match(page, /Output-reader weights/);
  assert.match(page, /This is technically a third linear layer, but not a hidden dense layer/);
  assert.match(page, /Where is dropout/);
  assert.match(page, /ChromBPNet.*preserve a long spatial grid/s);
  assert.match(page, /not attribution scores for the final prediction/i);

  const data = JSON.parse(await readFile(new URL("app/data/basset-demo.json", root), "utf8"));
  assert.equal(data.sequence.length, 600);
  assert.equal(data.outputs.all_labels.length, 164);
  assert.ok(data.verification.maximum_absolute_error < 1e-9);
  assert.deepEqual(data.architecture.map(stage => stage.shape), [
    [4, 600], [300, 582], [300, 194], [200, 184], [200, 46],
    [200, 40], [200, 10], [2000], [1000], [1000], [164],
  ]);
  assert.deepEqual(Object.fromEntries(Object.entries(data.dense_weight_assets).map(([name, asset]) => [name, [asset.rows, asset.columns]])), {
    dense1: [1000, 2000], dense2: [1000, 1000], output: [164, 1000],
  });
  for (const asset of Object.values(data.dense_weight_assets)) {
    const raw = gunzipSync(await readFile(new URL(`public${asset.url}`, root)));
    assert.equal(raw.byteLength, asset.rows * asset.columns * 4);
  }
  assert.equal(data.dense2_readout_example.contributions.length, 1000);
  assert.equal(data.outputs.k562_reader.contributions.length, 1000);
});
