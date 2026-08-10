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

for (const preset of ["k562-peak", "synthetic"]) {
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
  assert.match(page, /type="number" min="1" max="512"/);
  assert.doesNotMatch(page, /GC fraction/i);
});

test("the dilation evolution demo separates geometric reach from measured contribution", async () => {
  const page = await readFile(new URL("app/dilation-trace/page.tsx", root), "utf8");
  assert.match(page, /possible reach through the stacked kernels/i);
  assert.match(page, /does not claim that every reachable feature affected the prediction/i);
  assert.match(page, /PRESERVED SHORTCUT/);
  assert.match(page, /LEARNED CORRECTION/);
  assert.match(page, /Each tap reads all 512 channels/);
  assert.match(page, /38,400 activation × weight products/);
});
