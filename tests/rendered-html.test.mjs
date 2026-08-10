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
  assert.doesNotMatch(page, /GC fraction/i);
});
