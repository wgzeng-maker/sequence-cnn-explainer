/** Load a little-endian float32 tensor from a static .gz asset.
 * Some servers transparently decode gzip; others return the gzip bytes.
 */
export async function loadFloat32Tensor(url: string, expectedValues: number, signal?: AbortSignal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Tensor request failed (${response.status}) for ${url}`);

  let buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const isGzip = bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;

  if (isGzip) {
    if (typeof DecompressionStream === "undefined") {
      throw new Error("This browser cannot decode the compressed tensor data.");
    }
    buffer = await new Response(
      new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip")),
    ).arrayBuffer();
  }

  const expectedBytes = expectedValues * Float32Array.BYTES_PER_ELEMENT;
  if (buffer.byteLength !== expectedBytes) {
    throw new Error(`Tensor data has ${buffer.byteLength.toLocaleString()} bytes; expected ${expectedBytes.toLocaleString()}.`);
  }
  return new Float32Array(buffer);
}
