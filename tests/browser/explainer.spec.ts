import { expect, test } from "@playwright/test";

test("checkpoint-specific correlation language does not overclaim", async ({ page }) => {
  await page.goto("/model-audit");
  await page.waitForFunction(() => document.querySelector<HTMLCanvasElement>('canvas[aria-label^="Signed profile-head"]')?.width === 75);
  const checkpoint = page.getByTestId("checkpoint-select");
  await expect(checkpoint).toHaveValue("k562-peak");
  await checkpoint.selectOption("gm21515");
  await expect(checkpoint).toHaveValue("gm21515");
  const panel = page.getByText("COUNT–PROFILE CHANNEL-WEIGHT CORRELATION").locator("..").locator("..");
  await expect(panel).toContainText("−0.330");
  await expect(panel).toContainText("moderate negative");
  await expect(panel).toContainText("checkpoints and folds");
  await expect(panel).not.toContainText("very close to zero");
});

test("dilation lab opens on a balanced, non-degenerate merge", async ({ page }) => {
  await page.goto("/dilation-trace");
  const flow = page.getByTestId("residual-numeric-flow");
  await expect(flow).toBeVisible({ timeout: 30_000 });
  await expect(page.getByLabel("Example selection rule")).toHaveValue("balanced");
  const correction = Number(await flow.getByTestId("correction-value").textContent());
  const shortcut = Number(await flow.getByTestId("shortcut-value").textContent());
  expect(correction).toBeGreaterThan(0);
  expect(shortcut).toBeGreaterThan(0);
  await expect(page.getByLabel("Tracked input coordinate")).toHaveValue("948");
});

test("residual selection modes remain inside the shared profile region", async ({ page }) => {
  await page.goto("/dilation-trace");
  for (const mode of ["balanced", "correction", "output"]) {
    await page.getByLabel("Example selection rule").selectOption(mode);
    await page.getByRole("button", { name: /Show selected example/i }).click();
    const coordinate = Number(await page.getByLabel("Tracked input coordinate").inputValue());
    expect(coordinate).toBeGreaterThanOrEqual(558);
    expect(coordinate).toBeLessThanOrEqual(1557);
  }
});

test("softmax story explains shifted logits without changing rank", async ({ page }) => {
  await page.goto("/");
  const outputs = page.locator("#outputs");
  await expect(outputs).toContainText("logit − minimum logit");
  await expect(outputs).toContainText("Softmax preserves ordering");
  await expect(outputs).toContainText("same peak position");
});

test("main residual lesson defaults to a balanced two-path merge", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => !document.body.innerText.includes("Loading verified checkpoint data"));
  await expect(page.getByLabel("Teaching example")).toHaveValue("balanced");
  const merge = page.locator("#residual .merge-lane");
  const values = await merge.locator("b").allTextContents();
  expect(Number(values[0])).toBeGreaterThan(0);
  expect(Number(values[1])).toBeGreaterThan(0);
  await page.locator("#residual input[type=range]").fill("300");
  await expect(page.locator("#residual .residual-example-controls")).toContainText("This live cell was not selected by that rule");
  await page.getByRole("button", { name: "Return to selected example" }).click();
  await expect(page.locator("#residual .residual-example-controls")).toContainText("maximize min(ReLU correction, shortcut)");
});

test("tensor inspector discloses and switches its magnitude scale", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => document.querySelector<HTMLCanvasElement>("#tensor-inspector canvas")?.width === 2094);
  const scale = page.getByLabel("Magnitude scale");
  await expect(scale).toHaveValue("layer");
  await expect(page.locator("#tensor-inspector")).toContainText("Color is normalized within this layer");
  await scale.selectOption("shared");
  await expect(scale).toHaveValue("shared");
  await expect(page.locator("#tensor-inspector")).toContainText("All backbone layers share");
});

test("Basset readout keeps the narrated K562 target visible", async ({ page }) => {
  await page.goto("/basset");
  const outputs = page.getByText("Highest predicted accessibility probabilities").locator("..").locator("..");
  await expect(outputs).toContainText("K562");
  await expect(outputs).toContainText("rank 13");
  await expect(page.getByText("before batch norm", { exact: true }).first()).toBeVisible();
});
