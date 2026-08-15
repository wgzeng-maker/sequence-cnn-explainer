import { env } from "cloudflare:workers";

const COUNTER_KEY = "visualizer_visits";
const INITIAL_VISIBLE_COUNT = 45;

async function ensureCounterTable() {
  await env.DB.prepare(`
    CREATE TABLE IF NOT EXISTS site_counters (
      key TEXT PRIMARY KEY NOT NULL,
      value INTEGER NOT NULL,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
  `).run();
}

function json(visits: number, status = 200) {
  return Response.json(
    { visits },
    { status, headers: { "Cache-Control": "no-store" } },
  );
}

export async function GET() {
  await ensureCounterTable();
  const row = await env.DB.prepare(
    "SELECT value FROM site_counters WHERE key = ?",
  ).bind(COUNTER_KEY).first<{ value: number }>();
  return json(row?.value ?? INITIAL_VISIBLE_COUNT);
}

export async function POST(request: Request) {
  const requestOrigin = request.headers.get("Origin");
  if (requestOrigin && requestOrigin !== new URL(request.url).origin) {
    return Response.json({ error: "Cross-origin counting is not allowed." }, { status: 403 });
  }

  await ensureCounterTable();
  const row = await env.DB.prepare(`
    INSERT INTO site_counters (key, value, updated_at)
    VALUES (?, 45, CURRENT_TIMESTAMP)
    ON CONFLICT(key) DO UPDATE SET
      value = site_counters.value + 1,
      updated_at = CURRENT_TIMESTAMP
    RETURNING value
  `).bind(COUNTER_KEY).first<{ value: number }>();

  return json(row?.value ?? INITIAL_VISIBLE_COUNT);
}
