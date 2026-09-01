import { NextRequest, NextResponse } from "next/server";

const storeUrl = process.env.UPSTASH_REDIS_REST_URL || process.env.KV_REST_API_URL || "";
const storeToken = process.env.UPSTASH_REDIS_REST_TOKEN || process.env.KV_REST_API_TOKEN || "";
const stateKey = "athena-tennis:agent-state";

async function redis(command: unknown[]) {
  const response = await fetch(storeUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${storeToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(command),
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`Store returned ${response.status}`);
  return response.json();
}

export async function POST(request: NextRequest) {
  if (!storeUrl || !storeToken) {
    return NextResponse.json({ error: "No store configured" }, { status: 503 });
  }
  const agentKey = process.env.AGENT_KEY || "";
  if (!agentKey || request.headers.get("x-agent-key") !== agentKey) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const body = await request.text();
  if (body.length > 500_000) {
    return NextResponse.json({ error: "Payload too large" }, { status: 413 });
  }
  await redis(["SET", stateKey, body]);
  return NextResponse.json({ ok: true });
}

export async function GET() {
  if (!storeUrl || !storeToken) {
    return NextResponse.json({ error: "No store configured" }, { status: 503 });
  }
  const data = await redis(["GET", stateKey]);
  if (!data?.result) {
    return NextResponse.json({ error: "No session yet" }, { status: 404 });
  }
  return new NextResponse(data.result, {
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
