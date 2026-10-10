import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const SESSION_NAME = process.argv[2] || 'session_2_isolated_instance';

async function runSession() {
  const tempProfile = fs.mkdtempSync(path.join(os.tmpdir(), 'chrome_cdp_'));
  const port = 9220 + Math.floor(Math.random() * 50);

  console.log(`Starting Chrome session [${SESSION_NAME}] on port ${port}...`);
  const chromeProc = spawn(CHROME_PATH, [
    '--headless=new',
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${tempProfile}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-background-networking',
    '--disable-extensions',
    'http://localhost:5173/',
  ]);

  chromeProc.stderr.on('data', () => {});
  chromeProc.stdout.on('data', () => {});

  // Wait for Chrome CDP endpoint to be ready
  let targetWsUrl = null;
  for (let attempt = 0; attempt < 30; attempt++) {
    await new Promise((r) => setTimeout(r, 500));
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/list`);
      if (res.ok) {
        const list = await res.json();
        const page = list.find((item) => item.type === 'page');
        if (page && page.webSocketDebuggerUrl) {
          targetWsUrl = page.webSocketDebuggerUrl;
          break;
        }
      }
    } catch {
      // waiting
    }
  }

  if (!targetWsUrl) {
    chromeProc.kill();
    fs.rmSync(tempProfile, { recursive: true, force: true });
    throw new Error('Chrome CDP endpoint did not become ready');
  }

  console.log(`Connected to Chrome CDP: ${targetWsUrl}`);
  const ws = new WebSocket(targetWsUrl);

  let msgId = 1;
  const pendingRequests = new Map();

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.id && pendingRequests.has(data.id)) {
      const { resolve, reject } = pendingRequests.get(data.id);
      pendingRequests.delete(data.id);
      if (data.error) reject(new Error(JSON.stringify(data.error)));
      else resolve(data.result);
    }
  };

  await new Promise((resolve) => ws.onopen = resolve);

  function sendCommand(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = msgId++;
      pendingRequests.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async function evaluate(expression, awaitPromise = false) {
    const res = await sendCommand('Runtime.evaluate', {
      expression,
      awaitPromise,
      returnByValue: true,
    });
    return res.result ? res.result.value : null;
  }

  await sendCommand('Page.enable');
  await sendCommand('Runtime.enable');

  console.log('Waiting for page to initialize...');
  await new Promise((r) => setTimeout(r, 2000));

  // 1. Switch to Research Lab tab
  console.log('Switching to Research Lab tab...');
  await evaluate(`document.getElementById('tab-research-lab')?.click()`);
  await new Promise((r) => setTimeout(r, 1000));

  // 2. Click run full parity button
  console.log('Clicking run full parity button...');
  await evaluate(`document.getElementById('btn-run-full-parity')?.click()`);

  // 3. Poll for completion
  console.log('Waiting for Web Worker analysis on 16 samples...');
  let completed = false;
  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const btnText = await evaluate(`document.getElementById('btn-run-full-parity')?.innerText`);
    const status = await evaluate(`window.__researchParityReceipt?.summary?.status`);
    if (status === 'BROWSER_6LAYER_PARITY_COMPLETE' && btnText?.includes('Chạy Toàn Bộ')) {
      completed = true;
      break;
    }
  }

  if (!completed) {
    ws.close();
    chromeProc.kill();
    fs.rmSync(tempProfile, { recursive: true, force: true });
    throw new Error('Timeout waiting for 16-sample analysis to complete');
  }

  console.log('Analysis complete! Extracting receipt...');
  const receipt = await evaluate(`window.__researchParityReceipt`);

  // Save session payload
  const sessionData = {
    session_id: SESSION_NAME,
    cold_start_initialization_ms: receipt.summary.cold_start_initialization_ms,
    num_samples: receipt.samples.length,
    raw_timings_ms: {
      preprocessing: receipt.samples.map((s) => s.timing.preprocessing),
      dsp_extraction: receipt.samples.map((s) => s.timing.dsp),
      backbone_wasm: receipt.samples.map((s) => s.timing.backbone),
      outer_fold_scoring: receipt.samples.map((s) => s.timing.scoring),
      total_warm_pipeline: receipt.samples.map((s) => s.timing.total),
    },
    warm_statistics_ms: {
      mean: receipt.summary.latency_total.mean_ms,
      median_p50: receipt.summary.latency_total.p50_ms,
      p95: receipt.summary.latency_total.p95_ms,
      min: Math.min(...receipt.samples.map((s) => s.timing.total)),
      max: Math.max(...receipt.samples.map((s) => s.timing.total)),
    },
    decisions_matched: receipt.summary.decisions_matched_count,
    decisions_total: receipt.summary.total_predictions_evaluated,
  };

  const outDir = path.resolve(import.meta.dirname, '../../research/evidence/browser_fp32_parity');
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(
    path.join(outDir, `${SESSION_NAME}.json`),
    JSON.stringify(sessionData, null, 2),
    'utf8'
  );

  console.log(`Saved ${SESSION_NAME}.json successfully!`);
  console.log(`Cold Start: ${sessionData.cold_start_initialization_ms} ms`);
  console.log(`Mean Latency: ${sessionData.warm_statistics_ms.mean} ms, P50: ${sessionData.warm_statistics_ms.median_p50} ms, P95: ${sessionData.warm_statistics_ms.p95} ms`);
  console.log(`Decisions Matched: ${sessionData.decisions_matched} / ${sessionData.decisions_total}`);

  ws.close();
  chromeProc.kill();
  try {
    fs.rmSync(tempProfile, { recursive: true, force: true });
  } catch {}
}

runSession().catch((err) => {
  console.error('Session failed:', err);
  process.exit(1);
});
