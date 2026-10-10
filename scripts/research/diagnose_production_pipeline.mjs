import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const PREVIEW_URL = 'http://localhost:4173/';

function getSha256(filePath) {
  const buf = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(buf).digest('hex');
}

async function main() {
  console.log('='.repeat(70));
  console.log('DIAGNOSING PRODUCTION DEMO PIPELINE & RECONCILING CONFIDENCE');
  console.log('='.repeat(70));

  // 1. Hash verification
  const onnxPath = path.resolve('apps/web/public/models/mobilenet_v3_small_backbone_fp32.onnx');
  const distOnnxPath = path.resolve('apps/web/dist/models/mobilenet_v3_small_backbone_fp32.onnx');
  const bundlePath = path.resolve('packages/inference/src/research-models-bundle.json');
  
  const onnxHash = getSha256(onnxPath);
  const distOnnxHash = getSha256(distOnnxPath);
  const bundleHash = getSha256(bundlePath);

  console.log(`ONNX Public Hash : ${onnxHash}`);
  console.log(`ONNX Dist Hash   : ${distOnnxHash}`);
  console.log(`ONNX Match       : ${onnxHash === distOnnxHash ? 'YES' : 'NO'}`);
  console.log(`Bindings Bundle  : ${bundleHash}`);

  // 2. Launch headless Chrome with CDP
  const tempProfile = fs.mkdtempSync(path.join(os.tmpdir(), 'chrome_diag_'));
  const port = 9320 + Math.floor(Math.random() * 50);

  const chromeProc = spawn(CHROME_PATH, [
    '--headless=new',
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${tempProfile}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-background-networking',
    '--disable-extensions',
    PREVIEW_URL,
  ]);

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
    } catch {}
  }

  if (!targetWsUrl) {
    chromeProc.kill();
    fs.rmSync(tempProfile, { recursive: true, force: true });
    throw new Error('Chrome CDP endpoint failed to initialize');
  }

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

  async function evaluate(expression) {
    const res = await sendCommand('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    return res.result ? res.result.value : null;
  }

  await sendCommand('Page.enable');
  await sendCommand('Runtime.enable');

  console.log('\nWaiting for page load...');
  await new Promise((r) => setTimeout(r, 2000));

  // Check initial tab
  const activeTabClass = await evaluate(`document.getElementById('tab-research-lab')?.style.background`);
  console.log('Active tab state:', activeTabClass);

  // Switch to Research Lab
  console.log('Clicking Research Lab tab...');
  await evaluate(`document.getElementById('tab-research-lab')?.click()`);
  await new Promise((r) => setTimeout(r, 1000));

  // Check Disclaimer text
  const disclaimer = await evaluate(`document.getElementById('research-disclaimer-box')?.innerText`);
  console.log('\nDisclaimer box snippet:');
  console.log(disclaimer ? disclaimer.substring(0, 160) + '...' : 'NULL');

  // Evaluate Sample 0
  console.log('\n--- Evaluating Sample 0 (000000002261_authentic.png) ---');
  await evaluate(`document.getElementById('btn-sample-0')?.click()`);
  
  let sample0Result = null;
  for (let i = 0; i < 30; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const isBusy = await evaluate(`document.getElementById('btn-run-full-parity')?.disabled`);
    const elText = await evaluate(`document.getElementById('single-sample-result')?.innerText`);
    if (!isBusy && elText) {
      sample0Result = elText;
      break;
    }
  }

  console.log('Sample 0 DOM text:');
  console.log(sample0Result);

  // Evaluate Sample 1
  console.log('\n--- Evaluating Sample 1 (000000002261_ai_edited.png) ---');
  await evaluate(`document.getElementById('btn-sample-1')?.click()`);

  let sample1Result = null;
  for (let i = 0; i < 30; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const isBusy = await evaluate(`document.getElementById('btn-run-full-parity')?.disabled`);
    const elText = await evaluate(`document.getElementById('single-sample-result')?.innerText`);
    if (!isBusy && elText && elText.includes('AI EDITED')) {
      sample1Result = elText;
      break;
    }
  }

  console.log('Sample 1 DOM text:');
  console.log(sample1Result);

  // Save diagnostic output
  const diagReceipt = {
    timestamp: new Date().toISOString(),
    onnxHash,
    distOnnxHash,
    bundleHash,
    sample0_dom: sample0Result,
    sample1_dom: sample1Result,
  };

  const outPath = path.resolve('research/evidence/browser_fp32_parity/diagnose_pipeline_output.json');
  fs.writeFileSync(outPath, JSON.stringify(diagReceipt, null, 2), 'utf8');
  console.log(`\nDiagnostic output saved to: ${outPath}`);

  ws.close();
  chromeProc.kill();
  try {
    fs.rmSync(tempProfile, { recursive: true, force: true });
  } catch {}

  console.log('='.repeat(70));
}

main().catch((err) => {
  console.error('Diagnostic run failed:', err);
  process.exit(1);
});
