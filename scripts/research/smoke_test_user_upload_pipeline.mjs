import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '../..');

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';

// Parse command line arguments
let targetUrl = 'http://localhost:4173/forensics-web-lab/';
for (const arg of process.argv.slice(2)) {
  if (arg.startsWith('--url=')) {
    targetUrl = arg.slice(6);
  }
}

const sample0Path = path.resolve(repoRoot, 'research/reference_samples/000000002261_authentic.png');
const sample1Path = path.resolve(repoRoot, 'research/reference_samples/000000002261_ai_edited.png');

if (!fs.existsSync(sample0Path) || !fs.existsSync(sample1Path)) {
  console.error('ERROR: Reference samples not found at:', sample0Path, sample1Path);
  process.exit(1);
}

const RECEIPT_PATH = path.resolve(
  repoRoot,
  'research/evidence/browser_fp32_parity/production_upload_smoke_receipt.json'
);

async function runUploadPipelineSmokeTest() {
  console.log('='.repeat(70));
  console.log(`Running User Upload FP32 Smoke Test on: ${targetUrl}`);
  console.log('='.repeat(70));

  const tempProfile = fs.mkdtempSync(path.join(os.tmpdir(), 'chrome_upload_smoke_'));
  const port = 9250 + Math.floor(Math.random() * 40);

  console.log(`Starting headless Chrome on port ${port}...`);
  const chromeProc = spawn(CHROME_PATH, [
    '--headless=new',
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${tempProfile}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-background-networking',
    '--disable-extensions',
    targetUrl,
  ]);

  chromeProc.stderr.on('data', () => {});
  chromeProc.stdout.on('data', () => {});

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

  console.log(`Connected to Chrome CDP: ${targetWsUrl}`);
  const ws = new WebSocket(targetWsUrl);

  let msgId = 1;
  const pendingRequests = new Map();
  const networkRequests = [];
  const consoleLogs = [];

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.id && pendingRequests.has(data.id)) {
      const { resolve, reject } = pendingRequests.get(data.id);
      pendingRequests.delete(data.id);
      if (data.error) reject(new Error(JSON.stringify(data.error)));
      else resolve(data.result);
    } else if (data.method === 'Network.requestWillBeSent') {
      const reqUrl = data.params.request.url;
      networkRequests.push({
        url: reqUrl,
        method: data.params.request.method,
        type: data.params.type,
      });
    } else if (data.method === 'Runtime.consoleAPICalled') {
      consoleLogs.push({
        type: data.params.type,
        text: data.params.args.map((a) => a.value || a.description).join(' '),
      });
    }
  };

  await new Promise((resolve) => (ws.onopen = resolve));

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
  await sendCommand('Network.enable');
  await sendCommand('DOM.enable');

  console.log('1. Waiting for Application to mount...');
  for (let wait = 0; wait < 10; wait++) {
    await new Promise((r) => setTimeout(r, 500));
    const mounted = await evaluate(`Boolean(document.getElementById('scientific-limitations-banner'))`);
    if (mounted) break;
  }

  const pageTitle = await evaluate('document.title');
  console.log(`   Page title: "${pageTitle}"`);

  const rootHtml = await evaluate(`document.getElementById('root')?.innerHTML`);
  if (!rootHtml || rootHtml.trim() === '') {
    console.error('Console logs recorded:', consoleLogs);
    throw new Error(`React app failed to mount into #root. Console errors: ${JSON.stringify(consoleLogs)}`);
  }

  // Step 2: Verify Scientific Limitations Disclaimer
  console.log('2. Verifying Scientific Limitations & Forensic Boundaries Notice...');
  const disclaimerText = await evaluate(
    `document.getElementById('scientific-limitations-banner')?.innerText`
  );
  if (!disclaimerText) {
    console.error('Console logs:', consoleLogs);
    console.error('Root HTML:', rootHtml);
    throw new Error('Scientific limitations banner not found in DOM!');
  }

  const hasResearchNotice = disclaimerText.includes('nghiên cứu');
  const hasBinaryNotice = disclaimerText.includes('nhị phân') || disclaimerText.includes('authentic') || disclaimerText.includes('ai_edited');
  const hasNotAbsoluteReal = disclaimerText.includes('không chứng minh ảnh thật') || disclaimerText.includes('không xác nhận ảnh thật');
  const hasFullyGenBoundary = disclaimerText.includes('fully_generated') || disclaimerText.includes('toàn phần');

  console.log(`   - Research notice: ${hasResearchNotice ? 'PASS' : 'FAIL'}`);
  console.log(`   - Binary 2-class scope notice: ${hasBinaryNotice ? 'PASS' : 'FAIL'}`);
  console.log(`   - Authentic != absolute real notice: ${hasNotAbsoluteReal ? 'PASS' : 'FAIL'}`);
  console.log(`   - Fully-generated boundary notice: ${hasFullyGenBoundary ? 'PASS' : 'FAIL'}`);

  if (!hasResearchNotice || !hasBinaryNotice || !hasNotAbsoluteReal || !hasFullyGenBoundary) {
    throw new Error('Disclaimer banner does not meet all 4 mandatory criteria');
  }

  // Step 3: Upload Sample 0 (Authentic reference) via user file input
  console.log('3. Uploading Sample 0 (000000002261_authentic.png) via User File Input...');
  let doc = await sendCommand('DOM.getDocument');
  let fileInputNode = await sendCommand('DOM.querySelector', {
    nodeId: doc.root.nodeId,
    selector: 'input[type="file"]',
  });
  if (!fileInputNode.nodeId) {
    throw new Error('File input element not found in DOM');
  }

  await sendCommand('DOM.setFileInputFiles', {
    nodeId: fileInputNode.nodeId,
    files: [sample0Path],
  });

  console.log('   File attached. Waiting for FP32 ONNX pipeline & 5 folds execution...');
  let sample0Done = false;
  let sample0Verdict = null;
  let sample0VisProb = null;
  let sample0FusProb = null;
  let sample0Timing = null;

  for (let i = 0; i < 40; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const badge = await evaluate(`document.getElementById('binary-verdict-badge')?.innerText`);
    const visVal = await evaluate(`document.getElementById('prob-visual-value')?.innerText`);
    const fusVal = await evaluate(`document.getElementById('prob-fusion-value')?.innerText`);
    const timing = await evaluate(`document.getElementById('timing-total')?.innerText`);
    const hasError = await evaluate(`Boolean(document.querySelector('.glass-panel[style*="f43f5e"]'))`);

    if (hasError) {
      const errText = await evaluate(`document.querySelector('.glass-panel[style*="f43f5e"]')?.innerText`);
      throw new Error(`Inference error displayed in UI: ${errText}`);
    }

    if (badge && visVal && fusVal) {
      sample0Done = true;
      sample0Verdict = badge.trim();
      sample0VisProb = parseFloat(visVal.replace('%', '')) / 100;
      sample0FusProb = parseFloat(fusVal.replace('%', '')) / 100;
      sample0Timing = timing;
      console.log(`   Sample 0 Completed!`);
      console.log(`   - Verdict Badge: "${sample0Verdict}"`);
      console.log(`   - Visual Calibrated: ${(sample0VisProb * 100).toFixed(2)}%`);
      console.log(`   - Late Fusion DSP: ${(sample0FusProb * 100).toFixed(2)}%`);
      console.log(`   - Total Latency: ${sample0Timing}`);
      break;
    }
  }

  if (!sample0Done) {
    throw new Error('Timed out waiting for Sample 0 inference');
  }

  // Verify Sample 0 against locked reference
  // Reference: Visual = 0.4811, Fusion = 0.4908, Verdict = AUTHENTIC
  const sample0VerdictPass = sample0Verdict.includes('AUTHENTIC');
  const sample0VisPass = Math.abs(sample0VisProb - 0.4811) <= 0.005;
  const sample0FusPass = Math.abs(sample0FusProb - 0.4908) <= 0.005;

  console.log(`   Sample 0 Verdict Check (expected AUTHENTIC): ${sample0VerdictPass ? 'PASS' : 'FAIL'}`);
  console.log(`   Sample 0 Visual Prob Check (expected ~48.11%): ${sample0VisPass ? 'PASS' : 'FAIL'} (got ${(sample0VisProb * 100).toFixed(2)}%)`);
  console.log(`   Sample 0 Fusion Prob Check (expected ~49.08%): ${sample0FusPass ? 'PASS' : 'FAIL'} (got ${(sample0FusProb * 100).toFixed(2)}%)`);

  if (!sample0VerdictPass || !sample0VisPass || !sample0FusPass) {
    throw new Error('Sample 0 outputs deviate from locked reference tolerances!');
  }

  // Step 4: Test Session Reset / New Image
  console.log('4. Testing Session Reset ("Tải Ảnh Khác")...');
  await evaluate(`document.getElementById('btn-upload-new-image')?.click()`);
  await new Promise((r) => setTimeout(r, 1000));

  const isResetClean = await evaluate(`!document.getElementById('binary-verdict-badge') && Boolean(document.querySelector('input[type="file"]'))`);
  console.log(`   Reset state clean (Verdict dismissed, Dropzone restored): ${isResetClean ? 'PASS' : 'FAIL'}`);
  if (!isResetClean) {
    throw new Error('Failed to cleanly reset session state');
  }

  // Step 5: Upload Sample 1 (AI Edited reference) via user file input
  console.log('5. Uploading Sample 1 (000000002261_ai_edited.png) via User File Input...');
  doc = await sendCommand('DOM.getDocument');
  fileInputNode = await sendCommand('DOM.querySelector', {
    nodeId: doc.root.nodeId,
    selector: 'input[type="file"]',
  });
  if (!fileInputNode.nodeId) {
    throw new Error('File input element not found after reset');
  }

  await sendCommand('DOM.setFileInputFiles', {
    nodeId: fileInputNode.nodeId,
    files: [sample1Path],
  });

  console.log('   File attached. Waiting for FP32 ONNX pipeline execution on Sample 1...');
  let sample1Done = false;
  let sample1Verdict = null;
  let sample1VisProb = null;
  let sample1FusProb = null;
  let sample1Timing = null;

  for (let i = 0; i < 40; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const badge = await evaluate(`document.getElementById('binary-verdict-badge')?.innerText`);
    const visVal = await evaluate(`document.getElementById('prob-visual-value')?.innerText`);
    const fusVal = await evaluate(`document.getElementById('prob-fusion-value')?.innerText`);
    const timing = await evaluate(`document.getElementById('timing-total')?.innerText`);

    if (badge && visVal && fusVal) {
      sample1Done = true;
      sample1Verdict = badge.trim();
      sample1VisProb = parseFloat(visVal.replace('%', '')) / 100;
      sample1FusProb = parseFloat(fusVal.replace('%', '')) / 100;
      sample1Timing = timing;
      console.log(`   Sample 1 Completed!`);
      console.log(`   - Verdict Badge: "${sample1Verdict}"`);
      console.log(`   - Visual Calibrated: ${(sample1VisProb * 100).toFixed(2)}%`);
      console.log(`   - Late Fusion DSP: ${(sample1FusProb * 100).toFixed(2)}%`);
      console.log(`   - Total Latency: ${sample1Timing}`);
      break;
    }
  }

  if (!sample1Done) {
    throw new Error('Timed out waiting for Sample 1 inference');
  }

  // Verify Sample 1 against locked reference
  // Reference: Visual = 0.4941, Fusion = 0.5034, Verdict = AI EDITED
  const sample1VerdictPass = sample1Verdict.includes('AI EDITED');
  const sample1VisPass = Math.abs(sample1VisProb - 0.4941) <= 0.005;
  const sample1FusPass = Math.abs(sample1FusProb - 0.5034) <= 0.005;

  console.log(`   Sample 1 Verdict Check (expected AI EDITED): ${sample1VerdictPass ? 'PASS' : 'FAIL'}`);
  console.log(`   Sample 1 Visual Prob Check (expected ~49.41%): ${sample1VisPass ? 'PASS' : 'FAIL'} (got ${(sample1VisProb * 100).toFixed(2)}%)`);
  console.log(`   Sample 1 Fusion Prob Check (expected ~50.34%): ${sample1FusPass ? 'PASS' : 'FAIL'} (got ${(sample1FusProb * 100).toFixed(2)}%)`);

  if (!sample1VerdictPass || !sample1VisPass || !sample1FusPass) {
    throw new Error('Sample 1 outputs deviate from locked reference tolerances!');
  }

  // Step 6: Network Egress Audit
  console.log('6. Auditing Network Traffic for Zero Image/Pixel Data Egress...');
  const externalDataRequests = networkRequests.filter((req) => {
    try {
      if (req.url.startsWith('blob:') || req.url.startsWith('data:')) {
        return false; // Local memory blob/data URL
      }
      const u = new URL(req.url);
      const host = u.hostname;
      // Allow localhost, 127.0.0.1, nomozer.github.io, and Google fonts
      if (
        host === 'localhost' ||
        host === '127.0.0.1' ||
        host === 'nomozer.github.io' ||
        host.includes('fonts.googleapis.com') ||
        host.includes('fonts.gstatic.com')
      ) {
        return false;
      }
      return true;
    } catch {
      return false;
    }
  });

  console.log(`   Total Network Requests recorded: ${networkRequests.length}`);
  console.log(`   External data/server egress requests: ${externalDataRequests.length}`);
  const zeroEgressPass = externalDataRequests.length === 0;

  console.log(`   Zero Data Egress Status: ${zeroEgressPass ? 'PASS' : 'FAIL'}`);
  if (!zeroEgressPass) {
    console.error('External requests detected:', externalDataRequests);
    throw new Error('Network egress violation: external requests detected');
  }

  // Save audit receipt
  const receipt = {
    schema_version: '1.0.0',
    timestamp: new Date().toISOString(),
    target_url: targetUrl,
    environment: 'Chromium Headless via CDP',
    pipeline_type: 'FP32 Binary Forensic Detector (MobileNetV3 + 16-D DSP + 5 Folds)',
    results: {
      scientific_notice_verified: true,
      session_reset_verified: true,
      sample_0_authentic: {
        filename: '000000002261_authentic.png',
        verdict: sample0Verdict,
        visual_calibrated_probability: sample0VisProb,
        late_fusion_dsp_probability: sample0FusProb,
        latency_str: sample0Timing,
        expected_verdict: 'AUTHENTIC',
        tolerance_pass: sample0VerdictPass && sample0VisPass && sample0FusPass,
      },
      sample_1_ai_edited: {
        filename: '000000002261_ai_edited.png',
        verdict: sample1Verdict,
        visual_calibrated_probability: sample1VisProb,
        late_fusion_dsp_probability: sample1FusProb,
        latency_str: sample1Timing,
        expected_verdict: 'AI EDITED',
        tolerance_pass: sample1VerdictPass && sample1VisPass && sample1FusPass,
      },
      network_egress_audit: {
        total_requests: networkRequests.length,
        external_data_requests_count: externalDataRequests.length,
        zero_data_egress: zeroEgressPass,
      },
    },
    overall_status: 'PASS',
  };

  fs.mkdirSync(path.dirname(RECEIPT_PATH), { recursive: true });
  fs.writeFileSync(RECEIPT_PATH, JSON.stringify(receipt, null, 2), 'utf8');
  console.log(`Receipt successfully saved to: ${RECEIPT_PATH}`);

  ws.close();
  chromeProc.kill();
  await new Promise((r) => setTimeout(r, 500));
  try {
    fs.rmSync(tempProfile, { recursive: true, force: true });
  } catch {}

  console.log('='.repeat(70));
  console.log('PRODUCTION USER UPLOAD SMOKE TEST: ALL CHECKS PASSED');
  console.log('='.repeat(70));
}

runUploadPipelineSmokeTest().catch((err) => {
  console.error('TEST FAILED:', err);
  process.exit(1);
});
