/* HexaDepth workspace — focused, no frontend build dependencies. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const state = { mode: 'lossless', samples: [], file: null, result: null, link: null, busy: false };
  const el = {
    sample: $('sampleSelect'), file: $('fileInput'), uploadName: $('uploadName'), process: $('processBtn'), processText: $('processText'),
    transmit: $('transmitBtn'), bitrate: $('bitrate'), channels: $('channels'), corrupt: $('corrupt'), notice: $('notice'),
    inputImage: $('inputImage'), inputPlaceholder: $('inputPlaceholder'), inputSize: $('inputSize'), inputNote: $('inputNote'),
    outputImage: $('outputImage'), outputPlaceholder: $('outputPlaceholder'), outputSize: $('outputSize'), outputNote: $('outputNote'),
    outputCaption: $('outputCaption'), polygon: $('polygon'), objectCanvas: $('objectCanvas'), rawMetric: $('rawMetric'),
    payloadMetric: $('payloadMetric'), reductionMetric: $('reductionMetric'), deliveryMetric: $('deliveryMetric'),
    integrity: $('integrityBadge'), laneList: $('laneList'), linkSummary: $('linkSummary'), packetHex: $('packetHex'),
    copy: $('copyBtn'), packetTitle: $('packetTitle'), download: $('downloadOutput'), methodNote: $('methodNote'), modeDescription: $('modeDescription')
  };
  const sizes = (bytes) => bytes == null ? '—' : bytes >= 1_000_000 ? `${(bytes / 1_000_000).toFixed(2)} MB` : bytes >= 1_000 ? `${(bytes / 1_000).toFixed(1)} kB` : `${bytes} bytes`;
  const secs = (seconds) => seconds < 60 ? `${seconds.toFixed(2)} s` : seconds < 3600 ? `${(seconds / 60).toFixed(1)} min` : `${(seconds / 3600).toFixed(1)} h`;
  const setNotice = (text, type = 'neutral') => { el.notice.className = `notice notice-${type}`; el.notice.textContent = text; };
  const setIntegrity = (text, type = 'neutral') => { el.integrity.className = `integrity ${type}`; el.integrity.textContent = text; };
  const urlForSample = () => state.samples.find(s => s.name === el.sample.value)?.url;

  function clearOutput() {
    state.result = null; state.link = null;
    el.outputImage.hidden = true; el.objectCanvas.hidden = true; el.outputPlaceholder.hidden = false;
    el.outputCaption.textContent = 'Awaiting transmission'; el.outputNote.textContent = 'Lossless mode transfers the entire original file, packet by packet.';
    el.outputSize.textContent = state.mode === 'lossless' ? 'Bit-exact file' : (state.mode === 'visual64' ? '64 bits' : '64 bytes');
    el.rawMetric.textContent = '—'; el.payloadMetric.textContent = state.mode === 'lossless' ? '—' : (state.mode === 'visual64' ? '8 bytes' : '64 bytes');
    el.reductionMetric.textContent = '—'; el.deliveryMetric.textContent = '—';
    el.laneList.replaceChildren();
    const p = document.createElement('p'); p.className = 'lane-placeholder'; p.textContent = 'Channel information appears after encoding.'; el.laneList.append(p);
    el.linkSummary.textContent = 'Each fragment adds 5 framing bytes for sequence information and CRC-16. The acoustic modem itself is not simulated.';
    el.packetHex.textContent = '—'; el.copy.disabled = true; el.transmit.disabled = true; el.download.hidden = true; el.download.removeAttribute('href');
    setIntegrity('Not transmitted');
  }
  function sourceChanged() {
    clearOutput();
    if (state.file) {
      el.inputImage.src = URL.createObjectURL(state.file);
      el.inputNote.textContent = `${state.file.name} · Uploaded image, not a model-training example.`;
      el.uploadName.textContent = state.file.name;
    } else {
      el.uploadName.textContent = '';
      const sample = state.samples.find(s => s.name === el.sample.value);
      if (sample) {
        el.inputImage.src = sample.url;
        el.inputNote.textContent = sample.group === 'real' ? 'Real MILCO sonar frame · Santos et al. 2024 (CC BY 4.0)' : `Synthetic sample: ${sample.short}`;
      } else { el.inputImage.hidden = true; el.inputPlaceholder.hidden = false; return; }
    }
    el.inputImage.hidden = false; el.inputPlaceholder.hidden = true; el.inputSize.textContent = 'Source image';
    setNotice('Source ready. Choose a mode, then send the image through the simulated links.', 'neutral');
  }
  function setMode(mode) {
    state.mode = mode;
    document.querySelectorAll('[data-mode]').forEach(btn => { const chosen = btn.dataset.mode === mode; btn.classList.toggle('is-selected', chosen); btn.setAttribute('aria-pressed', String(chosen)); });
    el.modeDescription.textContent = mode === 'lossless'
      ? 'Exact-byte transfer of the entire image file. Compress if possible, split into 64-byte frames, reassemble and verify SHA-256. Model training is not required.'
      : mode === 'visual64'
      ? 'An 8×8 monochrome sketch from any sonar image. The receiver sees an approximation, not the original picture.'
      : 'A 64-byte shape and position message from the detector. Unseen objects may be missed or mislabeled.';
    el.processText.textContent = mode === 'lossless' ? 'Send image' : mode === 'visual64' ? 'Create sketch' : 'Detect and encode';
    el.methodNote.textContent = mode === 'lossless'
      ? 'Compressed source bytes are split into fixed 64-byte packets (46-byte data payload + 18-byte header/padding/CRC). Reassembly and SHA-256 verify exact file identity. Parallel channels are hypothetical independent acoustic links; not yet connected to hardware.'
      : mode === 'visual64'
      ? 'The image is intentionally reduced to an 8×8 binary intensity grid; all other pixels are lost.'
      : 'The model-derived polygon and class estimate can be wrong, especially when the sonar differs from the training dataset.';
    el.packetTitle.textContent = mode === 'lossless' ? 'First 64-byte frame' : 'Encoded payload';
    document.querySelector('.output-well').classList.toggle('lossless', mode === 'lossless');
    clearOutput();
    setNotice(mode === 'lossless' ? 'Lossless mode needs no trained detector; transmission time depends on file size and available bandwidth.' : mode === 'visual64' ? 'Sketch mode is lossy.' : 'Object mode relies on a trained detector.', 'neutral');
  }
  async function post(path, options) {
    const response = await fetch(path, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : (data.error || `Request failed: HTTP ${response.status}`));
    return data;
  }
  function formForSource() {
    const form = new FormData();
    if (state.file) form.append('file', state.file); else if (el.sample.value) form.append('sample', el.sample.value); else throw new Error('Choose a sample or upload an image');
    return form;
  }
  async function process() {
    if (state.busy) return;
    state.busy = true; el.process.disabled = true; clearOutput();
    setNotice('Encoding image…', 'neutral');
    try {
      if (state.mode === 'lossless') { state.result = {mode: 'lossless'}; await transmit(); return; }
      const route = state.mode === 'visual64' ? '/api/preview' : '/api/analyze';
      const data = await post(route, { method: 'POST', body: formForSource() });
      if (state.mode === 'object64' && !data.packet_hex) {
        setNotice('No object outline was detected. Switch to the 64-bit visual preview to transmit the scene without an AI classification.', 'warning');
        el.inputSize.textContent = sizes(data.raw_bytes);
        return;
      }
      state.result = { mode: state.mode, data, hex: state.mode === 'visual64' ? data.payload_hex : data.packet_hex,
        payloadBytes: state.mode === 'visual64' ? 8 : 64, rawBytes: data.raw_bytes };
      el.inputSize.textContent = `${data.width} × ${data.height}`;
      el.rawMetric.textContent = sizes(data.raw_bytes);
      el.payloadMetric.textContent = `${state.result.payloadBytes} bytes`;
      el.outputSize.textContent = state.mode === 'lossless' ? 'Bit-exact file' : (state.mode === 'visual64' ? '64 bits' : '64 bytes');
      el.reductionMetric.textContent = `${((1 - state.result.payloadBytes / data.raw_bytes) * 100).toFixed(4)}%`;
      el.packetHex.textContent = state.result.hex.match(/.{1,8}/g)?.join(' ') ?? state.result.hex;
      el.copy.disabled = false; el.transmit.disabled = false;
      await transmit();
    } catch (err) {
      setNotice(err.message, 'error');
    } finally { state.busy = false; el.process.disabled = false; }
  }
  function renderLanes(link) {
    el.laneList.replaceChildren();
    const max = Math.max(...link.fragments.map(f => f.airtime_seconds));
    for (const fragment of link.fragments) {
      const row = document.createElement('div'); row.className = 'lane-row';
      const label = document.createElement('span'); label.className = 'lane-label'; label.textContent = `Link ${fragment.lane}`;
      const track = document.createElement('div'); track.className = 'lane-track';
      const fill = document.createElement('span'); fill.className = 'lane-fill'; fill.style.width = `${Math.max(5, fragment.airtime_seconds / max * 100)}%`; track.append(fill);
      const info = document.createElement('span'); info.textContent = `${fragment.wire_bytes} B · ${secs(fragment.airtime_seconds)}`;
      row.append(label, track, info); el.laneList.append(row);
    }
    if (link.packet_count) {
      el.linkSummary.textContent = `${link.packet_count.toLocaleString()} fixed 64-byte frames: 46 data bytes per frame and 18 framing/check bytes. ${sizes(link.wire_bytes)} on wire across ${link.channels_used} independent simulated links at ${link.rate_bps_per_channel.toLocaleString()} bps each. ${secs(link.parallel_seconds)} estimated vs ${secs(link.single_link_seconds)} using one link (${link.estimated_speedup}×). Ignores real modem/FEC/ACK overhead, scheduling, interference and retries.`;
      return;
    }
    el.linkSummary.textContent = `${link.wire_bytes} bytes on wire including ${link.fragments.length * 5} bytes of fragmentation overhead. ${link.channels_used} independent link${link.channels_used > 1 ? 's' : ''} modeled at ${link.rate_bps_per_channel.toLocaleString()} bps each. ${secs(link.parallel_seconds)} estimated vs ${secs(link.single_link_seconds)} on one link (${link.estimated_speedup}×). Includes illustrative 0.15 s setup + 0.33 s propagation, excludes modem preambles, FEC, interference and retries.`;
  }
  async function transmitLossless() {
    el.transmit.disabled = true;
    try {
      const form = formForSource();
      form.append('channels', el.channels.value);
      form.append('rate_bps', el.bitrate.value);
      form.append('corrupt', String(el.corrupt.checked));
      const result = await post('/api/lossless', {method: 'POST', body: form});
      state.link = result;
      renderLanes(result);
      el.inputSize.textContent = `${result.width} × ${result.height}`;
      el.rawMetric.textContent = sizes(result.source_bytes);
      el.payloadMetric.textContent = sizes(result.encoded_bytes);
      el.outputSize.textContent = `${result.packet_count.toLocaleString()} × 64 B`;
      const relative = (1 - result.compressed_bytes / result.source_bytes) * 100;
      el.reductionMetric.textContent = relative > 0 ? `${relative.toFixed(1)}% smaller` : 'No size reduction';
      el.deliveryMetric.textContent = secs(result.parallel_seconds);
      el.packetHex.textContent = result.first_packet_hex.match(/.{1,8}/g)?.join(' ') || result.first_packet_hex;
      el.copy.disabled = false;
      if (!result.ok) {
        setIntegrity('CRC rejected', 'error');
        el.outputImage.hidden = true; el.objectCanvas.hidden = true; el.outputPlaceholder.hidden = false;
        el.outputPlaceholder.textContent = 'A packet was corrupted. Request a retry.';
        el.outputCaption.textContent = 'No trusted file'; el.outputNote.textContent = result.error;
        el.download.hidden = true;
        setNotice(`File rejected: ${result.error}`, 'warning');
        return;
      }
      setIntegrity('SHA-256 verified', 'success');
      el.outputImage.src = result.images.reconstructed;
      el.outputImage.hidden = false; el.objectCanvas.hidden = true; el.outputPlaceholder.hidden = true;
      el.outputCaption.textContent = 'Original file received · byte-for-byte';
      el.outputNote.textContent = `SHA-256 matched the source. Display is resized in the browser; the transferred file bytes were not modified. ${result.codec === 'zlib' ? 'Lossless zlib compression used.' : 'Source file already compressed; transmitted without additional compression.'}`;
      if (result.reconstructed_file_base64) {
        el.download.href = `data:application/octet-stream;base64,${result.reconstructed_file_base64}`;
        el.download.download = result.source;
        el.download.hidden = false;
      } else { el.download.hidden = true; }
      setNotice(`Exact image received. ${result.packet_count.toLocaleString()} packets transmitted; original and reconstructed SHA-256 digests match.`, 'success');
    } catch (err) { setNotice(err.message, 'error'); setIntegrity('Link error', 'error'); }
    finally { el.transmit.disabled = false; }
  }

  async function transmit() {
    if (!state.result) return;
    if (state.result.mode === 'lossless') return transmitLossless();
    const data = state.result;
    el.transmit.disabled = true;
    try {
      const link = await post('/api/link/simulate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
        payload_hex: data.hex, channels: Number(el.channels.value), rate_bps: Number(el.bitrate.value),
        corrupt: el.corrupt.checked, raw_bytes: data.rawBytes
      })});
      state.link = link; renderLanes(link);
      el.deliveryMetric.textContent = secs(link.parallel_seconds);
      if (!link.ok) {
        setIntegrity('CRC rejected', 'error'); el.outputImage.hidden = true; el.objectCanvas.hidden = true; el.outputPlaceholder.hidden = false;
        el.outputPlaceholder.textContent = 'Transmission rejected. No trusted reconstruction.';
        el.outputCaption.textContent = 'Corrupt fragment'; el.outputNote.textContent = link.error;
        setNotice('The receiver rejected the corrupted fragment. Turn off “Simulate one bit error” and transmit again.', 'warning');
        return;
      }
      setIntegrity('CRC verified', 'success');
      el.outputPlaceholder.hidden = true;
      if (data.mode === 'visual64') {
        el.objectCanvas.hidden = true; el.outputImage.hidden = false;
        el.outputImage.src = data.data.images.reconstructed;
        el.outputCaption.textContent = 'Rebuilt from 8 transmitted bytes';
        el.outputNote.textContent = '8 × 8 binary approximation. Edges and details are permanently lost.';
        setNotice('Preview transmitted and reassembled successfully. No model, object label or trained dataset was needed.', 'success');
      } else {
        el.outputImage.hidden = true; el.objectCanvas.hidden = false;
        const decoded = await post('/api/decode', { method:'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({hex: link.payload_hex}) });
        const points = decoded.vertices.map(([x, y]) => `${(x / 255 * 265 + 18).toFixed(2)},${(y / 255 * 172 + 14).toFixed(2)}`).join(' ');
        el.polygon.setAttribute('points', points);
        const uncertain = decoded.confidence < 70 || data.data.detector === 'classical';
        el.outputCaption.textContent = uncertain ? 'Geometry reconstructed · Label uncertain' : 'Geometry reconstructed · Estimated label';
        el.outputNote.textContent = `Class estimate: ${decoded.object_type} (${decoded.confidence}% detector score). ${uncertain ? 'Treat as unverified; detector may be unfamiliar with this scene.' : 'Confidence is not measured accuracy.'}`;
        setNotice('Shape packet transmitted and decoded. This confirms transport integrity, not the detector’s correctness.', 'success');
      }
    } catch (err) { setNotice(err.message, 'error'); setIntegrity('Link error', 'error'); }
    finally { el.transmit.disabled = false; }
  }

  for (const b of document.querySelectorAll('[data-mode]')) b.addEventListener('click', () => setMode(b.dataset.mode));
  el.sample.addEventListener('change', () => { if (state.file && el.inputImage.src.startsWith('blob:')) URL.revokeObjectURL(el.inputImage.src); state.file = null; el.file.value = ''; sourceChanged(); });
  el.file.addEventListener('change', () => { if (!el.file.files[0]) return; if (el.inputImage.src.startsWith('blob:')) URL.revokeObjectURL(el.inputImage.src); state.file = el.file.files[0]; sourceChanged(); });
  el.process.addEventListener('click', process); el.transmit.addEventListener('click', transmit);
  for (const input of [el.channels, el.bitrate, el.corrupt]) input.addEventListener('change', () => { if (state.result) transmit(); });
  el.copy.addEventListener('click', async () => { if (!state.result) return; try {await navigator.clipboard.writeText(state.result.hex); el.copy.textContent = 'Copied'; setTimeout(() => el.copy.textContent = 'Copy packet', 1500);} catch {setNotice('Clipboard permission unavailable. Select the packet text to copy it.', 'warning');} });
  document.querySelector('.output-well').classList.add('lossless');
  (async () => {
    try {
      state.samples = await (await fetch('/api/samples')).json();
      el.sample.replaceChildren();
      const synthetic = document.createElement('optgroup'); synthetic.label = 'Synthetic sonar';
      const real = document.createElement('optgroup'); real.label = 'Real sonar · MILCO';
      for (const s of state.samples) { const o = new Option(s.short, s.name); (s.group === 'real' ? real : synthetic).append(o); }
      if (real.children.length) el.sample.append(real);
      el.sample.append(synthetic);
      const selected = state.samples.find(s => s.group === 'real') || state.samples[0];
      if (selected) el.sample.value = selected.name;
      sourceChanged();
    } catch (err) { setNotice(`Could not load samples: ${err.message}`, 'error'); }
  })();
})();
