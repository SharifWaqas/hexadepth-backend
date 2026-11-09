(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const ui = {
    sample:$('samples'), upload:$('upload'), chosen:$('chosenFile'), rate:$('bitrate'), simulate:$('simulate'),
    status:$('status'), source:$('sourceImage'), sourceEmpty:$('sourceEmpty'), sourceSize:$('sourceSize'),
    receiver:$('receiverImage'), map:$('receiverMap'), empty:$('receiverEmpty'), caption:$('receiverCaption'),
    tag:$('receiverTag'), note:$('receiverNote'), exact:$('exactButton'), download:$('downloadExact'),
    first:$('timeFirst'), detail:$('timeDetail'), full:$('timeFull'), gain:$('timeGain'),
    stages:[$('metadataStage'),$('overviewStage'),$('detailStage')],
    stageTime:[$('metaTime'),$('overviewTime'),$('detailTime')], model:$('modelInfo'), timeline:$('timelineRows'),
    baselineTime:$('baselineTime'), proposedTime:$('proposedTime'), baselineBytes:$('baselineBytes'), proposedBytes:$('proposedBytes'),
    baselineBar:$('baselineBar'), proposedBar:$('proposedBar'), comparisonGain:$('comparisonGain'),
    modemReferenceText:$('modemReferenceText'), modemReferenceLink:$('modemReferenceLink'),
    findingsMode:$('findingsMode'), findingsList:$('findingsList')
  };
  let samples = [], file = null, result = null, lastBlob = null, replayTimer = null;
  const replayButton = $('replayButton'), playbackStatus = $('playbackStatus'), bytesDelivered = $('bytesDelivered'), sweep = $('transferSweep');
  function stopReplay() { clearTimeout(replayTimer); replayTimer = null; sweep.hidden = true; }
  function replay() {
    if (!result?.ok) return;
    stopReplay();
    const data = result;
    let i = 0, delivered = 0;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    function step() {
      if (result !== data) return;
      delivered += data.stages[i].wire_bytes;
      selectStage(i);
      bytesDelivered.textContent = `${size(delivered)} received`;
      playbackStatus.textContent = `${secs(data.stages[i].available_at_seconds)} modeled · accelerated playback`;
      if (i < 2) { i++; sweep.hidden = reduced; replayTimer = setTimeout(step, reduced ? 0 : 1800); }
      else { sweep.hidden = true; playbackStatus.textContent = 'Priority detail received · replay complete'; }
    }
    step();
  }
  replayButton.addEventListener('click', replay);
  const secs = n => n == null ? '—' : n < 60 ? `${n.toFixed(1)}s` : n < 3600 ? `${(n/60).toFixed(1)}m` : `${(n/3600).toFixed(2)}h`;
  const size = n => n == null ? '—' : n >= 1000000 ? `${(n/1000000).toFixed(2)} MB` : n >= 1000 ? `${(n/1000).toFixed(1)} KB` : `${n} B`;
  function status(message, bad=false) {ui.status.textContent=message;ui.status.classList.toggle('bad',bad);}
  async function post(path, form) {const response=await fetch(path,{method:'POST',body:form});const data=await response.json().catch(()=>({}));if(!response.ok)throw new Error(data.detail||`Request failed (${response.status})`);return data;}
  function form() {const f=new FormData();if(file)f.append('file',file);else if(ui.sample.value)f.append('sample',ui.sample.value);else throw new Error('Choose an image first');return f;}
  function clear() {$('sourceRegions')?.remove();stopReplay();replayButton.disabled=true;bytesDelivered.textContent='0 B received';playbackStatus.textContent='Three stages. One useful picture.';result=null;ui.receiver.hidden=true;ui.map.hidden=true;ui.empty.hidden=false;ui.tag.textContent='Awaiting data';ui.tag.classList.remove('verified');ui.caption.textContent='No packets received';ui.first.textContent=ui.detail.textContent=ui.full.textContent=ui.gain.textContent='—';ui.baselineTime.textContent=ui.proposedTime.textContent=ui.comparisonGain.textContent='—';ui.baselineBytes.textContent=ui.proposedBytes.textContent='Awaiting sample';ui.baselineBar.style.width=ui.proposedBar.style.width='0%';ui.findingsMode.textContent='Awaiting processing';ui.findingsList.innerHTML='<p class="muted">Run the pipeline to view ranked regions.</p>';ui.modemReferenceText.textContent='Choose a sample to calculate the comparison.';ui.timeline.innerHTML='<p class="muted">Run the simulation to compare each stage.</p>';ui.exact.disabled=true;ui.download.hidden=true;for(const [i,b] of ui.stages.entries()){b.disabled=true;b.classList.remove('active');ui.stageTime[i].textContent='—';}}
  function showSource(){clear();if(lastBlob)URL.revokeObjectURL(lastBlob);if(file){lastBlob=URL.createObjectURL(file);ui.source.src=lastBlob;ui.chosen.textContent=file.name;ui.sourceSize.textContent=size(file.size);}else{lastBlob=null;const src=samples.find(s=>s.name===ui.sample.value);ui.source.src=src?.url||'';ui.chosen.textContent='or use the selected sample';ui.sourceSize.textContent=src?.group==='real'?'MILCO real sonar':'Synthetic sample';}ui.source.hidden=!ui.source.src;ui.sourceEmpty.hidden=!!ui.source.src;status('Ready to build a priority-first transmission.');}
  function renderMap(){const c=ui.map,ctx=c.getContext('2d');if(!ctx)return;const w=c.width,h=c.height;ctx.fillStyle='#102f3d';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#224e5b';ctx.lineWidth=1;for(let x=0;x<=w;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}for(let y=0;y<=h;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  const regions=result.regions||[];ctx.strokeStyle='#8bdddf';ctx.lineWidth=3;regions.forEach((r,i)=>{const [x0,y0,x1,y1]=r.box;const xx=x0/result.width*w,yy=y0/result.height*h,ww=(x1-x0)/result.width*w,hh=(y1-y0)/result.height*h;ctx.strokeRect(xx,yy,ww,hh);ctx.fillStyle='#a9e3e8';ctx.font='14px Segoe UI';ctx.fillText(`Region ${i+1} / unverified`,xx+6,Math.max(18,yy-7));});}
  function selectStage(i){if(!result?.ok)return;ui.stages.forEach((b,j)=>b.classList.toggle('active',i===j));ui.receiver.hidden=true;ui.map.hidden=true;ui.empty.hidden=true;const stage=result.stages[i];if(i===0){ui.map.hidden=false;renderMap();ui.caption.textContent='Coordinates received · no image pixels';ui.note.textContent=`${result.regions.length} prioritized regions. ${result.model_used?'Trained detector predictions remain unverified.':'OpenCV saliency ranks visual contrast; it cannot classify objects.'}`;}else{ui.receiver.src=stage.image;ui.receiver.hidden=false;ui.caption.textContent=i===1?'Lossy full-scene overview':'Lossy overview + selected higher-detail patches';ui.note.textContent=i===1?'A low-resolution JPEG reconstructs the entire scene. Unknown objects remain visible as rough image content.':'Only prioritized regions gain detail. This is not the original image and does not prove semantic recognition.';}ui.tag.classList.add('verified');ui.tag.textContent=`CRC verified · ${stage.packet_count} packet${stage.packet_count===1?'':'s'}`;}
  function drawSourceRegions(){
    const source=ui.source;
    let layer=$('sourceRegions');
    if(!layer){layer=document.createElement('canvas');layer.id='sourceRegions';layer.setAttribute('aria-label','Prioritized region outlines on original sonar');source.parentNode.append(layer);}
    const draw=()=>{
      if(!result)return;
      const w=source.clientWidth,h=source.clientHeight;
      layer.width=w;layer.height=h;
      layer.style.cssText=`position:absolute;left:0;top:${source.offsetTop}px;width:${w}px;height:${h}px;pointer-events:none`;
      const ctx=layer.getContext('2d'),scale=Math.min(w/result.width,h/result.height),ox=(w-result.width*scale)/2,oy=(h-result.height*scale)/2;
      ctx.strokeStyle='#72d8c7';ctx.lineWidth=1.5;ctx.font='11px Segoe UI';
      result.regions.forEach((r,i)=>{const [a,b,c,d]=r.box,x=ox+a*scale,y=oy+b*scale;ctx.strokeRect(x,y,(c-a)*scale,(d-b)*scale);ctx.fillStyle='#15352f';ctx.fillRect(x,Math.max(0,y-19),42,18);ctx.fillStyle='#a9f1dd';ctx.fillText(`R-0${i+1}`,x+5,Math.max(13,y-6));});
    };
    draw();source.onload=draw;
  }
  window.addEventListener('resize',()=>{if(result)drawSourceRegions();});
  function renderFindings(){drawSourceRegions();
    ui.findingsList.replaceChildren();
    const regions = result?.regions || [];
    if(!regions.length){const p=document.createElement('p');p.className='muted';p.textContent='No candidate regions found. The whole-scene overview is still transmitted.';ui.findingsList.append(p);}
    regions.forEach((r,i)=>{
      const item=document.createElement('div');item.className='finding-row';
      const [x0,y0,x1,y1]=r.box;
      const label=document.createElement('strong');
      const detectorLabel = ({10:'Mine-like contact (MILCO)',11:'Other bottom object (NOMBO)'})[r.class_id];
      label.textContent=detectorLabel ? `Candidate ${i+1}: ${detectorLabel}` : `Region ${i+1}: unverified visual feature`;
      const desc=document.createElement('small');
      desc.textContent=`X ${x0}–${x1} px · Y ${y0}–${y1} px`;
      const score=document.createElement('span');score.className='finding-score';
      score.textContent = r.class_id != null ? `Model score ${Math.round(r.score/255*100)}% · uncalibrated` : `Visual priority ${Math.round(r.score/255*100)}/100`;
      item.append(label,desc,score);ui.findingsList.append(item);
    });
    ui.findingsMode.textContent=result.model_used?'Trained model + saliency fallback (unverified)':'Image saliency · not object recognition';
  }
  function renderComparison(){
    if(!result?.comparison)return;
    const c=result.comparison;
    ui.baselineTime.textContent=secs(c.full_file_seconds);
    ui.proposedTime.textContent=secs(c.first_view_seconds);
    ui.baselineBytes.textContent=`${size(c.full_wire_bytes)} on wire · exact file`;
    ui.proposedBytes.textContent=`${size(c.first_view_wire_bytes)} on wire · lossy overview`;
    ui.baselineBar.style.width='100%';
    ui.proposedBar.style.width=`${Math.max(.8,100*c.first_view_seconds/c.full_file_seconds)}%`;
    ui.comparisonGain.textContent=`${c.first_view_advantage.toFixed(1)}×`;
    if(c.modem_reference){
      ui.modemReferenceText.textContent=`${c.modem_reference.name}: ${c.modem_reference.bps.toLocaleString()} bps nominal. ${c.modem_reference.notes}`;
      ui.modemReferenceLink.href=c.modem_reference.url;
    }else{
      ui.modemReferenceText.textContent=`Illustrative ${c.link_bps.toLocaleString()} bps link; no hardware benchmark.`;
    }
  }
  function timeline(){const stages=result.stages;const values=[{label:'Region coordinates',seconds:stages[0].available_at_seconds},{label:'First scene overview',seconds:stages[1].available_at_seconds},{label:'Priority detail',seconds:stages[2].available_at_seconds},{label:'Full image (optional)',seconds:result.full_file_estimate_seconds}];const max=Math.max(...values.map(v=>v.seconds));ui.timeline.replaceChildren();values.forEach(v=>{const row=document.createElement('div');row.className='timeline-item';const a=document.createElement('span');a.textContent=v.label;const track=document.createElement('div');track.className='timeline-track';const bar=document.createElement('span');bar.className='timeline-fill';bar.style.width=`${Math.max(0.3,100*v.seconds/max)}%`;track.append(bar);const t=document.createElement('strong');t.textContent=secs(v.seconds);row.append(a,track,t);ui.timeline.append(row);});}
  async function simulate(){ui.simulate.disabled=true;clear();status('Encoding and validating staged 64-byte packets…');try{const f=form();f.append('rate_bps',ui.rate.value);const data=await post('/api/priority',f);if(!data.ok){status(`Packet integrity failed: ${data.error}`,true);return;}result=data;ui.first.textContent=secs(data.overview_seconds);ui.detail.textContent=secs(data.detail_seconds);ui.full.textContent=secs(data.full_file_estimate_seconds);ui.gain.textContent=`${data.estimated_first_view_gain.toFixed(1)}×`;ui.stageTime.forEach((t,i)=>t.textContent=`${secs(data.stages[i].available_at_seconds)} · ${data.stages[i].packet_count} pkt`);ui.stages.forEach(b=>b.disabled=false);ui.exact.disabled=false;ui.model.textContent=`${data.detector_note} The current result uses ${data.method}. The presence of a candidate region does not validate recognition on unfamiliar sonar.`;renderComparison();renderFindings();timeline();replayButton.disabled=false;replay();status(`Verified preview ready in ${secs(data.overview_seconds)} estimated airtime, using ${size(data.metadata_and_overview_wire_bytes)} on-wire. ${data.stages[1].packet_count} JPEG overview packets.`);}catch(err){status(err.message,true);}finally{ui.simulate.disabled=false;}}
  async function exact(){if(!result)return;stopReplay();ui.exact.disabled=true;status('Reassembling and SHA-256-checking the original file…');try{const f=form();f.append('channels','1');f.append('rate_bps',ui.rate.value);const r=await post('/api/lossless',f);if(!r.ok){status(r.error,true);return;}ui.receiver.src=r.images.reconstructed;ui.receiver.hidden=false;ui.map.hidden=true;ui.empty.hidden=true;ui.caption.textContent='Original image · SHA-256 verified';ui.note.textContent=`Exact original file reassembled. ${r.packet_count.toLocaleString()} × 64-byte frames, ${size(r.wire_bytes)} on wire. Airtime is modeled, not measured.`;ui.tag.textContent='SHA-256 verified';ui.tag.classList.add('verified');ui.stages.forEach(b=>b.classList.remove('active'));if(r.reconstructed_file_base64){ui.download.href=`data:application/octet-stream;base64,${r.reconstructed_file_base64}`;ui.download.download=r.source;ui.download.hidden=false;}status(`Exact file received: SHA-256 source and receiver match. ${secs(r.parallel_seconds)} estimated standalone transfer.`);}catch(err){status(err.message,true);}finally{ui.exact.disabled=false;}}
  ui.simulate.addEventListener('click',simulate);ui.exact.addEventListener('click',exact);ui.stages.forEach((b,i)=>b.addEventListener('click',()=>{stopReplay();selectStage(i);bytesDelivered.textContent=`${size(result.stages.slice(0,i+1).reduce((n,s)=>n+s.wire_bytes,0))} received`;playbackStatus.textContent='Stage inspection · replay paused';}));ui.rate.addEventListener('change',clear);
  ui.sample.addEventListener('change',()=>{file=null;ui.upload.value='';showSource();});ui.upload.addEventListener('change',()=>{if(ui.upload.files[0]){file=ui.upload.files[0];showSource();}});
  (async()=>{try{const r=await fetch('/api/samples');samples=await r.json();ui.sample.replaceChildren();const real=document.createElement('optgroup');real.label='Real MILCO sonar';const synth=document.createElement('optgroup');synth.label='Synthetic examples';for(const s of samples){const o=new Option(s.short,s.name);(s.group==='real'?real:synth).append(o);}if(real.children.length)ui.sample.append(real);ui.sample.append(synth);ui.sample.value=(samples.find(s=>s.group==='real')||samples[0])?.name||'';showSource();}catch(err){status(`Samples unavailable: ${err.message}`,true);}})();
})();
