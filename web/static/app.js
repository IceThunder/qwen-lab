'use strict';
const $ = id => document.getElementById(id);
const KEY = 'qwen-lab-session-v1';
const presets = {
  general: '你是一位严谨、友好的中文助手。请清晰回答问题，不确定时如实说明。',
  reason: '你是一位善于解释原理的中文助手。请给出关键依据和简洁的推理说明，区分事实与假设。',
  code: '你是一位经验丰富的软件工程师。优先给出可运行的代码，解释边界情况，避免编造接口。',
  writing: '你是一位中文编辑。保留原意，改善文字的清晰度与流畅度，不添加未经提供的事实。',
  custom: ''
};
let models = [], groups = [], turns = [], busy = false, activeJob = null, editing = null, pendingTurn = null;
let restoring = false;
let recipePreview = null, recipeSource = null, recipeFetching = false;
const settingsIds = ['model','adapter','preset','system','prefix','suffix','maxTokens','contextLimit','temperature','seed','thinking','compare','skipFirst'];
function notice(message = '') { uiBind($('notice'),()=>uiMessage(message),'textContent'); $('notice').hidden = !message; }
function el(tag, cls, text) { const x = document.createElement(tag); if (cls) x.className = cls; if (text !== undefined) x.textContent = text; return x; }
function currentModel() { return models.find(m => m.id === $('model').value); }
function supportsSteering() { return currentModel()?.supports_steering !== false; }
function syncCapabilities() {
  const supported = supportsSteering();
  if (!supported) { groups=[]; $('compare').checked=false; }
  document.querySelectorAll('#neuronControls input,#neuronControls select,#neuronControls button').forEach(x=>x.disabled=busy||!supported);
  $('compare').disabled=busy||!supported;
  $('adapter').disabled=busy||!supported;
  $('steeringUnavailable').hidden=supported;
  const count=groups.filter(g=>g.enabled).length;
  uiBind($('steeringStatus'),()=>!supported?uiT('GGUF · 仅对话'):count?uiTr`${count} 组干预已启用（收起仍生效）`:uiT('未启用神经元干预'),'textContent');
  $('steeringStatus').classList.toggle('active',supported&&count>0);
}
$('toggleNeurons').addEventListener('click',()=>{
  const open=$('neuronControls').hidden;
  $('neuronControls').hidden=!open;
  document.querySelector('main').classList.toggle('console-hidden',!open);
  $('toggleNeurons').setAttribute('aria-expanded',String(open));
  uiBind($('toggleNeurons'),()=>open?uiT('收起神经元控制台'):uiT('展开神经元控制台'),'textContent');
});
function settings() { return Object.fromEntries(settingsIds.map(id => [id, $(id).type === 'checkbox' ? $(id).checked : $(id).value])); }
function session() { return {version:1, settings:settings(), groups, turns, recipeSource, draft:$('prompt').value}; }
function save() {
  if (restoring) return;
  try { localStorage.setItem(KEY, JSON.stringify(session())); uiBind($('saveState'),()=>uiT('已保存在此浏览器'),'textContent'); }
  catch (_) { uiBind($('saveState'),()=>uiT('浏览器存储已满，请导出保存'),'textContent'); }
}
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await r.json();
  if (r.status===404 && (path.startsWith('/api/model-manager')||path==='/api/models/test-load')) throw new Error(uiT('当前后端尚未加载模型管理接口，请重启 bash scripts/web.sh 后刷新页面。'));
  if (r.status===404 && path==='/api/neuronary/preview') throw new Error(uiT('当前 Web 后端尚未加载 Neuronary 导入接口。请在启动终端按 Ctrl+C，再运行 bash scripts/web.sh，然后刷新页面。'));
  if (!r.ok) throw new Error(uiMessage(data.error) || uiTr`请求失败 ${r.status}`); return data;
}
function fillAdapters(value = '') {
  const m = currentModel(); $('adapter').replaceChildren(new Option(uiT('基础模型 · 不加载适配器'),''));
  for (const name of m?.adapters || []) $('adapter').add(new Option(name, name));
  $('adapter').value = m?.adapters.includes(value) ? value : '';
  uiBind($('modelInfo'),()=>m ? (m.supports_steering===false?uiTr`${m.format} / 文本多轮对话 / 不支持神经元干预`:uiTr`${m.format} / ${m.layers} 层 / 每层 ${m.width} 个 MLP 通道`) : uiT('未发现受支持的本地模型'),'textContent');
  uiBind($('layerInfo'),()=>m?.layers ? uiTr`0–${m.layers - 1} · 从 0 编号` : '','textContent');
}
function setBusy(value) {
  busy = value;
  document.querySelectorAll('[data-lock], #groups input, #groups select, #groups button, #layerGrid button').forEach(x => x.disabled = value);
  ['newChat','import','cancelEdit'].forEach(id => $(id).disabled = value);
  $('send').disabled = value || !currentModel(); $('stop').hidden = !value;
  document.querySelectorAll('.turn-label button,.adopt').forEach(x => x.disabled = value);
  syncCapabilities();
  syncRecipeButtons();
}
function renderLayers() {
  $('layerGrid').replaceChildren();
  for (let n=0; n<(supportsSteering() ? currentModel()?.layers || 0 : 0); n++) {
    const used = groups.some(g => g.enabled && Number(g.layer)===n);
    const b=el('button', used ? 'configured':'', String(n)); b.type='button'; b.disabled=busy;
    uiBind(b,()=>uiTr`第 ${n} 层${used ? uiT(' · 已有干预组') : uiT(' · 点击添加组')}`,'title');
    b.addEventListener('click', () => addGroup(n)); $('layerGrid').append(b);
  }
  uiBind($('groupCount'),()=>String(groups.filter(g => g.enabled).length),'textContent');
  syncCapabilities();
}
function renderGroups() {
  syncCapabilities();
  $('groups').replaceChildren();
  groups.forEach((g, i) => {
    g.mode = g.mode || 'scale';
    const additive = g.mode === 'normalized_add';
    const box=el('div', `group${g.enabled ? '':' off'}`), head=el('div','group-head'), label=el('label');
    const enabled=document.createElement('input'); enabled.type='checkbox'; enabled.checked=g.enabled; uiBind(enabled,()=>uiTr`启用第 ${i+1} 组`,'ariaLabel');
    const layer=document.createElement('input'); layer.type='number'; layer.className='layer-input'; layer.min=0; layer.max=(currentModel()?.layers || 32)-1; layer.value=g.layer; uiBind(layer,()=>uiTr`第 ${i+1} 组层号`,'ariaLabel');
    label.append(enabled,document.createTextNode(uiT('层')),layer);
    const del=el('button','','×'); uiBind(del,()=>uiT('删除此组'),'title'); uiBind(del,()=>uiTr`删除第 ${i+1} 组`,'ariaLabel'); head.append(label,del);
    const mode=document.createElement('select'); uiBind(mode,()=>uiTr`第 ${i+1} 组干预模式`,'ariaLabel');
    mode.add(new Option(uiT('乘法缩放 · 激活 × 倍率'),'scale'));
    mode.add(new Option(uiT('归一化加法 · 激活 + Δ'),'normalized_add')); mode.value=g.mode;
    mode.addEventListener('change',()=>{g.mode=mode.value;if(g.dose===undefined)g.dose=0;if(g.scale===undefined)g.scale=1;renderGroups();save();});
    const caption=el('label','',uiT('通道编号 / 区间')); const channels=document.createElement('input'); channels.className='channel-input'; channels.value=g.neurons; uiBind(channels,()=>uiT('例如 0,1,2,16-31'),'placeholder'); uiBind(channels,()=>uiTr`第 ${i+1} 组神经元编号`,'ariaLabel');
    const scaleLine=el('div','scale-line'); scaleLine.append(el('span','',additive?uiT('归一化剂量 dose（带正负号）'):uiT('激活缩放倍率 ×')));
    const value=additive?(g.dose??0):g.scale;
    const number=document.createElement('input'); number.type='number'; number.min=additive?-4:0; number.max=4; number.step=.1; number.value=value; uiBind(number,()=>uiTr`第 ${i+1} 组${additive?uiT('剂量'):uiT('倍率')}`,'ariaLabel');
    const slider=document.createElement('input'); slider.type='range'; slider.min=number.min; slider.max=4; slider.step=.1; slider.value=value; uiBind(slider,()=>number.ariaLabel+uiT('滑块'),'ariaLabel'); scaleLine.append(number);
    const choices=el('div','presets');
    const updateValue = value => {g[additive?'dose':'scale']=Number(value); number.value=value; slider.value=value; save();};
    const quick=additive?[[-1,uiT('负向 −1')],[0,uiT('不注入 0')],[1,uiT('正向 +1')]]:[[0,uiT('关闭 0×')],[1,uiT('原值 1×')],[2,uiT('增强 2×')]];
    for (const [v,text] of quick) { const b=el('button','',text); b.addEventListener('click',()=>updateValue(v)); choices.append(b); }
    enabled.addEventListener('change',()=>{g.enabled=enabled.checked; box.classList.toggle('off',!g.enabled); renderLayers(); save();});
    layer.addEventListener('input',()=>{g.layer=Number(layer.value); renderLayers(); save();});
    channels.addEventListener('input',()=>{g.neurons=channels.value;save();});
    number.addEventListener('input',()=>updateValue(number.value)); slider.addEventListener('input',()=>updateValue(slider.value));
    del.addEventListener('click',()=>{groups.splice(i,1);renderGroups();save();});
    box.append(head,mode,caption,channels,scaleLine,slider,choices);
    if(additive)box.append(el('p','hint',uiT('Δ = dose × 残差中位数 ÷ 权重列范数。0 表示不注入，不是关闭通道。')));
    $('groups').append(box);
  });
  renderLayers(); setBusy(busy);
}
function addGroup(layer=16) {
  if(!supportsSteering())return;
  if (groups.length>=32) return notice(uiT('最多配置 32 个干预组。'));
  groups.push({layer:Math.min(layer,(currentModel()?.layers||32)-1),neurons:'0,1,2,3',mode:'scale',scale:0,dose:0,enabled:true});
  renderGroups(); save();
}
function renderTurns() {
  const region=$('messages'); const nearBottom=region.scrollHeight-region.scrollTop-region.clientHeight<90;
  if (!turns.length && !pendingTurn) { region.replaceChildren(emptyTemplate.cloneNode(true)); bindExamples(); uiStatic(region); uiBind($('turnCount'),()=>uiT('0 轮对话'),'textContent');return; }
  region.replaceChildren();
  const visible = pendingTurn ? [...turns,pendingTurn] : turns;
  visible.forEach((turn,index)=>{
    const wrapper=el('article','turn'), title=el('div','turn-label',`YOU / ${String(index+1).padStart(2,'0')}`);
    if (index<turns.length) {
      const edit=el('button','',uiT('编辑并重跑')); edit.disabled=busy;
      edit.addEventListener('click',()=>{editing=index; $('prompt').value=turn.raw; $('editBanner').hidden=false; $('prompt').focus();}); title.append(edit);
    }
    wrapper.append(title,el('div','user-message',turn.user));
    const keys=turn.snapshot?.compare ? ['baseline','intervened']:['intervened'];
    const answers=el('div',`answers${keys.length>1?' compare':''}`);
    for (const key of keys) {
      const r=turn.results?.[key], chosen=turn.chosen===key && turn.status==='done';
      const card=el('div',`answer${chosen?' chosen':''}`), cap=el('div','answer-title',key==='baseline'?uiT('无干预 · 对照'):uiT('当前配置'));
      if (chosen) cap.append(el('span','',uiT('已用于上下文')));
      card.append(cap,el('div','answer-text',r?.text || (turn.status==='running'?uiT('等待生成…'):'—')));
      if (r) {const metric=el('div','answer-metrics');metric.append(el('span','',`${r.tokens} tokens`),el('span','',`${r.tps} tok/s`),el('span','',r.peak_gb==null?uiT('llama.cpp · 速率含预填充'):uiTr`MLX 峰值 ${r.peak_gb} GB`));if(r.finish_reason==='length')metric.append(el('span','',uiT('已到长度上限')));card.append(metric);}
      if (turn.status==='done' && !chosen && r) {const adopt=el('button','adopt',uiT('采用此回答继续'));adopt.disabled=busy;adopt.addEventListener('click',()=>{
        if(index<turns.length-1 && !confirm(uiT('更换这轮回答会删除后续对话，是否继续？')))return;
        turns=turns.slice(0,index+1);turn.chosen=key;editing=null;$('editBanner').hidden=true;renderTurns();save();
      });card.append(adopt);}
      answers.append(card);
    }
    wrapper.append(answers);
    if(turn.calibration)wrapper.append(calibrationView(turn));
    const snap=el('details','snapshot');snap.append(el('summary','',uiTr`${turn.snapshot?.model || ''} · 查看本轮提示词与干预快照`),el('pre','',JSON.stringify(turn.snapshot,null,2)));wrapper.append(snap);
    if(turn.status==='error'||turn.status==='cancelled')wrapper.append(el('div','failure',uiMessage(turn.error) || uiT('已停止：本轮未加入后续上下文，可编辑后重跑。')));
    if(turn.status==='running')wrapper.append(el('div','pending',uiMessage(turn.phase) || uiT('正在运行…')));
    region.append(wrapper);
  });
  uiBind($('turnCount'),()=>uiTr`${turns.filter(t=>t.status==='done').length} 轮对话`,'textContent');
  if(nearBottom || pendingTurn)region.scrollTop=region.scrollHeight;
}
function calibrationView(turn) {
  const c=turn.calibration, box=el('details','calibration');
  const actual=turn.activationAudit || [];
  box.append(el('summary','',uiTr`归一化校准 · ${c.calibration_tokens} tokens · ${c.rows.length} 个通道${actual.length?uiT(' · 已观测实际注入'):''}`));
  box.append(el('p','hint',uiTr`无干预预运行，${c.skip_first_token?uiT('跳过首 token'):uiT('包含首 token')}；每轮重新校准，固定 Δ 应用于全部输入及生成 token。`));
  const scroll=el('div','table-scroll'),table=el('table'),head=el('tr');
  for(const title of [uiT('位置'),'dose',uiT('残差中位数'),uiT('权重列范数'),uiT('计算 Δ'),uiT('首 token：前 → 后')])head.append(el('th','',title));
  table.append(head);
  const num=n=>Number.isFinite(n)?Number(n).toPrecision(5):'—';
  for(const row of c.rows){
    const a=actual.find(a=>a.layer===row.layer&&a.neuron===row.neuron),tr=el('tr');
    for(const text of [`L${row.layer} N${row.neuron}`,String(row.dose),num(row.residual_median),num(row.column_norm),num(row.delta),a?`${num(a.before)} → ${num(a.after)}`:uiT('等待干预分支')])tr.append(el('td','',text));
    table.append(tr);
  }
  scroll.append(table);box.append(scroll,el('p','hint',uiT('前后值取干预分支首个输入 token，含计算精度舍入；不是全部 token 的均值。')));
  return box;
}
function history(until) {
  return turns.slice(0,until).filter(t=>t.status==='done').flatMap(t=>[{role:'user',content:t.user},{role:'assistant',content:t.results[t.chosen].text}]);
}
async function send() {
  if(busy)return;
  const raw=$('prompt').value.trim(); if(!raw)return notice(uiT('请先输入问题。'));
  const opts=settings(); const user=[opts.prefix.trim(),raw,opts.suffix.trim()].filter(Boolean).join('\n\n');
  const messages=history(editing===null?turns.length:editing);
  if(opts.system.trim())messages.unshift({role:'system',content:opts.system.trim()});messages.push({role:'user',content:user});
  const request={model:opts.model,adapter:opts.adapter,messages,groups:structuredClone(groups),compare:opts.compare,thinking:opts.thinking,
    calibration_skip_first:opts.skipFirst,max_tokens:Number(opts.maxTokens),context_limit:Number(opts.contextLimit),temperature:Number(opts.temperature),seed:Number(opts.seed)};
  notice();setBusy(true);uiBind($('status'),()=>uiT('提交实验…'),'textContent');
  try {
    const job=await api('/api/chat',request);activeJob=job.id;
    if(editing!==null)turns=turns.slice(0,editing);
    editing=null;$('editBanner').hidden=true;
    pendingTurn={raw,user,status:'running',chosen:'intervened',results:{},snapshot:{...request,neuronary:recipeSource?structuredClone(recipeSource):undefined,messages:undefined,system:opts.system,prefix:opts.prefix,suffix:opts.suffix},job:job.id};
    $('prompt').value='';renderTurns();save();
    let errors=0;
    while(activeJob) {
      let state;
      try {state=await api(`/api/jobs/${activeJob}`); errors=0;}
      catch(error){if(++errors>=5)throw error;uiBind($('status'),()=>uiT('连接中断，正在重试…'),'textContent');await new Promise(r=>setTimeout(r,1000));continue;}
      pendingTurn.results=state.results;pendingTurn.calibration=state.calibration;pendingTurn.activationAudit=state.activation_audit;pendingTurn.phase=state.phase;uiBind($('status'),()=>uiMessage(state.phase),'textContent');renderTurns();
      if(['done','error','cancelled'].includes(state.status)) {
        pendingTurn.status=state.status;pendingTurn.error=state.error;
        turns.push(pendingTurn);pendingTurn=null;activeJob=null;save();renderTurns();
        if(state.error)notice(state.error);break;
      }
      await new Promise(r=>setTimeout(r,250));
    }
  } catch(error) {
    notice(error.message);uiBind($('status'),()=>uiT('请求未完成'),'textContent');
    if(activeJob)api(`/api/jobs/${activeJob}/cancel`,{}).catch(()=>{});
    if(pendingTurn){pendingTurn.status='error';pendingTurn.error=uiT('连接中断，本轮未加入后续上下文。');turns.push(pendingTurn);pendingTurn=null;renderTurns();save();}
    activeJob=null;
  } finally {setBusy(false);}
}
function validateSession(s) {
  if(!s||s.version!==1||!s.settings||!Array.isArray(s.groups)||!Array.isArray(s.turns)||s.groups.length>32||s.turns.length>100)throw new Error(uiT('会话文件格式不正确'));
  if(!models.some(m=>m.id===s.settings.model))throw new Error(uiT('此会话的模型未安装，无法导入'));
  if(s.groups.some(g=>!g||!Number.isInteger(g.layer)||!['scale','normalized_add'].includes(g.mode||'scale')||!Number.isFinite(g.mode==='normalized_add'?g.dose:g.scale)||typeof g.neurons!=='string'||typeof g.enabled!=='boolean'))throw new Error(uiT('干预组格式不正确'));
  for(const t of s.turns){
    if(!t||typeof t.raw!=='string'||typeof t.user!=='string'||!t.results||!t.snapshot||!['done','error','cancelled'].includes(t.status))throw new Error(uiT('会话内容格式不正确'));
    if(t.status==='done'&&(!['baseline','intervened'].includes(t.chosen)||typeof t.results[t.chosen]?.text!=='string'))throw new Error(uiT('会话回答不完整'));
  }
  if(s.recipeSource && (typeof s.recipeSource!=='object'||!/^[-A-Za-z0-9_]{4,64}$/.test(s.recipeSource.id)||typeof s.recipeSource.name!=='string'))throw new Error(uiT('配方来源格式不正确'));
  return s;
}
function restore(s) {
  validateSession(s);restoring=true;
  for(const id of settingsIds){if(id==='adapter')continue;if($(id).type==='checkbox')$(id).checked=id==='skipFirst'?(s.settings[id]??true):Boolean(s.settings[id]);else $(id).value=String(s.settings[id]??'');}
  fillAdapters(s.settings.adapter);recipeSource=s.recipeSource||null;renderRecipeSource();groups=s.groups;turns=s.turns;$('prompt').value=s.draft||'';editing=null;$('editBanner').hidden=true;
  restoring=false;renderGroups();renderTurns();save();
}
function bindExamples(){document.querySelectorAll('[data-example]').forEach(b=>b.addEventListener('click',()=>{$('prompt').value=uiT(b.dataset.example);$('prompt').focus();save();}));}
const emptyTemplate=$('empty').cloneNode(true);
$('send').addEventListener('click',send);
$('prompt').addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter'){e.preventDefault();send();}});
$('prompt').addEventListener('input',save);
$('stop').addEventListener('click',async()=>{if(activeJob){uiBind($('status'),()=>uiT('正在停止，等待当前计算结束…'),'textContent');try{await api(`/api/jobs/${activeJob}/cancel`,{});}catch(e){notice(e.message);}}});
$('addGroup').addEventListener('click',()=>addGroup());
$('resetGroups').addEventListener('click',()=>{recipeSource=null;renderRecipeSource();groups=[];renderGroups();save();});
$('cancelEdit').addEventListener('click',()=>{editing=null;$('editBanner').hidden=true;$('prompt').value='';save();});
$('preset').addEventListener('change',()=>{if($('preset').value!=='custom')$('system').value=uiT(presets[$('preset').value]);save();});
$('system').addEventListener('input',()=>{$('preset').value='custom';save();});
settingsIds.filter(id=>id!=='model').forEach(id=>$(id).addEventListener('change',save));
$('model').addEventListener('change',()=>{
  if(turns.length && !confirm(uiT('切换模型会清空当前对话和干预组。需要保留时请先导出。是否继续？'))){$('model').value=turns[0].snapshot.model;save();return;}
  turns=[];groups=[];recipeSource=null;renderRecipeSource();editing=null;$('editBanner').hidden=true;fillAdapters();renderGroups();renderTurns();save();
});
$('newChat').addEventListener('click',()=>{if(turns.length&&!confirm(uiT('清空当前对话？模型、提示词和干预设置会保留。')))return;turns=[];editing=null;$('editBanner').hidden=true;$('prompt').value='';notice();renderTurns();save();});
$('export').addEventListener('click',()=>{
  const data=new Blob([JSON.stringify(session(),null,2)],{type:'application/json'}),url=URL.createObjectURL(data),a=document.createElement('a');
  a.href=url;a.download=`qwen-lab-${new Date().toISOString().replaceAll(':','-')}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
$('import').addEventListener('click',()=>$('importFile').click());
$('importFile').addEventListener('change',async e=>{
  const file=e.target.files[0];if(!file)return;
  try {if(file.size>1048576)throw new Error(uiT('会话文件不能超过 1 MB'));const s=validateSession(JSON.parse(await file.text()));if(turns.length&&!confirm(uiT('导入会替换当前会话，是否继续？')))return;restore(s);notice();}
  catch(error){notice(uiTr`导入失败：${error.message}`);}finally{e.target.value='';}
});
window.addEventListener('beforeunload',e=>{if(busy){e.preventDefault();e.returnValue='';}});
function syncRecipeButtons() {
  $('neuronaryFetch').disabled=busy||recipeFetching;
  $('neuronaryApply').disabled=busy||recipeFetching||!recipePreview?.compatible_model;
}
function renderRecipeSource() {
  $('neuronarySource').hidden=!recipeSource;
  $('neuronarySource').replaceChildren();
  if(recipeSource){
    const link=el('a','',`${recipeSource.name} · ${recipeSource.id}`);
    link.href=`https://neuronary.org/r/${encodeURIComponent(recipeSource.id)}`;link.target='_blank';link.rel='noopener noreferrer';
    $('neuronarySource').append(document.createTextNode(uiT('初始配方来源：')),link,document.createTextNode(uiT('。当前参数可能已手动修改；来源及原始参数保存在实验快照中。')));
  }
}
function renderRecipeExample() {
  const ex=recipePreview?.examples[Number($('neuronaryExample').value)];
  uiBind($('neuronaryExampleText'),()=>ex?uiTr`提示词：\n${ex.prompt}\n\n网站无干预回答：\n${ex.base_output||uiT('未提供')}\n\n网站干预回答：\n${ex.output||uiT('未提供')}`:uiT('保留当前输入框中的问题，不导入网站回答。'),'textContent');
}
function renderRecipePreview(preserve=false) {
  if(!recipePreview)return;const d=recipePreview, selected=$('neuronaryExample').value;
    uiBind($('neuronaryName'),()=>d.name,'textContent');
    uiBind($('neuronaryModel'),()=>uiTr`网站模型：${d.model}`,'textContent');
    uiBind($('neuronaryRecipe'),()=>d.groups.map(g=>`L${g.layer} N${g.neurons} ${g.dose>=0?'+':''}${g.dose}`).join('\n'),'textContent');
    uiBind($('neuronaryDescription'),()=>d.description,'textContent');
    uiBind($('neuronaryCompatibility'),()=>uiMessage(d.compatibility),'textContent');
    $('neuronaryWarnings').replaceChildren(...d.warnings.map(w=>el('li','',uiMessage(w))));
    uiBind($('neuronaryNotes'),()=>d.notes||uiT('未提供备注'),'textContent');
    uiBind($('neuronarySampling'),()=>JSON.stringify(d.sampling,null,2),'textContent');
    $('neuronaryExample').replaceChildren(new Option(uiT('保留我当前输入的问题'),'-1'));
    d.examples.forEach((ex,i)=>$('neuronaryExample').add(new Option(`${i+1}. ${ex.prompt.slice(0,70)}`,String(i))));
    if(d.examples.length)$('neuronaryExample').value='0';
    if(preserve)$('neuronaryExample').value=selected;
    renderRecipeExample();$('neuronaryPreview').hidden=false;
}
$('neuronaryUrl').addEventListener('input',()=>{
  recipePreview=null;$('neuronaryPreview').hidden=true;uiBind($('neuronaryMessage'),()=>'','textContent');syncRecipeButtons();
});
$('neuronaryExample').addEventListener('change',renderRecipeExample);
$('neuronaryFetch').addEventListener('click',async()=>{
  if(busy||recipeFetching)return;
  const input=$('neuronaryUrl').value.trim();
  recipeFetching=true;recipePreview=null;$('neuronaryPreview').hidden=true;
  uiBind($('neuronaryMessage'),()=>uiT('正在读取公开配方…'),'textContent');syncRecipeButtons();
  try {
    const d=await api('/api/neuronary/preview',{url:input});
    if(input!==$('neuronaryUrl').value.trim())return;
    recipePreview=d;
    renderRecipePreview();
    uiBind($('neuronaryMessage'),()=>uiT('读取完成。请查看兼容性和参数后导入；导入不会自动发送问题。'),'textContent');
  }catch(e){uiBind($('neuronaryMessage'),()=>uiTr`读取失败：${e.message}`,'textContent');}
  finally{recipeFetching=false;syncRecipeButtons();}
});
$('neuronaryApply').addEventListener('click',()=>{
  const d=recipePreview;if(busy||!d?.compatible_model)return;
  const multiplier=Number($('neuronaryDoseFactor').value);
  if($('neuronaryDoseFactor').value.trim()===''||!Number.isFinite(multiplier)||multiplier<0||multiplier>1)return notice(uiT('导入剂量倍率必须在 0–1 之间。'));
  const ex=d.examples[Number($('neuronaryExample').value)];
  if((turns.length||groups.length||$('prompt').value.trim())&&!confirm(uiT('导入将新建对话、替换干预组，并按所选示例替换问题。现有会话需要保留时请先导出。继续？')))return;
  turns=[];groups=structuredClone(d.groups).map(g=>({...g,dose:Number((g.dose*multiplier).toPrecision(12))}));
  editing=null;$('editBanner').hidden=true;
  $('model').value=d.compatible_model;fillAdapters('');
  $('temperature').value=String(d.temperature);$('thinking').checked=false;$('compare').checked=true;$('skipFirst').checked=true;
  if($('neuronaryClean').checked){$('system').value='';$('prefix').value='';$('suffix').value='';$('preset').value='custom';}
  if(ex)$('prompt').value=ex.prompt;
  recipeSource={id:d.id,url:d.url,name:d.name,model:d.model,original_groups:structuredClone(d.groups),
    sampling:d.sampling,local_convention:d.local_convention,warnings:d.warnings,dose_multiplier:multiplier,
    selected_example:ex?structuredClone(ex):null,imported_at:new Date().toISOString()};
  renderRecipeSource();renderGroups();renderTurns();save();
  $('neuronControls').hidden=false;document.querySelector('main').classList.remove('console-hidden');
  $('toggleNeurons').setAttribute('aria-expanded','true');uiBind($('toggleNeurons'),()=>uiT('收起神经元控制台'),'textContent');
  notice(uiTr`已导入 ${d.name}：${groups.length} 个通道，剂量倍率 ${multiplier}。请点击“发送并运行”进行本地对照；网站示例不是本地验证结果。`);
});
async function init(){
  try{
    const data=await api('/api/models');models=data.models;$('model').replaceChildren();models.forEach(m=>$('model').add(new Option(m.name,m.id)));fillAdapters();$('system').value=uiT(presets.general);
    const stored=localStorage.getItem(KEY);if(stored){try{restore(JSON.parse(stored));}catch(e){notice(uiTr`未恢复旧会话：${e.message}`);}}
    renderGroups();bindExamples();setBusy(false);
    if(!models.length)notice(uiT('未发现可用模型。请先按照项目文档下载模型。'));
    if(data.active_job){notice(uiT('服务器还有一个实验正在运行。请等待完成；需要时可用“停止生成”结束该实验。'));activeJob=data.active_job;setBusy(true);
      const timer=setInterval(async()=>{try{const j=await api(`/api/jobs/${data.active_job}`);uiBind($('status'),()=>uiMessage(j.phase),'textContent');if(['done','error','cancelled'].includes(j.status)){clearInterval(timer);activeJob=null;setBusy(false);notice(uiT('此前实验已结束。请重新发送问题以继续。'));}}catch(e){clearInterval(timer);activeJob=null;setBusy(false);notice(e.message);}},1000);
    }
  }catch(e){notice(uiTr`连接失败：${e.message}`);uiBind($('status'),()=>uiT('服务不可用'),'textContent');$('send').disabled=true;}
}
init();

window.addEventListener('uilanguagechange',()=>{
  const scroll=$('messages').scrollTop;
  fillAdapters($('adapter').value);renderGroups();renderTurns();renderRecipeSource();renderRecipePreview(true);
  $('messages').scrollTop=scroll;
  uiBind($('toggleNeurons'),()=>uiT($('neuronControls').hidden?'展开神经元控制台':'收起神经元控制台'),'textContent');
});
