'use strict';
// Uses the main application's model list and job lock; no chat history is changed.
let repositoryPreview=null, managerState=null, managerTimer=null, modelTestId=null, repositoryReading=false;
const bytes=n=>`${(Number(n)/1024**3).toFixed(2)} GiB`;
function managerMessage(message=''){ uiBind($('managerMessage'),()=>uiMessage(message),'textContent'); }
async function refreshInstalledModels(){
  const d=await api('/api/models');const selected=$('model').value,adapter=$('adapter').value;
  models=d.models;$('model').replaceChildren(...models.map(m=>new Option(m.name,m.id)));
  if(models.some(m=>m.id===selected))$('model').value=selected;
  fillAdapters(adapter);setBusy(busy);renderInstalledModels();
}
function renderInstalledModels(){
  $('installedModels').replaceChildren();
  for(const m of models){
    const row=el('div','installed-model');const info=el('div');info.append(el('strong','',m.name),el('p','hint',m.format));
    if(m.last_load_test) info.append(el('p','hint',uiTr`上次测试：${{done:uiT('短文本推理通过'),error:uiT('失败'),cancelled:uiT('已取消')}[m.last_load_test.status]||m.last_load_test.status}${m.last_load_test.error?' · '+uiMessage(m.last_load_test.error):''}`));
    const test=el('button','',uiT('测试加载'));test.disabled=busy;test.addEventListener('click',()=>testInstalledModel(m.id,m.name));
    row.append(info,test);$('installedModels').append(row);
  }
}
function renderVariant(){
  const o=repositoryPreview?.options.find(o=>o.id===$('modelVariant').value);
  $('modelInstall').disabled=!o||Boolean(o.installed_model)||!o.disk_ok||Boolean(managerState?.active);
  uiBind($('variantInfo'),()=>o?uiTr`${o.format} · ${bytes(o.size)} · ${o.file_count} 个文件\n${uiMessage(o.capabilities)}\n${uiMessage(o.memory_note)}${o.installed_model?uiT('\n该版本已安装，可在下方测试加载。'):!o.disk_ok?uiT('\n当前磁盘余量不足，无法新下载。'):''}`:'','textContent');
}
function renderDownloads(){
  $('downloadTasks').replaceChildren();
  for(const t of [...(managerState?.tasks||[])].reverse()){
    const card=el('div','download-task');card.append(el('strong','',t.name));
    const progress=document.createElement('progress');progress.max=t.total;progress.value=t.downloaded;
    progress.setAttribute('aria-label',uiTr`${t.name} 下载进度`);
    card.append(progress,el('p','hint',`${uiMessage(t.phase)} · ${bytes(t.downloaded)} / ${bytes(t.total)}${t.error?'\n'+uiMessage(t.error):''}`));
    const action=el('button','',t.status==='done'?uiT('测试加载'):managerState.active===t.id?uiT('暂停下载'):uiT('继续下载'));
    if(t.status==='done'){
      action.disabled=busy;action.addEventListener('click',()=>testInstalledModel(t.model_id,t.name));
    }else if(managerState.active===t.id){
      action.addEventListener('click',async()=>{try{await api('/api/model-manager/pause',{id:t.id});managerMessage(uiT('正在暂停，已完成分块将保留。'));}catch(e){managerMessage(e.message);}});
    }else{
      action.disabled=Boolean(managerState.active);action.addEventListener('click',()=>installVariant(t.id));
    }
    card.append(action);$('downloadTasks').append(card);
  }
}
async function pollModelManager(){
  try{
    const previous=JSON.stringify(managerState?.tasks?.map(t=>[t.id,t.status]));
    managerState=await api('/api/model-manager');renderDownloads();renderVariant();
    uiBind($('managerDisk'),()=>uiTr`磁盘可用 ${bytes(managerState.free_bytes)}；安装至少保留 1 GiB 余量。`,'textContent');
    if(previous!==JSON.stringify(managerState.tasks.map(t=>[t.id,t.status]))&&managerState.tasks.some(t=>t.status==='done'))await refreshInstalledModels();
    if(!$('modelManagerDialog').open&&!managerState.active){clearInterval(managerTimer);managerTimer=null;}
  }catch(e){managerMessage(uiTr`状态读取失败：${e.message}`);}
}
function ensureManagerPolling(){if(!managerTimer)managerTimer=setInterval(pollModelManager,1500);}
$('modelManagerOpen').addEventListener('click',async()=>{
  $('modelManagerDialog').showModal();renderInstalledModels();await pollModelManager();ensureManagerPolling();
});
$('modelManagerClose').addEventListener('click',()=>$('modelManagerDialog').close());
$('modelRepo').addEventListener('input',()=>{repositoryPreview=null;$('repositoryPreview').hidden=true;});
$('modelInspect').addEventListener('click',async()=>{
  if(repositoryReading)return;repositoryReading=true;$('modelInspect').disabled=true;
  const input=$('modelRepo').value.trim();repositoryPreview=null;$('repositoryPreview').hidden=true;managerMessage(uiT('正在读取仓库元数据和配置…'));
  try{
    const d=await api('/api/model-manager/inspect',{url:input});
    if(input!==$('modelRepo').value.trim())return;
    repositoryPreview=d;
    renderRepositoryPreview();
    managerMessage(d.options.length?uiT('请选择版本后下载安装。下载不会自动加载模型。'):uiT('未找到第一版支持的模型文件；详见下方原因。'));
  }catch(e){managerMessage(uiTr`读取失败：${e.message}`);}finally{repositoryReading=false;$('modelInspect').disabled=false;}
});
$('modelVariant').addEventListener('change',renderVariant);
async function installVariant(id){
  try{await api('/api/model-manager/install',{id});managerMessage(uiT('下载已开始。可关闭此面板；关闭 Web 服务会暂停下载，下次可继续。'));await pollModelManager();ensureManagerPolling();}
  catch(e){managerMessage(uiTr`无法开始：${e.message}`);}
}
$('modelInstall').addEventListener('click',()=>installVariant($('modelVariant').value));
async function testInstalledModel(id,name){
  if(busy||modelTestId)return managerMessage(uiT('请等待当前推理完成后再测试加载。'));
  setBusy(true);renderInstalledModels();renderDownloads();uiBind($('modelTestOutput'),()=>uiTr`正在测试 ${name}…\n执行一次无干预短文本推理；不修改当前对话。`,'textContent');
  try{
    const started=await api('/api/models/test-load',{model:id});modelTestId=started.id;activeJob=started.id;
    $('stopModelTest').hidden=false;
    for(;;){
      const j=await api(`/api/jobs/${started.id}`);
      uiBind($('modelTestOutput'),()=>`${name}\n${uiMessage(j.phase)}\n${j.results?.intervened?.text||''}${j.error?'\n'+uiMessage(j.error):''}${j.status==='done'?'\n'+uiT('\n短文本推理通过；未验证长上下文、干预或微调。').trimStart():''}`,'textContent');
      if(['done','error','cancelled'].includes(j.status)){
        break;
      }
      await new Promise(resolve=>setTimeout(resolve,500));
    }
  }catch(e){
    uiBind($('modelTestOutput'),()=>uiTr`测试失败：${e.message}`,'textContent');
    if(modelTestId)await api(`/api/jobs/${modelTestId}/cancel`,{}).catch(()=>{});
  }finally{modelTestId=null;activeJob=null;$('stopModelTest').hidden=true;setBusy(false);await refreshInstalledModels().catch(()=>{});renderInstalledModels();renderDownloads();}
}
$('stopModelTest').addEventListener('click',async()=>{if(modelTestId){try{await api(`/api/jobs/${modelTestId}/cancel`,{});}catch(e){managerMessage(e.message);}}});

function renderRepositoryPreview(preserve=false){
if(!repositoryPreview)return;const d=repositoryPreview,selected=$('modelVariant').value;
    uiBind($('repositoryInfo'),()=>uiTr`${d.repo}\n固定版本 ${d.revision}\n统一内存 ${bytes(d.ram_bytes)}；磁盘可用 ${bytes(d.free_bytes)}\n${uiMessage(d.limitation)}`,'textContent');
    $('modelVariant').replaceChildren(...d.options.map(o=>new Option(`${o.recommended?uiT('推荐 · '):''}${o.installed_model?uiT('已安装 · '):''}${uiMessage(o.label)} — ${bytes(o.size)}`,o.id)));
    uiBind($('repositoryNotes'),()=>d.notes.map(note=>uiMessage(note)).join('\n'),'textContent');$('repositoryPreview').hidden=false;renderVariant();

if(preserve){$('modelVariant').value=selected;renderVariant();}
}
window.addEventListener('uilanguagechange',()=>{renderInstalledModels();renderDownloads();renderRepositoryPreview(true);renderVariant();});
