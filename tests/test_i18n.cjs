// Run with node tests/test_i18n.cjs; no npm dependencies.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const button={textContent:'',setAttribute(){},addEventListener(_,callback){this.click=callback;}};
const storage=new Map();
const context=vm.createContext({window:{dispatchEvent(){}},document:{documentElement:{},querySelectorAll(){return [];},getElementById(){return button;}},localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},Event:class{}});
for(const path of ['i18n-catalog.js','i18n.js'])vm.runInContext(fs.readFileSync('web/static/'+path,'utf8'),context);
const run=s=>vm.runInContext(s,context);
assert.equal(run('uiT("模型管理")'),'模型管理');button.click();
assert.equal(storage.get('qwen-lab-language'),'en');assert.equal(run('uiT("模型管理")'),'Model manager');
assert.equal(run('uiTr`网站模型：${"中文模型"}`'),'Website model: 中文模型');
assert.equal(run('uiMessage("任意引擎错误原文")'),'任意引擎错误原文');
assert.equal(run('uiMessage("最多配置 32 个干预组。")'),'You can configure up to 32 intervention groups.');
for(const [key,value] of Object.entries(context.window.UI_EN)){
 assert.deepEqual((key.match(/\{\d+\}/g)||[]).sort(),(value.match(/\{\d+\}/g)||[]).sort(),key);
}
const node={isConnected:true};context.testNode=node;run('uiBind(testNode,()=>uiT("模型管理"))');assert.equal(node.textContent,'Model manager');button.click();assert.equal(node.textContent,'模型管理');
context.localStorage.setItem=()=>{throw Error('blocked')};button.click();assert.equal(run('uiT("模型管理")'),'Model manager');
const markup=fs.readFileSync('web/static/index.html','utf8');
for(const match of markup.matchAll(/data-i18n(?:-(?:placeholder|title|aria-label))?="([^"]+)"/g)){
 const key=match[1].replaceAll('&quot;','"').replaceAll('&amp;','&');assert.ok(context.window.UI_EN[key],key);
}
console.log('i18n: catalog, interpolation, fallbacks, persistence, bindings and markup passed');
// Transient UI messages switch back too; interpolation values stay literal.
run('globalThis.testMessage = uiTr`网站模型：${"中文模型"}`');
button.click();assert.equal(run('uiMessage(testMessage)'),'网站模型：中文模型');
button.click();assert.equal(run('uiMessage(testMessage)'),'Website model: 中文模型');
