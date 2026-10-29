'use strict';
// Local UI translation only. Prompts, replies, names and experiment JSON stay intact.
let uiLanguage = 'zh';
try { if (localStorage.getItem('qwen-lab-language') === 'en') uiLanguage = 'en'; } catch (_) {}
const uiRendered = new Map();
function uiRemember(text, key, values = []) {
  if (uiRendered.size > 1000) uiRendered.delete(uiRendered.keys().next().value);
  uiRendered.set(text, {key, values}); return text;
}
function uiT(source) { return uiRemember( uiLanguage === 'en' ? (window.UI_EN[source] ?? source) : source, source); }
function uiTr(parts, ...values) {
  const key = parts.reduce((s, p, i) => s + p + (i < values.length ? `{${i}}` : ''), '');
  return uiRemember(uiT(key).replace(/\{(\d+)\}/g, (_, i) => String(values[Number(i)])), key, values);
}
const uiPatterns = Object.entries(window.UI_EN).filter(([key]) => /\{\d+\}/.test(key)).map(([key, value]) => {
  const ids = []; const pattern = key.split(/(\{\d+\})/).map(p => {
    if (/^\{\d+\}$/.test(p)) { ids.push(p); return '([\\s\\S]*?)'; }
    return p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }).join('');
  return {key, re: new RegExp('^' + pattern + '$'), ids, value};
});
function uiMessage(source, depth = 0) {
  if (typeof source !== 'string') return source;
  const remembered = uiRendered.get(source);
  if (remembered && depth < 4) return uiT(remembered.key).replace(/\{(\d+)\}/g, (_, i) => String(uiMessage(remembered.values[Number(i)], depth + 1)));
  if (uiLanguage !== 'en') return source;
  if (Object.hasOwn(window.UI_EN, source)) return uiT(source);
  for (const {key, re, ids, value} of uiPatterns) {
    const match = source.match(re);
    if (match) {
      const values = []; ids.forEach((id, i) => values[Number(id.slice(1,-1))] = match[i+1]);
      return uiRemember(value.replace(/\{\d+\}/g, id => match[ids.indexOf(id) + 1]), key, values);
    }
  }
  return source; // Unknown engine/server errors remain available verbatim.
}
const uiBindings = new Map();
function uiBind(node, render, property = 'textContent') {
  if (uiBindings.size > 1500) for (const n of uiBindings.keys()) if (!n.isConnected) uiBindings.delete(n);
  if (!uiBindings.has(node)) uiBindings.set(node, new Map());
  uiBindings.get(node).set(property, render);
  node[property] = render();
}
function uiStatic(root = document) {
  root.querySelectorAll('[data-i18n]').forEach(node => node.textContent = uiT(node.dataset.i18n));
  for (const attr of ['placeholder','title','aria-label']) {
    root.querySelectorAll(`[data-i18n-${attr}]`).forEach(node => node.setAttribute(attr, uiT(node.getAttribute(`data-i18n-${attr}`))));
  }
}
function uiRefresh() {
  document.documentElement.lang = uiLanguage === 'en' ? 'en' : 'zh-CN';
  uiStatic();
  for (const [node, properties] of uiBindings) {
    if (!node.isConnected) { uiBindings.delete(node); continue; }
    for (const [property, render] of properties) node[property] = render();
  }
  const button = document.getElementById('languageToggle');
  button.textContent = uiLanguage === 'en' ? '中文' : 'English';
  button.setAttribute('aria-label', uiLanguage === 'en' ? '切换为中文界面' : 'Switch to English');
}
document.getElementById('languageToggle').addEventListener('click', () => {
  uiLanguage = uiLanguage === 'en' ? 'zh' : 'en';
  try { localStorage.setItem('qwen-lab-language', uiLanguage); } catch (_) {}
  uiRefresh();
  window.dispatchEvent(new Event('uilanguagechange'));
});
uiRefresh();
