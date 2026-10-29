"""Public Hugging Face installs: pinned manifests, resumable transfers, atomic publish."""
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote, urlsplit
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 1024 ** 3
CHUNK = 8 * 1024 ** 2


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def repo_id(value):
    if not isinstance(value, str) or len(value) > 300:
        raise ValueError('请输入 Hugging Face 仓库链接或 owner/repo')
    value = value.strip().rstrip('/')
    if '://' in value:
        u = urlsplit(value)
        if u.scheme != 'https' or u.netloc != 'huggingface.co' or u.query or u.fragment:
            raise ValueError('仅支持 https://huggingface.co/owner/repo 仓库链接')
        value = u.path.lstrip('/')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('请粘贴仓库首页链接（不含 tree、blob 或文件路径）')
    return value


def safe_file(name):
    if not isinstance(name, str) or len(name) > 300 or '\\' in name or any(ord(c)<32 for c in name):
        raise ValueError('仓库包含无效文件路径')
    p = PurePosixPath(name)
    if p.is_absolute() or any(x in ('', '.', '..') or x.startswith('.') for x in name.split('/')):
        raise ValueError('仓库包含不安全文件路径')
    return name


def json_get(url, limit=8*1024*1024):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'QwenLab/1.0'}), timeout=20) as r:
            raw = r.read(limit+1)
        if len(raw) > limit:
            raise ValueError('仓库元数据过大')
        return json.loads(raw)
    except urllib.error.HTTPError as e:
        if e.code in (401,403):
            raise ValueError('第一版仅支持无需登录的公开仓库；私有或需要授权的模型暂不支持') from e
        if e.code == 404:
            raise ValueError('未找到仓库或必要的模型配置文件') from e
        raise ValueError(f'Hugging Face 返回 HTTP {e.code}，请稍后重试') from e
    except (OSError,UnicodeError,json.JSONDecodeError) as e:
        raise ValueError('读取 Hugging Face 失败，请检查网络后重试') from e


def resolve_url(repo, revision, filename):
    return f'https://huggingface.co/{repo}/resolve/{revision}/{quote(filename, safe="/")}'


def file_spec(item):
    name = safe_file(item['rfilename'])
    size = item.get('size')
    if type(size) is not int or size <= 0:
        raise ValueError(f'{name} 缺少可靠文件大小')
    lfs = item.get('lfs')
    kind = 'sha256' if lfs else 'git-sha1'
    digest = lfs.get('sha256') if lfs else item.get('blobId')
    if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}' if lfs else '[0-9a-f]{40}',digest):
        raise ValueError(f'{name} 缺少完整性校验值')
    return dict(name=name,size=size,hash_kind=kind,digest=digest)


def quant_label(name):
    m = re.search(r'(?:^|[-_.])(IQ\d[^.]*|Q\d[^.]*|BF16|F16|F32)(?:\.gguf)$',name,re.I)
    return m.group(1).upper() if m else '量化格式未标注'


def digest_file(path, spec, check=lambda:None):
    h = hashlib.sha256() if spec['hash_kind']=='sha256' else hashlib.sha1()
    if spec['hash_kind']=='git-sha1':
        h.update(f"blob {spec['size']}\0".encode())
    with path.open('rb') as f:
        while data := f.read(8*1024*1024):
            check();h.update(data)
    return h.hexdigest()


class Paused(Exception):
    pass


class ModelManager:
    def __init__(self, root=ROOT):
        self.root=Path(root)
        self.work=self.root/'work/model-downloads';self.work.mkdir(parents=True,exist_ok=True)
        (self.root/'models').mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='model-download')
        self.plans={};self.tasks={};self.active=None;self.cancel_event=threading.Event()
        self.download_lock=(self.work/'.download.lock').open('a')
        for folder in self.work.iterdir():
            try:
                state=json.loads((folder/'state.json').read_text())
                if not re.fullmatch('[0-9a-f]{16}',folder.name) or state['id']!=folder.name:continue
                if state['status'] in ('queued','downloading','verifying'):
                    state.update(status='paused',phase='服务重启后暂停，可继续下载')
                # Recover a crash between atomic directory publish and the final state write.
                destination=self.root/'models'/state['model_id']
                if (destination/'installation.json').is_file():
                    installed=json.loads((destination/'installation.json').read_text())
                    if installed.get('plan_id')==state['id'] and installed.get('complete'):
                        state.update(status='done',phase='下载校验完成；请测试加载',downloaded=state['total'],error='')
                self.tasks[folder.name]=state
            except (OSError,ValueError,KeyError):pass

    def installed(self, plan):
        for folder in (self.root/'models').iterdir():
            try:
                source=json.loads((folder/'source.json').read_text())
                if source.get('repo_id')!=plan['repo'] or source.get('revision')!=plan['revision']:continue
                if plan['backend']=='llama.cpp' and source.get('file')!=plan['files'][0]['name']:continue
                if all((folder/'weights'/f['name']).is_file() and (folder/'weights'/f['name']).stat().st_size==f['size'] for f in plan['files']):
                    return folder.name
            except (OSError,ValueError):pass
        return None

    def inspect(self, value):
        repo=repo_id(value)
        metadata=json_get(f'https://huggingface.co/api/models/{repo}?blobs=true')
        if not isinstance(metadata,dict):raise ValueError('仓库元数据格式无效')
        revision=metadata.get('sha','')
        if not isinstance(revision,str) or not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('无法固定仓库版本')
        if metadata.get('private') or metadata.get('gated'):
            raise ValueError('第一版仅支持无需登录或申请授权的公开模型')
        siblings=metadata.get('siblings',[])
        if not isinstance(siblings,list) or any(not isinstance(x,dict) for x in siblings):raise ValueError('仓库文件列表无效')
        items={x.get('rfilename'):x for x in siblings}
        candidates=[];notes=[]
        for item in siblings:
            name=item.get('rfilename','')
            if not name.lower().endswith('.gguf') or 'mmproj' in name.lower():continue
            if re.search(r'-\d{5}-of-\d{5}',name):
                notes.append('暂不支持分片 GGUF；请选择单文件 GGUF。');continue
            try:f=file_spec(item)
            except ValueError as e:notes.append(str(e));continue
            candidates.append(dict(backend='llama.cpp',label=name,format='GGUF · '+quant_label(name),files=[f],
                capabilities='文本对话；不支持神经元干预或 LoRA；是否支持该架构须测试加载',layers=None,width=0))
        if 'config.json' in items and any(str(n).endswith('.safetensors') for n in items):
            config=json_get(resolve_url(repo,revision,'config.json'),1024*1024)
            if not isinstance(config,dict):raise ValueError('模型配置格式无效')
            text=config.get('text_config',config);q=config.get('quantization',{})
            if not isinstance(text,dict):raise ValueError('模型文本配置格式无效')
            if config.get('model_type') not in ('qwen3_5','qwen3_5_text') or not isinstance(q,dict) or q.get('bits') not in (2,3,4,5,6,8):
                notes.append('safetensors 仓库未匹配已适配的 Qwen3.5 MLX 量化配置；不会当作 MLX 自动安装。')
            elif any(type(text.get(k)) is not int or text[k]<=0 for k in ('num_hidden_layers','intermediate_size')):
                notes.append('MLX 配置缺少有效层数或通道数。')
            else:
                try:
                    if 'model.safetensors.index.json' in items:
                        index=json_get(resolve_url(repo,revision,'model.safetensors.index.json'))
                        if not isinstance(index,dict) or not isinstance(index.get('weight_map'),dict):raise ValueError('无效权重分片索引')
                        names=set(index['weight_map'].values())
                        if not names or any(not n.endswith('.safetensors') for n in names):raise ValueError('无效权重分片索引')
                        names.add('model.safetensors.index.json')
                    elif 'model.safetensors' in items:names={'model.safetensors'}
                    else:raise ValueError('未找到标准 MLX 权重文件或分片索引')
                    if 'tokenizer.json' not in items or 'tokenizer_config.json' not in items:
                        raise ValueError('缺少 tokenizer.json 或 tokenizer_config.json')
                    names.update(n for n in ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                        'added_tokens.json','generation_config.json','chat_template.jinja','merges.txt','vocab.json','tokenizer.model') if n in items)
                    files=[file_spec(items[n]) for n in sorted(names)]
                    candidates.append(dict(backend='mlx',label=f'MLX 全套文件 · {q["bits"]}-bit',format=f'MLX · {q["bits"]}-bit',files=files,
                        layers=text['num_hidden_layers'],width=text['intermediate_size'],
                        capabilities='Qwen3.5 文本推理及 MLP 干预接口；下载后须测试加载，Neuronary 模型映射不自动扩展'))
                except (ValueError,KeyError,TypeError) as e:notes.append('MLX 文件集不完整：'+str(e))
        free=shutil.disk_usage(self.root).free
        try:ram=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True).strip())
        except (OSError,ValueError,subprocess.SubprocessError):ram=16*1024**3
        result=[]
        for candidate in candidates:
            token=hashlib.sha256(json.dumps([repo,revision,candidate['backend'],candidate['files']],sort_keys=True).encode()).hexdigest()[:16]
            slug=re.sub('[^a-z0-9-]+','-',repo.split('/')[-1].lower()).strip('-')[:55]
            model_id=f'{slug}-{token[:8]}'
            if repo == 'mlx-community/Qwen3.5-4B-4bit' and candidate['backend'] == 'mlx' and not (self.root/'models/qwen3.5-4b').exists():
                model_id='qwen3.5-4b'
            plan=dict(candidate,id=token,repo=repo,revision=revision,model_id=model_id)
            plan['total_bytes']=sum(f['size'] for f in plan['files'])
            plan['name']=repo.split('/')[-1]+' · '+candidate['format']
            with self.lock:
                if len(self.plans)>=200:self.plans.pop(next(iter(self.plans)))
                self.plans[token]=plan
            installed=self.installed(plan)
            result.append(dict(id=token,label=candidate['label'],format=candidate['format'],backend=candidate['backend'],
                size=plan['total_bytes'],file_count=len(plan['files']),capabilities=candidate['capabilities'],installed_model=installed,
                disk_ok=free>=plan['total_bytes']+RESERVE,
                memory_note='权重本身已超过统一内存的 60%，加载或长上下文可能吃紧。' if plan['total_bytes']>ram*.6 else '仅按文件体积估计；实际内存还包含缓存及运行开销。',
                recommended=False))
        eligible=[o for o in result if o['disk_ok'] and o['size']<ram*.6]
        if eligible:
            recommended=min(eligible,key=lambda o:(0 if 'Q4_K_M' in o['format'] else 1 if '4-bit' in o['format'] else 2,o['size']))
            recommended['recommended']=True
        result.sort(key=lambda x:(not x['recommended'],x['size']))
        return dict(repo=repo,revision=revision,options=result,notes=sorted(set(notes)),free_bytes=free,ram_bytes=ram,reserve_bytes=RESERVE,
                    limitation='仅公开仓库；GGUF 单文件文本模型、已适配 Qwen3.5 MLX 量化仓库。不下载或执行远程 Python 代码。')

    def snapshot(self):
        with self.lock:return dict(active=self.active,tasks=copy.deepcopy(list(self.tasks.values())),free_bytes=shutil.disk_usage(self.root).free)

    def update(self, token, **values):
        with self.lock:
            self.tasks[token].update(values)
            atomic_json(self.work/token/'state.json',self.tasks[token])

    def start(self, token):
        if not isinstance(token,str) or not re.fullmatch('[0-9a-f]{16}',token):raise ValueError('无效下载任务')
        with self.lock:
            if self.active:raise RuntimeError('已有模型下载进行中，请先暂停或等待完成')
            try:
                fcntl.flock(self.download_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('另一个 Web 服务正在下载，请使用原服务等待完成或暂停')
            try:
                folder=self.work/token
                if token in self.tasks and (folder/'plan.json').is_file():plan=json.loads((folder/'plan.json').read_text())
                elif token in self.plans:plan=self.plans[token]
                else:raise ValueError('仓库预览已过期，请重新读取仓库')
                installed=self.installed(plan)
                if installed:raise ValueError(f'该版本已经安装：{installed}，无需重复下载')
                destination=self.root/'models'/plan['model_id']
                if destination.exists():raise ValueError('目标模型目录已存在，不会覆盖；请检查目录或重新读取仓库')
                received=self.received(plan)
                if shutil.disk_usage(self.root).free<plan['total_bytes']-received+RESERVE:
                    raise ValueError('磁盘空间不足：需要剩余下载空间，并保留至少 1 GiB 余量')
                folder.mkdir(parents=True,exist_ok=True);atomic_json(folder/'plan.json',plan)
                self.tasks[token]=dict(id=token,model_id=plan['model_id'],name=plan['name'],repo=plan['repo'],revision=plan['revision'],
                    status='queued',phase='准备下载',downloaded=received,total=plan['total_bytes'],error='')
                self.active=token;self.cancel_event.clear();self.update(token)
                self.pool.submit(self._run,plan)
            except BaseException:
                fcntl.flock(self.download_lock,fcntl.LOCK_UN)
                raise
        return token

    def pause(self, token):
        with self.lock:
            if token!=self.active:raise ValueError('该任务当前未运行')
            self.cancel_event.set()

    def check(self):
        if self.cancel_event.is_set():raise Paused()

    def received(self, plan):
        count=0
        for spec in plan['files']:
            path=self.work/plan['id']/'package/weights'/spec['name']
            if path.exists() and path.stat().st_size==spec['size']:count+=spec['size'];continue
            partial=path.with_name(path.name+'.partial');state=path.with_name(path.name+'.ranges.json')
            if partial.exists() and state.exists():
                try:
                    chunks=json.loads(state.read_text())
                    count+=sum(min(CHUNK,spec['size']-int(i)*CHUNK) for i in chunks if 0<=int(i)*CHUNK<spec['size'])
                except (ValueError,TypeError):pass
        return count

    def transfer_chunk(self, url, start, end, total, tmp, header):
        for attempt in range(3):
            self.check()
            proc=subprocess.Popen(['curl','--proto','=https','--proto-redir','=https','-fLsS','--connect-timeout','15','--max-time','90',
                '--max-filesize',str(end-start+1),'--range',f'{start}-{end}','-D',str(header),'-o',str(tmp),
                url+f'?download=true&part={start//CHUNK}&retry={time.time_ns()}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                while proc.poll() is None:
                    if self.cancel_event.wait(.1):raise Paused()
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
            self.check()
            headers=header.read_text(errors='replace').lower() if header.exists() else ''
            if proc.returncode==0 and tmp.exists() and tmp.stat().st_size==end-start+1 and (
                f'content-range: bytes {start}-{end}/{total}' in headers or (start==0 and end+1==total)):
                return tmp.read_bytes()
            if attempt<2:
                if self.cancel_event.wait(attempt+1):raise Paused()
        statuses=re.findall(r'^http/[^ ]+ (\d+)',headers,re.M)
        raise RuntimeError(f'下载分块失败（curl {proc.returncode}，HTTP {statuses[-1] if statuses else "未知"}）；已保留进度，可继续重试')

    def download_file(self, plan, spec):
        self.check();folder=self.work/plan['id'];path=folder/'package/weights'/spec['name'];path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if path.stat().st_size==spec['size'] and digest_file(path,spec,self.check)==spec['digest']:return
            raise ValueError('已完成文件校验失败：'+spec['name'])
        partial=path.with_name(path.name+'.partial');state=path.with_name(path.name+'.ranges.json')
        done=json.loads(state.read_text()) if partial.exists() and state.exists() else []
        total_chunks=math.ceil(spec['size']/CHUNK)
        if not isinstance(done,list) or any(type(i)is not int or not 0<=i<total_chunks for i in done):raise ValueError('续传记录损坏')
        done=set(done);fd=os.open(partial,os.O_RDWR|os.O_CREAT,0o644);os.ftruncate(fd,spec['size'])
        ranges=folder/'ranges';ranges.mkdir(exist_ok=True)
        def chunk(i):
            start=i*CHUNK;end=min(spec['size'],start+CHUNK)-1
            tmp=ranges/f'{i}.part';header=ranges/f'{i}.headers'
            try:
                raw=self.transfer_chunk(resolve_url(plan['repo'],plan['revision'],spec['name']),start,end,spec['size'],tmp,header)
                offset=start
                while raw:
                    n=os.pwrite(fd,raw,offset);raw=raw[n:];offset+=n
                return i
            finally:
                tmp.unlink(missing_ok=True);header.unlink(missing_ok=True)
        try:
            # A bounded number of transfers; task state is persisted after each finished chunk.
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures=[pool.submit(chunk,i) for i in range(total_chunks) if i not in done]
                try:
                    for future in as_completed(futures):
                        i=future.result();done.add(i);atomic_json(state,sorted(done))
                        self.update(plan['id'],downloaded=self.received(plan),phase='下载 '+spec['name'])
                except BaseException:
                    self.cancel_event.set()
                    for future in futures:future.cancel()
                    raise
            os.fsync(fd)
        finally:os.close(fd)
        self.check();self.update(plan['id'],status='verifying',phase='完整性校验 '+spec['name'])
        if digest_file(partial,spec,self.check)!=spec['digest']:
            # Reset only this task-owned partial file so retry does not reuse corrupt chunks.
            partial.unlink();state.unlink(missing_ok=True)
            raise ValueError('完整文件校验失败，已重置此文件；请继续下载重试')
        partial.replace(path);state.unlink(missing_ok=True)
        self.update(plan['id'],status='downloading')

    def _run(self, plan):
        token=plan['id']
        try:
            self.update(token,status='downloading',phase='开始下载')
            for spec in plan['files']:self.download_file(plan,spec)
            self.check();package=self.work/token/'package'
            source=dict(repo_id=plan['repo'],revision=plan['revision'],format=plan['format'],files=plan['files'])
            if plan['backend']=='llama.cpp':source['file']=plan['files'][0]['name']
            atomic_json(package/'source.json',source)
            atomic_json(package/'checksum.json',dict(verified=True,files=plan['files']))
            atomic_json(package/'installation.json',dict(complete=True,managed=True,plan_id=token,name=plan['name'],backend=plan['backend']))
            if plan['backend']=='llama.cpp':
                f=plan['files'][0]
                atomic_json(package/'runtime.json',dict(file=f['name'],size=f['size'],name=plan['name'],format=plan['format'],
                    layers=None,backend='llama.cpp',supports_steering=False))
            self.check();package.rename(self.root/'models'/plan['model_id'])
            self.update(token,status='done',phase='下载校验完成；请测试加载',downloaded=plan['total_bytes'])
        except Paused:
            self.update(token,status='paused',phase='已暂停，继续时复用已完成分块')
        except Exception as e:
            self.update(token,status='error',phase='下载失败，进度保留',error=str(e))
        finally:
            with self.lock:
                self.active=None
                fcntl.flock(self.download_lock,fcntl.LOCK_UN)

    def close(self):
        self.cancel_event.set();self.pool.shutdown(wait=True)
        if not self.download_lock.closed:self.download_lock.close()
