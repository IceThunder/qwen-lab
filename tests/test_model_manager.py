import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'web'))
import model_manager as mm


def spec(name, raw):
    return dict(name=name,size=len(raw),hash_kind='sha256',digest=hashlib.sha256(raw).hexdigest())

def plan(raw=b'GGUF-test-data'):
    return dict(id='0123456789abcdef',repo='test/model',revision='a'*40,model_id='test-model-01234567',
        backend='llama.cpp',format='GGUF · Q4_K_M',name='Test model',files=[spec('test-Q4_K_M.gguf',raw)],total_bytes=len(raw))

def wait(manager):
    for _ in range(500):
        if not manager.snapshot()['active']:return manager.snapshot()['tasks'][0]
        time.sleep(.01)
    raise TimeoutError('test download')

class ModelManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name);self.manager=mm.ModelManager(self.root)
    def tearDown(self):
        self.manager.close();self.tmp.cleanup()

    def test_repo_and_path_constraints(self):
        self.assertEqual(mm.repo_id('https://huggingface.co/org/model/'),'org/model')
        for value in ['http://huggingface.co/org/model','https://evil.org/org/model','https://huggingface.co/org/model/tree/main','../model']:
            with self.assertRaises(ValueError):mm.repo_id(value)
        for name in ['../x.gguf','/tmp/x','sub/../../x','sub\\x','.hidden','a//x']:
            with self.assertRaises(ValueError):mm.safe_file(name)

    def test_hashes(self):
        p=self.root/'x';p.write_bytes(b'abc')
        self.assertEqual(mm.digest_file(p,spec('x',b'abc')),hashlib.sha256(b'abc').hexdigest())
        self.assertEqual(mm.digest_file(p,dict(size=3,hash_kind='git-sha1')),hashlib.sha1(b'blob 3\0abc').hexdigest())

    def test_inspect_and_existing(self):
        raw=b'GGUF';f=spec('a-Q4_K_M.gguf',raw)
        data=dict(sha='a'*40,siblings=[dict(rfilename=f['name'],size=f['size'],lfs={'sha256':f['digest']}),
            dict(rfilename='a-Q4_K_M-00001-of-00002.gguf'),dict(rfilename='mmproj.gguf')])
        with patch.object(mm,'json_get',return_value=data):d=self.manager.inspect('org/model')
        self.assertEqual(len(d['options']),1);self.assertTrue(d['options'][0]['recommended'])
        self.assertTrue(any('分片' in n for n in d['notes']))
        token=d['options'][0]['id'];p=self.manager.plans[token]
        dest=self.root/'models/existing';(dest/'weights').mkdir(parents=True);(dest/'weights'/f['name']).write_bytes(raw)
        mm.atomic_json(dest/'source.json',dict(repo_id='org/model',revision='a'*40,file=f['name']))
        self.assertEqual(self.manager.installed(p),'existing')
        with self.assertRaises(ValueError):self.manager.start(token)

    def test_success_atomic_publish_and_resume_restart(self):
        raw=b'GGUF'+b'0123456789'*10;p=plan(raw);self.manager.plans[p['id']]=p
        original_update=self.manager.update
        def update(token,**kw):
            original_update(token,**kw)
            if kw.get('downloaded',0)>0 and self.manager.active:self.manager.pause(token)
        def transfer(url,start,end,total,*args):return raw[start:end+1]
        with patch.object(mm,'CHUNK',8),patch.object(self.manager,'transfer_chunk',side_effect=transfer),patch.object(self.manager,'update',side_effect=update):
            self.manager.start(p['id']);state=wait(self.manager)
        self.assertEqual(state['status'],'paused')
        self.assertFalse((self.root/'models'/p['model_id']).exists())
        self.manager.close();self.manager=mm.ModelManager(self.root)
        with patch.object(mm,'CHUNK',8),patch.object(self.manager,'transfer_chunk',side_effect=transfer):
            self.manager.start(p['id']);state=wait(self.manager)
        self.assertEqual(state['status'],'done',state)
        dest=self.root/'models'/p['model_id']
        self.assertEqual((dest/'weights'/p['files'][0]['name']).read_bytes(),raw)
        self.assertTrue(json.loads((dest/'checksum.json').read_text())['verified'])
        self.assertIsNone(json.loads((dest/'runtime.json').read_text())['layers'])

    def test_hash_failure_resets_and_can_retry(self):
        raw=b'GGUF'+b'1'*30;p=plan(raw);self.manager.plans[p['id']]=p
        with patch.object(self.manager,'transfer_chunk',side_effect=lambda url,start,end,*_:b'X'*(end-start+1)):
            self.manager.start(p['id']);state=wait(self.manager)
        self.assertEqual(state['status'],'error');self.assertIn('校验失败',state['error'])
        self.assertFalse((self.root/'models'/p['model_id']).exists())
        with patch.object(self.manager,'transfer_chunk',side_effect=lambda url,start,end,*_:raw[start:end+1]):
            self.manager.start(p['id']);state=wait(self.manager)
        self.assertEqual(state['status'],'done',state)

    def test_disk_and_existing_directory_protection(self):
        p=plan();self.manager.plans[p['id']]=p
        with patch.object(mm.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaises(ValueError):self.manager.start(p['id'])
        (self.root/'models'/p['model_id']).mkdir()
        with self.assertRaises(ValueError):self.manager.start(p['id'])

    def test_mlx_file_set_and_unsupported(self):
        files=['config.json','tokenizer.json','tokenizer_config.json','model.safetensors']
        data=dict(sha='a'*40,siblings=[dict(rfilename=n,size=3,blobId='a'*40) for n in files])
        config=dict(model_type='qwen3_5',quantization=dict(bits=4),text_config=dict(num_hidden_layers=32,intermediate_size=9216))
        with patch.object(mm,'json_get',side_effect=[data,config]):d=self.manager.inspect('org/mlx')
        self.assertEqual(len(d['options']),1);self.assertEqual(d['options'][0]['file_count'],4)
        config['model_type']='other'
        with patch.object(mm,'json_get',side_effect=[data,config]):d=self.manager.inspect('org/mlx')
        self.assertEqual(d['options'],[]);self.assertTrue(d['notes'])

    def test_cross_service_download_lock(self):
        p=plan();self.manager.plans[p['id']]=p
        other=mm.ModelManager(self.root)
        try:
            mm.fcntl.flock(other.download_lock,mm.fcntl.LOCK_EX|mm.fcntl.LOCK_NB)
            with self.assertRaises(RuntimeError):self.manager.start(p['id'])
        finally:
            mm.fcntl.flock(other.download_lock,mm.fcntl.LOCK_UN);other.close()

    def test_recover_publish_before_final_state(self):
        p=plan();task=self.root/'work/model-downloads'/p['id'];task.mkdir()
        mm.atomic_json(task/'state.json',dict(id=p['id'],model_id=p['model_id'],status='downloading',total=p['total_bytes']))
        mm.atomic_json(self.root/'models'/p['model_id']/'installation.json',dict(complete=True,plan_id=p['id']))
        other=mm.ModelManager(self.root)
        try:self.assertEqual(other.snapshot()['tasks'][0]['status'],'done')
        finally:other.close()

if __name__=='__main__':unittest.main()
