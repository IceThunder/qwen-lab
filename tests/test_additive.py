import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'web'))
import mlx.core as mx
import mlx.nn as nn
from backend import validate_request, apply_groups
from steering import column_norms, calibrate

class Layer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor=factor
        self.is_linear=True
        self.mlp=nn.Module()
        self.mlp.down_proj=nn.Linear(4,2,bias=False)
        self.mlp.down_proj.weight=mx.array([[3.,0.,1.,1.],[4.,2.,1.,1.]])
    def __call__(self,x,*args,**kwargs):return x*self.factor

class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.layers=[Layer(1),Layer(2)]
    def __call__(self,tokens,cache=None):
        x=tokens.astype(mx.float32)[...,None]*mx.array([3.,4.])
        for layer in self.layers:x=layer(x)
        return x

class Model(nn.Module):
    def __init__(self):super().__init__();self.model=Backbone()
    @property
    def layers(self):return self.model.layers[:]
    def make_cache(self):return []

class AdditiveTests(unittest.TestCase):
    def test_quantized_column_orientation(self):
        linear=nn.Linear(64,3,bias=False)
        linear.weight=mx.arange(192,dtype=mx.float32).reshape(3,64)/100
        q=nn.QuantizedLinear.from_linear(linear,group_size=64,bits=4)
        ids=[1,23,63]
        norms=column_norms(q,ids)
        for n,got in zip(ids,norms):
            x=mx.eye(64)[n:n+1]
            expected=mx.linalg.norm(q(x)).item()
            self.assertAlmostEqual(got,expected,places=5)

    def test_lora_effective_columns(self):
        from mlx_lm.tuner.lora import LoRALinear
        p=LoRALinear(4,2,r=1,scale=2)
        p.eval();p.linear.weight=mx.ones((2,4));p.lora_a=mx.ones((4,1));p.lora_b=mx.array([[1.,2.]])
        self.assertAlmostEqual(column_norms(p,[0])[0],mx.linalg.norm(p(mx.array([[1.,0.,0.,0.]]))).item(),places=5)

    def test_calibration_global_skip_and_chunk_invariance(self):
        m=Model();old=list(m.layers)
        groups=[{'layer':0,'neurons':[0],'mode':'normalized_add','dose':1},
                {'layer':1,'neurons':[1],'mode':'normalized_add','dose':-1}]
        a,report=calibrate(m,[1,2,3,4],groups,chunk_size=1)
        b,report2=calibrate(m,[1,2,3,4],groups,chunk_size=3)
        self.assertEqual(report,report2)
        self.assertAlmostEqual(a[0]['offsets'][0],3.,places=5);self.assertAlmostEqual(a[1]['offsets'][0],-15.,places=5)
        self.assertEqual(report['calibration_tokens'],3)
        self.assertAlmostEqual(report['rows'][0]['residual_median'],15.,places=5)
        for i in range(2):self.assertIs(m.layers[i],old[i])
        _,all_tokens=calibrate(m,[1,2,3,4],groups,skip_first=False)
        self.assertAlmostEqual(all_tokens['rows'][0]['residual_median'],12.5,places=5)

    def test_cancel_restores_calibration_hooks(self):
        m=Model();old=list(m.layers);calls=[0]
        def check():
            calls[0]+=1
            if calls[0]>=2:raise RuntimeError('cancel')
        with self.assertRaises(RuntimeError):calibrate(m,[1,2,3],[{'layer':0,'neurons':[0],'mode':'normalized_add','dose':1}],check=check)
        for i in range(2):self.assertIs(m.layers[i],old[i])

    def test_addition_sign_mixed_modes_and_restore(self):
        m=Model();original=m.layers[0].mlp.down_proj;audit=[]
        groups=[{'layer':0,'neurons':[0],'mode':'normalized_add','dose':-1,'offsets':[-3.]},
                {'layer':0,'neurons':[1],'scale':0}]
        with apply_groups(m,groups,4,audit=audit):
            output=m.layers[0].mlp.down_proj(mx.array([[[1.,2.,3.,4.]]]))
            expected=original(mx.array([[[-2.,0.,3.,4.]]]))
            self.assertTrue(mx.allclose(output,expected).item())
        self.assertIs(m.layers[0].mlp.down_proj,original)
        self.assertEqual(audit[0]['actual_delta'],-3.)
        with self.assertRaises(ValueError):
            with apply_groups(m,[{'layer':0,'neurons':[0],'mode':'normalized_add','dose':1}],4):pass

    def test_zero_dose_identity(self):
        m=Model();groups=[{'layer':0,'neurons':[0],'mode':'normalized_add','dose':0}]
        resolved,_=calibrate(m,[1,2,3],groups)
        x=mx.ones((1,2,4));before=m.layers[0].mlp.down_proj(x)
        with apply_groups(m,resolved,4):after=m.layers[0].mlp.down_proj(x)
        self.assertTrue(mx.array_equal(before,after).item())

    def test_validation_and_legacy_scale(self):
        catalog=[{'id':'test','layers':2,'width':256,'adapters':[]}]
        req={'model':'test','messages':[{'role':'user','content':'hi'}],
             'groups':[{'layer':0,'neurons':'1','mode':'normalized_add','dose':-1}]}
        out=validate_request(req,catalog);self.assertEqual(out['groups'][0]['dose'],-1)
        self.assertTrue(out['calibration_skip_first'])
        req['groups'][0]['neurons']='0-128'
        with self.assertRaises(ValueError):validate_request(req,catalog)
        req['groups']=[{'layer':0,'neurons':'1','scale':1.8}]
        self.assertEqual(validate_request(req,catalog)['groups'][0]['mode'],'scale')

if __name__=='__main__':unittest.main()
