import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'web'))
from backend import parse_neurons, validate_request, apply_groups

CATALOG = [{'id':'test', 'layers':2, 'width':4, 'adapters':['demo']}]

def request(**kw):
    return dict(model='test', messages=[{'role':'user','content':'hello'}], **kw)


class WebValidationTests(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(parse_neurons('0，1-3,2',4),[0,1,2,3])
        for value in ['-1','3-5','3-1','bad','']:
            with self.assertRaises(ValueError):parse_neurons(value,4)

    def test_duplicate_and_invalid_values(self):
        groups=[{'layer':0,'neurons':'0-2','scale':0},{'layer':0,'neurons':'2,3','scale':2}]
        with self.assertRaises(ValueError):validate_request(request(groups=groups),CATALOG)
        groups[1]['enabled']=False
        self.assertEqual(len(validate_request(request(groups=groups),CATALOG)['groups']),1)
        for value in [float('nan'),-1,5]:
            with self.assertRaises(ValueError):validate_request(request(groups=[{'layer':0,'neurons':'0','scale':value}]),CATALOG)
        with self.assertRaises(ValueError):validate_request(request(adapter='../../foo'),CATALOG)

    def test_history(self):
        r=request();r['messages']=[{'role':'system','content':'sys'},{'role':'user','content':'a'},{'role':'assistant','content':'b'},{'role':'user','content':'c'}]
        self.assertEqual(len(validate_request(r,CATALOG)['messages']),4)
        r['messages'][2]['role']='user'
        with self.assertRaises(ValueError):validate_request(r,CATALOG)

    def test_multiple_layers_and_restore_after_exception(self):
        import mlx.core as mx
        import mlx.nn as nn
        layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=nn.Linear(4,2,bias=False))) for _ in range(2)]
        for layer in layers:layer.mlp.down_proj.weight=mx.array([[1.,1.,1.,1.],[2.,2.,2.,2.]])
        model=SimpleNamespace(layers=layers)
        originals=[l.mlp.down_proj for l in layers]
        groups=[{'layer':0,'neurons':[0,1],'scale':0},{'layer':0,'neurons':[2],'scale':2},{'layer':1,'neurons':[3],'scale':3}]
        with self.assertRaises(RuntimeError):
            with apply_groups(model,groups,4):
                self.assertEqual(layers[0].mlp.down_proj(mx.ones((1,4))).tolist(),[[3.,6.]])
                self.assertEqual(layers[1].mlp.down_proj(mx.ones((1,4))).tolist(),[[6.,12.]])
                raise RuntimeError('cancel')
        for layer,old in zip(layers,originals):self.assertIs(layer.mlp.down_proj,old)

if __name__=='__main__':unittest.main()
