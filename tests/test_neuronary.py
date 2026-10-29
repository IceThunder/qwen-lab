import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'web'))
from neuronary import recipe_id, normalize, fetch_recipe, API, NoRedirect
from backend import validate_request

FIXTURE=json.loads((Path(__file__).parent/'fixtures/neuronary_y2r85pwz.json').read_text())
CATALOG=[dict(id='qwen3.5-4b',layers=32,width=9216,adapters=[]),
         dict(id='qwen3.5-4b-hauhaucs-aggressive',layers=32,width=0,adapters=[],supports_steering=False)]

class NeuronaryTests(unittest.TestCase):
    def test_url_allowlist(self):
        for value in ['y2r85pwz','https://neuronary.org/r/y2r85pwz','https://neuronary.org/r/y2r85pwz/#example']:
            self.assertEqual(recipe_id(value),'y2r85pwz')
        for value in ['http://127.0.0.1:7860','https://neuronary.org.evil/r/y2r85pwz',
                      'https://evil@neuronary.org/r/y2r85pwz','https://neuronary.org/',
                      'https://neuronary.org/r/../about','https://neuronary.org/r/%2fetc',
                      'https://neuronary.org/r/y2r85pwz?url=evil',None]:
            with self.assertRaises(ValueError):recipe_id(value)

    def test_recipe_roundtrip(self):
        d=normalize(FIXTURE,CATALOG)
        self.assertEqual(d['compatible_model'],'qwen3.5-4b')
        self.assertEqual(d['groups'][0]['dose'],-.8)
        self.assertEqual(d['groups'][0]['neurons'],'4595')
        req=validate_request(dict(model=d['compatible_model'],groups=d['groups'],
            messages=[dict(role='user',content=d['examples'][0]['prompt'])]),CATALOG)
        self.assertEqual(req['groups'][0]['mode'],'normalized_add')
        self.assertEqual(d['temperature'],0)
        self.assertIn('网站',d['warnings'][0])

    def test_incompatible_model_not_guessed(self):
        for model in ['Qwen/Qwen3.5-9B','Qwen/Qwen3.8-4B','HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive']:
            d=normalize(dict(FIXTURE,model=model),CATALOG)
            self.assertIsNone(d['compatible_model'])
        self.assertIsNone(normalize(FIXTURE,CATALOG[1:])['compatible_model'])

    def test_invalid_neurons_and_limits(self):
        for change in [dict(expert=1),dict(layer=32),dict(neuron=9216),dict(neuron=True),
                       dict(dose=4.1),dict(dose=float('nan')),dict(dose=True)]:
            d=copy.deepcopy(FIXTURE);d['neurons'][0].update(change)
            with self.assertRaises(ValueError):normalize(d,CATALOG)
        d=copy.deepcopy(FIXTURE);d['neurons']*=2
        with self.assertRaises(ValueError):normalize(d,CATALOG)

    def test_sampling_is_explicit_not_silently_dropped(self):
        d=normalize(dict(FIXTURE,sampling=dict(temperature=.7,top_p=.9,repetition_penalty=1.1)),CATALOG)
        self.assertEqual(d['temperature'],.7)
        self.assertTrue(any('repetition_penalty' in w for w in d['warnings']))
        self.assertTrue(any('top_p' in w for w in d['warnings']))
        d=normalize(dict(FIXTURE,sampling=dict(temperature=9)),CATALOG)
        self.assertEqual(d['temperature'],0)
        self.assertTrue(any('temperature' in w for w in d['warnings']))

    def test_network_uses_only_fixed_api(self):
        with patch('neuronary.urllib.request.build_opener') as builder:
            builder.return_value.open.return_value.__enter__.return_value.read.return_value=json.dumps(FIXTURE).encode()
            self.assertEqual(fetch_recipe('https://neuronary.org/r/y2r85pwz')['id'],'y2r85pwz')
            args,kwargs=builder.return_value.open.call_args
            self.assertEqual(args[0].full_url,API+'y2r85pwz')
            self.assertEqual(kwargs['timeout'],15)
            self.assertIs(builder.call_args.args[0],NoRedirect)

if __name__=='__main__':unittest.main()
