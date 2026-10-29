import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'web'))
from backend import validate_request
from gguf_backend import specification

class GGUFTests(unittest.TestCase):
    def test_capability_validation(self):
        catalog=[dict(id='gguf',adapters=[],layers=32,width=0,supports_steering=False)]
        body=dict(model='gguf',messages=[dict(role='user',content='hello')])
        self.assertEqual(validate_request(body,catalog)['groups'],[])
        for changes in [dict(compare=True),dict(groups=[dict(enabled=False)]),dict(adapter='test')]:
            with self.assertRaises(ValueError):
                validate_request(dict(body,**changes),catalog)

    def test_incomplete_and_outside_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'weights').mkdir();(root/'weights/test.gguf').write_bytes(b'GGUF')
            def meta(file,size):
                (root/'runtime.json').write_text(json.dumps(dict(file=file,size=size)))
            meta('test.gguf',4)
            self.assertEqual(specification(root)[1].name,'test.gguf')
            for file,size in [('test.gguf',5),('../test.gguf',4),('missing.gguf',4)]:
                meta(file,size)
                with self.assertRaises(ValueError):specification(root)


class GenerationLimitTests(unittest.TestCase):
    def test_extended_limits(self):
        catalog=[dict(id='gguf',adapters=[],layers=32,width=0,supports_steering=False)]
        body=dict(model='gguf',messages=[dict(role='user',content='hi')],max_tokens=4096,context_limit=8192)
        self.assertEqual(validate_request(body,catalog)['max_tokens'],4096)
        for change in [dict(max_tokens=4097),dict(context_limit=8193)]:
            with self.assertRaises(ValueError):validate_request(dict(body,**change),catalog)

if __name__=='__main__':unittest.main()
