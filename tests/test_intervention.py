"""Small numerical checks without loading the 4B model."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mlx.core as mx
import mlx.nn as nn
from intervene import ScaledInput, intervention


class InterventionTests(unittest.TestCase):
    def test_zero_double_identity_and_dtype(self):
        base = nn.Linear(4, 2, bias=False)
        base.weight = mx.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=mx.float16)
        x = mx.array([[[1, 2, 3, 4], [4, 3, 2, 1]]], dtype=mx.float16)
        for scale in [0, 1, 2]:
            wrapped = ScaledInput(base, 4, [1, 3], scale)
            expected = base(x * mx.array([1, scale, 1, scale], dtype=x.dtype))
            actual = wrapped(x)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(mx.allclose(actual, expected).item())

    def test_restore_even_on_exception(self):
        base = nn.Linear(4, 2, bias=False)
        mlp = SimpleNamespace(down_proj=base)
        model = SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)],
                                args=SimpleNamespace(text_config={'intermediate_size': 4}))
        with self.assertRaises(RuntimeError):
            with intervention(model, 0, [1], 0):
                self.assertIsNot(mlp.down_proj, base)
                raise RuntimeError('intentional')
        self.assertIs(mlp.down_proj, base)


if __name__ == '__main__':
    unittest.main()
