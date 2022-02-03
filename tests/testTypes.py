from unittest import TestCase
from tdfTypes import Type
import numpy as np


class TestTypes(TestCase):
    def test_float32(self):
        f = Type(np.dtype("<f4"))
        self.assertEqual(f.btype.shape, ())
        self.assertEqual(f.btype.itemsize, 4)
        self.assertEqual(f.btype.kind, "f")
        self.assertEqual(f.btype.str, "<f4")
        self.assertEqual(f.btype.name, "float32")

        b = b"\x00\x00\x80?"
        self.assertEqual(f.read(b), np.float32(1))
        self.assertEqual(f.read(b).shape, np.float32(1).shape)

        self.assertEqual(f.read(b, n=1), np.float32([1]))
        self.assertEqual(f.read(b, n=1).shape, np.float32([1]).shape)
