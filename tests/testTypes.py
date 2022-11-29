from io import BytesIO
from unittest import TestCase
from basictdf.tdfTypes import TdfType
import numpy as np
from numpy.testing import assert_array_equal


class TestTypes(TestCase):
    def test_float32(self):
        f = TdfType(np.dtype("<f4"))
        self.assertEqual(f.btype.shape, ())
        self.assertEqual(f.btype.itemsize, 4)
        self.assertEqual(f.btype.kind, "f")
        self.assertEqual(f.btype.str, "<f4")
        self.assertEqual(f.btype.name, "float32")

        b = b"\x00\x00\x80?"
        self.assertEqual(f.read(b), np.float32(1))
        self.assertEqual(f.read(b).shape, (1,))

        self.assertEqual(type(f.read(b)), np.ndarray)

    def test_segment_data(self):
        dtype = np.dtype([("startFrame", "<i4"), ("nFrames", "<i4")])
        s = TdfType(dtype)
        b = b"\x01\x00\x00\x00\x01\x00\x00\x00"

        # read a single object. Assume
        read = s.read(b)

        self.assertEqual(read["startFrame"], 1)
        self.assertEqual(read["nFrames"], 1)
        self.assertEqual(read.dtype, dtype)
        self.assertEqual(read.shape, (1,))
        self.assertEqual(read, np.array([(1, 1)], dtype=dtype))

        bio = BytesIO(b)

        read = s.bread(bio)
        self.assertEqual(read["startFrame"], 1)
        self.assertEqual(read["nFrames"], 1)
        self.assertEqual(read.dtype, dtype)
        self.assertEqual(read.shape, ())
        self.assertEqual(read, np.array([(1, 1)], dtype=dtype)[0])

        bio.seek(0, 0)
        read = s.bread(bio, 1)
        self.assertEqual(read, np.array([(1, 1)], dtype=dtype))
