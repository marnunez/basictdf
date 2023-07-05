from io import BytesIO
from unittest import TestCase

import numpy as np

from basictdf.tdfForcePlatformsCalibration import (
    ForcePlatformsCalibrationDataBlock,
    ForcePlatform,
    ForcePlatformVertices,
    ForcePlatformCalibrationBlockFormat,
)


class TestForcePlatform(TestCase):
    def test_creation(self):
        # Create a 4x3 matrix of vertices
        vertices = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ],
            dtype=np.float32,
        )
        size = (1.0, 2.0)

        f = ForcePlatform("test", size, vertices)
        assert f.label == "test"
        np.testing.assert_equal(f.size, size)
        np.testing.assert_equal(f.position, vertices)

    def test_build(self):
        vertices = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ],
            dtype=np.float32,
        )
        size = np.array([1.0, 2.0], dtype=np.float32)

        b = b""
        # label
        b += b"test" + b"\x00" * 252
        # size
        b += np.array(size, dtype="<f4").tobytes()
        # vertices
        b += ForcePlatformVertices.write(vertices)
        # padding
        b += b"\x00" * 256

        c = BytesIO(b)
        f = ForcePlatform._build(c)
        self.assertEqual(f.label, "test")
        np.testing.assert_equal(f.size, size)
        np.testing.assert_equal(f.position, vertices)
        self.assertEqual(f.nBytes, len(b))

    def test_write(self):
        vertices = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ],
            dtype=np.float32,
        )
        size = np.array([1.0, 2.0], dtype=np.float32)
        a = ForcePlatform("test", size, vertices)

        b = b""
        # label
        b += b"test" + b"\x00" * 252
        # size
        b += np.array(size, dtype="<f4").tobytes()
        # vertices
        b += ForcePlatformVertices.write(vertices)
        # padding
        b += b"\x00" * 256

        c = BytesIO()
        a._write(c)
        self.assertEqual(c.getvalue(), b)
        self.assertEqual(a.nBytes, len(b))

        d = ForcePlatform._build(BytesIO(b))
        self.assertEqual(d.label, a.label)
        np.testing.assert_equal(d.size, a.size)
        np.testing.assert_equal(d.position, a.position)
        self.assertEqual(d.nBytes, len(b))


class TestForcePlatformsCalibrationDataBlock(TestCase):
    def test_creation(self):
        a = ForcePlatformsCalibrationDataBlock()
        self.assertEqual(len(a), 0)
        self.assertEqual(a.nBytes, 8)

        a.add_platform(ForcePlatform("test", (1.0, 2.0), np.zeros((4, 3))))
        self.assertEqual(len(a), 1)
        self.assertEqual(
            a.nBytes, ForcePlatform.nBytes + 8 + 2
        )  # 2 extra bytes from the platformMap

    def test_add_platform(self):
        a = ForcePlatformsCalibrationDataBlock()
        a.add_platform(ForcePlatform("test", (1.0, 2.0), np.zeros((4, 3))))
        self.assertEqual(len(a), 1)
        self.assertEqual(a.nBytes, ForcePlatform.nBytes + 8 + 2)

        # Shouldn't let me add it to the same channel
        with self.assertRaises(ValueError):
            a.add_platform(
                ForcePlatform("test", (1.0, 2.0), np.zeros((4, 3))), channel=0
            )

        # Should let me add a platform to an arbitrary channel
        a.add_platform(ForcePlatform("test", (1.0, 2.0), np.zeros((4, 3))), channel=36)
        self.assertEqual(len(a), 2)
        self.assertEqual(a._platformMap, [0, 36])

    def test_write(self):
        a = ForcePlatformsCalibrationDataBlock()
        fp = ForcePlatform("test", (1.0, 2.0), np.zeros((4, 3)))
        a.add_platform(fp)
        a.add_platform(fp, channel=36)

        c = BytesIO()
        a._write(c)
        c.seek(0)

        a = ForcePlatformsCalibrationDataBlock._build(
            c, ForcePlatformCalibrationBlockFormat.GRPFormat
        )

        self.assertEqual(len(a), 2)
        self.assertEqual(a._platformMap, [0, 36])
        self.assertEqual(
            a.nBytes,
            2 * ForcePlatform.nBytes
            + 8  # base
            + (2 * 2),  # platformMap for two platforms
        )

        # Check that the platforms are the same
        for i in range(2):
            self.assertEqual(a[i], fp)
