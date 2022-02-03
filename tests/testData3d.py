from io import BytesIO
from unittest import TestCase

import numpy as np
from tdfData3D import Data3D, Data3dBlockFormat, Track, TrackType


class TestTrack(TestCase):
    def test_creation(self):
        a = Track("marker", np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(a.label, "marker")
        np.testing.assert_equal(a.data, np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(a._segments, [slice(0, 2)])
        self.assertEqual(a.nFrames, 2)

    def test_track_properties(self):
        a = Track("marker", np.array([[1, 2, 3], [4, 5, 6]]))
        np.testing.assert_equal(a.X, np.array([1, 4]))
        np.testing.assert_equal(a.Y, np.array([2, 5]))
        np.testing.assert_equal(a.Z, np.array([3, 6]))

        a.X = np.array([10, 20])
        np.testing.assert_equal(a.X, np.array([10, 20]))
        a.Y = np.array([10, 20])
        np.testing.assert_equal(a.Y, np.array([10, 20]))
        a.Z = np.array([10, 20])
        np.testing.assert_equal(a.Z, np.array([10, 20]))

    def test_build(self):
        b = b""
        # label
        b += b"marker" + b"\x00" * 250
        # nSegments
        b += b"\x01\x00\x00\x00"
        # padding
        b += b"\x00\x00\x00\x00"
        # startFrame
        b += b"\x00\x00\x00\x00"
        # nFrames
        b += b"\x02\x00\x00\x00"
        # trackData
        b += TrackType.write(np.array([[1, 2, 3], [4, 5, 6]]))
        c = BytesIO(b)
        a = Track.build(c, 2)
        self.assertEqual(a.label, "marker")
        self.assertEqual(a.nFrames, 2)
        self.assertEqual(a.nBytes, len(b))
        self.assertEqual(a._segments, [slice(0, 2)])
        self.assertEqual(a.data.shape, (2, 3))
        d = BytesIO()
        a.write(d)
        self.assertEqual(d.getvalue(), b)

    def test_write(self):
        a = Track("marker", np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(len(a._segments), 1)
        self.assertEqual(a._segments[0], slice(0, 2))

        b = b""
        # label
        b += b"marker" + b"\x00" * 250
        # nSegments
        b += b"\x01\x00\x00\x00"
        # padding
        b += b"\x00\x00\x00\x00"
        # startFrame
        b += b"\x00\x00\x00\x00"
        # nFrames
        b += b"\x02\x00\x00\x00"
        # trackData
        b += TrackType.write(a.data)
        i = BytesIO(b)
        other = BytesIO()
        a.write(other)
        self.assertEqual(i.getvalue(), other.getvalue())
        self.assertEqual(a.nBytes, len(b))


class TestData3D(TestCase):
    def test_creation(self):
        t = Track("marker", np.array([[1, 2, 3], [4, 5, 6]]))

        a = Data3D(
            frequency=100,
            nFrames=2,
            volume=np.array([1, 2, 3]),
            translationVector=np.array([1, 2, 3]),
            rotationMatrix=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        )
        self.assertEqual(a.format, Data3dBlockFormat.byTrack)
        self.assertEqual(a.nFrames, 2)
        self.assertEqual(a.nTracks, 0)

        # different nFrames
        t2 = Track("marker", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        with self.assertRaises(ValueError):
            a.add_track(t2)

        with self.assertRaises(TypeError):
            a.add_track([])

        a.add_track(t)

        self.assertEqual(a.nTracks, 1)
        self.assertEqual(a.tracks, [t])

        a.tracks = []
        self.assertEqual(a.nTracks, 0)
        a.tracks = [t, t]
        self.assertEqual(a.nTracks, 2)

    def test_build(self):
        t = Track("marker", np.array([[1, 2, 3], [4, 5, 6]]))
        t2 = Track("marker2", np.array([[4, 5, 6], [7, 8, 9]]))

        dataBlock1 = Data3D(
            frequency=100,
            nFrames=2,
            volume=np.array([1, 2, 3]),
            translationVector=np.array([1, 2, 3]),
            rotationMatrix=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        )

        dataBlock1.tracks = [t, t2]

        buff1 = BytesIO()
        dataBlock1.write(buff1)
        buff1.seek(0, 0)
        dataBlock2 = Data3D.build(
            buff1,
            Data3dBlockFormat.byTrack,
        )
        buff2 = BytesIO()
        dataBlock2.write(buff2)
        self.assertEqual(dataBlock2.format, Data3dBlockFormat.byTrack)
        self.assertEqual(dataBlock1, dataBlock2)
        self.assertEqual(buff1.getvalue(), buff2.getvalue())
