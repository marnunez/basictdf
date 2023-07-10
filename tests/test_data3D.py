from io import BytesIO
from unittest import TestCase

import numpy as np

from basictdf.tdfData3D import (
    Data3D,
    Data3dBlockFormat,
    MarkerTrack,
    TrackType,
)


class TestMarkerTrack(TestCase):
    def test_creation(self):
        a = MarkerTrack("marker", np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(a.label, "marker")
        np.testing.assert_equal(a.data, np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(a._segments, [slice(0, 2)])
        self.assertEqual(a.nFrames, 2)

    def test_track_properties(self):
        a = MarkerTrack("marker", np.array([[1, 2, 3], [4, 5, 6]]))
        np.testing.assert_equal(a.X, np.array([1, 4]))
        np.testing.assert_equal(a.Y, np.array([2, 5]))
        np.testing.assert_equal(a.Z, np.array([3, 6]))

        a.X = np.array([10, 20])
        np.testing.assert_equal(a.X, np.array([10, 20]))
        np.testing.assert_equal(a.data, np.array([[10, 2, 3], [20, 5, 6]]))
        a.Y = np.array([10, 20])
        np.testing.assert_equal(a.data, np.array([[10, 10, 3], [20, 20, 6]]))
        np.testing.assert_equal(a.Y, np.array([10, 20]))
        a.Z = np.array([10, 20])
        np.testing.assert_equal(a.Z, np.array([10, 20]))
        np.testing.assert_equal(a.data, np.array([[10, 10, 10], [20, 20, 20]]))

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
        a = MarkerTrack._build(c, 2)
        self.assertEqual(a.label, "marker")
        self.assertEqual(a.nFrames, 2)
        self.assertEqual(a.nBytes, len(b))
        self.assertEqual(a._segments, [slice(0, 2)])
        self.assertEqual(a.data.shape, (2, 3))
        d = BytesIO()
        a._write(d)
        self.assertEqual(d.getvalue(), b)

    def test_write(self):
        a = MarkerTrack("marker", np.array([[1, 2, 3], [4, 5, 6]]))
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
        a._write(other)
        self.assertEqual(i.getvalue(), other.getvalue())
        self.assertEqual(a.nBytes, len(b))


class TestData3D(TestCase):
    def test_creation(self):
        t = MarkerTrack("marker", np.array([[1, 2, 3], [4, 5, 6]]))

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
        t2 = MarkerTrack("marker", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
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
        t = MarkerTrack("marker", np.array([[1, 2, 3], [4, 5, 6]]))
        t2 = MarkerTrack("marker2", np.array([[4, 5, 6], [7, 8, 9]]))

        dataBlock1 = Data3D(
            frequency=100,
            nFrames=2,
            volume=np.array([1, 2, 3]),
            translationVector=np.array([1, 2, 3]),
            rotationMatrix=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        )

        dataBlock1.tracks = [t, t2]

        buff1 = BytesIO()
        dataBlock1._write(buff1)
        buff1.seek(0, 0)
        dataBlock2 = Data3D._build(
            buff1,
            Data3dBlockFormat.byTrack,
        )
        buff2 = BytesIO()
        dataBlock2._write(buff2)
        self.assertEqual(dataBlock2.format, Data3dBlockFormat.byTrack)
        self.assertEqual(dataBlock1, dataBlock2)
        self.assertEqual(dataBlock1.tracks, dataBlock2.tracks)
        self.assertEqual(buff1.getvalue(), buff2.getvalue())
        self.assertEqual(dataBlock1.nBytes, len(buff2.getvalue()))

    # def test_files(self) -> None:
    #     with TemporaryDirectory() as tmp_dir:
    #         for file_name, data in test_file_feeder("data3d"):
    #             with Tdf(file_name) as tdf:
    #                 data3d = tdf.data3D
    #                 assert data3d.format == data["format"]
    #                 assert data3d.nFrames == data["nFrames"]
    #                 assert data3d.nTracks == data["nTracks"]
    #                 assert data3d.frequency == data["frequency"]
    #                 np.testing.assert_allclose(data3d.volume, data["volume"])
    #                 np.testing.assert_almost_equal(
    #                     data3d.translationVector, data["translationVector"]
    #                 )
    #                 np.testing.assert_almost_equal(
    #                     data3d.rotationMatrix, data["rotationMatrix"]
    #                 )
    #                 assert data3d.startTime == data["startTime"]
    #                 assert len(data3d.tracks) == len(data3d) == data["nTracks"]
    #                 assert len(data3d.links) == data["nLinks"]
    #                 assert data3d.nBytes == data["nBytes"]

    #             tmp_dir = Path(tmp_dir)
    #             new_tdf = Tdf.new(tmp_dir / "data3d.tdf")
    #             with new_tdf.allow_write():
    #                 new_tdf.data3D = data3d
    #             assert new_tdf.data3D == data3d
