from io import BytesIO
from unittest import TestCase, skip

import numpy as np

from basictdf.tdfForce3D import (
    ApplicationPointType,
    ForceTorque3D,
    ForceTorque3DBlockFormat,
    ForceTorqueTrack,
    ForceType,
    TorqueType,
)


class TestForceTorqueTrack(TestCase):
    @skip("Not implemented")
    def test_creation(self):
        cop = np.array([[1, 2, 3], [4, 5, 6]])
        force = np.array([[7, 8, 9], [10, 11, 12]])
        torque = np.array([[13, 14, 15], [16, 17, 18]])

        a = ForceTorqueTrack(
            "test_force_track",
            application_point=cop,
            force=force,
            torque=torque,
        )

        np.testing.assert_equal(a.application_point, cop)
        np.testing.assert_equal(a.force, force)
        np.testing.assert_equal(a.torque, torque)

        self.assertEqual(a.nFrames, 2)

        wrong_shape_cop = np.array([[1, 2, 3], [4, 5, 6, 7]])
        with self.assertRaises(ValueError):
            a = ForceTorqueTrack(
                "test_force_track",
                application_point=wrong_shape_cop,
                force=force,
                torque=torque,
            )

    def test_build(self):
        cop = np.array([[1, 2, 3], [4, 5, 6]])
        force = np.array([[5, 6, 7], [8, 9, 10]])
        torque = np.array([[11, 12, 13], [14, 15, 16]])
        b = b""
        # label
        b += b"r_gr" + b"\x00" * 252
        # nSegments
        b += b"\x01\x00\x00\x00"
        # padding
        b += b"\x00\x00\x00\x00"
        # startFrame
        b += b"\x00\x00\x00\x00"
        # nFrames
        b += b"\x02\x00\x00\x00"

        for i in range(2):
            # application point
            b += ApplicationPointType.write(cop[i])

            # force
            b += ForceType.write(force[i])

            # torque
            b += TorqueType.write(torque[i])

        c = BytesIO(b)
        a = ForceTorqueTrack._build(c, 2)
        self.assertEqual(a.label, "r_gr")
        self.assertEqual(a.nFrames, 2)
        np.testing.assert_almost_equal(a.application_point, cop)

        self.assertEqual(a.application_point.shape, (2, 3))
        self.assertEqual(a.force.shape, (2, 3))
        self.assertEqual(a.torque.shape, (2, 3))

        self.assertEqual(a._segments, [slice(0, 2)])
        self.assertEqual(a.nBytes, len(b))
        d = BytesIO()
        a._write(d)
        self.assertEqual(d.getvalue(), b)

    def test_equality(self):
        cop = np.array([[1, 2, 3], [4, 5, 6]])
        force = np.array([[5, 6, 7], [8, 9, 10]])
        torque = np.array([[11, 12, 13], [14, 15, 16]])
        a = ForceTorqueTrack(
            "track_label", application_point=cop, force=force, torque=torque
        )
        self.assertNotEqual(a, ForceTorqueTrack("other_label", cop, force, torque))
        self.assertNotEqual(a, ForceTorqueTrack("track_label", force, force, torque))
        self.assertNotEqual(a, ForceTorqueTrack("track_label", cop, cop, torque))
        self.assertNotEqual(a, ForceTorqueTrack("track_label", cop, force, force))
        self.assertEqual(a, ForceTorqueTrack("track_label", cop, force, torque))

    def test_write(self):
        cop = np.array([[1, 2, 3], [4, 5, 6]])
        force = np.array([[5, 6, 7], [8, 9, 10]])
        torque = np.array([[11, 12, 13], [14, 15, 16]])
        a = ForceTorqueTrack("r_gr", application_point=cop, force=force, torque=torque)
        self.assertEqual(len(a._segments), 1)
        self.assertEqual(a._segments[0], slice(0, 2))

        b = b""
        # label
        b += b"r_gr" + b"\x00" * 252
        # nSegments
        b += b"\x01\x00\x00\x00"
        # padding
        b += b"\x00\x00\x00\x00"
        # startFrame
        b += b"\x00\x00\x00\x00"
        # nFrames
        b += b"\x02\x00\x00\x00"

        for i in range(2):
            # application point
            b += ApplicationPointType.write(cop[i])

            # force
            b += ForceType.write(force[i])

            # torque
            b += TorqueType.write(torque[i])

        i = BytesIO(b)
        other = BytesIO()
        a._write(other)
        self.assertEqual(i.getvalue(), other.getvalue())
        self.assertEqual(a.nBytes, len(b))


class TestForceTorque3D(TestCase):
    def test_creation(self):
        cop = np.array([[1, 2, 3], [4, 5, 6]])
        force = np.array([[5, 6, 7], [8, 9, 10]])
        torque = np.array([[11, 12, 13], [14, 15, 16]])

        track = ForceTorqueTrack(
            "track_label", application_point=cop, force=force, torque=torque
        )

        a = ForceTorque3D(
            frequency=100,
            nFrames=2,
            volume=np.array([1, 2, 3]),
            translationVector=np.array([1, 2, 3]),
            rotationMatrix=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        )
        self.assertEqual(a.format, ForceTorque3DBlockFormat.byTrack)
        self.assertEqual(a.nFrames, 2)
        self.assertEqual(a.nTracks, 0)

        # different nFrames
        t2 = ForceTorqueTrack(
            "track_label",
            application_point=np.array([[1, 2, 3]]),
            force=np.array([[1, 2, 3]]),
            torque=np.array([[1, 2, 3]]),
        )
        with self.assertRaises(ValueError):
            a.add_track(t2)

        with self.assertRaises(TypeError):
            a.add_track([])

        a.add_track(track)

        self.assertEqual(a.nTracks, 1)
        self.assertEqual(a.tracks, [track])
        self.assertTrue("track_label" in a)
        self.assertTrue(track in a)

        a.tracks = []
        self.assertEqual(a.nTracks, 0)
        self.assertEqual(a.tracks, [])
        self.assertTrue("track_label" not in a)
        self.assertTrue(track not in a)

        a.tracks = [track, track]
        self.assertEqual(a.nTracks, 2)
        self.assertEqual(a.tracks, [track, track])
        self.assertTrue("track_label" in a)
        self.assertTrue(track in a)

    def test_build(self):
        cop = np.array([[1, 2, 3], [4, 5, 6]])
        force = np.array([[5, 6, 7], [8, 9, 10]])
        torque = np.array([[11, 12, 13], [14, 15, 16]])

        cop2 = np.array([[1, 2, 43], [4, 5, 62]])
        force2 = np.array([[5, 6, 72], [8, 92, 10]])
        torque2 = np.array([[11, 12, 13], [134, 15, 16]])

        t = ForceTorqueTrack(
            "track_label1", application_point=cop, force=force, torque=torque
        )
        t2 = ForceTorqueTrack(
            "track_label2",
            application_point=cop2,
            force=force2,
            torque=torque2,
        )

        dataBlock1 = ForceTorque3D(
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
        dataBlock2 = ForceTorque3D._build(
            buff1,
            ForceTorque3DBlockFormat.byTrack,
        )
        buff2 = BytesIO()
        dataBlock2._write(buff2)
        self.assertEqual(dataBlock2.format, ForceTorque3DBlockFormat.byTrack)
        self.assertEqual(dataBlock1, dataBlock2)
        self.assertEqual(buff1.getvalue(), buff2.getvalue())
