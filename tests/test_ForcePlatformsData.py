from io import BytesIO
from unittest import TestCase
import numpy as np

from basictdf.tdfForcePlatformsData import (
    ForcePlatformData,
    ForcePlatformBlockFormat,
)


class TestForcePlatformData(TestCase):
    def test_creation(self):
        # 2D array of application points (x,y)
        application_point = np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            dtype=np.float32,
        )
        # 3D array of forces (x,y,z)
        force = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ],
            dtype=np.float32,
        )
        # 1D array of torques (z)
        torque = np.array(
            [1, 2, 3, 4],
            dtype=np.float32,
        )
        f = ForcePlatformData(
            label="test",
            application_point=application_point,
            force=force,
            torque=torque,
        )
        b = BytesIO()

        f._write(b, format=ForcePlatformBlockFormat.byTrackISSFormat)
        b.seek(0, 0)
        new_f = ForcePlatformData._build(
            b, ForcePlatformBlockFormat.byTrackISSFormat, 4
        )
        c = BytesIO()
        new_f._write(c, format=ForcePlatformBlockFormat.byTrackISSFormat)
        c.seek(0, 0)

        np.testing.assert_array_almost_equal(
            f.application_point, new_f.application_point
        )
        np.testing.assert_array_almost_equal(f.force, new_f.force)
        np.testing.assert_array_almost_equal(f.torque, new_f.torque)
        self.assertEqual(f.nBytes, new_f.nBytes)

        self.assertEqual(b.getvalue(), c.getvalue())
        self.assertEqual(f.nBytes, len(b.getvalue()))
        self.assertEqual(new_f.nBytes, len(c.getvalue()))
