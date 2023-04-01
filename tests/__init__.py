from __future__ import annotations
from pathlib import Path
from typing import Generator, Tuple

from basictdf.tdfData3D import Data3dBlockFormat

test_files_data = {
    "2241_aa_Walking 01.tdf": {
        "nBlocks": 8,
        "version": 1,
        "data3d": {
            "nFrames": 710,
            "format": Data3dBlockFormat.byTrack,
            "frequency": 100,
            "startTime": 0,
            "nTracks": 18,
            "volume": [3.85, 1.65, 1.32],
            "rotationMatrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translationVector": [-0.958, 0.058, -0.246],
            "nLinks": 25,
            "nBytes": 134712,
        },
        "emg": {
            "nFrames": 710,
        },
    }
}


def test_file_feeder(
    block: str | None = None,
) -> Generator[Tuple[Path, dict], None, None]:
    tdf_files = (Path(__file__).parent / "test_files").glob("*.tdf")
    for tdf_file in tdf_files:
        tdf_metadata = (
            test_files_data[tdf_file.name].get(block, None)
            if block
            else test_files_data.get(tdf_file.name)
        )
        if not tdf_metadata:
            continue
        yield tdf_file, tdf_metadata
