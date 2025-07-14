import unittest
import cProfile
import pstats
import io
import SimpleITK as sitk

from ..registration import registration
from .test_registration import load_nifty


class TestProfiling(unittest.TestCase):
    def test_profile_registration(self):
        fixed = load_nifty('samples/fixed.nii.gz')
        moving = load_nifty('samples/moving.nii.gz')

        pr = cProfile.Profile()
        pr.enable()
        registration(fixed, moving)
        pr.disable()

        s = io.StringIO()
        stats = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        stats.print_stats(10)
        output = s.getvalue()

        # ensure profiling captured some execution time
        total_time = sum(v[2] for v in stats.stats.values())
        self.assertGreater(total_time, 0)

        # ensure profiler produced textual output
        self.assertTrue(len(output) > 0)

