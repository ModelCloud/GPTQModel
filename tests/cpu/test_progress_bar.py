import unittest
from time import sleep

from gptqmodel.utils.progress import ProgressBar


class TestBits(unittest.TestCase):
    def test_progress_bar(self):
        pb = ProgressBar(range(1,101))
        for i in pb:
            pb.info(f"Test run index {i} of 100")
            sleep(0.1)


