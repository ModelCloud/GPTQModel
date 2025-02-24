import unittest
from time import sleep

from gptqmodel.utils.progress import ProgressBar


class TestProgressBar(unittest.TestCase):
    def test_range_manual(self):
        pb = ProgressBar(range(100)).manual()
        for _ in pb:
            pb.draw()
            sleep(0.05)

    def test_range_auto(self):
        pb = ProgressBar(range(100))
        for _ in pb:
            sleep(0.05)

    def test_range_auto_disable_ui_left_steps(self):
        pb = ProgressBar(range(100)).set(show_left_steps=False)
        for _ in pb:
            sleep(0.05)

    def test_title(self):
        pb = ProgressBar(range(100))
        for _ in pb:
            pb.title(f"[Test run index]").draw()
            sleep(0.05)

    def test_title_subtitle(self):
        pb = ProgressBar(range(100)).title("[Title: Title]")
        for _ in pb:
            pb.subtitle(f"[Subtitle: Test run index]").draw()
            sleep(0.05)
