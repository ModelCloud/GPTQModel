import unittest
from time import sleep

from gptqmodel.utils.progress import ProgressBar


def generate_epanding_str_a_to_z():
    # Initialize an empty list to store the strings
    strings = []

    # Loop through the alphabet from 'A' to 'Z'
    for i in range(26):
        # Create a string from 'A' to the current character
        current_string = ''.join([chr(ord('A') + j) for j in range(i + 1)])
        strings.append(current_string)

    # Now, reverse the sequence from 'A...Y' to 'A'
    for i in range(25, 0, -1):
        # Create a string from 'A' to the current character
        current_string = ''.join([chr(ord('A') + j) for j in range(i)])
        strings.append(current_string)

    return strings

SAMPLES = generate_epanding_str_a_to_z()
REVERSED_SAMPLES = reversed(SAMPLES)

class TestProgressBar(unittest.TestCase):

    def test_title_fixed_subtitle_dynamic(self):
        pb = ProgressBar(SAMPLES).title("TITLE:").manual()
        for i in pb:
            pb.subtitle(f"[SUBTITLE: {i}]").draw()
            sleep(0.1)

    def test_title_dynamic_subtitle_fixed(self):
        pb = ProgressBar(SAMPLES).subtitle("SUBTITLE: FIXED").manual()
        for i in pb:
            pb.title(f"[TITLE: {i}]").draw()
            sleep(0.1)

    def test_title_dynamic_subtitle_dynamic(self):
        pb = ProgressBar(SAMPLES).manual()
        for i in pb:
            pb.title(f"[TITLE: {i}]").subtitle(f"[SUBTITLE: {i}]").draw()
            sleep(0.1)

    def test_range_manual(self):
        pb = ProgressBar(range(100)).manual()
        for _ in pb:
            pb.draw()
            sleep(0.1)

    def test_range_auto(self):
        pb = ProgressBar(range(100))
        for _ in pb:
            sleep(0.1)

    def test_range_auto_disable_ui_left_steps(self):
        pb = ProgressBar(range(100)).set(show_left_steps=False)
        for _ in pb:
            sleep(0.1)

    def test_title(self):
        pb = ProgressBar(range(100)).title("TITLE: FIXED")
        for _ in pb:
            sleep(0.1)
    #
    def test_title_subtitle(self):
        pb = ProgressBar(range(100)).title("[TITLE: FIXED]").manual()
        for _ in pb:
            pb.subtitle(f"[SUBTITLE: FIXED]").draw()
            sleep(0.1)
