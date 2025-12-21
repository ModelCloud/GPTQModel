# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import threading
import time
from unittest.mock import Mock, patch
import sys

import pytest

from gptqmodel.utils.pause_resume import PauseResumeController, PauseResumeState


class TestPauseResumeController:
    """Test the pause/resume controller functionality."""

    def test_initial_state(self):
        """Test that controller starts in running state."""
        controller = PauseResumeController()

        assert controller.get_state() == PauseResumeState.RUNNING
        assert controller.is_running()
        assert not controller.is_paused()

        controller.cleanup()

    def test_pause_requested_state(self):
        """Test pause request functionality."""
        controller = PauseResumeController()

        # Request pause
        controller.pause()
        assert controller.get_state() == PauseResumeState.PAUSE_REQUESTED
        assert controller.is_paused()
        assert not controller.is_running()

        controller.cleanup()

    def test_pause_resume_cycle(self):
        """Test complete pause/resume cycle."""
        controller = PauseResumeController()

        # Start running
        assert controller.is_running()

        # Request pause
        controller.pause()
        assert controller.get_state() == PauseResumeState.PAUSE_REQUESTED

        # Resume
        controller.resume()
        assert controller.get_state() == PauseResumeState.RUNNING
        assert controller.is_running()
        assert not controller.is_paused()

        controller.cleanup()

    def test_toggle_functionality(self):
        """Test toggle between pause and resume."""
        controller = PauseResumeController()

        # Toggle to pause
        controller.toggle_pause_resume()
        assert controller.get_state() == PauseResumeState.PAUSE_REQUESTED
        assert controller.is_paused()

        # Toggle to resume
        controller.toggle_pause_resume()
        assert controller.get_state() == PauseResumeState.RUNNING
        assert controller.is_running()

        controller.cleanup()

    def test_check_pause_point_when_running(self):
        """Test check_pause_point when running (should not block)."""
        controller = PauseResumeController()

        start_time = time.time()
        result = controller.check_pause_point("test layer")
        elapsed = time.time() - start_time

        assert result is True
        assert elapsed < 0.1  # Should not block

        controller.cleanup()

    def test_check_pause_point_with_pause_requested(self):
        """Test check_pause_point when pause was requested."""
        controller = PauseResumeController()

        # Request pause
        controller.pause()
        assert controller.get_state() == PauseResumeState.PAUSE_REQUESTED

        # Function to resume after delay
        def delayed_resume():
            time.sleep(0.2)
            controller.resume()

        # Start resume thread
        resume_thread = threading.Thread(target=delayed_resume)
        resume_thread.start()

        # Check pause point - should block until resume
        start_time = time.time()
        result = controller.check_pause_point("test layer")
        elapsed = time.time() - start_time

        assert result is True
        assert elapsed >= 0.1  # Should have blocked
        assert controller.get_state() == PauseResumeState.RUNNING

        resume_thread.join()
        controller.cleanup()

    def test_status_callback(self):
        """Test status callback functionality."""
        callback = Mock()
        controller = PauseResumeController()
        controller.set_status_callback(callback)

        # Trigger state change
        controller.pause()
        callback.assert_called_once_with(PauseResumeState.PAUSE_REQUESTED)

        # Reset mock
        callback.reset_mock()

        # Another state change
        controller.resume()
        callback.assert_called_once_with(PauseResumeState.RUNNING)

        controller.cleanup()

    def test_pause_context_manager(self):
        """Test pause_context context manager."""
        controller = PauseResumeController()

        # Test when running
        with controller.pause_context("test operation"):
            assert True  # Should not raise

        # Test with pause requested
        controller.pause()

        def delayed_resume():
            time.sleep(0.1)
            controller.resume()

        resume_thread = threading.Thread(target=delayed_resume)
        resume_thread.start()

        start_time = time.time()
        with controller.pause_context("test operation"):
            elapsed = time.time() - start_time
            assert elapsed >= 0.05  # Should have blocked

        resume_thread.join()
        controller.cleanup()

    @patch('sys.stdin.isatty', return_value=True)
    def test_keyboard_input_toggles_pause(self, mock_isatty):
        """Test that keyboard input 'p' toggles the pause state."""
        
        state_changed = threading.Event()
        
        def state_change_callback(new_state):
            # This callback will be executed in the listener thread
            state_changed.set()

        if sys.platform == "win32":
            # On each call to kbhit, we want to simulate a key press only on the first and third check
            # The listener thread loops roughly every 0.1s
            kbhit_mock = Mock(side_effect=[True, False, True, False, False, False, False, False])
            getch_mock = Mock(return_value=b'p')
            
            with patch('msvcrt.kbhit', kbhit_mock), \
                 patch('msvcrt.getch', getch_mock):
                
                controller = PauseResumeController()
                controller.set_status_callback(state_change_callback)
                
                # Wait for the listener thread to detect the first key press
                assert state_changed.wait(timeout=1), "Timeout waiting for first state change"
                assert controller.get_state() == PauseResumeState.PAUSE_REQUESTED
                assert getch_mock.call_count == 1

                # Reset event for the next state change
                state_changed.clear()
                
                # Wait for the listener thread to detect the second key press
                assert state_changed.wait(timeout=1), "Timeout waiting for second state change"
                assert controller.get_state() == PauseResumeState.RUNNING
                assert getch_mock.call_count == 2

                controller.cleanup()
        else:  # POSIX
            # This generator will be the side_effect for select.select
            # It yields a keypress, then nothing, then a keypress, then nothing forever.
            def select_generator():
                yield ([sys.stdin], [], [])  # First press
                yield ([], [], [])           # No press, allows 0.1s delay
                yield ([sys.stdin], [], [])  # Second press
                while True:
                    yield ([], [], [])       # No press thereafter

            select_gen = select_generator()
            select_mock = Mock(side_effect=lambda *args: next(select_gen))
            read_mock = Mock(return_value='p')
            
            # Mock tcgetattr to return a valid list to prevent TypeError in the listener's finally block
            tcgetattr_mock = Mock(return_value=["iflag", "oflag", "cflag", "lflag", "ispeed", "ospeed", "cc"])

            with patch('select.select', select_mock), \
                 patch('sys.stdin.read', read_mock), \
                 patch('termios.tcgetattr', tcgetattr_mock), \
                 patch('termios.tcsetattr'), \
                 patch('tty.setcbreak'):
                
                controller = PauseResumeController()
                controller.set_status_callback(state_change_callback)

                # Wait for the listener thread to detect the first key press
                assert state_changed.wait(timeout=2), "Timeout waiting for first state change"
                assert controller.get_state() == PauseResumeState.PAUSE_REQUESTED
                assert read_mock.call_count == 1

                # Reset event for the next state change
                state_changed.clear()
                
                # Wait for the listener thread to detect the second key press
                assert state_changed.wait(timeout=2), "Timeout waiting for second state change"
                assert controller.get_state() == PauseResumeState.RUNNING
                assert read_mock.call_count == 2
                
                controller.cleanup()

    def test_thread_safety(self):
        """Test thread safety of pause/resume operations."""
        controller = PauseResumeController()
        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    state = controller.get_state()
                    results.append(f"Worker {worker_id}: {state.value}")

                    # Toggle pause/resume
                    if i % 2 == 0:
                        controller.pause()
                    else:
                        controller.resume()

                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) > 0, "No operations completed"

        controller.cleanup()

    def test_cleanup(self):
        """Test cleanup functionality."""
        controller = PauseResumeController()

        # Use controller
        controller.pause()
        controller.resume()

        # Cleanup should not raise errors
        controller.cleanup()

        # After cleanup, should be in running state
        assert controller.get_state() == PauseResumeState.RUNNING