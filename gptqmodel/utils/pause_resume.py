# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Simple pause/resume functionality for GPTQModel quantization.

Provides thread-safe pause/resume capabilities with keyboard input handling.
"""

import logging
import sys
import threading
import time
import atexit
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Dict, List, Optional

from .colors import ANSIColor, color_text


log = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    import msvcrt
else:
    try:
        import select
        import termios
        import tty
    except ImportError:
        log.warning("termios, tty, or select not available. Pause/resume from keyboard will not work.")


_original_termios_settings = None
_terminal_restored = False
_terminal_restore_lock = threading.Lock()


def _restore_terminal_settings_on_exit():
    """Restore terminal settings on exit. Idempotent, thread-safe, and safe to call multiple times."""
    global _original_termios_settings, _terminal_restored

    with _terminal_restore_lock:
        # Idempotent: skip if already restored or nothing to restore
        if _terminal_restored or _original_termios_settings is None:
            return

        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _original_termios_settings)
            # Show cursor explicitly (in case it was hidden by progress bars)
            # ANSI escape sequence DECSET: Show cursor
            sys.stdout.write('\x1b[?25h')
            _original_termios_settings = None
            _terminal_restored = True
            log.debug("Terminal settings restored on exit.")
        except Exception as e:
            log.warning(f"Failed to restore terminal settings on exit: {e}")
            # Mark as restored even on failure to prevent retry loops
            _terminal_restored = True


if not _IS_WINDOWS and 'termios' in sys.modules:
    if sys.stdin.isatty():
        atexit.register(_restore_terminal_settings_on_exit)


class PauseResumeState(Enum):
    """Enumeration of possible pause/resume states."""
    RUNNING = "running"
    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"


class PauseResumeController:
    """
    Simple thread-safe controller for managing pause/resume during quantization.

    Provides:
    - Thread-safe state management
    - Keyboard input handling from stdin ('p' key)
    - Integration with progress tracking
    """

    def __init__(self):
        """
        Initialize the pause/resume controller.
        """
        self._state = PauseResumeState.RUNNING
        self._state_lock = threading.RLock()
        self._pause_event = threading.Event()
        self._resume_event = threading.Event()

        # Stdin input handling
        self._input_thread = None
        self._stop_event = threading.Event()

        # Callbacks for status updates
        self._status_callback: Optional[Callable[[PauseResumeState], None]] = None

        # Progress bar references for immediate title updates
        self._progress_bars: List[Dict] = []  # List of {"pb": progress_bar, "title_func": callable}

        # Initialize events
        self._resume_event.set()  # Allow execution to start

    def _play_icon(self) -> str:
        return color_text(">", ANSIColor.GREEN)

    def _pause_icon(self) -> str:
        return color_text("||", ANSIColor.YELLOW)

    def get_status_hint(self) -> str:
        """Get status hint for main progress bar."""
        state = self.get_state()
        if state == PauseResumeState.RUNNING:
            return f"['p' to {self._pause_icon()}]"
        elif state == PauseResumeState.PAUSE_REQUESTED:
            return f"[will {self._pause_icon()} at layer end]"
        elif state == PauseResumeState.PAUSED:
            return f"['p' to {self._play_icon()}]"
        else:
            return ""

    def status_icon(self) -> str:
        """Get current status icon (|| for pause states, > for running)."""
        state = self.get_state()
        if state in [PauseResumeState.RUNNING, PauseResumeState.PAUSE_REQUESTED]:
            return self._play_icon()
        elif state in [PauseResumeState.PAUSED]:
            return self._pause_icon()
        else:
            return ""

    def wrap_text(self, text: str) -> str:
        """Wrap text with status icon and hint for progress bar display."""
        icon = self.status_icon()
        hint = self.get_status_hint()

        if icon:
            text = f"{icon} {text}"
        if hint:
            text = f"{text} {hint}"
        return text

    def register_and_draw_progress_bar(self, pb, title: Optional[str] = None, subtitle: Optional[str] = None):
        """
        Register a progress bar for immediate title updates when pause/resume state changes.

        Args:
            pb: Progress bar instance
            title: Optional title string without status icons/hints
            subtitle: Optional subtitle string
        """
        # update progress bar subtitle
        pb.subtitle(subtitle)
        # register or update base title for progress bar
        try:
            with self._state_lock:
                # Check if this progress bar is already registered
                for item in self._progress_bars:
                    if item["pb"] == pb:
                        # Update the title and subtitle if already registered
                        item["title"] = title
                        return

                # Register new progress bar
                self._progress_bars.append({"pb": pb, "title": title})
        finally:
            self._update_progress_bars()

    def unregister_progress_bar(self, pb):
        """
        Unregister a progress bar from immediate title updates.

        Args:
            pb: Progress bar instance to remove
        """
        with self._state_lock:
            self._progress_bars = [item for item in self._progress_bars if item["pb"] != pb]

    def _update_progress_bars(self):
        """Update all registered progress bars with current pause/resume status."""
        for item in self._progress_bars:
            pb = item["pb"]
            title = item.get("title")

            if title:
                wrapped_title = self.wrap_text(title)
                try:
                    pb.title(wrapped_title)
                    pb.draw()
                except Exception as e:
                    log.warning(f"Failed to update progress bar title: {e}")

    def _windows_stdin_listener(self):
        while not self._stop_event.is_set():
            try:
                if msvcrt.kbhit():
                    char = msvcrt.getch().decode('utf-8').lower()
                    if char == 'p':
                        self.toggle_pause_resume()
                time.sleep(0.1)
            except Exception as e:
                log.warning(f"Error in stdin listener: {e}")
                break

    def _posix_stdin_listener(self):
        global _original_termios_settings
        old_settings = None
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            if _original_termios_settings is None:
                _original_termios_settings = old_settings
            tty.setcbreak(sys.stdin.fileno())

            while not self._stop_event.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char.lower() == 'p':
                        self.toggle_pause_resume()
        except Exception as e:
            log.warning(f"Error in stdin listener: {e}")
        finally:
            # Defense-in-depth: restore terminal settings when listener thread exits
            # This ensures cleanup even if cleanup() is never called
            if old_settings:
                _restore_terminal_settings_on_exit()

    def _setup_keyboard_handler(self):
        """Setup keyboard event handlers for pause/break keys."""
        if not sys.stdin.isatty():
            log.info("Not a TTY. Pause/resume from keyboard is disabled.")
            return

        target = None
        if _IS_WINDOWS:
            target = self._windows_stdin_listener
        else:
            if 'termios' in sys.modules:
                target = self._posix_stdin_listener
            else:
                log.warning("termios module not found, cannot listen for keyboard input on POSIX.")
                return

        self._input_thread = threading.Thread(target=target, daemon=True)
        self._input_thread.start()

    def set_status_callback(self, callback: Callable[[PauseResumeState], None]):
        """Set callback for state changes."""
        with self._state_lock:
            self._status_callback = callback

    def _set_state(self, new_state: PauseResumeState):
        """Internal method to change state with notification."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state

            # Update events based on state
            if new_state == PauseResumeState.PAUSED:
                self._pause_event.set()
                self._resume_event.clear()
            elif new_state == PauseResumeState.RUNNING:
                self._pause_event.clear()
                self._resume_event.set()

            # Update progress bars immediately when state changes
            if old_state != new_state:
                self._update_progress_bars()

            # Notify callback
            if self._status_callback and old_state != new_state:
                try:
                    self._status_callback(new_state)
                except Exception as e:
                    log.warning(f"Status callback error: {e}")

    def get_state(self) -> PauseResumeState:
        """Get current state."""
        with self._state_lock:
            return self._state

    def pause(self):
        """Request pause - will pause at next safe point."""
        with self._state_lock:
            if self._state == PauseResumeState.RUNNING:
                self._set_state(PauseResumeState.PAUSE_REQUESTED)

    def resume(self):
        """Resume quantization."""
        with self._state_lock:
            if self._state == PauseResumeState.PAUSED or self._state == PauseResumeState.PAUSE_REQUESTED:
                self._set_state(PauseResumeState.RUNNING)

    def toggle_pause_resume(self):
        """Toggle between pause and resume states."""
        current_state = self.get_state()
        if current_state == PauseResumeState.RUNNING:
            self.pause()
        elif current_state in [PauseResumeState.PAUSED, PauseResumeState.PAUSE_REQUESTED]:
            self.resume()

    def check_pause_point(self, layer_info: Optional[str] = None) -> bool:
        """
        Check if we should pause at this point.

        Called between layers or other safe points.

        Args:
            layer_info: Optional description of current layer for logging

        Returns:
            Always True (execution continues)
        """
        # Check if pause was requested
        with self._state_lock:
            if self._state == PauseResumeState.PAUSE_REQUESTED:
                self._set_state(PauseResumeState.PAUSED)

        # Wait if paused
        if self._pause_event.is_set():
            while self._pause_event.is_set():
                if self._resume_event.wait(timeout=0.1):
                    with self._state_lock:
                        if self._state == PauseResumeState.PAUSED:
                            self._set_state(PauseResumeState.RUNNING)
                        break

        return True

    def is_paused(self) -> bool:
        """Check if currently paused or pause requested."""
        with self._state_lock:
            return self._state in [PauseResumeState.PAUSED, PauseResumeState.PAUSE_REQUESTED]

    def is_running(self) -> bool:
        """Check if currently running."""
        with self._state_lock:
            return self._state == PauseResumeState.RUNNING

    @contextmanager
    def pause_context(self, operation_name: str):
        """
        Context manager for operations that should respect pause state.

        Args:
            operation_name: Name of the operation for logging
        """
        self.check_pause_point(operation_name)
        try:
            yield
        finally:
            # Check pause point again after operation
            self.check_pause_point(f"after {operation_name}")

    @contextmanager
    def lifecycle(self):
        """
        Context manager for the controller lifecycle.

        Usage:
            with controller.lifecycle():
                # quantization code here
                pass
            # cleanup is guaranteed here
        """
        self._setup_keyboard_handler()

        try:
            yield
        finally:
            # This ALWAYS runs, even on exceptions or KeyboardInterrupt
            self.cleanup()

    def cleanup(self):
        """
        Cleanup resources and keyboard handlers.

        This method is idempotent - safe to call multiple times.
        Ensures terminal settings are restored on all exit paths.
        """
        # Stop the input listener thread
        self._stop_event.set()
        if self._input_thread:
            self._input_thread.join(timeout=0.5)
            self._input_thread = None

        # Clear progress bar references
        with self._state_lock:
            self._progress_bars.clear()

        # Set final state to running for clean shutdown
        with self._state_lock:
            if self._state != PauseResumeState.RUNNING:
                self._set_state(PauseResumeState.RUNNING)

        # Ensure terminal settings are restored (idempotent)
        if not _IS_WINDOWS and 'termios' in sys.modules:
            _restore_terminal_settings_on_exit()
