# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""GDDR thermal pacing for the Tenstorrent backends.

A Blackhole p300c throttles its GDDR at 90 C and shuts the card down at 110 C.
A load/unload benchmark loop keeps the parts hot between iterations because the
chips never idle long enough to shed the heat the previous model left behind:
on this host (4x p300c) a warm reboot after a sustained run left one card in
``D3cold`` and only three of four root ports enumerated, which a ``reboot``
could not recover -- it needed ``poweroff`` plus a cold start.

This module gives the runtime two cheap, measured gates around every device
ownership transition:

``before_device_open``
    Read the peak GDDR temperature over PCI telemetry, and if it is above the
    soft limit, wait (bounded) for it to fall before the next model starts
    hammering the parts. Also sleeps the configured cooldown so the wait is not
    spent with the device already spinning.

``after_device_close``
    Sleep the cooldown after the device was actually closed, so the idle time
    is real idle time and not time stolen from the next load.

The telemetry read uses ``pyluwen`` over PCI only (``noc_safe=True``); it takes
about two milliseconds and never opens a TT-Metal device, so it is safe here --
these hooks only run while no mesh is open. Nothing in this module raises on a
missing driver or a missing ``pyluwen``: an unreadable temperature disables the
guard instead of failing a load.

Configuration (environment):

``GPTQMODEL_TT_COOLDOWN_SECS``
    Idle seconds after a device closes and before the next one opens. Default
    ``5``. This is a *teardown settle* delay, not a cooling delay: measured GDDR
    cooldown is ~2 C per minute, so five seconds of sleep sheds about 0.2 C.
    Heat is handled by the temperature gate below; this only stops a load from
    re-opening the mesh the instant the previous one released it. Set ``0`` to
    disable.

``GPTQMODEL_TT_GDDR_SOFT_LIMIT_C``
    Peak GDDR temperature, in Celsius, above which a load waits for cooldown.
    Default ``80``. Measured on this host, a TT-Metal model load raises peak
    GDDR by 20-34 C above idle, and the card throttles at 90 C and shuts down at
    110 C, so a load started above ~80 C can reach the shutdown limit on its own.
    Set ``0`` to disable the guard and only use the fixed sleep.

``GPTQMODEL_TT_GDDR_MAX_WAIT_SECS``
    Upper bound on that wait, in seconds. Default ``900``. GDDR sheds heat at
    only about 2 C per minute here, so cooling from 98 C to 80 C legitimately
    takes ~9 minutes; the budget allows that once. When it is reached the load
    proceeds and logs the temperature rather than hanging a job forever.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional, Sequence

from .logger import setup_logger


log = setup_logger()

DEFAULT_COOLDOWN_SECS = 5.0
DEFAULT_SOFT_LIMIT_C = 80.0
DEFAULT_MAX_WAIT_SECS = 900.0
# Poll interval while waiting for the parts to shed heat. Measured cooldown on a
# p300c is about 2 C per minute at idle, so a sub-second poll would only burn
# PCI transactions between two readings that differ by a fraction of a degree.
_POLL_SECS = 15.0


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning(f"TTThermal: ignoring {name}={raw!r}; not a number, using {default}.")
        return default


def cooldown_secs() -> float:
    """Return the configured idle delay between device ownership transitions."""

    return max(_env_float("GPTQMODEL_TT_COOLDOWN_SECS", DEFAULT_COOLDOWN_SECS), 0.0)


def soft_limit_c() -> float:
    """Return the GDDR temperature above which a load waits for cooldown."""

    return max(_env_float("GPTQMODEL_TT_GDDR_SOFT_LIMIT_C", DEFAULT_SOFT_LIMIT_C), 0.0)


def max_wait_secs() -> float:
    """Return the upper bound on a thermal wait."""

    return max(_env_float("GPTQMODEL_TT_GDDR_MAX_WAIT_SECS", DEFAULT_MAX_WAIT_SECS), 0.0)


def gddr_temps() -> Optional[Sequence[int]]:
    """Return the peak GDDR temperature (Celsius) of every visible card.

    ``None`` means the value is unavailable -- no Tenstorrent driver, no
    ``pyluwen``, or the telemetry read failed. Callers must treat that as
    "cannot measure" and skip the guard rather than guessing.

    This reads PCI telemetry only. Do not call it while a TT-Metal mesh is open:
    ``detect_chips_fallible`` enumerates from scratch, so the ownership gates in
    :mod:`gptqmodel.utils.ttnn_runtime` are the only supported call sites.
    """

    try:
        from pyluwen import detect_chips_fallible
    except Exception:
        return None

    try:
        chips = detect_chips_fallible(
            local_only=True, continue_on_failure=False, noc_safe=True
        )
    except Exception as exc:  # pragma: no cover - depends on host driver state
        log.debug(f"TTThermal: GDDR telemetry unavailable: {exc}")
        return None

    temps = []
    for chip in chips:
        try:
            telemetry = chip.force_upgrade().as_bh().get_telemetry()
            temps.append(int(telemetry.max_gddr_temp))
        except Exception as exc:  # pragma: no cover - per-chip driver hiccup
            log.debug(f"TTThermal: skipping a card whose telemetry failed: {exc}")
    return tuple(temps) if temps else None


def gddr_peak() -> Optional[int]:
    """Return the hottest GDDR temperature across all cards, or ``None``."""

    temps = gddr_temps()
    return max(temps) if temps else None


# --- board power and idle clock ------------------------------------------
#
# A p300c ships with a 125 W per-board TDP limit and the firmware THROTTLES BY
# INSERTING NOP CYCLES once that limit is reached, so power, not temperature, is
# what actually caps sustained throughput. Two levers reach the firmware
# directly, and neither needs a specific tt-metal build:
#
#   SET_TDP_LIMIT = 0x22   arg0 = watts, arg1 = bit 0 restores the default
#   AICLK_GO_LONG_IDLE = 0x54
#
# Both are documented in tt-umd's ``blackhole_arc.hpp``; ``SET_TDP_LIMIT`` has
# been in the firmware since 19.8.0 and this host runs 19.17.1. The newer
# ``TT_METAL_TDP_LIMIT_WATTS`` environment variable is *not* present in every
# release (it is absent from the 1.5.0 plugin build installed here), which is
# why this module drives the message itself.

BLACKHOLE_ARC_SET_TDP_LIMIT = 0x22
BLACKHOLE_ARC_AICLK_GO_LONG_IDLE = 0x54


def _blackhole_chips():
    """Return the local Blackhole chips, or ``None`` when telemetry is unavailable."""

    try:
        from pyluwen import detect_chips_fallible
    except Exception:
        return None
    try:
        chips = detect_chips_fallible(
            local_only=True, continue_on_failure=False, noc_safe=True
        )
    except Exception as exc:  # pragma: no cover - depends on host driver state
        log.debug(f"TTThermal: chip enumeration failed: {exc}")
        return None

    out = []
    for chip in chips:
        try:
            out.append(chip.force_upgrade().as_bh())
        except Exception as exc:  # pragma: no cover - non-Blackhole or dead card
            log.debug(f"TTThermal: skipping a card that is not a live Blackhole: {exc}")
    return out or None


def tdp_limit_watts() -> Optional[Sequence[int]]:
    """Return the effective per-board TDP limit of every visible card, in watts."""

    chips = _blackhole_chips()
    if chips is None:
        return None
    out = []
    for chip in chips:
        try:
            out.append(int(chip.get_telemetry().tdp_limit_max))
        except Exception:  # pragma: no cover - per-chip driver hiccup
            pass
    return tuple(out) if out else None


def set_tdp_limit(watts: int, *, restore_default: bool = False) -> Optional[Sequence[int]]:
    """Set the per-board TDP limit on every local card; return the new limits.

    ``restore_default`` asks the firmware for the board default instead of a
    specific wattage. The limit is a firmware state, so it outlives this
    process until the chip is reset or something else writes it -- callers that
    only want a temporary cap must restore it themselves.
    """

    if not restore_default and watts <= 0:
        raise ValueError(f"tdp limit must be a positive wattage, got {watts}")

    chips = _blackhole_chips()
    if chips is None:
        return None

    for index, chip in enumerate(chips):
        try:
            rc = chip.arc_msg(
                BLACKHOLE_ARC_SET_TDP_LIMIT,
                arg0=0 if restore_default else int(watts),
                arg1=1 if restore_default else 0,
            )
        except Exception as exc:  # pragma: no cover - firmware rejected the message
            log.warning(f"TTThermal: card {index} rejected SET_TDP_LIMIT: {exc}")
            continue
        if isinstance(rc, tuple) and rc and rc[0] != 0:
            log.warning(f"TTThermal: card {index} returned exit code {rc[0]} for SET_TDP_LIMIT.")

    after = tdp_limit_watts()
    if after is not None:
        requested = "board default" if restore_default else f"{watts} W"
        log.info(f"TTThermal: requested TDP limit {requested}; effective limits now {list(after)} W.")
    return after


def park_idle_clock() -> bool:
    """Return the cards to their 800 MHz long-idle clock after a device closes.

    Closing the last ``MeshDevice`` does *not* drop AICLK: measured on this
    host, the cards stayed at 1350 MHz and ~50 W per board for more than 50 s
    after ``close_device``, and GDDR kept climbing the whole time. The clock
    only falls back when the firmware is told to, so a cooldown that merely
    closes the device is not actually idle.

    Returns ``True`` when at least one card accepted the message.
    """

    chips = _blackhole_chips()
    if chips is None:
        return False
    sent = 0
    for index, chip in enumerate(chips):
        try:
            chip.arc_msg(BLACKHOLE_ARC_AICLK_GO_LONG_IDLE)
            sent += 1
        except Exception as exc:  # pragma: no cover - firmware rejected the message
            log.debug(f"TTThermal: card {index} rejected AICLK_GO_LONG_IDLE: {exc}")
    if sent:
        log.info(f"TTThermal: asked {sent} card(s) to return to the long-idle clock after close.")
    return sent > 0


def _default_sleep(seconds: float) -> None:
    time.sleep(seconds)


#: ``time.monotonic()`` at which the last idle window ended. A load may nest
#: several ownership gates (a model-level pace, then the mesh open), so the
#: cooldown is a rate limit: at most one idle window per ``cooldown_secs()``.
_last_idle_at: Optional[float] = None


def _record_idle_end(now: Optional[float]) -> None:
    global _last_idle_at
    _last_idle_at = now


def reset_cooldown_state() -> None:
    """Forget the last idle window. Intended for tests."""

    _record_idle_end(None)


def _idle_already_paid(now: float, delay: float) -> bool:
    """Return whether an idle window of ``delay`` finished within the last ``delay``."""

    return _last_idle_at is not None and (now - _last_idle_at) < delay


def wait_for_cooldown(
    *,
    sleep: Callable[[float], None] = _default_sleep,
    read_temps: Callable[[], Optional[Sequence[int]]] = gddr_temps,
    monotonic: Callable[[], float] = time.monotonic,
) -> Optional[int]:
    """Pace a load until the GDDR is below the soft limit or the wait expires.

    Returns the last measured peak, or ``None`` when telemetry was unavailable.
    The wait is bounded by ``GPTQMODEL_TT_GDDR_MAX_WAIT_SECS`` so a card with a
    broken fan fails the load loudly in the logs instead of hanging forever.
    """

    limit = soft_limit_c()
    temps = read_temps()
    if temps is None:
        log.debug("TTThermal: GDDR temperature unavailable; thermal guard skipped.")
        return None
    peak = max(temps)
    if limit <= 0:
        return peak

    budget = max_wait_secs()
    start = monotonic()
    waited = 0.0
    while peak > limit:
        elapsed = monotonic() - start
        if elapsed >= budget:
            log.warning(
                f"TTThermal: GDDR still at {peak} C after {elapsed:.0f}s of waiting "
                f"(soft limit {limit:.0f} C, budget {budget:.0f}s); proceeding."
            )
            return peak
        pause = min(_POLL_SECS, budget - elapsed)
        log.info(
            f"TTThermal: GDDR peak {peak} C is above the {limit:.0f} C soft limit; "
            f"waiting {pause:.0f}s for cooldown ({elapsed:.0f}s/{budget:.0f}s)."
        )
        sleep(pause)
        waited += pause
        temps = read_temps()
        if temps is None:
            log.debug("TTThermal: telemetry vanished mid-wait; thermal guard stopped.")
            return None
        peak = max(temps)

    if waited:
        log.info(f"TTThermal: GDDR cooled to {peak} C after {waited:.0f}s; resuming load.")
    return peak


def before_device_open(
    *,
    owner: str = "gptqmodel",
    sleep: Callable[[float], None] = _default_sleep,
    read_temps: Callable[[], Optional[Sequence[int]]] = gddr_temps,
    monotonic: Callable[[], float] = time.monotonic,
) -> Optional[int]:
    """Apply the thermal guard, then the remainder of the cooldown, before opening.

    A load that follows a close immediately only waits the part of the cooldown
    that the close did not already cover, so one idle window separates two
    models instead of two stacked ones.
    """

    peak = wait_for_cooldown(sleep=sleep, read_temps=read_temps, monotonic=monotonic)
    delay = cooldown_secs()
    if delay <= 0:
        return peak
    now = monotonic()
    if _idle_already_paid(now, delay):
        return peak
    log.info(
        f"TTThermal: cooling {delay:.1f}s before {owner!r} opens the Tenstorrent "
        f"devices (GDDR peak {peak if peak is not None else 'unknown'} C)."
    )
    sleep(delay)
    _record_idle_end(monotonic())
    return peak


def after_device_close(
    *,
    owner: str = "gptqmodel",
    sleep: Callable[[float], None] = _default_sleep,
    read_temps: Callable[[], Optional[Sequence[int]]] = gddr_temps,
    monotonic: Callable[[], float] = time.monotonic,
) -> Optional[int]:
    """Sleep the cooldown after a device closed, then report the peak GDDR."""

    delay = cooldown_secs()
    if delay > 0:
        log.info(
            f"TTThermal: {owner!r} closed the Tenstorrent devices; idling {delay:.1f}s "
            f"so the GDDR can shed heat."
        )
        sleep(delay)
    _record_idle_end(monotonic())
    return gddr_peak()


__all__ = [
    "DEFAULT_COOLDOWN_SECS",
    "DEFAULT_MAX_WAIT_SECS",
    "DEFAULT_SOFT_LIMIT_C",
    "after_device_close",
    "before_device_open",
    "cooldown_secs",
    "gddr_peak",
    "gddr_temps",
    "max_wait_secs",
    "park_idle_clock",
    "reset_cooldown_state",
    "set_tdp_limit",
    "soft_limit_c",
    "tdp_limit_watts",
    "wait_for_cooldown",
]
