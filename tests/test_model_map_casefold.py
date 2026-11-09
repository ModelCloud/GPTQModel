# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the case-insensitive ``MODEL_MAP`` lookup table."""

from gptqmodel.models.auto import MODEL_MAP, MODEL_MAP_CASEFOLD


def test_model_map_casefold_has_no_case_collisions():
    # Two registrations that differ only by case would silently shadow one
    # another: the dict comprehension keeps the last definition.
    assert len(MODEL_MAP_CASEFOLD) == len(MODEL_MAP)


def test_model_map_casefold_resolves_every_registration():
    for model_type, definition in MODEL_MAP.items():
        assert MODEL_MAP_CASEFOLD[model_type.lower()] is definition
