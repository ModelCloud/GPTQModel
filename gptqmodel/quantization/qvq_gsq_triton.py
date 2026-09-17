"""Fused CUDA primitives for sparse local-path GSQ relaxation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp32_multiply(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [left, right],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _fp32_add(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", [left, right],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _candidate_gradient_product(grad_matrix, matrix_indices, sparse_deltas,
                                tile, tile_mask, choice,
                                CHOICES: tl.constexpr,
                                WIDTH: tl.constexpr,
                                SPARSE_SLOT: tl.constexpr):
    sparse_offset = (
        (tile * (CHOICES - 1) + choice - 1) * WIDTH + SPARSE_SLOT
    )
    matrix_offset = tl.load(matrix_indices + sparse_offset,
                            mask=tile_mask, other=0)
    gradient = tl.load(grad_matrix + matrix_offset,
                       mask=tile_mask, other=0.0).to(tl.float32)
    delta = tl.load(sparse_deltas + sparse_offset,
                    mask=tile_mask, other=0.0).to(tl.float32)
    return _fp32_multiply(gradient, delta)


@triton.jit
def _gumbel_softmax_kernel(logits, uniform, probabilities, tau_or_schedule,
                            kappa_or_schedule, step_pointer,
                            TILE_COUNT: tl.constexpr, CHOICES: tl.constexpr,
                            UNIFORM_CHUNK: tl.constexpr, SCHEDULED: tl.constexpr,
                            BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    if SCHEDULED:
        step = tl.load(step_pointer)
        tau = tl.load(tau_or_schedule + step)
        kappa = tl.load(kappa_or_schedule + step)
        uniform_base = (step % UNIFORM_CHUNK) * TILE_COUNT * CHOICES
    else:
        tau = tau_or_schedule
        kappa = kappa_or_schedule
        uniform_base = 0
    choice = tl.arange(0, BLOCK)
    mask = choice < CHOICES
    offset = tile * CHOICES + choice
    u = tl.load(uniform + uniform_base + offset, mask=mask, other=0.5)
    u = tl.maximum(1.0e-6, tl.minimum(u, 1.0 - 1.0e-6))
    score = (kappa * tl.load(logits + offset, mask=mask, other=0.0)
             - tl.log(-tl.log(u))) / tau
    score = tl.where(mask, score, -float("inf"))
    score -= tl.max(score, axis=0)
    numerator = tl.exp(score)
    probability = numerator / tl.sum(numerator, axis=0)
    tl.store(probabilities + offset, probability, mask=mask)


@triton.jit
def _sparse_error_kernel(probabilities, baseline, indices, deltas, target, error,
                         N: tl.constexpr, OUTPUT_TILES: tl.constexpr,
                         CHOICES: tl.constexpr, WIDTH: tl.constexpr,
                         BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    scalar_mask = lane < 256
    tile_row = tile // OUTPUT_TILES
    tile_col = tile - tile_row * OUTPUT_TILES
    row = lane // 16
    column = lane - row * 16
    weight_offset = (tile_row * 16 + row) * N + tile_col * 16 + column
    base = tl.load(baseline + tile * 256 + lane, mask=scalar_mask, other=0.0)
    teacher = tl.load(target + weight_offset, mask=scalar_mask, other=0.0)
    tl.store(error + weight_offset, base - teacher, mask=scalar_mask)
    tl.debug_barrier()

    entries = (CHOICES - 1) * WIDTH
    entry_mask = lane < entries
    alternative = lane // WIDTH
    sparse_offset = tile * entries + lane
    scalar = tl.load(indices + sparse_offset, mask=entry_mask, other=0)
    row = scalar // 16
    column = scalar - row * 16
    weight_offset = (tile_row * 16 + row) * N + tile_col * 16 + column
    probability = tl.load(
        probabilities + tile * CHOICES + alternative + 1,
        mask=entry_mask, other=0.0,
    )
    delta = tl.load(deltas + sparse_offset, mask=entry_mask, other=0.0)
    tl.atomic_add(error + weight_offset, probability * delta, mask=entry_mask)


@triton.jit
def _sparse_lion_kernel(probabilities, metric_error, indices, deltas,
                        denominator, logits, momentum, norm_square,
                        tau_or_schedule, kappa_or_schedule, step_pointer,
                        learning_rate, decay, SCHEDULED: tl.constexpr,
                        N: tl.constexpr, OUTPUT_TILES: tl.constexpr,
                        CHOICES: tl.constexpr, WIDTH: tl.constexpr,
                        BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    if SCHEDULED:
        step = tl.load(step_pointer)
        tau = tl.load(tau_or_schedule + step)
        kappa = tl.load(kappa_or_schedule + step)
    else:
        tau = tau_or_schedule
        kappa = kappa_or_schedule
    choice = tl.arange(0, BLOCK)
    choice_mask = choice < CHOICES
    alternative_mask = (choice > 0) & choice_mask
    tile_row = tile // OUTPUT_TILES
    tile_col = tile - tile_row * OUTPUT_TILES
    derivative = tl.zeros((BLOCK,), tl.float32)
    alternative = choice - 1
    for sparse_slot in tl.static_range(0, WIDTH):
        sparse_offset = ((tile * (CHOICES - 1) + alternative) * WIDTH
                         + sparse_slot)
        scalar = tl.load(indices + sparse_offset, mask=alternative_mask, other=0)
        row = scalar // 16
        column = scalar - row * 16
        weight_offset = (tile_row * 16 + row) * N + tile_col * 16 + column
        metric = tl.load(metric_error + weight_offset, mask=alternative_mask, other=0.0)
        delta = tl.load(deltas + sparse_offset, mask=alternative_mask, other=0.0)
        derivative += tl.where(alternative_mask, metric * delta, 0.0)
    derivative *= 2.0 / tl.load(denominator)
    probability_offset = tile * CHOICES + choice
    probability = tl.load(probabilities + probability_offset, mask=choice_mask, other=0.0)
    center = tl.sum(probability * derivative, axis=0)
    gradient = probability * (derivative - center) * (kappa / tau)
    gradient = tl.where(choice_mask, gradient, 0.0)
    tl.atomic_add(norm_square, tl.sum(gradient * gradient, axis=0))

    old_momentum = tl.load(momentum + probability_offset, mask=choice_mask, other=0.0)
    direction = 0.9 * old_momentum + 0.1 * gradient
    direction = tl.where(direction > 0, 1.0, tl.where(direction < 0, -1.0, 0.0))
    parameter = tl.load(logits + probability_offset, mask=choice_mask, other=0.0)
    parameter = parameter * decay - learning_rate * direction
    new_momentum = 0.99 * old_momentum + 0.01 * gradient
    tl.store(logits + probability_offset, parameter, mask=choice_mask)
    tl.store(momentum + probability_offset, new_momentum, mask=choice_mask)


@triton.jit
def _finish_norm_kernel(norm_square, norm_minimum, norm_maximum, finite_state):
    square = tl.load(norm_square)
    value = tl.sqrt(square)
    tl.store(norm_minimum, tl.minimum(tl.load(norm_minimum), value))
    tl.store(norm_maximum, tl.maximum(tl.load(norm_maximum), value))
    finite = (value == value) & (tl.abs(value) != float("inf"))  # noqa: PLR0124
    tl.store(finite_state, tl.load(finite_state) & finite)
    tl.store(norm_square, 0.0)


@triton.jit
def _advance_step_kernel(step_pointer):
    tl.store(step_pointer, tl.load(step_pointer) + 1)


@triton.jit
def _build_position_map_kernel(indices, deltas, counters, position_choices,
                               position_deltas, TOTAL: tl.constexpr,
                               ENTRIES: tl.constexpr, WIDTH: tl.constexpr,
                               OVERLAP: tl.constexpr, BLOCK: tl.constexpr):
    entry = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = entry < TOTAL
    tile = entry // ENTRIES
    tile_entry = entry - tile * ENTRIES
    scalar = tl.load(indices + entry, mask=mask, other=0)
    counter_offset = tile * 256 + scalar
    slot = tl.atomic_add(counters + counter_offset, 1, mask=mask)
    output_offset = counter_offset * OVERLAP + slot
    choice = tile_entry // WIDTH + 1
    tl.store(position_choices + output_offset, choice,
             mask=mask & (slot < OVERLAP))
    tl.store(position_deltas + output_offset,
             tl.load(deltas + entry, mask=mask, other=0.0),
             mask=mask & (slot < OVERLAP))


@triton.jit
def _candidate_probability_gradient_kernel(grad_matrix, matrix_indices,
                                           sparse_deltas, output,
                                           TILE_COUNT,
                                           CHOICES: tl.constexpr,
                                           WIDTH: tl.constexpr,
                                           BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    choice = tl.arange(0, BLOCK)
    choice_mask = (tile < TILE_COUNT) & (choice < CHOICES)
    alternative_mask = choice_mask & (choice > 0)
    product0 = _candidate_gradient_product(
        grad_matrix, matrix_indices, sparse_deltas, tile, alternative_mask,
        choice, CHOICES, WIDTH, 0,
    )
    product1 = _candidate_gradient_product(
        grad_matrix, matrix_indices, sparse_deltas, tile, alternative_mask,
        choice, CHOICES, WIDTH, 1,
    )
    product2 = _candidate_gradient_product(
        grad_matrix, matrix_indices, sparse_deltas, tile, alternative_mask,
        choice, CHOICES, WIDTH, 2,
    )
    product3 = _candidate_gradient_product(
        grad_matrix, matrix_indices, sparse_deltas, tile, alternative_mask,
        choice, CHOICES, WIDTH, 3,
    )
    product4 = _candidate_gradient_product(
        grad_matrix, matrix_indices, sparse_deltas, tile, alternative_mask,
        choice, CHOICES, WIDTH, 4,
    )
    product5 = _candidate_gradient_product(
        grad_matrix, matrix_indices, sparse_deltas, tile, alternative_mask,
        choice, CHOICES, WIDTH, 5,
    )
    # Match PyTorch's six-lane CUDA reduction tree exactly. P32 legal
    # neighbours always expose six sparse deltas per alternative.
    even = _fp32_add(_fp32_add(product0, product4), product2)
    odd = _fp32_add(_fp32_add(product1, product5), product3)
    value = _fp32_add(even, odd)
    value = tl.where(choice > 0, value, 0.0)
    tl.store(output + tile * CHOICES + choice, value, mask=choice_mask)


@triton.jit
def _scheduled_gumbel_grouped_kernel(logits, uniform, probabilities,
                                     temperature_schedule, kappa_schedule,
                                     step_pointer, TILE_COUNT: tl.constexpr,
                                     CHOICES: tl.constexpr,
                                     UNIFORM_CHUNK: tl.constexpr,
                                     TILES: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0) * TILES + tl.arange(0, TILES)[:, None]
    choice = tl.arange(0, BLOCK)[None, :]
    mask = (tile < TILE_COUNT) & (choice < CHOICES)
    step = tl.load(step_pointer)
    offset = tile * CHOICES + choice
    uniform_base = (step % UNIFORM_CHUNK) * TILE_COUNT * CHOICES
    u = tl.load(uniform + uniform_base + offset, mask=mask, other=0.5)
    u = tl.maximum(1.0e-6, tl.minimum(u, 1.0 - 1.0e-6))
    tau = tl.load(temperature_schedule + step)
    kappa = tl.load(kappa_schedule + step)
    score = (kappa * tl.load(logits + offset, mask=mask, other=0.0)
             - tl.log(-tl.log(u))) / tau
    score = tl.where(mask, score, -float("inf"))
    score -= tl.max(score, axis=1)[:, None]
    numerator = tl.exp(score)
    probability = numerator / tl.sum(numerator, axis=1)[:, None]
    tl.store(probabilities + offset, probability, mask=mask)


@triton.jit
def _position_error_grouped_kernel(probabilities, baseline, position_choices,
                                   position_deltas, target, error,
                                   TILE_COUNT: tl.constexpr, N: tl.constexpr,
                                   OUTPUT_TILES: tl.constexpr,
                                   CHOICES: tl.constexpr,
                                   OVERLAP: tl.constexpr,
                                   TILES: tl.constexpr):
    tile = tl.program_id(0) * TILES + tl.arange(0, TILES)[:, None]
    scalar = tl.arange(0, 256)[None, :]
    mask = tile < TILE_COUNT
    tile_row = tile // OUTPUT_TILES
    tile_col = tile - tile_row * OUTPUT_TILES
    row = scalar // 16
    column = scalar - row * 16
    weight_offset = (tile_row * 16 + row) * N + tile_col * 16 + column
    value = tl.load(baseline + tile * 256 + scalar, mask=mask, other=0.0).to(tl.float32)
    value -= tl.load(target + weight_offset, mask=mask, other=0.0).to(tl.float32)
    position_offset = (tile * 256 + scalar) * OVERLAP
    for slot in tl.static_range(0, OVERLAP):
        choice = tl.load(position_choices + position_offset + slot,
                         mask=mask, other=0)
        active = mask & (choice > 0)
        probability = tl.load(probabilities + tile * CHOICES + choice,
                              mask=active, other=0.0)
        delta = tl.load(position_deltas + position_offset + slot,
                        mask=active, other=0.0)
        value += probability * delta
    tl.store(error + weight_offset, value, mask=mask)


@triton.jit
def _position_error_compact_kernel(probabilities, baseline, position_indices,
                                   position_choices, position_deltas, target,
                                   error, TILE_COUNT: tl.constexpr,
                                   N: tl.constexpr,
                                   OUTPUT_TILES: tl.constexpr,
                                   CHOICES: tl.constexpr,
                                   POSITIONS: tl.constexpr,
                                   OVERLAP: tl.constexpr,
                                   TILES: tl.constexpr,
                                   BLOCK: tl.constexpr):
    """Build E from a dense baseline and only the positions candidates edit.

    P32 W3's 32 legal alternatives touch 64 unique scalars per tile, with
    three alternatives sharing each scalar.  Keeping only those 64 positions
    avoids streaming a [tile, 256, 3] mostly-empty choice map every update.
    """
    tile = tl.program_id(0) * TILES + tl.arange(0, TILES)[:, None]
    scalar = tl.arange(0, BLOCK)[None, :]
    scalar_mask = (tile < TILE_COUNT) & (scalar < 256)
    tile_row = tile // OUTPUT_TILES
    tile_col = tile - tile_row * OUTPUT_TILES
    row = scalar // 16
    column = scalar - row * 16
    weight_offset = (tile_row * 16 + row) * N + tile_col * 16 + column
    value = tl.load(baseline + tile * 256 + scalar,
                    mask=scalar_mask, other=0.0).to(tl.float32)
    value -= tl.load(target + weight_offset,
                     mask=scalar_mask, other=0.0).to(tl.float32)
    tl.store(error + weight_offset, value, mask=scalar_mask)
    tl.debug_barrier()

    compact_offset = (tile * POSITIONS + scalar) * OVERLAP
    first_choice = tl.load(
        position_choices + compact_offset,
        mask=(tile < TILE_COUNT) & (scalar < POSITIONS), other=0,
    ).to(tl.int32)
    position_mask = (
        (tile < TILE_COUNT) & (scalar < POSITIONS) & (first_choice > 0)
    )
    changed_scalar = tl.load(position_indices + tile * POSITIONS + scalar,
                             mask=position_mask, other=0).to(tl.int32)
    row = changed_scalar // 16
    column = changed_scalar - row * 16
    changed_weight_offset = (
        (tile_row * 16 + row) * N + tile_col * 16 + column
    )
    changed_value = tl.load(baseline + tile * 256 + changed_scalar,
                            mask=position_mask, other=0.0).to(tl.float32)
    changed_value -= tl.load(target + changed_weight_offset,
                             mask=position_mask, other=0.0).to(tl.float32)
    position_offset = compact_offset
    for slot in tl.static_range(0, OVERLAP):
        choice = tl.load(position_choices + position_offset + slot,
                         mask=position_mask, other=0).to(tl.int32)
        probability = tl.load(probabilities + tile * CHOICES + choice,
                              mask=position_mask, other=0.0)
        delta = tl.load(position_deltas + position_offset + slot,
                        mask=position_mask, other=0.0)
        changed_value += probability * delta
    tl.store(error + changed_weight_offset, changed_value, mask=position_mask)


@triton.jit
def _sparse_mixture_compact_kernel(probabilities, baseline, position_indices,
                                   position_choices, position_deltas, output,
                                   TILE_COUNT: tl.constexpr,
                                   IN_FEATURES: tl.constexpr,
                                   OUT_FEATURES: tl.constexpr,
                                   OUTPUT_TILES: tl.constexpr,
                                   CHOICES: tl.constexpr,
                                   POSITIONS: tl.constexpr,
                                   OVERLAP: tl.constexpr,
                                   TRANSPOSED: tl.constexpr,
                                   BLOCK: tl.constexpr):
    """Materialize the exact tile mixture from its compact position map."""
    tile = tl.program_id(0)
    scalar = tl.arange(0, BLOCK)
    scalar_mask = (tile < TILE_COUNT) & (scalar < 256)
    tile_row = tile // OUTPUT_TILES
    tile_col = tile - tile_row * OUTPUT_TILES
    if TRANSPOSED:
        row = scalar % 16
        column = scalar // 16
    else:
        row = scalar // 16
        column = scalar - row * 16
    source_scalar = row * 16 + column
    if TRANSPOSED:
        weight_offset = ((tile_col * 16 + column) * IN_FEATURES
                         + tile_row * 16 + row)
    else:
        weight_offset = ((tile_row * 16 + row) * OUT_FEATURES
                         + tile_col * 16 + column)
    value = tl.load(baseline + tile * 256 + source_scalar,
                    mask=scalar_mask, other=0.0).to(tl.float32)
    tl.store(output + weight_offset, value, mask=scalar_mask)
    tl.debug_barrier()

    position_mask = (tile < TILE_COUNT) & (scalar < POSITIONS)
    position_offset = (tile * POSITIONS + scalar) * OVERLAP
    changed_scalar = tl.load(position_indices + tile * POSITIONS + scalar,
                             mask=position_mask, other=0).to(tl.int32)
    row = changed_scalar // 16
    column = changed_scalar - row * 16
    if TRANSPOSED:
        changed_weight_offset = (
            (tile_col * 16 + column) * IN_FEATURES + tile_row * 16 + row
        )
    else:
        changed_weight_offset = (
            (tile_row * 16 + row) * OUT_FEATURES + tile_col * 16 + column
        )
    changed_value = tl.load(baseline + tile * 256 + changed_scalar,
                            mask=position_mask, other=0.0).to(tl.float32)
    choice0 = tl.load(position_choices + position_offset,
                      mask=position_mask, other=0).to(tl.int32)
    choice1 = tl.load(position_choices + position_offset + 1,
                      mask=position_mask, other=0).to(tl.int32)
    choice2 = tl.load(position_choices + position_offset + 2,
                      mask=position_mask, other=0).to(tl.int32)
    active0 = position_mask & (choice0 > 0)
    active1 = position_mask & (choice1 > 0)
    active2 = position_mask & (choice2 > 0)
    contribution0 = _fp32_multiply(
        tl.load(probabilities + tile * CHOICES + choice0,
                mask=active0, other=0.0).to(tl.float32),
        tl.load(position_deltas + position_offset,
                mask=active0, other=0.0).to(tl.float32),
    )
    contribution1 = _fp32_multiply(
        tl.load(probabilities + tile * CHOICES + choice1,
                mask=active1, other=0.0).to(tl.float32),
        tl.load(position_deltas + position_offset + 1,
                mask=active1, other=0.0).to(tl.float32),
    )
    contribution2 = _fp32_multiply(
        tl.load(probabilities + tile * CHOICES + choice2,
                mask=active2, other=0.0).to(tl.float32),
        tl.load(position_deltas + position_offset + 2,
                mask=active2, other=0.0).to(tl.float32),
    )
    # Match deterministic scatter_add: accumulate the sorted duplicate run
    # from zero, then add that reduction to the existing baseline.
    contribution = _fp32_add(
        _fp32_add(contribution0, contribution1), contribution2,
    )
    changed_value = _fp32_add(changed_value, contribution)
    tl.store(output + changed_weight_offset, changed_value, mask=position_mask)


@triton.jit
def _scheduled_sparse_lion_grouped_kernel(
        probabilities, metric_error, indices, deltas, denominator, logits,
        momentum, norm_square, temperature_schedule, kappa_schedule,
        step_pointer, learning_rate, decay, TILE_COUNT: tl.constexpr,
        N: tl.constexpr, OUTPUT_TILES: tl.constexpr, CHOICES: tl.constexpr,
        WIDTH: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0) * TILES + tl.arange(0, TILES)[:, None]
    choice = tl.arange(0, BLOCK)[None, :]
    choice_mask = (tile < TILE_COUNT) & (choice < CHOICES)
    alternative_mask = choice_mask & (choice > 0)
    tile_row = tile // OUTPUT_TILES
    tile_col = tile - tile_row * OUTPUT_TILES
    derivative = tl.zeros((TILES, BLOCK), tl.float32)
    alternative = choice - 1
    for sparse_slot in tl.static_range(0, WIDTH):
        sparse_offset = ((tile * (CHOICES - 1) + alternative) * WIDTH
                         + sparse_slot)
        scalar = tl.load(indices + sparse_offset,
                         mask=alternative_mask, other=0)
        row = scalar // 16
        column = scalar - row * 16
        weight_offset = (tile_row * 16 + row) * N + tile_col * 16 + column
        metric = tl.load(metric_error + weight_offset,
                         mask=alternative_mask, other=0.0)
        delta = tl.load(deltas + sparse_offset,
                        mask=alternative_mask, other=0.0)
        derivative += tl.where(alternative_mask, metric * delta, 0.0)
    derivative *= 2.0 / tl.load(denominator)
    probability_offset = tile * CHOICES + choice
    probability = tl.load(probabilities + probability_offset,
                          mask=choice_mask, other=0.0)
    center = tl.sum(probability * derivative, axis=1)[:, None]
    step = tl.load(step_pointer)
    tau = tl.load(temperature_schedule + step)
    kappa = tl.load(kappa_schedule + step)
    gradient = probability * (derivative - center) * (kappa / tau)
    gradient = tl.where(choice_mask, gradient, 0.0)
    tile_norm = tl.sum(gradient * gradient, axis=1)
    tl.atomic_add(norm_square, tl.sum(tile_norm, axis=0))
    old_momentum = tl.load(momentum + probability_offset,
                           mask=choice_mask, other=0.0)
    direction = 0.9 * old_momentum + 0.1 * gradient
    direction = tl.where(direction > 0, 1.0,
                         tl.where(direction < 0, -1.0, 0.0))
    parameter = tl.load(logits + probability_offset,
                        mask=choice_mask, other=0.0)
    tl.store(logits + probability_offset,
             parameter * decay - learning_rate * direction, mask=choice_mask)
    tl.store(momentum + probability_offset,
             0.99 * old_momentum + 0.01 * gradient, mask=choice_mask)


def gumbel_softmax(logits, uniform, output, tau: float, kappa: float):
    choices = logits.shape[1]
    block = triton.next_power_of_2(choices)
    _gumbel_softmax_kernel[(logits.shape[0],)](
        logits, uniform, output, tau, kappa, logits,
        TILE_COUNT=logits.shape[0], CHOICES=choices,
        UNIFORM_CHUNK=1, SCHEDULED=False, BLOCK=block,
        num_warps=1,
    )


def scheduled_gumbel_softmax(logits, uniform_chunk, output, temperature_schedule,
                             kappa_schedule, step_pointer):
    choices = logits.shape[1]
    block = triton.next_power_of_2(choices)
    _gumbel_softmax_kernel[(logits.shape[0],)](
        logits, uniform_chunk, output, temperature_schedule, kappa_schedule,
        step_pointer, TILE_COUNT=logits.shape[0], CHOICES=choices,
        UNIFORM_CHUNK=uniform_chunk.shape[0], SCHEDULED=True, BLOCK=block,
        num_warps=1,
    )


def sparse_error(probabilities, baseline, indices, deltas, target, output):
    choices = probabilities.shape[1]
    width = indices.shape[2]
    block = triton.next_power_of_2(max(256, (choices - 1) * width))
    _sparse_error_kernel[(probabilities.shape[0],)](
        probabilities, baseline, indices, deltas, target, output,
        N=target.shape[1], OUTPUT_TILES=target.shape[1] // 16,
        CHOICES=choices, WIDTH=width, BLOCK=block, num_warps=4,
    )


def sparse_lion(probabilities, metric_error, indices, deltas, denominator,
                logits, momentum, norm_square, norm_minimum, norm_maximum,
                finite_state, tau: float, kappa: float, learning_rate: float,
                weight_decay: float):
    choices = probabilities.shape[1]
    block = triton.next_power_of_2(choices)
    _sparse_lion_kernel[(probabilities.shape[0],)](
        probabilities, metric_error, indices, deltas, denominator,
        logits, momentum, norm_square, tau, kappa, logits, learning_rate,
        1.0 - learning_rate * weight_decay,
        SCHEDULED=False,
        N=metric_error.shape[1], OUTPUT_TILES=metric_error.shape[1] // 16,
        CHOICES=choices, WIDTH=indices.shape[2], BLOCK=block, num_warps=1,
    )
    _finish_norm_kernel[(1,)](norm_square, norm_minimum, norm_maximum, finite_state,
                              num_warps=1)


def scheduled_sparse_lion(probabilities, metric_error, indices, deltas,
                          denominator, logits, momentum, norm_square,
                          norm_minimum, norm_maximum, finite_state,
                          temperature_schedule, kappa_schedule, step_pointer,
                          learning_rate: float, weight_decay: float):
    choices = probabilities.shape[1]
    block = triton.next_power_of_2(choices)
    _sparse_lion_kernel[(probabilities.shape[0],)](
        probabilities, metric_error, indices, deltas, denominator,
        logits, momentum, norm_square, temperature_schedule, kappa_schedule,
        step_pointer, learning_rate, 1.0 - learning_rate * weight_decay,
        SCHEDULED=True,
        N=metric_error.shape[1], OUTPUT_TILES=metric_error.shape[1] // 16,
        CHOICES=choices, WIDTH=indices.shape[2], BLOCK=block, num_warps=1,
    )
    _finish_norm_kernel[(1,)](norm_square, norm_minimum, norm_maximum,
                              finite_state, num_warps=1)
    _advance_step_kernel[(1,)](step_pointer, num_warps=1)


def build_position_map(indices, deltas):
    tile_count, alternatives, width = indices.shape
    overlap = width // 2
    counters = torch.zeros((tile_count, 256), dtype=torch.int32,
                           device=indices.device)
    choices = torch.zeros((tile_count, 256, overlap), dtype=torch.int32,
                          device=indices.device)
    values = torch.zeros((tile_count, 256, overlap), dtype=deltas.dtype,
                         device=deltas.device)
    entries = alternatives * width
    total = tile_count * entries
    block = 256
    _build_position_map_kernel[(triton.cdiv(total, block),)](
        indices, deltas, counters, choices, values,
        TOTAL=total, ENTRIES=entries, WIDTH=width, OVERLAP=overlap,
        BLOCK=block, num_warps=4,
    )
    return choices, values


def build_compact_position_map(indices, deltas, matrix_indices=None):
    """Compress P32's repeated candidate entries into unique tile positions."""
    tile_count, alternatives, width = indices.shape
    overlap = width // 2
    entries = alternatives * width
    flat_indices = indices.flatten(1)
    sorted_indices, order = flat_indices.sort(dim=1)
    is_new = torch.ones_like(sorted_indices, dtype=torch.bool)
    is_new[:, 1:] = sorted_indices[:, 1:] != sorted_indices[:, :-1]
    position_slot = is_new.cumsum(1) - 1
    positions = int(is_new.sum(1).max())
    entry = torch.arange(entries, device=indices.device).expand(tile_count, -1)
    run_start = torch.where(is_new, entry, 0).cummax(1).values
    overlap_slot = entry - run_start
    if bool((overlap_slot >= overlap).any()):
        raise ValueError("P32 sparse candidate position overlap exceeds compact capacity")
    compact_indices = torch.zeros(
        (tile_count, positions), dtype=torch.uint8, device=indices.device,
    )
    compact_indices.scatter_(1, position_slot, sorted_indices.to(torch.uint8))
    flat_deltas = deltas.flatten(1)
    entry_choices = torch.arange(
        1, alternatives + 1, device=indices.device, dtype=torch.int64,
    ).repeat_interleave(width).expand(tile_count, -1)
    if matrix_indices is not None:
        if matrix_indices.shape != indices.shape:
            raise ValueError("matrix indices do not match sparse candidate metadata")
        global_indices = matrix_indices.flatten()
        sorted_global_indices, global_order = global_indices.sort()
        global_new = torch.ones_like(sorted_global_indices, dtype=torch.bool)
        global_new[1:] = sorted_global_indices[1:] != sorted_global_indices[:-1]
        global_entry = torch.arange(len(global_order), device=indices.device)
        global_run_start = torch.where(global_new, global_entry, 0).cummax(0).values
        global_overlap_slot = global_entry - global_run_start
        if bool((global_overlap_slot >= overlap).any()):
            raise ValueError("P32 global candidate overlap exceeds compact capacity")
        ordered_tile = global_order // entries
        ordered_entry = global_order - ordered_tile * entries
        ordered_scalar = indices.flatten()[global_order]
        ordered_choice = ordered_entry // width + 1
        dense_slot = (
            (ordered_tile * 256 + ordered_scalar) * overlap + global_overlap_slot
        )
        dense_choices = torch.zeros(
            tile_count * 256 * overlap, dtype=torch.uint8, device=indices.device,
        )
        dense_deltas = torch.zeros_like(dense_choices, dtype=deltas.dtype)
        dense_choices.scatter_(0, dense_slot, ordered_choice.to(torch.uint8))
        dense_deltas.scatter_(0, dense_slot, deltas.flatten()[global_order])
        gather = compact_indices.long().unsqueeze(-1).expand(-1, -1, overlap)
        compact_choices = dense_choices.reshape(tile_count, 256, overlap).gather(1, gather)
        compact_deltas = dense_deltas.reshape(tile_count, 256, overlap).gather(1, gather)
        return compact_indices, compact_choices, compact_deltas
    sorted_deltas = flat_deltas.gather(1, order)
    sorted_choices = entry_choices.gather(1, order)
    compact_choices = torch.zeros(
        (tile_count, positions * overlap), dtype=torch.uint8,
        device=indices.device,
    )
    compact_deltas = torch.zeros(
        (tile_count, positions * overlap), dtype=deltas.dtype,
        device=deltas.device,
    )
    compact_slot = position_slot * overlap + overlap_slot
    compact_choices.scatter_(1, compact_slot, sorted_choices.to(torch.uint8))
    compact_deltas.scatter_(1, compact_slot, sorted_deltas)
    return (
        compact_indices,
        compact_choices.reshape(tile_count, positions, overlap),
        compact_deltas.reshape(tile_count, positions, overlap),
    )


def transpose_compact_position_map(indices, choices, deltas):
    """Order compact tile positions for coalesced transposed matrix stores."""
    active = choices[..., 0] > 0
    transpose_key = ((indices % 16) * 16 + indices // 16).to(torch.int16)
    padding_order = torch.arange(
        indices.shape[1], device=indices.device, dtype=torch.int16,
    )
    transpose_key = torch.where(
        active, transpose_key, 256 + padding_order.unsqueeze(0),
    )
    order = transpose_key.argsort(1)
    gather = order.unsqueeze(-1).expand_as(choices)
    return (
        indices.gather(1, order),
        choices.gather(1, gather),
        deltas.gather(1, gather),
    )


def candidate_probability_gradient(grad_matrix, matrix_indices, sparse_deltas,
                                   output):
    """Apply the exact sparse-mixture adjoint without an indexed gather op."""
    tile_count, alternatives, width = matrix_indices.shape
    if width != 6:
        raise ValueError("fused P32 candidate gradients require sparse width six")
    block = triton.next_power_of_2(alternatives + 1)
    _candidate_probability_gradient_kernel[(tile_count,)](
        grad_matrix, matrix_indices, sparse_deltas, output, tile_count,
        CHOICES=alternatives + 1, WIDTH=width,
        BLOCK=block, num_warps=1,
    )


def compact_sparse_mixture(probabilities, baseline, position_indices,
                           position_choices, position_deltas, output, *,
                           in_features=None, out_features=None,
                           transposed=False):
    """Materialize a P32 sparse mixture without deterministic scatter sorting."""
    tile_count = probabilities.shape[0]
    choices = probabilities.shape[1]
    positions = position_indices.shape[1]
    overlap = position_choices.shape[2]
    if overlap != 3:
        raise ValueError("fused P32 sparse mixtures require overlap three")
    if in_features is None or out_features is None:
        in_features, out_features = output.shape
    expected = ((out_features, in_features) if transposed
                else (in_features, out_features))
    if output.shape != expected:
        raise ValueError("fused P32 sparse mixture output shape is invalid")
    _sparse_mixture_compact_kernel[(tile_count,)](
        probabilities, baseline, position_indices, position_choices,
        position_deltas, output,
        TILE_COUNT=tile_count, IN_FEATURES=in_features,
        OUT_FEATURES=out_features, OUTPUT_TILES=out_features // 16,
        CHOICES=choices, POSITIONS=positions, OVERLAP=overlap,
        TRANSPOSED=transposed, BLOCK=256, num_warps=4,
    )


def scheduled_grouped_gumbel_softmax(logits, uniform_chunk, output,
                                     temperature_schedule, kappa_schedule,
                                     step_pointer):
    choices = logits.shape[1]
    tiles = 4
    block = triton.next_power_of_2(choices)
    _scheduled_gumbel_grouped_kernel[(triton.cdiv(logits.shape[0], tiles),)](
        logits, uniform_chunk, output, temperature_schedule, kappa_schedule,
        step_pointer, TILE_COUNT=logits.shape[0], CHOICES=choices,
        UNIFORM_CHUNK=uniform_chunk.shape[0], TILES=tiles, BLOCK=block,
        num_warps=4,
    )


def grouped_position_error(probabilities, baseline, position_choices,
                           position_deltas, target, output):
    tiles = 4
    _position_error_grouped_kernel[(triton.cdiv(probabilities.shape[0], tiles),)](
        probabilities, baseline, position_choices, position_deltas, target,
        output, TILE_COUNT=probabilities.shape[0], N=target.shape[1],
        OUTPUT_TILES=target.shape[1] // 16, CHOICES=probabilities.shape[1],
        OVERLAP=position_choices.shape[2], TILES=tiles, num_warps=8,
    )


def compact_position_error(probabilities, baseline, position_indices,
                           position_choices, position_deltas, target, output):
    block = 256
    tiles = 2
    _position_error_compact_kernel[(triton.cdiv(probabilities.shape[0], tiles),)](
        probabilities, baseline, position_indices, position_choices,
        position_deltas, target, output, TILE_COUNT=probabilities.shape[0],
        N=target.shape[1],
        OUTPUT_TILES=target.shape[1] // 16, CHOICES=probabilities.shape[1],
        POSITIONS=position_indices.shape[1], OVERLAP=position_choices.shape[2],
        TILES=tiles, BLOCK=block, num_warps=8,
    )


def scheduled_grouped_sparse_lion(
        probabilities, metric_error, indices, deltas, denominator, logits,
        momentum, norm_square, norm_minimum, norm_maximum, finite_state,
        temperature_schedule, kappa_schedule, step_pointer,
        learning_rate: float, weight_decay: float):
    choices = probabilities.shape[1]
    tiles = 4
    block = triton.next_power_of_2(choices)
    _scheduled_sparse_lion_grouped_kernel[(
        triton.cdiv(probabilities.shape[0], tiles),
    )](
        probabilities, metric_error, indices, deltas, denominator, logits,
        momentum, norm_square, temperature_schedule, kappa_schedule,
        step_pointer, learning_rate, 1.0 - learning_rate * weight_decay,
        TILE_COUNT=probabilities.shape[0], N=metric_error.shape[1],
        OUTPUT_TILES=metric_error.shape[1] // 16, CHOICES=choices,
        WIDTH=indices.shape[2], TILES=tiles, BLOCK=block, num_warps=4,
    )
    _finish_norm_kernel[(1,)](norm_square, norm_minimum, norm_maximum,
                              finite_state, num_warps=1)
    _advance_step_kernel[(1,)](step_pointer, num_warps=1)
