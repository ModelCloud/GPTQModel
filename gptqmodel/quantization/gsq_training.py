"""Primitives for the staged scalar GSQ training path (integration in progress).

Reference: IST-DASLab/GSQ 03fc16484c369e3127225615d5e03e8d3a6043e3.
These helpers alone do not implement the paper's staged training procedure.
"""

import math

import torch


def sampling_schedule(step, total_steps, *, temperature=(2., .5), multiplier=(10., 50.)):
    """Author training loop's linear per-update schedules, including endpoints.

    A single update uses the initial values; upstream divides by zero in that
    corner case. Reject invalid counters rather than extrapolating silently.
    """
    if (isinstance(step, bool) or isinstance(total_steps, bool)
            or not isinstance(step, int) or not isinstance(total_steps, int)
            or total_steps < 1 or not 0 <= step < total_steps):
        raise ValueError('GSQ schedule requires 0 <= step < positive total_steps')
    if any(not math.isfinite(v) or v <= 0 for pair in (temperature, multiplier) for v in pair):
        raise ValueError('GSQ sampling endpoints must be finite and positive')
    fraction = step / max(total_steps - 1, 1)
    return tuple(pair[0] + (pair[1]-pair[0])*fraction for pair in (temperature, multiplier))


def relaxed_scalar_weights(logits, scales, candidates, group_index, *, uniform, temperature, multiplier, initial=None):
    """Differentiable author-form scalar relaxation with explicit random draws.

    Explicit draws permit matched-noise gradient comparisons without disturbing
    global RNG state. Candidates have [choices,out,in] geometry; scales [out,group].
    """
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('GSQ temperature and multiplier must be finite and positive')
    if uniform.shape != logits.shape or candidates.shape != logits.shape:
        raise ValueError('GSQ logits, candidates and uniform draws must have identical shapes')
    if not torch.isfinite(uniform).all() or (uniform < 0).any() or (uniform > 1).any():
        raise ValueError('GSQ uniform draws must be finite in [0,1]')
    return _ScalarRelaxation.apply(logits, scales, candidates, group_index, uniform,
                                   temperature, multiplier, initial)


class _ScalarRelaxation(torch.autograd.Function):
    """Explicit author-order first derivative; draws are owned by the caller."""

    @staticmethod
    def forward(ctx, logits, scales, candidates, group_index, uniform, temperature, multiplier, initial):
        noise = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
        if initial is None:
            noise = noise.to(candidates.dtype)
            effective_logits = logits.to(candidates.dtype)
        else:
            effective_logits = logits
        probability = ((effective_logits*multiplier+noise)/temperature).softmax(0).to(candidates.dtype)
        expected = (probability*candidates).sum(0)
        if initial is not None:
            expected = initial+expected
        ctx.save_for_backward(probability, candidates, group_index, scales, expected)
        ctx.temperature, ctx.multiplier, ctx.logit_dtype = temperature, multiplier, logits.dtype
        return expected*scales[:, group_index].to(expected.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        probability, candidates, groups, scales, expected = ctx.saved_tensors
        scale_gradient = torch.zeros_like(scales)
        scale_gradient.scatter_add_(1, groups.unsqueeze(0).expand_as(expected),
                                    (grad_output*expected).to(scales.dtype))
        output_gradient = grad_output*scales[:, groups].to(expected.dtype)
        categorical_gradient = output_gradient.unsqueeze(0)*candidates
        dot = (categorical_gradient*probability).sum(0, keepdim=True)
        logit_gradient = probability*(categorical_gradient-dot)
        logit_gradient = logit_gradient*ctx.multiplier/ctx.temperature
        return logit_gradient.to(ctx.logit_dtype), scale_gradient, None, None, None, None, None, None


class GSQLion(torch.optim.Optimizer):
    """Lion update used by staged GSQ, with ordinary per-group weight decay.

    Matches lion-pytorch's default (non-Triton, decoupled_weight_decay=False)
    behavior. Scales belong in a separate group with zero weight decay.
    """

    def __init__(self, params, lr=1e-4, betas=(.9, .99), weight_decay=0.):
        if not math.isfinite(lr) or lr <= 0 or len(betas) != 2 or any(not 0 <= b <= 1 for b in betas):
            raise ValueError('GSQ Lion requires positive learning rate and two betas in [0,1]')
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError('GSQ Lion requires finite nonnegative weight decay')
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise ValueError('GSQ Lion requires dense gradients')
                state = self.state[parameter]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(parameter)
                momentum = state['exp_avg']
                if (parameter.is_cuda and parameter.dtype == torch.float32
                        and parameter.is_contiguous() and gradient.is_contiguous()
                        and momentum.is_contiguous()):
                    from .qvq_gsq_triton import lion_update

                    lion_update(
                        parameter, gradient, momentum,
                        learning_rate=group['lr'], beta1=beta1, beta2=beta2,
                        weight_decay=group['weight_decay'],
                    )
                    continue
                direction = momentum.clone().mul_(beta1).add(gradient, alpha=1-beta1).sign_()
                parameter.mul_(1-group['lr']*group['weight_decay'])
                parameter.add_(direction, alpha=-group['lr'])
                momentum.mul_(beta2).add_(gradient, alpha=1-beta2)
        return loss


def scalar_training_candidates(weight, scales, group_size, *, bits, std=.01, strength=6., noise):
    """Prepare the pinned author's W2 or W3/W4 initialization with explicit noise.

    ``weight`` is the already quantized initializer, not the dense teacher.
    Returned candidates are unscaled; invalid local shifts must stay masked.
    """
    if bits not in (2, 3, 4) or isinstance(bits, bool):
        raise ValueError('Staged scalar GSQ currently requires W2, W3 or W4')
    if (weight.ndim != 2 or not weight.is_floating_point() or not torch.isfinite(weight).all()
            or isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0):
        raise ValueError('GSQ initializer requires finite rank-2 weights and positive group size')
    rows, columns = weight.shape
    groups = torch.arange(columns, device=weight.device) // group_size
    if (scales.shape != (rows, (columns+group_size-1)//group_size) or scales.device != weight.device
            or not torch.isfinite(scales).all() or (scales == 0).any()):
        raise ValueError('GSQ initializer requires matching nonzero finite scales')
    initial = weight / scales[:, groups]
    if bits == 2:
        levels = torch.arange(-2, 2, dtype=weight.dtype, device=weight.device)[:, None, None]
        candidates = levels.expand(4, rows, columns)
        valid = torch.ones_like(candidates, dtype=torch.bool)
        prior = -.5*(initial.unsqueeze(0)-candidates).square()
        prior = prior-prior.mean(0, keepdim=True)
    else:
        shifts = torch.arange(-2, 3, dtype=weight.dtype, device=weight.device)[:, None, None]
        candidates = initial.unsqueeze(0)+shifts
        valid = (candidates >= -(2**(bits-1))) & (candidates <= 2**(bits-1)-1)
        if not valid.any(0).all():
            raise ValueError('GSQ initializer is outside the signed scalar grid')
        prior = (-.5*shifts.square()).expand_as(candidates)
        prior = prior-(prior*valid).sum(0, keepdim=True)/valid.sum(0, keepdim=True)
    if noise.shape != candidates.shape or noise.device != weight.device or not torch.isfinite(noise).all():
        raise ValueError('GSQ initializer noise must match candidate geometry and be finite')
    return dict(candidates=candidates.detach(), valid=valid, group_index=groups,
                logits=(std*(noise+prior*strength)).detach(), scales=scales.float().detach().clone())


class GSQScalarTrainingModule(torch.nn.Module):
    """Trainable scalar assignments and group scales for staged reconstruction.

    This module owns weights only; the caller supplies the actual attention or
    block objective. No per-projection hard-loss guard substitutes for that loss.
    """

    def __init__(self, weight, scales, group_size, *, bits, noise, std=.01, strength=6., logits_dtype=None):
        super().__init__()
        prepared = scalar_training_candidates(weight, scales, group_size, bits=bits,
                                               noise=noise, std=std, strength=strength)
        self.register_buffer('initial', None if bits == 2 else
                             (weight/scales[:, prepared['group_index']]).detach().clone())
        self.logits = torch.nn.Parameter(prepared['logits'].to(logits_dtype or weight.dtype))
        self.scales = torch.nn.Parameter(prepared['scales'])
        for name in ('candidates', 'valid', 'group_index'):
            self.register_buffer(name, prepared[name])

    def forward(self, *, uniform, temperature, multiplier):
        masked = self.logits.masked_fill(~self.valid, -torch.inf)
        candidates = self.candidates
        if self.initial is not None:
            candidates = torch.arange(-2, 3, device=candidates.device, dtype=candidates.dtype)[:, None, None]
            candidates = candidates.expand_as(self.candidates)
        return relaxed_scalar_weights(masked, self.scales, candidates, self.group_index,
                                      uniform=uniform, temperature=temperature, multiplier=multiplier,
                                      initial=self.initial)

    @torch.no_grad()
    def hard_weight(self):
        selected = self.logits.masked_fill(~self.valid, -torch.inf).argmax(0, keepdim=True)
        assignments = self.candidates.gather(0, selected).squeeze(0)
        return assignments * self.scales[:, self.group_index].to(assignments.dtype)

    def optimizer_groups(self, *, assignment_lr, scale_lr, weight_decay):
        return [{'params': [self.logits], 'lr': assignment_lr, 'weight_decay': weight_decay},
                {'params': [self.scales], 'lr': scale_lr, 'weight_decay': 0.}]


def reconstruction_stage_loss(module, args, kwargs, *, student_weights, teacher_weights=None, output_select=None,
                              output_mask=None):
    """Evaluate an actual stage with differentiable weight substitutions.

    The caller chooses the stage module/output (linear, attention-to-MLP, or
    complete block) and supplies saved teacher attention weights when needed.
    Dense parameters are detached for both paths; only supplied student weights
    receive gradients. Functional substitution restores module state on errors.
    """
    parameters = {name: value.detach() for name, value in module.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in module.named_buffers()}
    teacher_weights = {} if teacher_weights is None else teacher_weights
    for replacements in (student_weights, teacher_weights):
        for name, value in replacements.items():
            if name not in parameters or value.shape != parameters[name].shape:
                raise ValueError(f'GSQ stage replacement does not match parameter {name}')
    def match_parameter(name, value):
        parameter = parameters[name]
        if value.device != parameter.device or value.dtype != parameter.dtype:
            return value.to(device=parameter.device, dtype=parameter.dtype)
        return value

    teacher_state = {**parameters, **{name: match_parameter(name, value.detach())
                                      for name, value in teacher_weights.items()}}
    with torch.no_grad():
        teacher = torch.func.functional_call(module, (teacher_state, buffers), args, kwargs)
        teacher = teacher if output_select is None else output_select(teacher)
    student_state = {**parameters, **{name: match_parameter(name, value) for name, value in student_weights.items()}}
    student = torch.func.functional_call(module, (student_state, buffers), args, kwargs)
    student = student if output_select is None else output_select(student)
    if not isinstance(student, torch.Tensor) or not isinstance(teacher, torch.Tensor):
        raise ValueError('GSQ stage requires tensor outputs or an output selector')
    if output_mask is not None:
        if (not isinstance(output_mask, torch.Tensor) or output_mask.dtype != torch.bool
                or output_mask.shape != student.shape[:-1] or output_mask.device != student.device
                or not output_mask.any()):
            raise ValueError('GSQ output mask must select valid tokens with an aligned boolean tensor')
        # Select before subtraction: excluded padding cannot affect the loss
        # denominator or introduce nonfinite arithmetic into reconstruction.
        student, teacher = student[output_mask], teacher[output_mask]
    return torch.nn.functional.mse_loss(student, teacher)


def reconstruction_stage_student_loss(module, args, kwargs, *, student_weights,
                                      teacher, output_mask=None):
    """Evaluate only the trainable path against an exact cached teacher output."""
    parameters = {name: value.detach() for name, value in module.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in module.named_buffers()}
    for name, value in student_weights.items():
        if name not in parameters or value.shape != parameters[name].shape:
            raise ValueError(f'GSQ stage replacement does not match parameter {name}')
    student_state = {**parameters}
    for name, value in student_weights.items():
        parameter = parameters[name]
        student_state[name] = value.to(
            device=parameter.device, dtype=parameter.dtype,
        ) if value.device != parameter.device or value.dtype != parameter.dtype else value
    student = torch.func.functional_call(
        module, (student_state, buffers), args, kwargs,
    )
    if not isinstance(student, torch.Tensor) or not isinstance(teacher, torch.Tensor):
        raise TypeError('GSQ cached stage requires tensor student and teacher outputs')
    if student.shape != teacher.shape or student.device != teacher.device:
        raise ValueError('GSQ cached teacher output does not match the student stage')
    if output_mask is not None:
        if (not isinstance(output_mask, torch.Tensor) or output_mask.dtype != torch.bool
                or output_mask.shape != student.shape[:-1] or output_mask.device != student.device
                or not output_mask.any()):
            raise ValueError('GSQ output mask must select valid tokens with an aligned boolean tensor')
        student, teacher = student[output_mask], teacher[output_mask]
    return torch.nn.functional.mse_loss(student, teacher)


def train_stage_update(quantizers, optimizer, microbatches, objective, *, generator,
                       temperature, multiplier):
    """One accumulated staged update; objective(batch, weights) returns mean MSE.

    Microbatches are (batch, element_count) pairs. Counts describe reconstructed
    output elements, so partial batches contribute proportionally. Unlike the
    pinned author's floor-division loop, no remainder is silently discarded.
    """
    microbatches = list(microbatches)
    if not microbatches or any(isinstance(count, bool) or not isinstance(count, int) or count <= 0
                              for _, count in microbatches):
        raise ValueError('GSQ accumulation requires positive output element counts')
    total = sum(count for _, count in microbatches)
    optimizer.zero_grad(set_to_none=True)
    reported = None
    finite = torch.ones((), dtype=torch.bool, device=next(iter(quantizers.values())).logits.device)
    try:
        for batch, count in microbatches:
            weights = {}
            for name, quantizer in quantizers.items():
                uniform = torch.rand(
                    quantizer.logits.shape, dtype=quantizer.logits.dtype,
                    device=quantizer.logits.device, generator=generator,
                )
                weights[name] = quantizer(
                    uniform=uniform, temperature=temperature,
                    multiplier=multiplier,
                )
            loss = objective(batch, weights)
            if loss.ndim != 0:
                raise ValueError('GSQ stage objective must be a scalar')
            finite.logical_and_(torch.isfinite(loss))
            fraction = count/total
            (loss*fraction).backward()
            contribution = loss.detach()*fraction
            reported = contribution if reported is None else reported+contribution
        gradients = [
            parameter.grad
            for group in optimizer.param_groups
            for parameter in group['params']
            if parameter.grad is not None
        ]
        if gradients and all(
            gradient.is_cuda and gradient.is_floating_point()
            and not gradient.is_sparse
            for gradient in gradients
        ):
            # GSQ's logits and scales are dense CUDA tensors. The AMP
            # multi-tensor primitive checks the whole list in one pass; an
            # inverse scale of one leaves every gradient byte unchanged.
            state = getattr(optimizer, '_gsq_finite_state', None)
            if state is None or state[0].device != finite.device:
                state = (
                    torch.zeros((), dtype=torch.float32, device=finite.device),
                    torch.ones((), dtype=torch.float32, device=finite.device),
                )
                optimizer._gsq_finite_state = state
            found_inf, inverse_scale = state
            found_inf.zero_()
            torch._amp_foreach_non_finite_check_and_unscale_(
                gradients, found_inf, inverse_scale,
            )
            finite.logical_and_(found_inf == 0)
        else:
            for gradient in gradients:
                finite.logical_and_(torch.isfinite(gradient).all())
        # Keep nonfinite detection on-device. Converting each check to a Python
        # bool serializes the CUDA stream once per loss and trainable tensor.
        torch._assert_async(finite, 'GSQ stage objective or gradient is nonfinite')
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    return reported


@torch.no_grad()
def _hard_stage_weights(quantizers):
    """Materialize one reusable set of deterministic hard stage weights."""
    return {
        name: (
            quantizer.hard_weight_for_evaluation()
            if hasattr(quantizer, "hard_weight_for_evaluation")
            else quantizer.hard_weight()
        )
        for name, quantizer in quantizers.items()
    }


@torch.no_grad()
def evaluate_hard_stage(quantizers, batches, objective, *, weights=None):
    """Measure deterministic hard assignments over the complete stage dataset."""
    import logging
    import time

    # Candidate banks validate every serialized payload when they are built.
    # Their training-only materializer can therefore avoid repeating the much
    # more expensive pack/unpack legality audit at every held-out checkpoint.
    # Public/export materialization deliberately continues to use hard_weight.
    if weights is None:
        weights = _hard_stage_weights(quantizers)
    elif set(weights) != set(quantizers):
        raise ValueError('GSQ hard stage weights do not match quantizers')
    losses = []
    counts = []
    elements = 0
    progress_at = time.monotonic()+60
    for batch_index in range(len(batches)):
        for batch, count in batches[batch_index]:
            loss = objective(batch, weights)
            if loss.ndim != 0:
                raise ValueError('GSQ hard stage objective must be a finite scalar')
            losses.append(loss.detach())
            counts.append(count)
            elements += count
        if time.monotonic() >= progress_at:
            logging.getLogger(__name__).info(
                'GSQ hard-stage evaluation batches=%d/%d',
                batch_index+1,
                len(batches),
            )
            progress_at = time.monotonic()+60
    if not elements:
        raise ValueError('GSQ hard stage evaluation requires output elements')
    # Transfer all scalar losses together. Converting every microbatch loss to
    # ``float`` above would serialize the CUDA stream once per microbatch. The
    # CPU weighted sum retains the historical binary64 operation order.
    loss_values = torch.stack(losses).cpu().tolist()
    if not all(math.isfinite(value) for value in loss_values):
        raise ValueError('GSQ hard stage objective must be a finite scalar')
    return sum(value*count for value, count in zip(loss_values, counts))/elements


class LlamaGSQAttentionStage(torch.nn.Module):
    """Llama author objective: residual plus attention, before post-attention norm."""

    def __init__(self, decoder_layer):
        super().__init__()
        self.input_layernorm = decoder_layer.input_layernorm
        self.self_attn = decoder_layer.self_attn

    def forward(self, hidden_states, **kwargs):
        if kwargs.get('use_cache', False) or kwargs.get('past_key_values') is not None:
            raise ValueError('GSQ reconstruction requires cache-free attention')
        output, _ = self.self_attn(self.input_layernorm(hidden_states), **kwargs)
        return hidden_states+output


class LlamaGSQMLPStage(torch.nn.Module):
    """Llama MLP objective beginning at the fixed post-attention residual."""

    def __init__(self, decoder_layer):
        super().__init__()
        self.post_attention_layernorm = decoder_layer.post_attention_layernorm
        self.mlp = decoder_layer.mlp

    def forward(self, hidden_states):
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


def stage_learning_rate(step, total_steps, *, base_lr, warmup_steps=0, min_lr=0., decay='linear'):
    """Pinned author's scheduler, applied before the corresponding Lion update."""
    if total_steps < 1 or not 0 <= step < total_steps or not 0 <= warmup_steps < total_steps:
        raise ValueError('GSQ learning-rate schedule has invalid step or warmup bounds')
    if decay not in ('linear', 'cosine', 'constant') or not 0 <= min_lr <= 1:
        raise ValueError('GSQ learning-rate decay or minimum fraction is invalid')
    if step < warmup_steps:
        return base_lr*(min_lr+(1-min_lr)*step/warmup_steps)
    if decay == 'constant':
        return base_lr
    remaining = total_steps-1-warmup_steps
    fraction = (step-warmup_steps)/remaining if remaining else 1.
    factor = 1-fraction if decay == 'linear' else .5*(1+math.cos(math.pi*fraction))
    return base_lr*(min_lr+(1-min_lr)*factor)


def fit_reconstruction_stage(quantizers, batches, objective, *, epochs, seed=7,
                             assignment_lr=2e-4, scale_lr=1e-4, weight_decay=1., betas=(.9, .95),
                             temperature=(2., .5), multiplier=(10., 50.), warmup_steps=0,
                             min_lr=0., decay='linear', optimizer='lion',
                             validation_batches=None, restore_best=False,
                             fp32_tail_epochs=0, validation_start_epoch=0,
                             export_weights=True):
    """Train one stage with an optional held-out guard and hard-weight export.

    Each batch contains (microbatch, output_element_count) entries. The caller
    owns capture, teacher state and stage ordering. When validation batches are
    supplied, hard checkpoints are selected only on that disjoint objective.
    Epochs shuffle whole batches, as in the author trainer, using private RNG.
    ``validation_start_epoch`` can defer per-epoch hard checkpoints while the
    initial held-out baseline remains eligible for restoration.
    ``export_weights=False`` avoids a redundant dense materialization when the
    caller exports the quantizer's legal payload directly.
    """
    if not quantizers or not batches or isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError('GSQ stage fitting requires quantizers, batches and positive epochs')
    if restore_best and not validation_batches:
        raise ValueError('GSQ restore_best requires nonempty held-out validation batches')
    if (isinstance(fp32_tail_epochs, bool) or not isinstance(fp32_tail_epochs, int)
            or not 0 <= fp32_tail_epochs <= epochs):
        raise ValueError('GSQ FP32 tail epochs must be an integer in [0, epochs]')
    if fp32_tail_epochs and any(not hasattr(quantizer, 'training_dtype')
                               for quantizer in quantizers.values()):
        raise ValueError('GSQ FP32 tail requires quantizers with a training dtype')
    if (isinstance(validation_start_epoch, bool)
            or not isinstance(validation_start_epoch, int)
            or not 0 <= validation_start_epoch < epochs):
        raise ValueError('GSQ validation start epoch must be an integer in [0, epochs)')
    if not isinstance(export_weights, bool):
        raise ValueError('GSQ export_weights must be boolean')
    import time

    started = time.perf_counter()
    initial_hard_weights = _hard_stage_weights(quantizers)
    hard_loss_before = evaluate_hard_stage(
        quantizers, batches, objective, weights=initial_hard_weights,
    )
    validation_hard_loss_before = (
        evaluate_hard_stage(
            quantizers, validation_batches, objective,
            weights=initial_hard_weights,
        )
        if validation_batches else None
    )
    del initial_hard_weights

    def parameter_snapshot():
        return {
            quantizer_name: {
                parameter_name: parameter.detach().clone()
                for parameter_name, parameter in quantizer.named_parameters()
            }
            for quantizer_name, quantizer in quantizers.items()
        }

    best_validation_loss = validation_hard_loss_before
    best_epoch = -1
    best_parameters = parameter_snapshot() if restore_best else None
    groups = []
    for quantizer in quantizers.values():
        groups.extend(quantizer.optimizer_groups(assignment_lr=assignment_lr, scale_lr=scale_lr,
                                                weight_decay=weight_decay))
    if optimizer not in ('lion', 'adamw'):
        raise ValueError('GSQ optimizer must be lion or adamw')
    optimizer = (GSQLion(groups, betas=betas) if optimizer == 'lion' else
                 torch.optim.AdamW(groups, betas=betas, eps=1e-8, foreach=False, fused=False))
    initial_lrs = [group['lr'] for group in optimizer.param_groups]
    device = next(iter(quantizers.values())).logits.device
    sampling_rng = torch.Generator(device=device).manual_seed(seed)
    shuffle_rng = torch.Generator().manual_seed(seed)
    total_steps = epochs*len(batches)
    history = []
    validation_history = []
    progress_at = time.monotonic()+60
    for epoch in range(epochs):
        if fp32_tail_epochs and epoch == epochs-fp32_tail_epochs:
            for quantizer in quantizers.values():
                quantizer.training_dtype = torch.float32
        for index in torch.randperm(len(batches), generator=shuffle_rng).tolist():
            step = len(history)
            tau, kappa = sampling_schedule(step, total_steps, temperature=temperature, multiplier=multiplier)
            for group, base_lr in zip(optimizer.param_groups, initial_lrs):
                group['lr'] = stage_learning_rate(step, total_steps, base_lr=base_lr, warmup_steps=warmup_steps,
                                                  min_lr=min_lr, decay=decay)
            loss = train_stage_update(quantizers, optimizer, batches[index], objective, generator=sampling_rng,
                                      temperature=tau, multiplier=kappa)
            if step == 0 or (step+1) % 100 == 0 or step+1 == total_steps or time.monotonic() >= progress_at:
                import logging

                logging.getLogger(__name__).info("GSQ stage update %d/%d loss=%g", step+1, total_steps, loss)
                progress_at = time.monotonic()+60
            history.append(dict(epoch=epoch, step=step, batch=index, loss=loss, temperature=tau,
                                multiplier=kappa, learning_rates=[group['lr'] for group in optimizer.param_groups]))
        if validation_batches and epoch >= validation_start_epoch:
            validation_loss = evaluate_hard_stage(quantizers, validation_batches, objective)
            validation_history.append(dict(epoch=epoch, hard_loss=validation_loss))
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                if restore_best:
                    best_parameters = parameter_snapshot()
    # Preserve the public CPU-scalar history with one device transfer instead
    # of one synchronizing scalar conversion per optimizer update.
    history_losses = torch.stack([entry['loss'] for entry in history]).cpu().tolist()
    for entry, loss in zip(history, history_losses):
        entry['loss'] = loss
    if restore_best:
        with torch.no_grad():
            for quantizer_name, quantizer in quantizers.items():
                parameters = dict(quantizer.named_parameters())
                for parameter_name, value in best_parameters[quantizer_name].items():
                    parameters[parameter_name].copy_(value)
    result = dict(weights=(
                      {name: quantizer.hard_weight().detach().clone()
                       for name, quantizer in quantizers.items()}
                      if export_weights else None
                  ),
                  scales={name: quantizer.scales.detach().clone() for name, quantizer in quantizers.items()},
                  history=history, hard_loss_before=hard_loss_before)
    final_hard_weights = _hard_stage_weights(quantizers)
    result['hard_loss_after'] = evaluate_hard_stage(
        quantizers, batches, objective, weights=final_hard_weights,
    )
    result['hard_loss_delta'] = result['hard_loss_after']-hard_loss_before
    result['validation_history'] = validation_history
    result['validation_hard_loss_before'] = validation_hard_loss_before
    if not validation_batches:
        result['validation_hard_loss_after'] = None
    elif restore_best:
        result['validation_hard_loss_after'] = best_validation_loss
    else:
        result['validation_hard_loss_after'] = evaluate_hard_stage(
            quantizers, validation_batches, objective,
            weights=final_hard_weights,
        )
    del final_hard_weights
    result['best_validation_hard_loss'] = best_validation_loss
    result['best_validation_epoch'] = best_epoch
    result['restored_best_validation_checkpoint'] = bool(restore_best)
    result['fp32_tail_epochs'] = fp32_tail_epochs
    result['validation_start_epoch'] = validation_start_epoch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    result['elapsed_seconds'] = time.perf_counter()-started
    return result


def fit_llama_stages(layer, initializers, batches, *, bits, group_size, epochs, seed=7,
                     qk_steps=2000, qk_damp_percent=.01, reinitialize_mlp=True, initializer='gptq',
                     batch_size=1, microbatch_size=1, affine_initializers=False, **training):
    """Fit a Llama block in author stage order from supplied scalar initializers.

    Batches are (hidden_states, attention_kwargs) pairs without padding. Caller
    owns GPTQ initialization, disjoint data preparation, packing and evaluation.
    The source layer is never mutated. Returns its fitted copy and stage records.
    """
    import copy

    if initializer not in ('gptq', 'gptq_signed', 'rtn'):
        raise ValueError('Unknown staged GPTQ initializer')
    if affine_initializers and reinitialize_mlp:
        raise ValueError('Affine staged fitting must retain supplied MLP initialization, not replace it with GPTQ')
    for value in (batch_size, microbatch_size):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('GSQ batch sizes must be positive integers')
    if microbatch_size > batch_size:
        raise ValueError('GSQ microbatch size exceeds optimizer batch size')
    attention_implementation = layer.self_attn.config._attn_implementation
    if attention_implementation not in ('eager', 'sdpa'):
        raise ValueError('Staged GSQ requires eager or SDPA Llama attention')
    device = next(layer.parameters()).device
    offloaded = getattr(batches, 'offloaded', None)
    if offloaded is None:
        offloaded = any(hidden.device != device for hidden, _ in batches)
    if attention_implementation == 'sdpa':
        fixed_length = getattr(batches, 'fixed_sequence_length', None)
        lengths = {fixed_length} if fixed_length is not None else {hidden.shape[1] for hidden, _ in batches}
        has_attention_mask = getattr(batches, 'has_attention_mask', None)
        if has_attention_mask is None:
            has_attention_mask = any(kwargs.get('attention_mask') is not None for _, kwargs in batches)
        if len(lengths) != 1 or has_attention_mask:
            raise ValueError('SDPA staged GSQ requires equal-length documents without explicit attention masks')
    names = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
             'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    if set(initializers) != set(names) or not batches:
        raise ValueError('Llama GSQ requires all seven projection initializers and nonempty batches')
    initializers = dict(initializers)
    fitted = copy.deepcopy(layer).eval()
    records = {}

    def quantizer(name):
        if affine_initializers:
            codes, scales, zeros = initializers[name]
            weight = codes.to(scales.dtype)
        else:
            weight, scales = initializers[name]
        count = 4 if bits == 2 else 5
        rng = torch.Generator(device=weight.device).manual_seed(seed)
        noise = torch.randn((count, *weight.shape), dtype=weight.dtype, device=weight.device, generator=rng)
        if affine_initializers:
            from .gsq_training_affine import GSQAffineTrainingModule

            return GSQAffineTrainingModule(codes, scales, zeros, group_size, bits=bits, noise=noise,
                                           logits_dtype=torch.float32 if name in names[:2] else weight.dtype)
        return GSQScalarTrainingModule(weight, scales, group_size, bits=bits, noise=noise,
                                       logits_dtype=torch.float32 if name in names[:2] else weight.dtype)

    def run(stage_name, stage, selected, stage_batches, teacher=None):
        quantizers = {name+'.weight': quantizer(name) for name in selected}

        def objective(batch, weights):
            inputs, kwargs, mask = batch
            return reconstruction_stage_loss(stage, (inputs,), kwargs, student_weights=weights,
                                             teacher_weights=teacher, output_mask=mask)
        result = fit_reconstruction_stage(quantizers, stage_batches, objective,
                                          epochs=epochs, seed=seed, **training)
        if affine_initializers:
            result['zeros'] = {name: quant.zeros.detach().clone() for name, quant in quantizers.items()}
            result['codes'] = {name: quant.hard_codes().detach().clone() for name, quant in quantizers.items()}
        with torch.no_grad():
            for name in selected:
                fitted.get_submodule(name).weight.copy_(result['weights'][name+'.weight'])
        records[stage_name] = result

    # Keep names consistent with the containing block for functional replacement.
    qk_inputs = batches.iter_hidden_batches() if hasattr(batches, 'iter_hidden_batches') else (
        hidden for hidden, _ in batches)
    factor, dead = prepare_qk_calibration_factor(
        qk_inputs,
        damp_percent=qk_damp_percent,
        device=device,
        transform=fitted.input_layernorm,
    )
    for name in names[:2]:
        projection = fitted.get_submodule(name)
        quant = quantizer(name)
        qk_option_names = (
            'assignment_lr', 'scale_lr', 'betas', 'weight_decay', 'temperature', 'multiplier', 'optimizer',
        )
        qk_options = {key: value for key, value in training.items() if key in qk_option_names}
        result = fit_qk_projection(
            quant,
            projection.weight,
            factor=factor,
            dead=dead,
            steps=qk_steps,
            damp_percent=qk_damp_percent,
            seed=seed,
            **qk_options,
        )
        if affine_initializers:
            result['zeros'] = {'weight': quant.zeros.detach().clone()}
            result['codes'] = {'weight': quant.hard_codes().detach().clone()}
        result['objective'] = 'prepared_qk_quadratic_sum'
        result['damp_percent'] = qk_damp_percent
        with torch.no_grad():
            projection.weight.copy_(result['weights']['weight'])
        records[name] = result
    if batch_size == microbatch_size == 1 and not offloaded:
        staged_batches = [[((hidden, kwargs, None), hidden.numel())] for hidden, kwargs in batches]
    else:
        from .gsq_batching import llama_stage_batches

        lazy_stage_batches = offloaded or (
            hasattr(batches, 'iter_hidden_batches') and hasattr(batches, 'iter_batches')
        )
        staged_batches = llama_stage_batches(
            batches,
            batch_size=batch_size,
            microbatch_size=microbatch_size,
            device=device,
            implicit_causal=attention_implementation == 'sdpa',
            lazy=lazy_stage_batches,
        )
    # Q/K have already been fitted above.  The attention stage therefore uses
    # fitted Q/K plus dense V/O as its teacher, matching the paper's sequential
    # attention objective; overriding all attention weights with the original
    # dense snapshot would train V/O against a state the final block does not
    # use.
    run('attention', LlamaGSQAttentionStage(fitted), names[2:4], staged_batches)
    mlp_metadata = None
    if reinitialize_mlp:
        refreshed, mlp_metadata = initialize_llama_gptq(fitted, batches, bits=bits, group_size=group_size,
                                                       damp_percent=qk_damp_percent, projections=names[4:],
                                                       initializer=initializer)
        initializers.update(refreshed)
    # The MLP stage must preserve the already-fitted attention on both sides
    # of its reconstruction target.  Passing ``teacher_attention`` here would
    # silently replace the fitted Q/K/V/O weights with the dense attention,
    # while the student still used the fitted attention, so the optimizer
    # would train the MLP against a teacher that the final staged block could
    # never reproduce.  This is the paper's sequential objective: fitted
    # attention + dense MLP -> fitted attention + trainable MLP.
    run('mlp', fitted, names[4:], staged_batches)
    records['mlp']['initializer_timing'] = 'after_attention' if reinitialize_mlp else 'before_attention'
    records['mlp']['initializer_metadata'] = mlp_metadata
    return fitted, records


def initialize_llama_gptq(layer, batches, *, bits, group_size, damp_percent=.01, projections=None, initializer='gptq'):
    """Capture real projection inputs and prepare symmetric GPTQ stage seeds.

    This initializer uses this repository's GPTQ, not the author's fork. Its
    numerical parity must be assessed separately from GSQ training parity.
    """
    import copy
    import logging
    import time

    from .config import GPTQConfig, HessianConfig, LengthAwareConfig, LengthAwareMode
    from .gptq import GPTQ

    if bits not in (2, 3, 4) or not batches:
        raise ValueError('Staged Llama GPTQ requires W2/W3/W4 and calibration batches')
    if initializer not in ('gptq', 'gptq_signed', 'rtn'):
        raise ValueError('Unknown staged GPTQ initializer')
    working = copy.deepcopy(layer).eval()
    tasks, handles = {}, []
    names = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
             'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    if projections is not None:
        if not projections or not set(projections).issubset(names):
            raise ValueError('Unsupported Llama GPTQ projection subset')
        names = tuple(projections)
    if initializer == 'rtn':
        from .config import RTNConfig
        from .rtn import RTN

        initializers, metadata = {}, {}
        for name in names:
            module = working.get_submodule(name)
            config = RTNConfig(
                bits=bits,
                group_size=group_size,
                sym=True,
                desc_act=False,
                offload_to_disk=False,
            )
            quantizer = RTN(module, config)
            weight, scales, zeros, groups, _, loss, damp, samples = quantizer.quantize()
            if not torch.all(zeros == 2**(bits - 1)):
                raise ValueError('RTN initializer zero point does not match the signed GSQ grid')
            expected = torch.arange(weight.shape[1], device=groups.device) // group_size
            if not torch.equal(groups, expected.to(groups.dtype)):
                raise ValueError('RTN initializer grouping does not match contiguous GSQ groups')
            initializers[name] = (weight.detach().clone(), scales.detach().clone())
            metadata[name] = dict(loss=loss, damp=damp, samples=samples)
        return initializers, metadata
    try:
        for name in names:
            module = working.get_submodule(name)
            prior = dict(mse=2.4, scale_search='mse') if initializer == 'gptq_signed' else {}
            config = GPTQConfig(bits=bits, group_size=group_size, sym=True, desc_act=False,
                                damp_percent=damp_percent, gsq=None, act_group_aware=False,
                                offload_to_disk=False,
                                # The GSQ author GPTQ uses H = 2/N_seq * X^T X.
                                # The repository default is a bucketed length-aware
                                # estimator that is not materialized here and falls
                                # back to token-count normalization.
                                hessian=HessianConfig(
                                    length_aware=LengthAwareConfig(
                                        mode=LengthAwareMode.SEQUENCE_COUNT
                                    )
                                ),
                                **prior)
            task = GPTQ(module, config)
            if initializer == 'gptq_signed':
                from .gsq_initialization import SignedGSQQuantizer

                task.quantizer = SignedGSQQuantizer(config)
            task.quantizer.configure(perchannel=True)
            tasks[name] = task

            def capture(_module, inputs, output, task=task):
                task.add_batch(inputs[0].detach(), output.detach())
            handles.append(module.register_forward_hook(capture))
        device = next(working.parameters()).device
        moved_metadata = {}

        def move(value):
            identity = id(value)
            if identity in moved_metadata:
                return moved_metadata[identity]
            if isinstance(value, torch.Tensor):
                result = value.detach().to(device)
            elif isinstance(value, tuple):
                result = tuple(move(item) for item in value)
            elif isinstance(value, list):
                result = [move(item) for item in value]
            elif isinstance(value, dict):
                result = {key: move(item) for key, item in value.items()}
            else:
                result = value
            moved_metadata[identity] = result
            return result

        with torch.no_grad():
            progress_at = time.monotonic()+60
            source = batches.iter_batches() if hasattr(batches, 'iter_batches') else batches
            for batch_index, (hidden, kwargs) in enumerate(source):
                moved_hidden = hidden.to(device, non_blocking=True)
                output = working(moved_hidden, **move(kwargs))
                del output, moved_hidden, hidden, kwargs
                if time.monotonic() >= progress_at:
                    logging.getLogger(__name__).info(
                        'GSQ GPTQ initialization projections=%s batches=%d/%d',
                        ','.join(names),
                        batch_index+1,
                        len(batches),
                    )
                    progress_at = time.monotonic()+60
        for handle in handles:
            handle.remove()
        handles.clear()
        initializers, metadata = {}, {}
        for name, task in tasks.items():
            weight, scales, zeros, groups, _, loss, damp, samples = task.quantize()
            if not torch.all(zeros == 2**(bits-1)):
                raise ValueError('GPTQ initializer zero point does not match the signed GSQ grid')
            expected = torch.arange(weight.shape[1], device=groups.device)//group_size
            if not torch.equal(groups, expected.to(groups.dtype)):
                raise ValueError('GPTQ initializer grouping does not match contiguous GSQ groups')
            initializers[name] = (weight.detach().clone(), scales.detach().clone())
            metadata[name] = dict(loss=loss, damp=damp, samples=samples)
        return initializers, metadata
    finally:
        for handle in handles:
            handle.remove()
        for task in tasks.values():
            task.free()


def prepare_qk_calibration_factor(inputs, *, damp_percent=.01, device=None, transform=None):
    """Author Q/K metric: 2/sequence_count Gram, dead diagonal repair, damping."""
    if not math.isfinite(damp_percent) or damp_percent < 0:
        raise ValueError('GSQ Q/K factor requires inputs and nonnegative damping')
    import logging
    import time

    iterator = iter(inputs)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError('GSQ Q/K factor requires inputs and nonnegative damping') from error
    width = first.shape[-1]
    device = first.device if device is None else torch.device(device)
    gram = torch.zeros(width, width, device=device, dtype=torch.float32)
    sequences = 0
    progress_at = time.monotonic()+60
    total = len(inputs) if hasattr(inputs, '__len__') else None
    input_index = 0
    while True:
        batch = first if input_index == 0 else next(iterator, None)
        if batch is None:
            break
        if batch.ndim != 3 or batch.shape[-1] != width or not torch.isfinite(batch).all():
            raise ValueError('GSQ Q/K inputs require finite [batch,tokens,in] geometry')
        batch = batch.detach().to(device, non_blocking=True)
        if transform is not None:
            with torch.no_grad():
                batch = transform(batch)
        flattened = batch.reshape(-1, width).float()
        gram.add_(flattened.T @ flattened)
        sequences += batch.shape[0]
        input_index += 1
        if time.monotonic() >= progress_at:
            logging.getLogger(__name__).info(
                'GSQ Q/K calibration inputs=%d/%s',
                input_index,
                total if total is not None else '?',
            )
            progress_at = time.monotonic()+60
        del batch, flattened
    if not sequences:
        raise ValueError('GSQ Q/K factor has no sequences')
    gram.mul_(2/sequences)
    dead = gram.diagonal() == 0
    gram.diagonal()[dead] = 1
    gram.diagonal().add_(damp_percent*gram.diagonal().mean())
    return torch.linalg.cholesky(gram), dead


def fit_qk_projection(quantizer, teacher, inputs=None, *, factor=None, dead=None, steps=2000,
                      damp_percent=.01, seed=7,
                      assignment_lr=1e-4, scale_lr=5e-5, betas=(.9, .95), weight_decay=1.,
                      temperature=(2., .05), multiplier=(100., 500.), optimizer='lion'):
    """Dedicated constant-LR Q/K training, using the prepared quadratic sum."""
    if factor is None or dead is None:
        factor, dead = prepare_qk_calibration_factor(inputs, damp_percent=damp_percent)
    target = teacher.detach().float().clone()
    target[:, dead] = 0

    def objective(_batch, weights):
        return ((target-weights['weight']) @ factor).square().sum()
    return fit_reconstruction_stage({'weight': quantizer}, [[(None, 1)]], objective, epochs=steps, seed=seed,
                                    assignment_lr=assignment_lr, scale_lr=scale_lr, betas=betas,
                                    weight_decay=weight_decay, temperature=temperature, multiplier=multiplier,
                                    decay='constant', optimizer=optimizer)


def pack_llama_staged_block(fitted, records, *, bits, group_size):
    """Export a fitted Llama block to portable Torch GPTQ modules on CPU.

    This is a block export helper, not a complete checkpoint writer. Scale
    storage rounding is the existing TorchLinear contract and must be included
    in downstream replay/evaluation.
    """
    import copy

    from ..nn_modules.qlinear.torch import TorchLinear
    from ..utils.backend import BACKEND

    exported = copy.deepcopy(fitted).cpu()
    names = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
             'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    for name in names:
        stage = name if name in records else ('attention' if name.startswith('self_attn.') else 'mlp')
        key = 'weight' if stage == name else name+'.weight'
        scales = records[stage]['scales'][key].detach().cpu()
        if not torch.isfinite(scales).all() or (scales == 0).any():
            raise ValueError('Staged GPTQ export requires finite nonzero learned scales')
        linear = exported.get_submodule(name)
        groups = torch.arange(linear.in_features, dtype=torch.int32)//group_size
        packed = TorchLinear(bits=bits, group_size=group_size, sym=True, desc_act=False,
                             in_features=linear.in_features, out_features=linear.out_features,
                             bias=linear.bias is not None, backend=BACKEND.TORCH)
        packed.pack_original(linear, scales, torch.full_like(scales, 2**(bits-1)), groups)
        parent, leaf = name.rsplit('.', 1)
        setattr(exported.get_submodule(parent), leaf, packed)
    return exported


def quantize_llama_gsq_block(layer, batches, *, bits, group_size, gsq=None, pack=True):
    """Quantize one captured Llama block, optionally train GSQ, then export.

    The default performs ordinary GPTQ initialization. Enabled staged GSQ has
    a distinct serialized configuration and uses late MLP initialization. This
    entry point does not own full-model capture or checkpoint writing.
    """
    import copy

    from .gsq_training_config import GSQTrainingConfig

    if gsq is None:
        gsq = GSQTrainingConfig()
    elif isinstance(gsq, dict):
        gsq = GSQTrainingConfig(**gsq)
    if not isinstance(gsq, GSQTrainingConfig):
        raise TypeError('Staged GSQ requires GSQTrainingConfig, a dictionary or None')
    effective = gsq.to_dict()
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4):
        raise ValueError('Staged scalar GSQ supports W2/W3/W4')
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError('Staged scalar GSQ requires a positive contiguous group size')
    if not isinstance(pack, bool):
        raise TypeError('pack must be boolean')
    initializers, metadata = initialize_llama_gptq(layer, batches, bits=bits, group_size=group_size,
                                                  damp_percent=gsq.damp_percent, initializer=gsq.initializer)
    if gsq.enabled:
        fitted, records = fit_llama_stages(layer, initializers, batches, bits=bits, group_size=group_size,
                                           **gsq.training_kwargs())
    else:
        fitted = copy.deepcopy(layer).eval()
        with torch.no_grad():
            for name, (weight, _) in initializers.items():
                fitted.get_submodule(name).weight.copy_(weight)
        records = {name: {'scales': {'weight': scales}} for name, (_, scales) in initializers.items()}
    exported = pack_llama_staged_block(fitted, records, bits=bits, group_size=group_size) if pack else fitted
    return exported, dict(gsq_training=effective, bits=bits, group_size=group_size,
                          packed=pack, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                          initializers=initializers, initializer_metadata=metadata, stages=records)
