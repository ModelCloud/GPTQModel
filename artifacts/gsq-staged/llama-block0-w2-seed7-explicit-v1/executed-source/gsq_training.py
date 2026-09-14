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
            or not torch.isfinite(scales).all() or (scales <= 0).any()):
        raise ValueError('GSQ initializer requires matching positive finite scales')
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
        masked = self.logits.masked_fill(~self.valid, -1e9)
        candidates = self.candidates
        if self.initial is not None:
            candidates = torch.arange(-2, 3, device=candidates.device, dtype=candidates.dtype)[:, None, None]
            candidates = candidates.expand_as(self.candidates)
        return relaxed_scalar_weights(masked, self.scales, candidates, self.group_index,
                                      uniform=uniform, temperature=temperature, multiplier=multiplier,
                                      initial=self.initial)

    @torch.no_grad()
    def hard_weight(self):
        selected = self.logits.masked_fill(~self.valid, -1e9).argmax(0, keepdim=True)
        assignments = self.candidates.gather(0, selected).squeeze(0)
        return assignments * self.scales[:, self.group_index].to(assignments.dtype)

    def optimizer_groups(self, *, assignment_lr, scale_lr, weight_decay):
        return [{'params': [self.logits], 'lr': assignment_lr, 'weight_decay': weight_decay},
                {'params': [self.scales], 'lr': scale_lr, 'weight_decay': 0.}]


def reconstruction_stage_loss(module, args, kwargs, *, student_weights, teacher_weights=None, output_select=None):
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
    teacher_state = {**parameters, **{name: value.detach() for name, value in teacher_weights.items()}}
    with torch.no_grad():
        teacher = torch.func.functional_call(module, (teacher_state, buffers), args, kwargs)
        teacher = teacher if output_select is None else output_select(teacher)
    student = torch.func.functional_call(module, ({**parameters, **student_weights}, buffers), args, kwargs)
    student = student if output_select is None else output_select(student)
    if not isinstance(student, torch.Tensor) or not isinstance(teacher, torch.Tensor):
        raise ValueError('GSQ stage requires tensor outputs or an output selector')
    return torch.nn.functional.mse_loss(student, teacher)


def train_stage_update(quantizers, optimizer, microbatches, objective, *, generator, temperature, multiplier):
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
    reported = 0.
    try:
        for batch, count in microbatches:
            weights = {}
            for name, quantizer in quantizers.items():
                uniform = torch.rand(quantizer.logits.shape, dtype=quantizer.logits.dtype,
                                     device=quantizer.logits.device, generator=generator)
                weights[name] = quantizer(uniform=uniform, temperature=temperature, multiplier=multiplier)
            loss = objective(batch, weights)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError('GSQ stage objective must be a finite scalar')
            fraction = count/total
            (loss*fraction).backward()
            reported += float(loss.detach())*fraction
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise ValueError('GSQ stage gradient is nonfinite')
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    return reported


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
                             min_lr=0., decay='linear'):
    """Train one stage and export final hard weights, without a surrogate guard.

    Each batch contains (microbatch, output_element_count) entries. The caller
    owns capture, teacher state, stage ordering and held-out evaluation. Epochs
    shuffle whole batches, as in the author trainer, using private RNG state.
    """
    if not quantizers or not batches or isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError('GSQ stage fitting requires quantizers, batches and positive epochs')
    groups = []
    for quantizer in quantizers.values():
        groups.extend(quantizer.optimizer_groups(assignment_lr=assignment_lr, scale_lr=scale_lr,
                                                weight_decay=weight_decay))
    optimizer = GSQLion(groups, betas=betas)
    initial_lrs = [group['lr'] for group in optimizer.param_groups]
    device = next(iter(quantizers.values())).logits.device
    sampling_rng = torch.Generator(device=device).manual_seed(seed)
    shuffle_rng = torch.Generator().manual_seed(seed)
    total_steps = epochs*len(batches)
    history = []
    for epoch in range(epochs):
        for index in torch.randperm(len(batches), generator=shuffle_rng).tolist():
            step = len(history)
            tau, kappa = sampling_schedule(step, total_steps, temperature=temperature, multiplier=multiplier)
            for group, base_lr in zip(optimizer.param_groups, initial_lrs):
                group['lr'] = stage_learning_rate(step, total_steps, base_lr=base_lr, warmup_steps=warmup_steps,
                                                  min_lr=min_lr, decay=decay)
            loss = train_stage_update(quantizers, optimizer, batches[index], objective, generator=sampling_rng,
                                      temperature=tau, multiplier=kappa)
            if step == 0 or (step+1) % 100 == 0 or step+1 == total_steps:
                import logging

                logging.getLogger(__name__).info("GSQ stage update %d/%d loss=%g", step+1, total_steps, loss)
            history.append(dict(epoch=epoch, step=step, batch=index, loss=loss, temperature=tau,
                                multiplier=kappa, learning_rates=[group['lr'] for group in optimizer.param_groups]))
    return dict(weights={name: quantizer.hard_weight().detach().clone() for name, quantizer in quantizers.items()},
                scales={name: quantizer.scales.detach().clone() for name, quantizer in quantizers.items()},
                history=history)


def fit_llama_stages(layer, initializers, batches, *, bits, group_size, epochs, seed=7,
                     qk_steps=2000, qk_damp_percent=.01, **training):
    """Fit a Llama block in author stage order from supplied scalar initializers.

    Batches are (hidden_states, attention_kwargs) pairs without padding. Caller
    owns GPTQ initialization, disjoint data preparation, packing and evaluation.
    The source layer is never mutated. Returns its fitted copy and stage records.
    """
    import copy

    names = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
             'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    if set(initializers) != set(names) or not batches:
        raise ValueError('Llama GSQ requires all seven projection initializers and nonempty batches')
    fitted = copy.deepcopy(layer).eval()
    teacher_attention = {name: value.detach().clone() for name, value in layer.named_parameters()
                         if name.startswith('self_attn.')}
    records = {}

    def quantizer(name):
        weight, scales = initializers[name]
        count = 4 if bits == 2 else 5
        rng = torch.Generator(device=weight.device).manual_seed(seed)
        noise = torch.randn((count, *weight.shape), dtype=weight.dtype, device=weight.device, generator=rng)
        return GSQScalarTrainingModule(weight, scales, group_size, bits=bits, noise=noise,
                                       logits_dtype=torch.float32 if name in names[:2] else weight.dtype)

    def run(stage_name, stage, selected, stage_batches, teacher=None):
        quantizers = {name+'.weight': quantizer(name) for name in selected}

        def objective(batch, weights):
            inputs, kwargs = batch
            return reconstruction_stage_loss(stage, (inputs,), kwargs, student_weights=weights,
                                             teacher_weights=teacher)
        result = fit_reconstruction_stage(quantizers, stage_batches, objective,
                                          epochs=epochs, seed=seed, **training)
        with torch.no_grad():
            for name in selected:
                fitted.get_submodule(name).weight.copy_(result['weights'][name+'.weight'])
        records[stage_name] = result

    # Keep names consistent with the containing block for functional replacement.
    for name in names[:2]:
        projection = fitted.get_submodule(name)
        quant = quantizer(name)
        with torch.no_grad():
            normalized = [fitted.input_layernorm(hidden) for hidden, _ in batches]
        qk_options = {key: value for key, value in training.items()
                      if key in ('assignment_lr', 'scale_lr', 'betas', 'weight_decay', 'temperature', 'multiplier')}
        result = fit_qk_projection(quant, projection.weight, normalized, steps=qk_steps,
                                   damp_percent=qk_damp_percent, seed=seed, **qk_options)
        result['objective'] = 'prepared_qk_quadratic_sum'
        result['damp_percent'] = qk_damp_percent
        with torch.no_grad():
            projection.weight.copy_(result['weights']['weight'])
        records[name] = result
    staged_batches = [[((hidden, kwargs), hidden.numel())] for hidden, kwargs in batches]
    run('attention', LlamaGSQAttentionStage(fitted), names[2:4], staged_batches, teacher_attention)
    run('mlp', fitted, names[4:], staged_batches, teacher_attention)
    return fitted, records


def initialize_llama_gptq(layer, batches, *, bits, group_size, damp_percent=.1):
    """Capture real projection inputs and prepare symmetric GPTQ stage seeds.

    This initializer uses this repository's GPTQ, not the author's fork. Its
    numerical parity must be assessed separately from GSQ training parity.
    """
    import copy

    from .config import GPTQConfig
    from .gptq import GPTQ

    if bits not in (2, 3, 4) or not batches:
        raise ValueError('Staged Llama GPTQ requires W2/W3/W4 and calibration batches')
    working = copy.deepcopy(layer).eval()
    tasks, handles = {}, []
    names = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
             'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    try:
        for name in names:
            module = working.get_submodule(name)
            config = GPTQConfig(bits=bits, group_size=group_size, sym=True, desc_act=False,
                                damp_percent=damp_percent, gsq=None, act_group_aware=False)
            task = GPTQ(module, config)
            task.quantizer.configure(perchannel=True)
            tasks[name] = task

            def capture(_module, inputs, output, task=task):
                task.add_batch(inputs[0].detach(), output.detach())
            handles.append(module.register_forward_hook(capture))
        with torch.no_grad():
            for hidden, kwargs in batches:
                working(hidden, **kwargs)
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


def prepare_qk_calibration_factor(inputs, *, damp_percent=.01):
    """Author Q/K metric: 2/sequence_count Gram, dead diagonal repair, damping."""
    if not inputs or not math.isfinite(damp_percent) or damp_percent < 0:
        raise ValueError('GSQ Q/K factor requires inputs and nonnegative damping')
    width = inputs[0].shape[-1]
    gram = torch.zeros(width, width, device=inputs[0].device, dtype=torch.float32)
    sequences = 0
    for batch in inputs:
        if batch.ndim != 3 or batch.shape[-1] != width or not torch.isfinite(batch).all():
            raise ValueError('GSQ Q/K inputs require finite [batch,tokens,in] geometry')
        flattened = batch.detach().reshape(-1, width).float()
        gram.add_(flattened.T @ flattened)
        sequences += batch.shape[0]
    if not sequences:
        raise ValueError('GSQ Q/K factor has no sequences')
    gram.mul_(2/sequences)
    dead = gram.diagonal() == 0
    gram.diagonal()[dead] = 1
    gram.diagonal().add_(damp_percent*gram.diagonal().mean())
    return torch.linalg.cholesky(gram), dead


def fit_qk_projection(quantizer, teacher, inputs, *, steps=2000, damp_percent=.01, seed=7,
                      assignment_lr=1e-4, scale_lr=5e-5, betas=(.9, .95), weight_decay=1.,
                      temperature=(2., .05), multiplier=(100., 500.)):
    """Dedicated constant-LR Q/K training, using the prepared quadratic sum."""
    factor, dead = prepare_qk_calibration_factor(inputs, damp_percent=damp_percent)
    target = teacher.detach().float().clone()
    target[:, dead] = 0

    def objective(_batch, weights):
        return ((target-weights['weight']) @ factor).square().sum()
    return fit_reconstruction_stage({'weight': quantizer}, [[(None, 1)]], objective, epochs=steps, seed=seed,
                                    assignment_lr=assignment_lr, scale_lr=scale_lr, betas=betas,
                                    weight_decay=weight_decay, temperature=temperature, multiplier=multiplier,
                                    decay='constant')
