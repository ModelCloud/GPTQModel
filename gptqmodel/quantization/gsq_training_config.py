"""Explicit configuration for staged scalar GSQ, separate from GSQ refinement."""

from dataclasses import asdict, dataclass
import math


@dataclass
class GSQTrainingConfig:
    """Opt-in staged training settings; not a claim of full paper reproduction.

    Calibration capture and precision remain caller-owned. Optimizer/schedule
    defaults follow the pinned author configuration. Initializer and singleton
    batching defaults preserve the earlier experimental API; select
    ``initializer='gptq_signed', batch_size=64, microbatch_size=16`` explicitly
    to exercise those author-style components. Variable-length token weighting,
    precision and the complete lifecycle still require separate validation.
    """

    enabled: bool = False
    initializer: str = 'gptq'
    optimizer: str = 'lion'
    seed: int = 7
    epochs: int = 10
    batch_size: int = 1
    microbatch_size: int = 1
    qk_steps: int = 2000
    damp_percent: float = .01
    assignment_lr: float = 1e-4
    scale_lr: float = 5e-5
    weight_decay: float = 1.
    betas: tuple[float, float] = (.9, .95)
    temperature: tuple[float, float] = (2., .05)
    multiplier: tuple[float, float] = (100., 500.)
    warmup_steps: int = 0
    min_lr: float = .1
    decay: str = 'cosine'

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError('GSQTrainingConfig: enabled must be boolean')
        if self.optimizer not in ('lion', 'adamw'):
            raise ValueError('GSQTrainingConfig: optimizer must be lion or adamw')
        if self.initializer not in ('gptq', 'gptq_signed', 'awq'):
            raise ValueError('GSQTrainingConfig: initializer must be gptq, gptq_signed or awq')
        for name, minimum in (('seed', 0), ('epochs', 1), ('qk_steps', 1), ('warmup_steps', 0),
                              ('batch_size', 1), ('microbatch_size', 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'GSQTrainingConfig: {name} must be an integer >= {minimum}')
        if self.microbatch_size > self.batch_size:
            raise ValueError('GSQTrainingConfig: microbatch_size exceeds batch_size')
        for name in ('damp_percent', 'assignment_lr', 'scale_lr', 'weight_decay', 'min_lr'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'GSQTrainingConfig: {name} must be finite numeric')
            if value < 0 or (name in ('assignment_lr', 'scale_lr') and value == 0):
                raise ValueError(f'GSQTrainingConfig: invalid {name}')
        if self.min_lr > 1 or self.decay not in ('cosine', 'linear', 'constant'):
            raise ValueError('GSQTrainingConfig: invalid learning-rate schedule')
        for name in ('betas', 'temperature', 'multiplier'):
            pair = getattr(self, name)
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f'GSQTrainingConfig: {name} requires two endpoints')
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   or (not 0 <= v < 1 if name == 'betas' else v <= 0) for v in pair):
                raise ValueError(f'GSQTrainingConfig: invalid {name}')
            setattr(self, name, tuple(pair))

    def to_dict(self):
        self.__post_init__()
        return asdict(self)

    def training_kwargs(self):
        values = self.to_dict()
        values.pop('enabled')
        values['qk_damp_percent'] = values.pop('damp_percent')
        return values
