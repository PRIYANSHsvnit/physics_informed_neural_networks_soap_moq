import math

import torch
from torch.optim import Optimizer


class PaperSOAPStyle(Optimizer):
    """Reconstruction of the paper's SOAP-style preconditioner.

    The paper does not specify these implementation values explicitly; this
    experiment records them as reconstruction assumptions:
    beta2=0.95, damping=1e-3, precondition_frequency=10,
    max_precond_dim=10000.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        beta2=0.95,
        damping=1e-3,
        precondition_frequency=10,
        max_precond_dim=10000,
        initial_factor_mode='ema_zero',
        update_order='before_update',
    ):
        if lr <= 0:
            raise ValueError('lr must be positive')
        if not 0.0 < beta2 < 1.0:
            raise ValueError('beta2 must be in (0, 1)')
        if damping <= 0:
            raise ValueError('damping must be positive')
        if precondition_frequency < 1:
            raise ValueError('precondition_frequency must be >= 1')
        if max_precond_dim < 1:
            raise ValueError('max_precond_dim must be >= 1')
        if initial_factor_mode not in {'ema_zero', 'first_gradient'}:
            raise ValueError("initial_factor_mode must be 'ema_zero' or 'first_gradient'")
        if update_order not in {'before_update', 'after_update'}:
            raise ValueError("update_order must be 'before_update' or 'after_update'")

        defaults = dict(
            lr=lr,
            beta2=beta2,
            damping=damping,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            initial_factor_mode=initial_factor_mode,
            update_order=update_order,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _matrix_view(gradient):
        if gradient.ndim < 2:
            return None
        rows = gradient.shape[0]
        columns = gradient.numel() // rows
        return gradient.reshape(rows, columns)

    @staticmethod
    def _inverse_sqrt(matrix, damping):
        original_dtype = matrix.dtype
        matrix64 = matrix.to(torch.float64)
        identity = torch.eye(matrix64.shape[0], device=matrix64.device, dtype=torch.float64)
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix64 + damping * identity)
        eigenvalues = torch.clamp(eigenvalues, min=damping)
        inverse_sqrt = torch.rsqrt(eigenvalues)
        result = eigenvectors @ torch.diag(inverse_sqrt) @ eigenvectors.T
        return result.to(original_dtype)

    def _init_state(self, parameter, gradient):
        state = self.state[parameter]
        if state:
            return

        state['step'] = 0
        if gradient.ndim < 2:
            state['mode'] = 'diagonal'
            state['v'] = torch.zeros_like(gradient)
            return

        matrix = self._matrix_view(gradient)
        rows, columns = matrix.shape
        if max(rows, columns) > self.defaults['max_precond_dim']:
            state['mode'] = 'diagonal'
            state['v'] = torch.zeros_like(gradient)
            return

        state['mode'] = 'structured'
        state['A'] = torch.zeros((rows, rows), device=gradient.device, dtype=gradient.dtype)
        state['B'] = torch.zeros((columns, columns), device=gradient.device, dtype=gradient.dtype)
        state['inv_sqrt_A'] = None
        state['inv_sqrt_B'] = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta2 = group['beta2']
            damping = group['damping']
            frequency = group['precondition_frequency']
            initial_factor_mode = group['initial_factor_mode']
            update_order = group['update_order']

            for parameter in group['params']:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad
                state = self.state[parameter]
                self._init_state(parameter, gradient)
                state['step'] += 1
                step = state['step']

                if state['mode'] == 'diagonal':
                    state['v'].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                    preconditioned = gradient / torch.sqrt(state['v'] + damping)
                    state['precond_gain'] = None
                    state['staleness_error'] = None
                    parameter.add_(preconditioned, alpha=-lr)
                    continue

                matrix_gradient = self._matrix_view(gradient)
                factor_outer_left = matrix_gradient @ matrix_gradient.T
                factor_outer_right = matrix_gradient.T @ matrix_gradient
                if state['inv_sqrt_A'] is None:
                    if initial_factor_mode == 'first_gradient':
                        state['A'].copy_(factor_outer_left)
                        state['B'].copy_(factor_outer_right)
                    else:
                        state['A'].mul_(beta2).add_(factor_outer_left, alpha=1.0 - beta2)
                        state['B'].mul_(beta2).add_(factor_outer_right, alpha=1.0 - beta2)
                    state['inv_sqrt_A'] = self._inverse_sqrt(state['A'], damping)
                    state['inv_sqrt_B'] = self._inverse_sqrt(state['B'], damping)
                elif update_order == 'before_update':
                    state['A'].mul_(beta2).add_(factor_outer_left, alpha=1.0 - beta2)
                    state['B'].mul_(beta2).add_(factor_outer_right, alpha=1.0 - beta2)
                    if step % frequency == 0:
                        state['inv_sqrt_A'] = self._inverse_sqrt(state['A'], damping)
                        state['inv_sqrt_B'] = self._inverse_sqrt(state['B'], damping)

                stale_preconditioned = (
                    state['inv_sqrt_A']
                    @ matrix_gradient
                    @ state['inv_sqrt_B']
                )
                fresh_inv_A = self._inverse_sqrt(state['A'], damping)
                fresh_inv_B = self._inverse_sqrt(state['B'], damping)
                fresh_preconditioned = fresh_inv_A @ matrix_gradient @ fresh_inv_B
                gradient_norm = torch.linalg.vector_norm(matrix_gradient)
                stale_norm = torch.linalg.vector_norm(stale_preconditioned)
                fresh_norm = torch.linalg.vector_norm(fresh_preconditioned)
                state['precond_gain'] = float(stale_norm / (gradient_norm + 1e-12))
                state['staleness_error'] = float(
                    torch.linalg.vector_norm(stale_preconditioned - fresh_preconditioned)
                    / (fresh_norm + 1e-12)
                )

                preconditioned = stale_preconditioned.reshape_as(gradient)
                parameter.add_(preconditioned, alpha=-lr)

                if update_order == 'after_update':
                    state['A'].mul_(beta2).add_(factor_outer_left, alpha=1.0 - beta2)
                    state['B'].mul_(beta2).add_(factor_outer_right, alpha=1.0 - beta2)
                    if step % frequency == 0:
                        state['inv_sqrt_A'] = self._inverse_sqrt(state['A'], damping)
                        state['inv_sqrt_B'] = self._inverse_sqrt(state['B'], damping)

        return loss
