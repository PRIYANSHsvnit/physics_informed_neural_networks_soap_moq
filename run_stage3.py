import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, width: int = 128, hidden_layers: int = 4) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sample_collocation(n: int, device: torch.device) -> torch.Tensor:
    return torch.rand(n, 2, device=device, requires_grad=True)

def sample_boundary(n_each: int, device: torch.device) -> torch.Tensor:
    t = torch.rand(4 * n_each, device=device)
    x = torch.empty(4 * n_each, 2, device=device)
    x[:n_each] = torch.stack([torch.zeros(n_each, device=device), t[:n_each]], dim=1)
    x[n_each:2*n_each] = torch.stack([torch.ones(n_each, device=device), t[n_each:2*n_each]], dim=1)
    x[2*n_each:3*n_each] = torch.stack([t[2*n_each:3*n_each], torch.zeros(n_each, device=device)], dim=1)
    x[3*n_each:] = torch.stack([t[3*n_each:], torch.ones(n_each, device=device)], dim=1)
    return x

def make_batch_schedule(steps: int, K: int, n_pde: int, n_boundary_each: int, device: torch.device) -> List[Dict[str, torch.Tensor]]:
    schedule: List[Dict[str, torch.Tensor]] = []
    pde: Optional[torch.Tensor] = None
    bc: Optional[torch.Tensor] = None

    for step in range(1, steps + 1):
        if step == 1 or (step - 1) % K == 0:
            pde = sample_collocation(n_pde, device)
            bc = sample_boundary(n_boundary_each, device)

        assert pde is not None and bc is not None
        schedule.append({
            "pde": pde.detach().clone(),
            "bc": bc.detach().clone(),
        })

    return schedule

def poisson_residual(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    u = model(xy)
    grad_u = torch.autograd.grad(
        u, xy, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]

    u_xx = torch.autograd.grad(
        u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True
    )[0][:, 1:2]

    forcing = 2.0 * math.pi**2 * torch.sin(math.pi * xy[:, 0:1]) * torch.sin(math.pi * xy[:, 1:2])
    return -(u_xx + u_yy) - forcing

def relative_l2(model: nn.Module, device: torch.device, n: int = 81) -> float:
    grid = torch.linspace(0.0, 1.0, n, device=device)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    with torch.no_grad():
        pred = model(xy).reshape(-1)
    ref = torch.sin(math.pi * xy[:, 0]) * torch.sin(math.pi * xy[:, 1])
    return float(torch.linalg.vector_norm(pred - ref) / (torch.linalg.vector_norm(ref) + 1e-12))

def flat_grad(model: nn.Module) -> torch.Tensor:
    chunks = []
    for p in model.parameters():
        if p.grad is not None:
            chunks.append(p.grad.detach().reshape(-1))
    if not chunks:
        raise RuntimeError("No gradients available")
    return torch.cat(chunks)

def get_flat_params(model: nn.Module) -> torch.Tensor:
    chunks = []
    for p in model.parameters():
        chunks.append(p.detach().reshape(-1))
    return torch.cat(chunks)

def set_flat_grad(model: nn.Module, flat_g: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        if p.grad is not None:
            p.grad.data.copy_(flat_g[offset:offset+numel].reshape(p.shape))
        offset += numel

def make_optimizer(name: str, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "soap":
        import sys
        sys.path.append(str(Path(__file__).parent))
        from official_soap import SOAP
        return SOAP(
            model.parameters(),
            lr=lr,
            betas=(0.95, 0.95),
            weight_decay=0.0,
            precondition_frequency=10,
        )
    raise ValueError(f"Unsupported optimizer: {name}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["soap"], default="soap")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--n-pde", type=int, default=1000)
    parser.add_argument("--n-boundary-each", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-moq", action="store_true")
    parser.add_argument("--mu", type=float, default=0.85)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu") # ensure CPU as requested
    model = PINN().to(device)
    
    # We load or generate schedule
    schedule_path = Path(f"c:/Users/HP/Downloads/PINN/poisson_seed{args.seed}_K{args.K}_schedule.pt")
    if schedule_path.exists():
        schedule = torch.load(schedule_path, map_location=device, weights_only=True)
    else:
        schedule = make_batch_schedule(max(2000, args.steps), args.K, args.n_pde, args.n_boundary_each, device)
        torch.save(schedule, schedule_path)

    opt = make_optimizer(args.optimizer, model, args.lr)
    
    prev_grad: Optional[torch.Tensor] = None
    history: List[Dict[str, float]] = []

    model.train()

    for step in range(1, args.steps + 1):
        current_pde = schedule[step - 1]["pde"].detach().clone().requires_grad_(True)
        current_bc = schedule[step - 1]["bc"]
        opt.zero_grad(set_to_none=True)

        r = poisson_residual(model, current_pde)
        loss_pde = torch.mean(r.square())
        u_bc = model(current_bc)
        loss_bc = torch.mean(u_bc.square())
        loss = loss_pde + loss_bc
        loss.backward()

        g_t = flat_grad(model)
        
        # Calculate logging quantities
        grad_norm = float(torch.linalg.vector_norm(g_t))
        
        grad_diff_norm = float("nan")
        grad_change_ratio = float("nan")
        grad_cosine = float("nan")
        MoQ_gradient_norm = float("nan")
        MoQ_raw_gradient_norm = float("nan") # MoQ/raw_gradient_norm
        cos_gt_gMoQ = float("nan")
        
        # Determine the gradient actually used for optimization
        g_opt = g_t.clone()
        
        if prev_grad is not None:
            diff = g_t - prev_grad
            grad_diff_norm = float(torch.linalg.vector_norm(diff))
            grad_change_ratio = grad_diff_norm / (grad_norm + 1e-12)
            grad_cosine = float(torch.dot(g_t, prev_grad) / (grad_norm * torch.linalg.vector_norm(prev_grad) + 1e-12))
            
            # MoQ gradient
            g_MoQ = g_t + args.mu * diff
            MoQ_gradient_norm = float(torch.linalg.vector_norm(g_MoQ))
            MoQ_raw_gradient_norm = MoQ_gradient_norm / (grad_norm + 1e-12)
            cos_gt_gMoQ = float(torch.dot(g_t, g_MoQ) / (grad_norm * MoQ_gradient_norm + 1e-12))
            
            if args.use_moq:
                g_opt = g_MoQ
        else:
            # step 1, MoQ gradient is just g_t
            MoQ_gradient_norm = grad_norm
            MoQ_raw_gradient_norm = 1.0
            cos_gt_gMoQ = 1.0

        if args.use_moq:
            set_flat_grad(model, g_opt)
            
        params_before = get_flat_params(model)
        opt.step()
        params_after = get_flat_params(model)
        
        parameter_update_norm = float(torch.linalg.vector_norm(params_after - params_before))
        preconditioned_update_norm = parameter_update_norm / args.lr # since weight_decay=0
        
        prev_grad = g_t.clone() # ALWAYS save the raw gradient, not the MoQ gradient, because MoQ extrapolates raw gradients!

        # Evaluation
        err = relative_l2(model, device)
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "rel_l2": err,
            "grad_norm": grad_norm,
            "grad_diff_norm": grad_diff_norm,
            "grad_change_ratio": grad_change_ratio,
            "grad_cosine": grad_cosine,
            "MoQ_gradient_norm": MoQ_gradient_norm,
            "MoQ_raw_gradient_norm": MoQ_raw_gradient_norm,
            "preconditioned_update_norm": preconditioned_update_norm,
            "parameter_update_norm": parameter_update_norm,
            "cos_g_t_g_MoQ": cos_gt_gMoQ
        }
        history.append(row)
        print(f"step={step} loss={row['loss']:.4e} rel_l2={row['rel_l2']:.4e}")

    result = {
        "stage": 3,
        "optimizer": "soap",
        "use_moq": args.use_moq,
        "mu": args.mu,
        "seed": args.seed,
        "K": args.K,
        "lr": args.lr,
        "steps": args.steps,
        "history": history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"Saved: {args.out.resolve()}")

if __name__ == "__main__":
    main()
