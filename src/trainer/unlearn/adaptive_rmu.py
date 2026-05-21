"""Adaptive-RMU activation-steering unlearning trainer."""

from __future__ import annotations

import torch

from trainer.unlearn.rmu import RMU


class AdaptiveRMU(RMU):
    """RMU with an activation-norm-scaled steering target.

    The base RMU objective pushes forget-token activations toward a fixed random
    direction. Adaptive-RMU keeps that direction but scales its magnitude by the
    observed forget activation norm:

        target = random_unit_direction * steering_coeff
                 * mean(||h_forget||_2) * adaptive_scale

    By default the coefficient is estimated from the first training batch and
    reused for the run. Set ``adaptive_coeff_mode=batch`` to recompute it every
    batch.
    """

    def __init__(
        self,
        adaptive_scale: float = 5.0,
        adaptive_coeff_mode: str = "first_batch",
        adaptive_coeff_eps: float = 0.0,
        steering_coeff: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(steering_coeff=steering_coeff, *args, **kwargs)
        self.adaptive_scale = float(adaptive_scale)
        self.adaptive_coeff_mode = str(adaptive_coeff_mode)
        self.adaptive_coeff_eps = float(adaptive_coeff_eps)
        self._adaptive_coeff_cache: torch.Tensor | None = None

        if self.adaptive_coeff_mode not in {"first_batch", "batch"}:
            raise ValueError(
                "adaptive_coeff_mode must be either 'first_batch' or 'batch', "
                f"got {self.adaptive_coeff_mode!r}"
            )

    def _adaptive_coeff(self, activations: torch.Tensor) -> torch.Tensor:
        if (
            self.adaptive_coeff_mode == "first_batch"
            and self._adaptive_coeff_cache is not None
        ):
            return self._adaptive_coeff_cache.to(
                device=activations.device,
                dtype=activations.dtype,
            )

        coeff = activations.detach().norm(dim=-1).mean(dim=1).mean()
        coeff = coeff * self.adaptive_scale
        if self.adaptive_coeff_eps > 0:
            coeff = coeff.clamp_min(self.adaptive_coeff_eps)

        coeff = coeff.detach()
        if self.adaptive_coeff_mode == "first_batch":
            self._adaptive_coeff_cache = coeff.float().cpu()
        return coeff.to(device=activations.device, dtype=activations.dtype)

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        model_forget_activations, forget_outputs = self.forward_with_cache(
            model,
            forget_inputs,
            self.model_module,
            no_grad=False,
        )

        base_control_vec = self.get_control_vector(
            model_forget_activations.shape[-1],
            device=model_forget_activations.device,
            dtype=model_forget_activations.dtype,
        )
        adaptive_coeff = self._adaptive_coeff(model_forget_activations)
        control_vec = (base_control_vec * adaptive_coeff).expand_as(
            model_forget_activations
        )

        forget_mask = forget_inputs["labels"] != -100
        forget_loss = self.compute_activation_loss(
            model_forget_activations,
            control_vec,
            forget_mask,
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        try:
            self.log(
                {
                    "adaptive_rmu_forget_loss": float(forget_loss.detach().item()),
                    "adaptive_rmu_retain_loss": float(retain_loss.detach().item()),
                    "adaptive_rmu_total_loss": float(loss.detach().item()),
                    "adaptive_rmu_activation_norm": float(
                        model_forget_activations.detach().norm(dim=-1).mean().item()
                    ),
                    "adaptive_rmu_control_norm": float(
                        control_vec.detach().norm(dim=-1).mean().item()
                    ),
                    "adaptive_rmu_coeff": float(adaptive_coeff.detach().float().item()),
                    "adaptive_rmu_scale": float(self.adaptive_scale),
                    "adaptive_rmu_steering_coeff": float(self.steering_coeff),
                }
            )
        except Exception:
            pass

        return (loss, forget_outputs) if return_outputs else loss
