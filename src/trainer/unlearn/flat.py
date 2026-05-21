from __future__ import annotations

import torch

from trainer.unlearn.grad_diff import GradDiff


class FLAT(GradDiff):
    """FLAT: LLM Unlearning via Loss Adjustment with Only Forget Data.

    This repo-fit implementation consumes the standard QA forget batch,
    reconstructs a template-answer batch from the prompt prefix, and applies
    the released FLAT f-divergence objective. The optional retain branch reuses
    GradDiff's retain loss for explicitly tagged FLAT+Retain comparison runs.
    """

    def __init__(
        self,
        divergence: str = "Total-Variation",
        template_text: str = "I don't know.",
        template_add_eos: bool = True,
        eps: float = 1e-5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.divergence = divergence
        self.template_text = template_text
        self.template_add_eos = bool(template_add_eos)
        self.eps = float(eps)
        self._template_token_ids_cache: list[int] | None = None
        self._last_flat_log_step: int | None = None

    @staticmethod
    def _as_model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    def _tokenizer(self):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            raise RuntimeError("FLAT requires Trainer.tokenizer / processing_class.")
        return tokenizer

    def _template_token_ids(self) -> list[int]:
        if self._template_token_ids_cache is not None:
            return self._template_token_ids_cache

        tokenizer = self._tokenizer()
        token_ids = tokenizer.encode(self.template_text, add_special_tokens=False)

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if self.template_add_eos and eos_token_id is not None:
            if not token_ids or token_ids[-1] != eos_token_id:
                token_ids.append(eos_token_id)

        if not token_ids:
            raise ValueError(
                "FLAT template_text produced no tokens. Set trainer.method_args.template_text."
            )

        self._template_token_ids_cache = list(token_ids)
        return self._template_token_ids_cache

    def _pad_token_id(self) -> int:
        tokenizer = self._tokenizer()
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            return int(eos_token_id)
        return 0

    def _build_template_batch(
        self,
        forget_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        input_ids = forget_inputs["input_ids"]
        labels = forget_inputs["labels"]
        attention_mask = forget_inputs["attention_mask"]

        if input_ids.ndim != 2:
            raise ValueError(f"FLAT expects 2D input_ids, got shape={tuple(input_ids.shape)}")

        batch_size, max_length = input_ids.shape
        device = input_ids.device
        template_token_ids = torch.tensor(
            self._template_token_ids(),
            device=device,
            dtype=input_ids.dtype,
        )

        template_input_ids = torch.full_like(input_ids, fill_value=self._pad_token_id())
        template_attention_mask = torch.zeros_like(attention_mask)
        template_labels = torch.full_like(labels, fill_value=-100)

        for row_idx in range(batch_size):
            supervised_positions = torch.nonzero(labels[row_idx] != -100, as_tuple=False).flatten()
            if supervised_positions.numel() == 0:
                prefix_len = int(attention_mask[row_idx].sum().item())
            else:
                prefix_len = int(supervised_positions[0].item())

            prefix_len = max(0, min(prefix_len, max_length))
            remaining = max_length - prefix_len

            prefix = input_ids[row_idx, :prefix_len]
            if remaining > 0:
                suffix = template_token_ids[:remaining]
                sequence = torch.cat([prefix, suffix], dim=0)
            else:
                sequence = prefix

            sequence = sequence[:max_length]
            seq_len = int(sequence.numel())
            if seq_len == 0:
                continue

            template_input_ids[row_idx, :seq_len] = sequence
            template_attention_mask[row_idx, :seq_len] = 1

            supervised_len = max(0, seq_len - prefix_len)
            if supervised_len > 0:
                template_labels[row_idx, prefix_len:seq_len] = template_input_ids[
                    row_idx, prefix_len:seq_len
                ]

        return {
            "input_ids": template_input_ids,
            "attention_mask": template_attention_mask,
            "labels": template_labels,
        }

    @staticmethod
    def _negative_mean_true_token_probability(
        model,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, object]:
        """Return the FLAT probability proxy used by the released implementation."""
        outputs = model(**batch)
        logits = outputs.logits[..., :-1, :]
        labels = batch["labels"][..., 1:]
        valid_mask = labels != -100

        safe_labels = labels.masked_fill(~valid_mask, 0)
        probs = torch.softmax(logits.float(), dim=-1)
        true_token_probs = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        true_token_probs = true_token_probs.to(logits.dtype)

        counts = valid_mask.sum(dim=-1).clamp_min(1)
        mean_true_prob = (true_token_probs * valid_mask).sum(dim=-1) / counts
        return -mean_true_prob, outputs

    def _f_divergence_loss(
        self,
        prob_sum_unlearn: torch.Tensor,
        prob_sum_good: torch.Tensor,
    ) -> torch.Tensor:
        """Port of FLAT get_contrastive_loss()."""
        div = self.divergence
        x_good = (-prob_sum_good).clamp(min=0.0, max=1.0)
        x_unlearn = (-prob_sum_unlearn).clamp(min=0.0, max=1.0)
        two_log = torch.log(torch.tensor(2.0, device=x_good.device, dtype=x_good.dtype))

        if div == "KL":
            loss_regular = -torch.mean(x_good)
            loss_peer = -torch.mean(torch.exp(x_unlearn - 1.0))
        elif div == "Reverse-KL":
            loss_regular = -torch.mean(-torch.exp(x_good))
            loss_peer = -torch.mean(-1.0 - x_unlearn)
        elif div == "Jeffrey":
            loss_regular = -torch.mean(x_good)
            loss_peer = -torch.mean(
                x_unlearn + x_unlearn.pow(2) / 4.0 + x_unlearn.pow(3) / 16.0
            )
        elif div == "Squared-Hellinger":
            exp_good = torch.exp(x_good)
            exp_unlearn = torch.exp(x_unlearn)
            loss_regular = -torch.mean(1.0 - exp_good)
            loss_peer = -torch.mean((1.0 - exp_unlearn) / exp_unlearn.clamp_min(self.eps))
        elif div == "Pearson":
            loss_regular = -torch.mean(x_good)
            loss_peer = -torch.mean(x_unlearn.pow(2) / 4.0 + x_unlearn)
        elif div == "Neyman":
            loss_regular = -torch.mean(1.0 - torch.exp(x_good))
            loss_peer = -torch.mean(
                2.0 - 2.0 * torch.sqrt((1.0 - x_unlearn).clamp_min(self.eps))
            )
        elif div in {"Jenson-Shannon", "Jensen-Shannon"}:
            loss_regular = -torch.mean(-torch.log1p(torch.exp(-x_good))) - two_log
            loss_peer = -torch.mean(x_unlearn + torch.log1p(torch.exp(-x_unlearn))) + two_log
        elif div == "Total-Variation":
            loss_regular = -torch.mean(torch.tanh(x_good) / 2.0)
            loss_peer = -torch.mean(torch.tanh(x_unlearn) / 2.0)
        else:
            raise NotImplementedError(f"Unsupported FLAT f-divergence: {div}")

        return loss_regular - loss_peer

    def _zero_loss(self, model) -> torch.Tensor:
        return next(model.parameters()).new_zeros(())

    def _maybe_log(
        self,
        flat_loss: torch.Tensor,
        retain_loss: torch.Tensor,
        total_loss: torch.Tensor,
        prob_sum_unlearn: torch.Tensor,
        prob_sum_good: torch.Tensor,
    ) -> None:
        step = int(getattr(self.state, "global_step", 0) or 0)
        if self._last_flat_log_step == step:
            return
        self._last_flat_log_step = step
        self.log(
            {
                "flat_forget_loss": float(flat_loss.detach().item()),
                "flat_retain_loss": float(retain_loss.detach().item()),
                "flat_total_loss": float(total_loss.detach().item()),
                "flat_forget_prob": float((-prob_sum_unlearn).mean().detach().item()),
                "flat_template_prob": float((-prob_sum_good).mean().detach().item()),
                "flat_alpha": float(self.alpha),
                "flat_gamma": float(self.gamma),
            }
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self._as_model_inputs(inputs["forget"])
        template_inputs = self._build_template_batch(forget_inputs)

        prob_sum_unlearn, forget_outputs = self._negative_mean_true_token_probability(
            model,
            forget_inputs,
        )
        prob_sum_good, _ = self._negative_mean_true_token_probability(
            model,
            template_inputs,
        )

        flat_loss = self._f_divergence_loss(
            prob_sum_unlearn=prob_sum_unlearn,
            prob_sum_good=prob_sum_good,
        )

        retain_loss = self._zero_loss(model)
        if self.alpha != 0.0 and "retain" in inputs and inputs["retain"] is not None:
            retain_inputs = self._as_model_inputs(inputs["retain"])
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * flat_loss + self.alpha * retain_loss
        self._maybe_log(flat_loss, retain_loss, loss, prob_sum_unlearn, prob_sum_good)
        return (loss, forget_outputs) if return_outputs else loss
