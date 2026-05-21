from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import _compute_shifted_token_ce, _filter_model_inputs


_DEFAULT_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "for",
    "nor",
    "of",
    "in",
    "on",
    "at",
    "to",
    "from",
    "by",
    "with",
    "about",
    "as",
    "into",
    "like",
    "through",
    "after",
    "over",
    "between",
    "out",
    "against",
    "during",
    "without",
    "before",
    "under",
    "around",
    "among",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "do",
    "does",
    "did",
    "doing",
    "have",
    "has",
    "had",
    "having",
    "can",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "will",
    "would",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "their",
    "our",
    "your",
    "my",
    "i",
    "you",
    "we",
    "me",
    "him",
    "who",
    "whom",
    "whose",
    "which",
    "what",
    "where",
    "when",
    "why",
    "how",
    "not",
    "no",
    "yes",
    "also",
    "only",
    "just",
    "than",
    "so",
    "such",
}


class TPO(GradDiff):
    """
    Targeted Preference Optimization for QA unlearning.

    The forget branch splits supervised answer tokens into target/unwanted
    tokens and preserve/general tokens. The target tokens receive a logit
    preference loss against a frozen pre-unlearning reference model, while the
    preserve tokens receive CE preservation loss.
    """

    def __init__(
        self,
        beta: float = 0.2,
        pl_coeff: float = 1.0,
        identifier_mode: str = "stopword",
        preserve_token_ids: Iterable[int] | None = None,
        normalize_lpl_by_tokens: bool = True,
        normalize_pl_by_tokens: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta = float(beta)
        self.pl_coeff = float(pl_coeff)
        self.identifier_mode = str(identifier_mode)
        self.preserve_token_ids = {
            int(token_id) for token_id in (preserve_token_ids or []) if token_id is not None
        }
        self.normalize_lpl_by_tokens = bool(normalize_lpl_by_tokens)
        self.normalize_pl_by_tokens = bool(normalize_pl_by_tokens)

        if self.beta <= 0:
            raise ValueError(f"TPO beta must be > 0, got {self.beta}.")
        if self.identifier_mode not in {"stopword", "all_target", "all_preserve"}:
            raise ValueError(
                "identifier_mode must be one of {'stopword', 'all_target', 'all_preserve'}, "
                f"got {self.identifier_mode!r}."
            )

        # TPO compares current logits against the pre-unlearning model even
        # when the retain branch uses NLL, so it always needs a reference.
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def _as_model_inputs(self, batch: dict) -> dict:
        if isinstance(batch, dict) and "original" in batch:
            batch = batch["original"]
        return _filter_model_inputs(batch)

    def _get_tokenizer(self):
        processing_class = getattr(self, "processing_class", None)
        if processing_class is not None:
            return processing_class
        return getattr(self, "tokenizer", None)

    @staticmethod
    def _clean_token_piece(piece: str) -> str:
        piece = piece.replace("Ġ", "").replace("▁", "").replace("Ċ", "")
        piece = piece.replace("</w>", "")
        piece = piece.strip().lower()
        return re.sub(r"^[#]+", "", piece)

    def _is_general_token_id(self, token_id: int) -> bool:
        if token_id in self.preserve_token_ids:
            return True

        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return False

        try:
            piece = tokenizer.convert_ids_to_tokens(int(token_id))
        except Exception:
            return False

        piece = self._clean_token_piece(str(piece))
        if not piece:
            return True
        if piece in _DEFAULT_STOPWORDS:
            return True
        if re.fullmatch(r"[\W_]+", piece):
            return True
        return False

    def _build_token_masks(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_mask = labels != -100
        preserve_mask = torch.zeros_like(valid_mask, dtype=torch.bool)

        if self.identifier_mode == "all_target":
            return valid_mask, preserve_mask

        if self.identifier_mode == "all_preserve":
            preserve_mask = valid_mask.clone()
        else:
            unique_ids = input_ids[valid_mask].detach().unique().tolist()
            general_ids = [
                int(token_id)
                for token_id in unique_ids
                if self._is_general_token_id(int(token_id))
            ]
            if general_ids:
                general_tensor = torch.tensor(
                    general_ids,
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
                preserve_mask = torch.isin(input_ids, general_tensor) & valid_mask

        target_mask = valid_mask & ~preserve_mask

        # Keep the forget branch active if a batch contains only tokens that
        # the identifier marked as general.
        empty_target_rows = target_mask.sum(dim=-1) == 0
        if empty_target_rows.any():
            target_mask[empty_target_rows] = valid_mask[empty_target_rows]
            preserve_mask[empty_target_rows] = False

        return target_mask, preserve_mask

    @staticmethod
    def _labels_from_mask(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, labels, torch.full_like(labels, -100))

    def _preservation_loss(
        self,
        logits: torch.Tensor,
        preserve_labels: torch.Tensor,
    ) -> torch.Tensor:
        token_loss, shifted_labels = _compute_shifted_token_ce(logits, preserve_labels)
        valid = shifted_labels != -100
        if not valid.any():
            return logits.new_zeros(())

        per_sample = token_loss.sum(dim=-1)
        if self.normalize_pl_by_tokens:
            denom = valid.sum(dim=-1).clamp_min(1).to(dtype=per_sample.dtype)
            per_sample = per_sample / denom

        rows = valid.any(dim=-1)
        return per_sample[rows].mean()

    def _logit_preference_loss(
        self,
        student_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        input_ids: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        shifted_mask = target_labels[..., 1:] != -100
        if not shifted_mask.any():
            return student_logits.new_zeros(())

        gathered_ids = input_ids[..., 1:].unsqueeze(-1)
        student_true_logits = student_logits[..., :-1, :].gather(-1, gathered_ids).squeeze(-1)
        ref_true_logits = ref_logits[..., :-1, :].gather(-1, gathered_ids).squeeze(-1)

        logit_gap = (ref_true_logits - student_true_logits) * shifted_mask.to(
            dtype=student_logits.dtype
        )
        per_sample = logit_gap.sum(dim=-1)
        if self.normalize_lpl_by_tokens:
            denom = shifted_mask.sum(dim=-1).clamp_min(1).to(dtype=per_sample.dtype)
            per_sample = per_sample / denom

        rows = shifted_mask.any(dim=-1)
        return -F.logsigmoid(self.beta * per_sample[rows]).mean() * 2.0 / self.beta

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self._as_model_inputs(inputs["forget"])
        labels = forget_inputs["labels"]
        input_ids = forget_inputs["input_ids"]

        target_mask, preserve_mask = self._build_token_masks(
            input_ids=input_ids,
            labels=labels,
        )
        target_labels = self._labels_from_mask(labels, target_mask)
        preserve_labels = self._labels_from_mask(labels, preserve_mask)

        student_forget_inputs = dict(forget_inputs)
        student_forget_inputs["labels"] = target_labels
        forget_outputs = model(**student_forget_inputs)

        with torch.no_grad():
            ref_outputs = self.ref_model(**student_forget_inputs)

        lpl_loss = self._logit_preference_loss(
            student_logits=forget_outputs.logits,
            ref_logits=ref_outputs.logits,
            input_ids=input_ids,
            target_labels=target_labels,
        )
        pl_loss = self._preservation_loss(
            logits=forget_outputs.logits,
            preserve_labels=preserve_labels,
        )
        forget_loss = lpl_loss + self.pl_coeff * pl_loss

        retain_inputs = self._as_model_inputs(inputs["retain"])
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        try:
            valid = labels != -100
            target_counts = (target_labels != -100).sum(dim=-1).float()
            preserve_counts = (preserve_labels != -100).sum(dim=-1).float()
            valid_counts = valid.sum(dim=-1).clamp_min(1).float()
            self.log(
                {
                    "tpo_lpl_loss": float(lpl_loss.detach().item()),
                    "tpo_preservation_loss": float(pl_loss.detach().item()),
                    "tpo_forget_loss": float(forget_loss.detach().item()),
                    "tpo_retain_loss": float(retain_loss.detach().item()),
                    "tpo_total_loss": float(loss.detach().item()),
                    "tpo_target_tokens_mean": float(target_counts.mean().detach().item()),
                    "tpo_preserve_tokens_mean": float(preserve_counts.mean().detach().item()),
                    "tpo_preserve_token_frac": float(
                        (preserve_counts / valid_counts).mean().detach().item()
                    ),
                    "tpo_beta": self.beta,
                    "tpo_pl_coeff": self.pl_coeff,
                    "tpo_alpha": float(self.alpha),
                    "tpo_gamma": float(self.gamma),
                }
            )
        except Exception:
            pass

        return (loss, forget_outputs) if return_outputs else loss
