from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_wga_loss


class WGA(GradDiff):
    def __init__(self, beta=1.0, gamma=1.0, alpha=1.0, *args, **kwargs):
        # GradDiff prepares a reference model only for KL retain loss. Plain
        # WGA production runs use NLL retain loss and should not allocate a
        # second full model copy.
        super().__init__(gamma=gamma, alpha=alpha, *args, **kwargs)
        self.beta = beta
        self.uses_lora = hasattr(self.model, "disable_adapter")
        self._input_require_grads_handle = None
        if self.uses_lora:
            self._prepare_lora_for_gradient_checkpointing()
            self._ensure_trainable_lora_adapter()

    def _prepare_lora_for_gradient_checkpointing(self):
        enable_input_require_grads = getattr(
            self.model, "enable_input_require_grads", None
        )
        if callable(enable_input_require_grads):
            enable_input_require_grads()
            return

        get_input_embeddings = getattr(self.model, "get_input_embeddings", None)
        if not callable(get_input_embeddings):
            return
        input_embeddings = get_input_embeddings()
        if input_embeddings is None:
            return

        def make_inputs_require_grad(_module, _inputs, output):
            output.requires_grad_(True)

        self._input_require_grads_handle = input_embeddings.register_forward_hook(
            make_inputs_require_grad
        )

    def _trainable_param_count(self):
        return sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )

    def _ensure_trainable_lora_adapter(self):
        if self._trainable_param_count() > 0:
            return

        active_adapter = getattr(self.model, "active_adapter", None)
        if callable(active_adapter):
            active_adapter = active_adapter()
        set_adapter = getattr(self.model, "set_adapter", None)
        if active_adapter is not None and callable(set_adapter):
            set_adapter(active_adapter)

        if self._trainable_param_count() == 0:
            raise ValueError(
                "WGA LoRA training found no trainable adapter parameters. "
                "If loading an existing adapter, make sure it is loaded as trainable."
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss, forget_outputs = compute_wga_loss(
            model=model, inputs=forget_inputs, beta=self.beta
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        if model.training and not loss.requires_grad:
            raise RuntimeError(
                "WGA loss has no gradient path. This usually means LoRA input "
                "gradients or trainable adapter parameters are disabled."
            )
        return (loss, forget_outputs) if return_outputs else loss
