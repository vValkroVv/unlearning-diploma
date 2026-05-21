from trainer.unlearn.dpo import DPO


class AltPO(DPO):
    """
    AltPO implemented as counterfactual preference optimization.

    Expected forget batch contract:
      inputs["forget"]["alternate"] = preferred / win answer
      inputs["forget"]["original"] = rejected / lose answer

    The loss is inherited from DPO. This wrapper keeps the trainer name,
    Hydra configs, save parsing, and result tables explicit for AltPO.
    """

    pass
