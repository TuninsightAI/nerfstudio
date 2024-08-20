import typing as t
from abc import ABCMeta, abstractmethod
from jaxtyping import Float
from torch import Tensor


class _MatchInterface(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def predict_correspondence(
        prev_image: Float[Tensor, "*batch 3 H W"],
        next_image: Float[Tensor, "*batch 3 H W"],
        **kwargs: t.Dict[str, t.Any],
    ) -> t.Tuple[
        Float[Tensor, "*batch 2"],
        Float[Tensor, "*batch 2"],
        Float[Tensor, "*batch"],
    ]:
        ...
