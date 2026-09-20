"""DeepMD-kit 3 TensorFlow implementation of the custom Si3N4 descriptor.

The descriptor keeps DeepMD's differentiable radial environment matrix, pools
each atom's neighbour features, and maps the pooled feature through the
user-defined SiLU MLP.  By inheriting from :class:`DescrptSeR`, it also keeps
the native statistics, force, and virial implementations required for
energy/force training.
"""

from __future__ import annotations

from typing import Any

from deepmd.tf.common import cast_precision
from deepmd.tf.descriptor import Descriptor
from deepmd.tf.descriptor.se_r import DescrptSeR
from deepmd.tf.env import tf
from deepmd.tf.utils.network import embedding_net
from deepmd.utils.argcheck import (
    descrpt_args_plugin,
    descrpt_se_r_args,
)


@descrpt_args_plugin.register(
    "my_embedding_net",
    doc="Custom radial neighbour-pooling descriptor with a SiLU embedding MLP.",
)
def my_embedding_net_args():
    """Expose the compatible radial-descriptor options to DeepMD validation."""

    return descrpt_se_r_args()


@Descriptor.register("my_embedding_net")
class MyEmbeddingNet(DescrptSeR):
    """Neighbour-mean pooling followed by the user-defined SiLU MLP.

    The input is DeepMD's normalized radial environment matrix.  Pooling is
    permutation invariant over the selected neighbours, while the inherited
    radial descriptor provides differentiable energy-to-force propagation.
    """

    def __init__(
        self,
        rcut: float = 6.0,
        rcut_smth: float = 0.5,
        sel: list[int] | None = None,
        neuron: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        if sel is None:
            sel = [100, 200]
        if neuron is None:
            neuron = [256, 128, 64]

        # The network architecture deliberately uses SiLU (Swish), matching
        # the original custom-network intent.  The remaining DeepMD options
        # (precision, seed, trainable, etc.) remain configurable through
        # ``kwargs`` and are handled by the proven radial descriptor backend.
        super().__init__(
            rcut=rcut,
            rcut_smth=rcut_smth,
            sel=list(sel),
            neuron=list(neuron),
            **kwargs,
        )

    @cast_precision
    def _filter_r(
        self,
        inputs: tf.Tensor,
        type_input: int,
        natoms: tf.Tensor,
        activation_fn: Any = None,
        stddev: float = 1.0,
        bavg: float = 0.0,
        name: str = "linear",
        reuse: bool | None = None,
        trainable: bool = True,
    ) -> tf.Tensor:
        """Apply the custom pooled-neighbour SiLU embedding network.

        ``inputs`` has one normalized radial feature per selected neighbour.
        The mean is therefore invariant to the neighbour ordering.  Its
        derivative remains connected to the radial environment matrix, so the
        inherited ``prod_force_virial`` routine can compute forces correctly.
        """

        del type_input, natoms, activation_fn
        with tf.variable_scope(name, reuse=reuse):
            atom_feature = tf.reduce_mean(
                inputs, axis=1, keepdims=True, name="pooled_neighbor_feature"
            )
            return embedding_net(
                atom_feature,
                self.filter_neuron,
                self.filter_precision,
                activation_fn=tf.nn.swish,
                resnet_dt=self.filter_resnet_dt,
                name_suffix="_my_embedding",
                stddev=stddev,
                bavg=bavg,
                seed=self.seed,
                trainable=trainable,
                uniform_seed=self.uniform_seed,
            )
