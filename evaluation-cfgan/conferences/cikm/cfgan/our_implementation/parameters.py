from typing import Any, Optional

import attr
from skopt.space import Categorical, Integer, Real

from conferences.cikm.cfgan.our_implementation.constants import CFGANMode, CFGANOptimizer, \
    CFGANBenchmarks, CFGANActivation, CFGANConditionType, CFGANType, CFGANMaskType

VALID_CFGAN_MASK_TYPES = [
    CFGANMaskType.ZERO_RECONSTRUCTION,
    CFGANMaskType.PARTIAL_MASKING,
    CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING,
]


def density_percentage_validator(instance: Any, attribute: Any, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"density_percentage attribute must be between 0 and 1.")


def cfgan_mask_type_validator(instance: Any, attribute: Any, value: CFGANMaskType) -> None:
    if value not in VALID_CFGAN_MASK_TYPES:
        raise ValueError(f"CFGAN only accepts {[VALID_CFGAN_MASK_TYPES]} as valid mask types.")


def batch_size_validator(instance: Any, attribute: Any, value: int) -> None:
    if value <= 1:
        raise ValueError(
            f"CFGAN only accepts batch sizes greater than 1 due to an error in the original code that causes NaN values "
            f"if the batch size is 1."
        )


def fill_reproducibility_seed_validator(instance: Any, attribute: Any, value: Optional[int]) -> None:
    if value is None:
        instance.__set_attribute__(attribute, 1234567890)


@attr.s(frozen=True, kw_only=True)
class CFGANHyperParameters:
    benchmark: CFGANBenchmarks = attr.ib()
    epochs: int = attr.ib()
    mode: CFGANMode = attr.ib()
    generator_hidden_features: int = attr.ib()
    generator_regularization: float = attr.ib()
    generator_learning_rate: float = attr.ib()
    generator_batch_size: int = attr.ib(
        validator=[
            batch_size_validator,
        ]
    )
    generator_hidden_layers: int = attr.ib()
    generator_steps: int = attr.ib()
    generator_optimizer: CFGANOptimizer = attr.ib()
    generator_activation: CFGANActivation = attr.ib()
    discriminator_hidden_features: int = attr.ib()
    discriminator_regularization: float = attr.ib()
    discriminator_learning_rate: float = attr.ib()
    discriminator_batch_size: int = attr.ib(
        validator=[
            batch_size_validator,
        ]
    )
    discriminator_hidden_layers: int = attr.ib()
    discriminator_steps: int = attr.ib()
    discriminator_optimizer: CFGANOptimizer = attr.ib()
    discriminator_activation: CFGANActivation = attr.ib()

    mask_type: CFGANMaskType = attr.ib(
        validator=[
            attr.validators.instance_of(CFGANMaskType),
            cfgan_mask_type_validator
        ]
    )
    coefficient_zero_reconstruction: float = attr.ib()
    ratio_zero_reconstruction: int = attr.ib()
    ratio_partial_masking: int = attr.ib()

    reproducibility_seed: Optional[int] = attr.ib(
        default=None
    )


@attr.s(frozen=True, kw_only=True)
class RandomNoiseCFGANHyperParameters(CFGANHyperParameters):
    noise_size: int = attr.ib()


@attr.s(frozen=True, kw_only=True)
class CFGANEarlyStoppingSearchHyperParameters:
    benchmark: Categorical = attr.ib()
    mode: Categorical = attr.ib()

    generator_steps: Integer = attr.ib()
    generator_batch_size: Integer = attr.ib()
    generator_regularization: Real = attr.ib()
    generator_hidden_features: Integer = attr.ib()
    generator_learning_rate: Real = attr.ib()
    generator_hidden_layers: Integer = attr.ib()
    generator_optimizer: Categorical = attr.ib()
    generator_activation: Categorical = attr.ib()

    discriminator_steps: Integer = attr.ib()
    discriminator_batch_size: Integer = attr.ib()
    discriminator_regularization: Real = attr.ib()
    discriminator_hidden_features: Integer = attr.ib()
    discriminator_learning_rate: Real = attr.ib()
    discriminator_hidden_layers: Integer = attr.ib()
    discriminator_optimizer: Categorical = attr.ib()
    discriminator_activation: Categorical = attr.ib()

    mask_type: Categorical = attr.ib()
    coefficient_zero_reconstruction: Real = attr.ib()
    ratio_zero_reconstruction: Integer = attr.ib()
    ratio_partial_masking: Integer = attr.ib()


@attr.s(frozen=True, kw_only=True)
class CFGANSearchHyperParameters(CFGANEarlyStoppingSearchHyperParameters):
    epochs: Categorical = attr.ib()


@attr.s(frozen=True, kw_only=True)
class RandomNoiseCFGANSearchHyperParameters(CFGANSearchHyperParameters):
    noise_size: Integer = attr.ib()


def get_cfgan_code_hyper_parameters(benchmark: CFGANBenchmarks) -> CFGANHyperParameters:
    if benchmark == CFGANBenchmarks.CIAO:
        return CFGANHyperParameters(
            benchmark=benchmark,
            epochs=550,  # 550,  # Original 550. This epochs = 550 / generator_steps
            mode=CFGANMode.ITEM_BASED,
            mask_type=CFGANMaskType.ZERO_RECONSTRUCTION,
            generator_hidden_features=250,
            discriminator_hidden_features=50,

            generator_regularization=0.001,
            discriminator_regularization=0.001,

            generator_learning_rate=0.0001,
            discriminator_learning_rate=0.0001,

            generator_batch_size=128,
            discriminator_batch_size=128,

            generator_optimizer=CFGANOptimizer.ADAM,
            discriminator_optimizer=CFGANOptimizer.ADAM,

            generator_hidden_layers=1,
            discriminator_hidden_layers=4,

            generator_steps=1,
            discriminator_steps=1,

            generator_activation=CFGANActivation.SIGMOID,
            discriminator_activation=CFGANActivation.SIGMOID,

            ratio_zero_reconstruction=40,
            ratio_partial_masking=0,
            coefficient_zero_reconstruction=0.1,
        )

    elif benchmark == CFGANBenchmarks.ML100K:
        return CFGANHyperParameters(
            benchmark=benchmark,
            epochs=250,  # Original 1000. This epochs = 1000 / generator_steps
            mode=CFGANMode.ITEM_BASED,
            mask_type=CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING,
            generator_hidden_features=400,
            generator_regularization=0.001,
            generator_learning_rate=0.0001,
            generator_batch_size=32,
            generator_hidden_layers=1,
            generator_steps=4,
            generator_optimizer=CFGANOptimizer.ADAM,
            generator_activation=CFGANActivation.SIGMOID,
            discriminator_hidden_features=125,
            discriminator_regularization=0,
            discriminator_learning_rate=0.0001,
            discriminator_batch_size=64,
            discriminator_hidden_layers=1,
            discriminator_steps=2,
            discriminator_optimizer=CFGANOptimizer.ADAM,
            discriminator_activation=CFGANActivation.SIGMOID,

            ratio_zero_reconstruction=70,
            ratio_partial_masking=70,
            coefficient_zero_reconstruction=0.03,
        )

    elif benchmark == CFGANBenchmarks.ML1M:
        return CFGANHyperParameters(
            benchmark=benchmark,
            epochs=1550,  # Original 1550. This epochs = 1550 / generator_steps
            mode=CFGANMode.ITEM_BASED,
            mask_type=CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING,
            generator_hidden_features=300,
            generator_regularization=0.001,
            generator_learning_rate=0.0001,
            generator_batch_size=256,
            generator_hidden_layers=1,
            generator_steps=1,
            generator_optimizer=CFGANOptimizer.ADAM,
            generator_activation=CFGANActivation.SIGMOID,
            discriminator_hidden_features=250,
            discriminator_regularization=0.00005,
            discriminator_learning_rate=0.00005,
            discriminator_batch_size=512,
            discriminator_hidden_layers=1,
            discriminator_steps=1,
            discriminator_optimizer=CFGANOptimizer.ADAM,
            discriminator_activation=CFGANActivation.SIGMOID,

            ratio_zero_reconstruction=90,
            ratio_partial_masking=90,
            coefficient_zero_reconstruction=0.03
        )

    else:
        raise NotImplementedError(f"Benchmark {benchmark} not available. Accepted benchmarks: "
                                  f"{list(CFGANBenchmarks)}")


def get_cfgan_paper_hyper_parameters(benchmark: CFGANBenchmarks) -> CFGANHyperParameters:
    """
    Full hyper-parameters are not provided in the paper. We copied the code
    hyper-parameters and changed the ones that were reported in the paper. Specifically,
    we set all epochs to 1000, all mask types to
    `CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING`, all ZR ratios to 70, all PM ratios
    to 70, and ZR coefficient (alpha in the paper) as 0.1
    """

    if benchmark == CFGANBenchmarks.CIAO:
        return CFGANHyperParameters(
            benchmark=benchmark,
            epochs=1000,
            mode=CFGANMode.ITEM_BASED,
            mask_type=CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING,
            generator_hidden_features=250,
            generator_regularization=0.001,
            generator_learning_rate=0.0001,
            generator_batch_size=128,
            generator_hidden_layers=1,
            generator_steps=1,
            generator_optimizer=CFGANOptimizer.ADAM,
            generator_activation=CFGANActivation.SIGMOID,
            discriminator_hidden_features=50,
            discriminator_regularization=0.001,
            discriminator_learning_rate=0.0001,
            discriminator_batch_size=128,
            discriminator_hidden_layers=4,
            discriminator_steps=1,
            discriminator_optimizer=CFGANOptimizer.ADAM,
            discriminator_activation=CFGANActivation.SIGMOID,

            ratio_zero_reconstruction=70,
            ratio_partial_masking=70,
            coefficient_zero_reconstruction=0.1,
        )

    elif benchmark == CFGANBenchmarks.ML100K:
        return CFGANHyperParameters(
            benchmark=benchmark,
            epochs=1000,
            mode=CFGANMode.ITEM_BASED,
            mask_type=CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING,
            generator_hidden_features=400,
            generator_regularization=0.001,
            generator_learning_rate=0.0001,
            generator_batch_size=32,
            generator_hidden_layers=1,
            generator_steps=4,
            generator_optimizer=CFGANOptimizer.ADAM,
            generator_activation=CFGANActivation.SIGMOID,
            discriminator_hidden_features=125,
            discriminator_regularization=0,
            discriminator_learning_rate=0.0001,
            discriminator_batch_size=64,
            discriminator_hidden_layers=1,
            discriminator_steps=2,
            discriminator_optimizer=CFGANOptimizer.ADAM,
            discriminator_activation=CFGANActivation.SIGMOID,

            ratio_zero_reconstruction=70,
            ratio_partial_masking=70,
            coefficient_zero_reconstruction=0.1,
        )

    elif benchmark == CFGANBenchmarks.ML1M:
        return CFGANHyperParameters(
            benchmark=benchmark,
            epochs=1000,
            mode=CFGANMode.ITEM_BASED,
            mask_type=CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING,
            generator_hidden_features=300,
            generator_regularization=0.001,
            generator_learning_rate=0.0001,
            generator_batch_size=256,
            generator_hidden_layers=1,
            generator_steps=1,
            generator_optimizer=CFGANOptimizer.ADAM,
            generator_activation=CFGANActivation.SIGMOID,
            discriminator_hidden_features=250,
            discriminator_regularization=0.00005,
            discriminator_learning_rate=0.00005,
            discriminator_batch_size=512,
            discriminator_hidden_layers=1,
            discriminator_steps=1,
            discriminator_optimizer=CFGANOptimizer.ADAM,
            discriminator_activation=CFGANActivation.SIGMOID,

            ratio_zero_reconstruction=70,
            ratio_partial_masking=70,
            coefficient_zero_reconstruction=0.1
        )

    else:
        raise NotImplementedError(f"Benchmark {benchmark} not available. Accepted benchmarks: "
                                  f"{list(CFGANBenchmarks)}")
