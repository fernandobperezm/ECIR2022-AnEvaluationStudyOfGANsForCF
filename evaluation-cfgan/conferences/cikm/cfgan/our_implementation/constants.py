from enum import Enum
from recsys_framework.Recommenders.DataIO import attach_to_extended_json_decoder


@attach_to_extended_json_decoder
class CFGANType(Enum):
    NOT_CONDITIONED = "NOT_CONDITIONED"
    CONDITIONED_XOR = "CONDITIONED_XOR"
    CONDITIONED_CONCAT = "CONDITIONED_CONCAT"
    CONDITIONED_INVERTED_CONCAT = "CONDITIONED_INVERTED_CONCAT"


@attach_to_extended_json_decoder
class CFGANConditionType(Enum):
    REAL_INTERACTIONS = "REAL_INTERACTIONS"
    ENUMERATED = "ENUMERATED"


@attach_to_extended_json_decoder
class CFGANDatasetMode(Enum):
    ORIGINAL = "ORIGINAL"
    OURS = "OURS"
    COLD_USERS = "COLD_USERS"


@attach_to_extended_json_decoder
class CFGANBenchmarks(Enum):
    CIAO = "Ciao"
    ML100K = "ML100K"
    ML1M = "ML1M"


@attach_to_extended_json_decoder
class CFGANMode(Enum):
    ITEM_BASED = "ITEM_BASED"
    USER_BASED = "USER_BASED"


@attach_to_extended_json_decoder
class CFGANScheme(Enum):
    ZR = "ZR"
    ZP = "ZP"
    PM = "PM"


@attach_to_extended_json_decoder
class CFGANOptimizer(Enum):
    ADAM = "ADAM"


@attach_to_extended_json_decoder
class CFGANActivation(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"


@attach_to_extended_json_decoder
class CFGANMaskType(Enum):
    NO_MASK = "NO_MASK"
    REAL_INTERACTIONS = "REAL_INTERACTIONS"
    ROUND_AND_RELU_ZERO_OR_ONE = "RELU_ZERO_OR_ONE"
    ZERO_RECONSTRUCTION = "ZERO_RECONSTRUCTION"
    PARTIAL_MASKING = "PARTIAL_MASKING"
    ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING = "ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING"
