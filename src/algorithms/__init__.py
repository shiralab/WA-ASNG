from .asng import ASNG
from .pbil import PBIL
from .waasng import WAASNG


def get_algorithm(name: str, *args, **kwargs):
    ALGORITHMS = {"ASNG": ASNG, "WAASNG": WAASNG, "PBIL": PBIL}

    if name not in ALGORITHMS:
        raise ValueError(f"Algorithm '{name}' is not implemented.")

    return ALGORITHMS[name](*args, **kwargs)
