from .bench import (
    BinVal,
    LeadingOnes,
    NoisyBinVal,
    NoisyLeadingOnes,
    NoisyOneMax,
    OneMax,
)


def get_problem(name, *args, **kwargs):
    name = name.lower()
    PROBLEMS = {
        "onemax": OneMax,
        "leadingones": LeadingOnes,
        "binval": BinVal,
        "noisyonemax": NoisyOneMax,
        "noisyleadingones": NoisyLeadingOnes,
        "noisybinval": NoisyBinVal,
    }

    return PROBLEMS[name](*args, **kwargs)
