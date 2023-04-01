from typing import List
import pickle


def load_model(path):

    with open(path, "rb") as handle:
        model = pickle.load(handle)

    return model


def _get_param_storage_key_prefix(with_guide: bool = False) -> str:
    # Doesn't do anything, just a central place to store the key prefix

    if with_guide:
        return "FactorModel._guide."
    else:
        return "FactorModel."


def _get_keys_from_model(model) -> List[str]:
    # Returns the keys used by the model, not pulling them from the pyro param storage

    keys = [_get_param_storage_key_prefix() + k[0] for k in model.named_pyro_params()]

    return keys
