import torch
import os
import pickle


def load_model(filename: str):

    if not isinstance(filename, str):
        raise TypeError("Parameter 'filename' must be a string.")

    try:
        with open(filename, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError as e:
        raise e

    # Try to load the state_dict corresponding to the model
    _, file_ending = os.path.splitext(filename)
    state_dict_name = filename.replace(file_ending, ".state_dict")
    try:
        model.load_state_dict(torch.load(state_dict_name))
    except FileNotFoundError as e:
        print(f"No state_dict with name '{state_dict_name}' found, loading model without. {e}")
        
    return model
