import os
import json

def save_dict_as_json(path_to_save, dictionary, file_name='dict.json'):
    # Check if path_to_save exists
    os.makedirs(path_to_save, exist_ok=True)

    # Join the path and filename
    full_path = os.path.join(path_to_save, file_name)

    # Ensure that the directory containing the file exists
    if not os.path.exists(os.path.dirname(full_path)):
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Save the dictionary to the JSON file
    with open(full_path, 'w') as file:
        json.dump(dictionary, file, default=custom_serializer, indent=4)

def load_json_file(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def custom_serializer(obj):

    import numpy as np
    if callable(obj):
        return obj.__name__ if hasattr(obj, '__name__') else str(obj)  # Devuelve el nombre de la función si existe
    elif isinstance(obj, type):
        return obj.__module__ + '.' + obj.__name__  # Devuelve el nombre completo de la clase
    # elif isinstance(obj, torch.device):
    #     return str(obj)
    # elif isinstance(obj, torch.Tensor):
    #     return obj.tolist()  # Convertir tensores a listas
    # elif isinstance(obj, torch.nn.Module):
    #     return str(obj)  # Convertir módulos a string
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError("Object of type %s is not JSON serializable" % type(obj))