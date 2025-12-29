import importlib

def import_class(class_name, module_name=None):
    """
    Imports and returns a class or function given its name.
    
    :param class_name: Name of the class or function to import.
    :param module_name: Name of the module where the class or function is located (optional).
                        If not provided, it searches in the current module.
    :return: The imported class or function.
    """
    if module_name:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            print(f"Error: The module '{module_name}' is not in sys.path.")
            raise
    return getattr(module, class_name)

def import_func(func_name, module_name=None):
    """
    Imports and returns a function given its name.
    
    :param func_name: Name of the function to import.
    :param module_name: Name of the module where the function is located (optional).
                        If not provided, it searches in the current module.
    :return: The imported function.
    """
    return import_class(func_name, module_name)
