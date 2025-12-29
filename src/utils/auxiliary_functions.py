def convert_labels_to_int(labels, dict_labels={}):
    import numpy as np
    return np.vectorize(dict_labels.get)(labels)

def binarize_y(y):
    import numpy as np
    """
    Convierte un vector y con 2 clases a binario (0 y 1), 
    devolviendo también el mapeo inverso para saber a qué clase original corresponde cada valor.
    """
    unique_classes = np.unique(y)
    assert len(unique_classes) == 2, "y debe tener exactamente dos clases distintas"
    
    # Asignar la clase menor al 0, y la mayor al 1 (orden fijo)
    class_to_bin = {unique_classes[0]: 0, unique_classes[1]: 1}
    bin_to_class = {v: k for k, v in class_to_bin.items()}
    
    y_bin = np.vectorize(class_to_bin.get)(y)
    
    return y_bin, bin_to_class

def zip_files_in_directory(source_directory, output_directory, prefix="compressed_files"):
    import os
    import zipfile
    from datetime import datetime
    """
    Compress all files in the specified source directory into a ZIP file,
    and save it to the specified output directory.
    
    Args:
        source_directory (str): The path to the directory containing the files to compress.
        output_directory (str): The path to the directory where the ZIP file will be saved.
    
    Returns:
        str: The path to the created ZIP file.
    """
    # Create a timestamp for the ZIP file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"{prefix}_{timestamp}.zip"
    zip_filepath = os.path.join(output_directory, zip_filename)

    # Create the ZIP file and add files from the source directory
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(source_directory):
            for file in files:
                full_path = os.path.join(root, file)
                # Add the file to the ZIP file using its relative path to preserve directory structure
                zip_file.write(full_path, os.path.relpath(full_path, source_directory))

    print(f"Files compressed into {zip_filepath}")
    return zip_filepath
