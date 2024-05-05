import numpy as np

def convert_to_c_array(filename, var_name, is_2d=True):
    # Load data from CSV
    data = np.loadtxt(filename, delimiter=',')

    # Check if the data is 1D or 2D and format accordingly
    if is_2d:
        # Convert 2D data to C array string
        c_array = f"const float {var_name}[{data.shape[0]}][{data.shape[1]}] = {{\n"
        for i, row in enumerate(data):
            row_str = "{" + ", ".join(f"{x:.6f}" for x in row) + "}"
            if i < len(data) - 1:
                row_str += ","
            c_array += row_str + "\n"
    else:
        # Convert 1D data to C array string
        c_array = f"const float {var_name}[{data.shape[0]}] = {{\n"
        c_array += ", ".join(f"{x:.6f}" for x in data) + "\n};\n"

    c_array += "};\n"
    return c_array

# Paths to your CSV files
matrix_filename = 'transformation_matrix_400.csv'
mean_filename = 'mean_vector_400.csv'

# Generate C arrays
matrix_var_name = 'pca_transformation_matrix'
mean_var_name = 'pca_mean_vector'
matrix_shape = (100, 4096)  # 120 components, each of 4096 elements
mean_shape = (4096,)         # Mean vector is a 1D array with 4096 elements

matrix_c_array = convert_to_c_array(matrix_filename, matrix_var_name, is_2d=True)
mean_c_array = convert_to_c_array(mean_filename, mean_var_name, is_2d=False)

# Save the arrays to a .h file
with open('pca_parameters_400.h', 'w') as f:
    f.write(matrix_c_array + "\n" + mean_c_array)
