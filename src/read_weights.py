import h5py

# Specify the path to your H5 weights file
weights_file_path = "2saved_weights.h5"

# Open the H5 file in read-only mode
with h5py.File(weights_file_path, "r") as file:

    # Print the keys of the groups in the H5 file
    print("Keys in the H5 file:", list(file.keys()))

    # Access the 'model_weights' group (assuming default Keras structure)
    model_weights_group = file["dense2"]

    # Print the keys within the 'model_weights' group
    print("\nKeys in 'model_weights' group:", list(model_weights_group.keys()))

    # Access the specific layer's weights
    # 'dense' is the name of the layer
    layer_weights = model_weights_group["dense2"]
    print("\nKeys in 'dense' layer group:", list(layer_weights.keys()))

    # Access and print the actual weight values
    # 'kernel:0' is the weight tensor
    weights_values = layer_weights["kernel:0"][:]
    print("\nWeight values:")
    print(weights_values)
