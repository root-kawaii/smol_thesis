import h5py

weights_sums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Specify the path to your H5 weights file
for i in range(10, 40):
    weights_file_path = str(i) + "saved_weights.h5"

    # Open the H5 file in read-only mode
    with h5py.File(weights_file_path, "r") as file:

        # Print the keys of the groups in the H5 file
        # print("Keys in the H5 file:", list(file.keys()))

        # Access the 'model_weights' group (assuming default Keras structure)
        model_weights_group = file["dense2"]

        # Print the keys within the 'model_weights' group
        # print("\nKeys in 'model_weights' group:", list(model_weights_group.keys()))

        # Access the specific layer's weights
        # 'dense' is the name of the layer
        layer_weights = model_weights_group["dense2"]
        # print("\nKeys in 'dense' layer group:", list(layer_weights.keys()))

        # Access and print the actual weight values
        # 'kernel:0' is the weight tensor
        weights_values = layer_weights["kernel:0"][:]
        # print("\nWeight values:")
        # weights_values = weights_values[0]
        # Sort the indices based on the values
        for j, k in enumerate(weights_values):
            weights_sums[j] += k

sorted_indices = sorted(
    range(len(weights_sums)), key=lambda i: weights_sums[i][0], reverse=True
)
# Return the sorted indices
print(weights_sums)
print(sorted_indices)
