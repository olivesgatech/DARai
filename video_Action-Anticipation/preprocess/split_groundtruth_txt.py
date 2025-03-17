import os

# Define paths
src_path = "FUTR_proposed/datasets/darai/groundTruth_all/"
dst_path = "FUTR_proposed/datasets/darai/groundTruth_nov11/"

# Helper function to read a file
def read_file(filepath):
    with open(filepath, 'r') as file:
        return file.readlines()

# Helper function to write to a file
def write_file(filepath, lines):
    with open(filepath, 'w') as file:
        file.writelines(lines)

# Iterate through files in the source path
for src_file in os.listdir(src_path):
    if src_file.endswith(".txt"):
        # Construct the full source filepath
        src_filepath = os.path.join(src_path, src_file)

        # Read the source file
        src_content = read_file(src_filepath)

        # Extract the base filename for destination mapping
        base_name = "_".join(src_file.split("_")[:-1]) + ".txt"
        dst_filepath = os.path.join(dst_path, base_name)

        # Ensure the destination file exists
        if not os.path.exists(dst_filepath):
            print(f"Destination file {dst_filepath} does not exist. Skipping...")
            continue

        # Read the destination file
        dst_content = read_file(dst_filepath)

        # Filter destination content based on source content (jpg sequence matching)
        filtered_content = [line for line in dst_content if any(jpg in line for jpg in [row.split(",")[0].split("/")[-1] for row in src_content])]

        # Construct the new destination filename
        new_dst_filename = src_file
        new_dst_filepath = os.path.join(dst_path, new_dst_filename)

        # Write the filtered content to the new destination file
        write_file(new_dst_filepath, filtered_content)

        print(f"Created {new_dst_filepath} with {len(filtered_content)} lines.")
