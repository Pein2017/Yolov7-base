import os
import re


def clean_ckpt(base_dir):
    """
    Traverse through all subdirectories within the given base directory
    and delete all *.pt files except for 'best.pt' and the 'best_xx.pt'
    where xx is the largest number within each subdirectory.

    Args:
        base_dir (str): The base directory to start the search.
    """
    # Traverse through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Filter .pt files in the directory
        pt_files = [f for f in files if f.endswith(".pt")]

        if pt_files:  # Process only if there are .pt files in the directory
            # Find best.pt and best_xx.pt files
            best_pt = None
            best_numbered_pt = None
            highest_number = -1

            for pt_file in pt_files:
                if pt_file == "best.pt":
                    best_pt = pt_file
                elif pt_file.startswith("best_") and pt_file.endswith(".pt"):
                    # Extract the number from the best_xx.pt filename
                    match = re.search(r"best_(\d+)\.pt", pt_file)
                    if match:
                        number = int(match.group(1))
                        if number > highest_number:
                            highest_number = number
                            best_numbered_pt = pt_file

            # Determine which files to keep
            keep_files = set([best_pt, best_numbered_pt])

            # Remove other .pt files
            for pt_file in pt_files:
                pt_path = os.path.join(root, pt_file)
                if pt_file not in keep_files:
                    print(f"Deleting file: {pt_path}")
                    os.remove(pt_path)

    print("Cleanup complete!")


if __name__ == "__main__":
    # Replace with your target directory
    base_directory = "/data/training_code/yolov7/runs/train"
    clean_ckpt(base_directory)
