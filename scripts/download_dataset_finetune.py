import objaverse
import os
import json
import shutil

def main():
    # Step 1: Load LVIS annotations
    print("Loading LVIS annotations...")
    lvis_annotations = objaverse.load_lvis_annotations()
    print(f"Total labels available: {len(lvis_annotations.keys())}")

    # Step 2: Sort labels by the number of objects (length of UID list) in descending order
    sorted_labels = sorted(lvis_annotations.keys(), key=lambda x: len(lvis_annotations[x]), reverse=True)
    print(f"Labels sorted by number of objects: {sorted_labels[:10]}")  # Display top 10 labels for context

    # Choose the first 100 labels
    selected_labels = sorted_labels[:100]
    print(f"Selected {len(selected_labels)} labels: {selected_labels}")

    # Step 3: Find UIDs for the selected labels
    uids_to_download = []
    label_to_uid_map = {}  # To map label to UIDs
    for label in selected_labels:
        uids = lvis_annotations[label]
        uids_to_download.extend(uids)
        label_to_uid_map[label] = uids

    print(f"Found {len(uids_to_download)} objects to download.")

    # Step 4: Prepare output directory
    output_dir = "./objaverse_finetune"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize mapping dictionary
    file_to_label_map = {}

    # Step 5: Download objects in batches
    batch_size = 1000
    total_objects = len(uids_to_download)

    for i in range(0, total_objects, batch_size):
        batch_uids = uids_to_download[i:i + batch_size]
        print(f"Downloading batch {i // batch_size + 1} containing {len(batch_uids)} objects...")

        # Download the current batch of objects
        objects = objaverse.load_objects(uids=batch_uids, download_processes=os.cpu_count())

        # Move downloaded files and update mapping
        for uid, filepath in objects.items():
            filename = f"{uid}.glb"
            new_filepath = os.path.join(output_dir, filename)
            shutil.move(filepath, new_filepath)  # Move file to output directory
            label = next((key for key, value in label_to_uid_map.items() if uid in value), None)
            if label:
                file_to_label_map[filename] = label

        # Save the mapping file after each batch
        mapping_file = os.path.join(output_dir, "file_to_label_map.json")
        with open(mapping_file, "w") as f:
            json.dump(file_to_label_map, f, indent=4)
        print(f"Mapping file saved to '{mapping_file}'.")

    print(f"Downloaded and processed {len(file_to_label_map)} objects in total.")

if __name__ == '__main__':
    main()
