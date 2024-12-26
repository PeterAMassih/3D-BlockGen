import objaverse
import os
import json

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

    # Step 4: Download GLB files
    output_dir = "./downloaded_objects"
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading objects...")
    objects = objaverse.load_objects(uids=uids_to_download, download_processes=os.cpu_count())

    # Step 5: Save the files locally and create a file-to-label mapping
    file_to_label_map = {}
    for label, uids in label_to_uid_map.items():
        for uid in uids:
            if uid in objects:
                filepath = objects[uid]
                filename = f"{uid}.glb"
                new_filepath = os.path.join(output_dir, filename)
                os.rename(filepath, new_filepath)
                file_to_label_map[filename] = label

    # Save the mapping as a JSON file
    mapping_file = os.path.join(output_dir, "file_to_label_map.json")
    with open(mapping_file, "w") as f:
        json.dump(file_to_label_map, f, indent=4)

    print(f"Downloaded {len(objects)} objects.")
    print(f"File-to-label mapping saved to '{mapping_file}'.")

if __name__ == '__main__':
    main()
