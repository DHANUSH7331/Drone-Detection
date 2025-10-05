import os

base_dir = '.'  # run this script inside 'drone detection' folder
weights_main_folder = 'Weights'
target_weight_file = 'yolov8_model.pt'

print(f"Checking weights inside '{weights_main_folder}' folder:")
main_weights_path = os.path.join(base_dir, weights_main_folder, target_weight_file)
if os.path.isfile(main_weights_path):
    print(f"  FOUND main weight file: {main_weights_path}")
else:
    print(f"  NOT FOUND main weight file: {main_weights_path}")

print("\nScanning runs/train* folders for weights...")

runs_dir = os.path.join(base_dir, 'runs')
if not os.path.isdir(runs_dir):
    print("No 'runs' folder found. Please check your folder structure.")
else:
    for root, dirs, files in os.walk(runs_dir):
        if 'weights' in dirs:
            weights_path = os.path.join(root, 'weights')
            pt_files = [f for f in os.listdir(weights_path) if f.endswith('.pt')]
            if pt_files:
                print(f"\nFound weights in: {weights_path}")
                for f in pt_files:
                    print(f"  - {f}")

print("\nDone.")
