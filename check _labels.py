import os

def check_labels(image_dir, label_dir):
    errors = []

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        name = os.path.splitext(filename)[0]
        label_path = os.path.join(label_dir, name + ".txt")

        # Check if label exists
        if not os.path.exists(label_path):
            errors.append(f"❌ Missing label: {label_path}")
            continue

        # Check if label is empty
        if os.path.getsize(label_path) == 0:
            errors.append(f"⚠️ Empty label: {label_path}")
            continue

        # Validate label content
        with open(label_path, 'r') as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    errors.append(f"❌ Invalid format in {label_path} (Line {i}): {line.strip()}")
                    continue
                try:
                    class_id = int(parts[0])
                    nums = list(map(float, parts[1:]))
                    if not all(0 <= n <= 1 for n in nums):
                        errors.append(f"❌ Values out of range in {label_path} (Line {i}): {line.strip()}")
                except ValueError:
                    errors.append(f"❌ Non-numeric value in {label_path} (Line {i}): {line.strip()}")

    if errors:
        print("🚨 Issues found:")
        for e in errors:
            print(e)
    else:
        print("✅ All label files are valid.")

# === USAGE ===
check_labels("train/img", "train/labels")
check_labels("val/img", "val/labels")
