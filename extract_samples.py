"""Extract one sample image per class (43 total) and save as PNG."""
import sys
import os
from pathlib import Path

# Force unbuffered stdout/stderr
sys.stdout.reconfigure(encoding='utf-8')

print("Starting sample extraction...", flush=True)

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}", flush=True)
except ImportError as e:
    print(f"Error importing cv2: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

project_root = Path(__file__).resolve().parent
print(f"Project root: {project_root}", flush=True)

train_root = project_root / "dataset" / "train" / "Final_Training" / "Images"
print(f"Looking for dataset in: {train_root}", flush=True)

if not train_root.exists():
    print(f"ERROR: Dataset not found at {train_root}", file=sys.stderr, flush=True)
    sys.exit(1)

output_dir = project_root / "sample_signs"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}", flush=True)


GTSRB_CLASSES = {
    0: "Speed_limit_20kmh",
    1: "Speed_limit_30kmh",
    2: "Speed_limit_50kmh",
    3: "Speed_limit_60kmh",
    4: "Speed_limit_70kmh",
    5: "Speed_limit_80kmh",
    6: "End_speed_limit_80kmh",
    7: "Speed_limit_100kmh",
    8: "Speed_limit_120kmh",
    9: "No_passing",
    10: "No_passing_over_3.5t",
    11: "Right_of_way",
    12: "Priority_road",
    13: "Yield",
    14: "Stop",
    15: "No_vehicles",
    16: "Vehicles_over_3.5t_prohibited",
    17: "No_entry",
    18: "General_caution",
    19: "Dangerous_curve_left",
    20: "Dangerous_curve_right",
    21: "Double_curve",
    22: "Bumpy_road",
    23: "Slippery_road",
    24: "Road_narrows_right",
    25: "Road_work",
    26: "Traffic_signals",
    27: "Pedestrians",
    28: "Children_crossing",
    29: "Bicycles_crossing",
    30: "Beware_ice_snow",
    31: "Wild_animals_crossing",
    32: "End_all_limits",
    33: "Turn_right_ahead",
    34: "Turn_left_ahead",
    35: "Ahead_only",
    36: "Go_straight_or_right",
    37: "Go_straight_or_left",
    38: "Keep_right",
    39: "Keep_left",
    40: "Roundabout_mandatory",
    41: "End_no_passing",
    42: "End_no_passing_3.5t",
}

count = 0
for class_id in range(43):
    class_dir = train_root / f"{class_id:05d}"
    if not class_dir.exists():
        print(f"[SKIP] Class {class_id}: folder not found")
        continue

    # Find the first PPM image
    images = sorted(class_dir.glob("*.ppm"))
    if not images:
        print(f"[SKIP] Class {class_id}: no PPM images")
        continue

    # Pick a sample from the middle (usually better quality/size)
    sample = images[len(images) // 2]
    img = cv2.imread(str(sample))
    if img is None:
        print(f"[SKIP] Class {class_id}: could not read {sample.name}")
        continue

    # Resize to a nice display size
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    name = GTSRB_CLASSES.get(class_id, f"class_{class_id}")
    out_path = output_dir / f"{class_id:02d}_{name}.png"
    cv2.imwrite(str(out_path), img)
    count += 1
    print(f"[OK] Class {class_id:02d}: {name} -> {out_path.name}")

print(f"\nDone! Saved {count} sample images to: {output_dir}")
