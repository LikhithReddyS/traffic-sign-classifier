"""Quick camera diagnostic — run this to check which camera backends work."""
import cv2
import sys

print(f"Python: {sys.version}")
print(f"OpenCV: {cv2.__version__}")
print()

backends = [
    ("Default (auto)", None),
    ("DirectShow (DSHOW)", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
]

for idx in range(3):
    for name, backend in backends:
        try:
            if backend is not None:
                cap = cv2.VideoCapture(idx, backend)
            else:
                cap = cv2.VideoCapture(idx)
            opened = cap.isOpened()
            if opened:
                ret, frame = cap.read()
                if ret:
                    print(f"  [OK]   Camera {idx} + {name}: opened & read frame ({frame.shape})")
                else:
                    print(f"  [WARN] Camera {idx} + {name}: opened but failed to read frame")
            else:
                print(f"  [FAIL] Camera {idx} + {name}: could not open")
            cap.release()
        except Exception as e:
            print(f"  [ERR]  Camera {idx} + {name}: {e}")
    print()
