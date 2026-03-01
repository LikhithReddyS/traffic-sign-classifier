
import os
try:
    with open("d:\\traffic sign classifier\\test_log.txt", "w") as f:
        f.write("Python file I/O works.")
    print("Success")
except Exception as e:
    print(f"Error: {e}")
