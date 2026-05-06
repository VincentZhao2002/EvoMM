import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    print("cv2 imported successfully")
except ImportError as e:
    print(f"Error importing cv2: {e}")

try:
    import pandas as pd
    print("pandas imported successfully")
except ImportError as e:
    print(f"Error importing pandas: {e}")

try:
    from eval.memgallery_test_mm import MultimodalVLLMGenerator
    print("MultimodalVLLMGenerator imported successfully")
except ImportError as e:
    print(f"Error importing MultimodalVLLMGenerator: {e}")
    # Try alternative import if running as script in eval dir
    try:
        from memgallery_test_mm import MultimodalVLLMGenerator
        print("MultimodalVLLMGenerator imported from local module")
    except ImportError as e2:
        print(f"Error importing from local module: {e2}")

try:
    from evomm import InMemoryPageStore
    print("evomm imported successfully")
except ImportError as e:
    print(f"Error importing evomm: {e}")
