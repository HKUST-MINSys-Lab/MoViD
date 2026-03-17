import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python archive/retired_tools/analysis/inspect_npz.py <path_to_npz_file>")
    sys.exit(1)

npz_path = sys.argv[1]

print(f"Inspecting: {npz_path}")
print("="*60)

data = np.load(npz_path, allow_pickle=True)

print(f"\nKeys in NPZ file: {list(data.keys())}")
print("\nDetailed structure:")
print("="*60)

for key in data.keys():
    value = data[key]
    print(f"\nKey: '{key}'")
    print(f"  Type: {type(value)}")
    
    if hasattr(value, 'shape'):
        print(f"  Shape: {value.shape}")
        print(f"  Dtype: {value.dtype}")
    
    if hasattr(value, 'item') and value.shape == ():
        # Scalar array, might contain a dict or other object
        item = value.item()
        print(f"  Item type: {type(item)}")
        if isinstance(item, dict):
            print(f"  Dict keys: {list(item.keys())}")
            for k, v in item.items():
                if hasattr(v, 'shape'):
                    print(f"    '{k}': shape {v.shape}, dtype {v.dtype}")
                else:
                    print(f"    '{k}': {type(v)}")
    
    # Show first few values if small array
    if hasattr(value, 'shape') and value.size <= 10:
        print(f"  Values: {value}")

print("\n" + "="*60)
