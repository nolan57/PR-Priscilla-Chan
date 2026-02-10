## Problem Analysis
The error occurs because:
1. `extract_perfect_f0.py` imports `get_pitch_parselmouth` from `DiffSinger.utils.binarizer_utils`
2. This triggers the import of `DiffSinger/utils/__init__.py` which has `from basics.base_module import CategorizedModule`
3. The `basics` module is not found because it's a subdirectory of `DiffSinger`, not a top-level module
4. While `extract_perfect_f0.py` adds the root directory to the Python path, this doesn't help because `basics` isn't directly under the root

## Solution
Modify `extract_perfect_f0.py` to add the `DiffSinger` directory to the Python path instead of the root directory, and update the import statement accordingly:

1. Change the Python path setup to add the `DiffSinger` directory
2. Update the import statement to use the correct path relative to the new Python path

## Implementation Steps
1. Modify the Python path setup in `extract_perfect_f0.py` to add the `DiffSinger` directory instead of the root directory
2. Update the import statement from `from DiffSinger.utils.binarizer_utils import get_pitch_parselmouth` to `from utils.binarizer_utils import get_pitch_parselmouth`

This solution maintains the existing import structure in `DiffSinger` while making it work when called from outside the directory.