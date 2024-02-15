import os
import shutil
cache_dir = os.path.expanduser('~/.cache/torch')  # Default cache directory

# Check if the cache directory exists and delete it
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("PyTorch cache directory has been deleted.")
else:
    print("Cache directory does not exist.")