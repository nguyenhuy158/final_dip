import os
import random

# List of possible commit message prefixes
prefixes = ['Fix', 'Update', 'Add', 'Remove', 'Refactor']

# List of possible commit message suffixes
suffixes = ['bug', 'typo', 'style', 'feature', 'doc']

# Generate a random commit message
prefix = random.choice(prefixes)
suffix = random.choice(suffixes)
message = f"{prefix} {suffix}"

# Commit the changes with the generated message
os.system(f"git add . && git commit -m '{message}' && git push")
