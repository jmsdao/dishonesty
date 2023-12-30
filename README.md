## Handy Dandy Snippets
```bash
# Install some linux tools
apt-get update && \
apt-get -y install vim && \
apt-get -y install less && \
apt-get -y install tmux
tmux
```

```bash
# Install mamba
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh && \
bash Mambaforge-Linux-x86_64.sh -b && \
mambaforge/bin/mamba init bash && \
exec bash
```

```bash
# Clone and cd into repo
git clone https://github.com/jmsdao/dishonesty.git && cd dishonesty
```

```bash
# Install and activate env (fresh)
mamba env create -f environment.yml && mamba activate dishonesty
```

```bash
# Activate the env from a runpod workspace
mamba activate /workspace/mamba-envs/dishonesty
```

```bash
# Install the repo's packages
pip install -e .
```

```python
# Handy snippet to get repo root from anywhere in the repo
import sys
from subprocess import check_output
ROOT = check_output('git rev-parse --show-toplevel', shell=True).decode("utf-8").strip()
if ROOT not in sys.path: sys.path.append(ROOT)
```

## Some ideas for later
- Definitely need to test out more prompts, especially ones that are negations of one another
- Try some new ideas:
  - Training SAEs on the MLPs that stands out ie mlp_20
  - Investigating what's up with layer 16
  - Trying EAP, path patching or ACDC
  - Playing around with patching corrupted into clean (noising)
  - What impact does this direction have, outside of the distribution of our prompts ivolving honesty/dishonesty
  - Why exactly has this direction been learned:
    - Look at where CE loss gets worse the most when we remove this direction
    - Checking max activating dataset examples to see where this direction occurs "in the wild"
    - Looking into how these directions are formed
