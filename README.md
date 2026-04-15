# TSP Resolver

Travelling Salesman Problem solver with random graph generation.

## Setup

### 1. Create virtual environment in VS Code

Open the project in VS Code, then open the Command Palette (`Cmd+Shift+P`) and run:

```
Python: Create Environment
```

Select **Venv** → select your Python interpreter. VS Code creates `.venv/` at the project root.

### 2. Install dependencies

Open a terminal (`Ctrl+`` `) — VS Code auto-activates `.venv`. Then run:

```bash
pip install networkx numpy matplotlib ipykernel
```

### 3. Select kernel in notebook

Open `tsp_resolver.ipynb`, click the kernel picker (top right), and select:

```
.venv (Python 3.x.x)
```

If `.venv` doesn't appear, run `Python: Select Interpreter` from the Command Palette and pick `.venv/bin/python`.
