![memento](https://github.com/Guilliu/Memento/blob/main/visual/dragons_wp.jpg)
# Memento

**Memento** is a python package with the aim of providing the necessary tools for:

- Grouping variables (both numerical and categorical) in an automatic and interactive way.
- Development of customizable scorecards adaptable to the requirements of each user.

## Installation
You can install Memento using pip
```
pip install memento-scorecard
```

## Documentation
Check out the official documentation: https://guilliu.github.io

## Simple template
```python
# Import the modules
import numpy as np, pandas as pd, memento as me

# Load the data
from sklearn.datasets import load_breast_cancer as lbc
X, y = pd.DataFrame(lbc().data, columns=lbc().feature_names), lbc().target 

# Get the auto-scorecard
model = me.scorecard().fit(X, y)

# Display the scorecard
me.pretty_scorecard(model)
```

## Examples
In the folder `/examples` there are notebooks that explore different use cases in detail.

## Code style
The code tries to be as minimalist as possible. The maximum characters per line is set to 100, since the 80 characters of the PEP 8 standard are considered worse for readability. For all other questions, it is recommended to follow the PEP 8 standards, with a slight preference for the use of single quotes.

