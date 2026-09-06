"""Generate native LIBLINEAR WLRN fixtures and reference predictions.

Run with the core Python package on PYTHONPATH and liblinear-official installed:
    PYTHONPATH=../wlearn/py python test/fixtures/generate.py
"""
import json
from pathlib import Path

import numpy as np
from wlearn.liblinear import LinearModel

FIXTURES_DIR = Path(__file__).parent
rng = np.random.RandomState(42)


def save_fixture(name, X, y, params):
    model = LinearModel.create(params)
    try:
        model.fit(X, y)
        model.save(FIXTURES_DIR / f'{name}.wlrn')
        data = {'X': X.tolist(), 'y': y.tolist(),
                'predictions': model.predict(X).tolist(), 'params': params}
        (FIXTURES_DIR / f'{name}.data.json').write_text(json.dumps(data, indent=2))
    finally:
        model.dispose()
    print(f'Saved {name}: native WLRN + reference predictions')


X = rng.randn(100, 2)
save_fixture('classification', X, (X[:, 0] + X[:, 1] > 0).astype(float),
             {'solver': 0, 'C': 1.0})
X = rng.randn(150, 2)
sums = X[:, 0] + X[:, 1]
save_fixture('multiclass', X, np.where(sums < -0.5, 0, np.where(sums < 0.5, 1, 2)),
             {'solver': 0, 'C': 1.0})
X = rng.randn(100, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + rng.randn(100) * 0.5
save_fixture('regression', X, y, {'solver': 12, 'C': 1.0, 'p': 0.1})
