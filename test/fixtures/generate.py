"""
Generate test fixtures for @wlearn/liblinear cross-runtime parity tests.

Requires: scikit-learn (which wraps LIBLINEAR internally)

Usage:
    python test/fixtures/generate.py
"""

import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Ridge
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent
np.random.seed(42)


def save_fixture(name, X, y, predictions, params):
    data = {
        'X': X.tolist(),
        'y': y.tolist(),
        'predictions': predictions.tolist(),
        'params': params
    }
    with open(FIXTURES_DIR / f'{name}.data.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  Saved {name}.data.json ({len(X)} samples, {X.shape[1]} features)')


def save_liblinear_model(model, name):
    """Save the underlying LIBLINEAR model in native format."""
    # sklearn wraps liblinear but doesn't expose save_model directly.
    # For cross-runtime parity, we save predictions instead.
    # Native model files can be generated with liblinear's CLI tool.
    pass


# --- Binary classification ---
print('Binary classification (LogisticRegression / L2R_LR):')
X = np.random.randn(100, 2).astype(np.float64)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)

clf = LogisticRegression(solver='liblinear', C=1.0, random_state=42, max_iter=1000)
clf.fit(X, y)
preds = clf.predict(X)

save_fixture('classification', X, y, preds, {
    'solver': 'L2R_LR',
    'C': 1.0,
})

# --- Multi-class classification ---
print('Multi-class classification (LogisticRegression / L2R_LR):')
X_mc = np.random.randn(150, 2).astype(np.float64)
sums = X_mc[:, 0] + X_mc[:, 1]
y_mc = np.where(sums < -0.5, 0, np.where(sums < 0.5, 1, 2)).astype(np.float64)

clf_mc = LogisticRegression(solver='liblinear', C=1.0, random_state=42, max_iter=1000)
clf_mc.fit(X_mc, y_mc)
preds_mc = clf_mc.predict(X_mc)

save_fixture('multiclass', X_mc, y_mc, preds_mc, {
    'solver': 'L2R_LR',
    'C': 1.0,
})

# --- Regression (SVR) ---
print('Regression (LinearSVR / L2R_L2LOSS_SVR_DUAL):')
X_reg = np.random.randn(100, 2).astype(np.float64)
y_reg = (2 * X_reg[:, 0] + 3 * X_reg[:, 1] + np.random.randn(100) * 0.5).astype(np.float64)

from sklearn.svm import LinearSVR
reg = LinearSVR(C=1.0, epsilon=0.1, random_state=42, max_iter=10000)
reg.fit(X_reg, y_reg)
preds_reg = reg.predict(X_reg)

save_fixture('regression', X_reg, y_reg, preds_reg, {
    'solver': 'L2R_L2LOSS_SVR_DUAL',
    'C': 1.0,
    'p': 0.1,
})

print('Done.')
