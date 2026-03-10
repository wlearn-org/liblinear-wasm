# Changelog

## 0.2.0 (unreleased)

- Wrap LinearModel with `createModelClass` for unified task detection
- Add `task` parameter: `'classification'` or `'regression'`, auto-detected from labels if omitted

## 0.1.0 (unreleased)

- Initial release
- LIBLINEAR v2.50 compiled to WASM via Emscripten
- Unified sklearn-style API: `create()`, `fit()`, `predict()`, `score()`, `save()`, `dispose()`
- Linear classifiers (logistic regression, SVC) and regressors (SVR)
- Buffer-based model I/O (no filesystem dependency)
- Accepts both typed matrices and number[][] with configurable coercion
- `predictProba()` for logistic regression solvers
- `decisionFunction()` for decision values
- `getParams()`/`setParams()` for AutoML integration
- `defaultSearchSpace()` for hyperparameter search
- `FinalizationRegistry` safety net for leak detection
- BSD-3-Clause license (same as upstream LIBLINEAR)
