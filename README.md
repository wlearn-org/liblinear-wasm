# @wlearn/liblinear

LIBLINEAR v2.50 compiled to WebAssembly. Linear classification and regression in browsers and Node.js.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)). Based on [LIBLINEAR v2.50](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) (BSD-3-Clause). Zero dependencies. CommonJS.

## Install

```bash
npm install @wlearn/liblinear
```

## Quick start

```js
const { LinearModel } = require('@wlearn/liblinear')

const model = await LinearModel.create({
  solver: 'L2R_LR',
  C: 1.0
})

// Train -- accepts number[][] or { data: Float64Array, rows, cols }
model.fit(
  [[1, 2], [3, 4], [5, 6], [7, 8]],
  [0, 0, 1, 1]
)

// Predict
const preds = model.predict([[2, 3], [6, 7]])  // Float64Array

// Probabilities (logistic regression solvers only)
const probs = model.predictProba([[2, 3], [6, 7]])  // Float64Array (nrow * nclass)

// Score
const accuracy = model.score([[2, 3], [6, 7]], [0, 1])

// Save / load
const buf = model.save()  // Uint8Array
const model2 = await LinearModel.load(buf)

// Clean up -- required, WASM memory is not garbage collected
model.dispose()
model2.dispose()
```

## Typed matrix input (fast path)

For performance-critical code and AutoML integration, pass typed matrices directly:

```js
const X = {
  data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]),
  rows: 4,
  cols: 2
}
const y = new Float64Array([0, 0, 1, 1])

model.fit(X, y)
```

This avoids the conversion cost of nested arrays.

## Input coercion policy

Control how non-typed inputs are handled:

```js
// Default: convert silently
const model = await LinearModel.create({ coerce: 'auto' })

// Warn once per instance when conversion happens
const model = await LinearModel.create({ coerce: 'warn' })

// Throw if input is not already a typed matrix
const model = await LinearModel.create({ coerce: 'error' })
```

## API

### `LinearModel.create(params?)`

Async factory. Loads WASM module, returns a ready-to-use model.

Parameters:
- `task` -- `'classification'` or `'regression'`. Auto-detected from labels if omitted.
- `solver` -- solver type string or number (default: `'L2R_LR'`)
- `C` -- regularization parameter (default: `1.0`)
- `eps` -- stopping tolerance (default: `0.01`)
- `bias` -- bias term, < 0 disables (default: `-1`)
- `p` -- epsilon in SVR loss (default: `0.1`)
- `coerce` -- input coercion: `'auto'` | `'warn'` | `'error'` (default: `'auto'`)

### `model.fit(X, y)`

Train on data. Returns `this` for chaining.
- `X` -- `number[][]` or `{ data: Float64Array, rows, cols }`
- `y` -- `number[]` or `Float64Array`

### `model.predict(X)`

Returns `Float64Array` of predicted labels.

### `model.predictProba(X)`

Returns `Float64Array` of shape `nrow * nclass` (row-major probabilities).
Only available for logistic regression solvers (L2R_LR, L1R_LR, L2R_LR_DUAL).

### `model.decisionFunction(X)`

Returns `Float64Array` of decision values.

### `model.score(X, y)`

Returns accuracy (classification) or R-squared (regression).

### `model.save()`

Returns `Uint8Array` (native LIBLINEAR model format).

### `LinearModel.load(buffer)`

Loads from `Uint8Array`. Returns `Promise<LinearModel>`.

### `model.dispose()`

Free WASM memory. Required. Idempotent.

### `model.getParams()` / `model.setParams(p)`

Get/set hyperparameters. Enables AutoML grid search and cloning.

### `LinearModel.defaultSearchSpace()`

Returns default hyperparameter search space for AutoML.

## Solver types

| Name | Code | Task |
|------|------|------|
| L2R_LR | 0 | L2-regularized logistic regression |
| L2R_L2LOSS_SVC_DUAL | 1 | L2-loss SVM (dual) |
| L2R_L2LOSS_SVC | 2 | L2-loss SVM (primal) |
| L2R_L1LOSS_SVC_DUAL | 3 | L1-loss SVM (dual) |
| MCSVM_CS | 4 | Multi-class SVM (Crammer-Singer) |
| L1R_L2LOSS_SVC | 5 | L1-regularized L2-loss SVM |
| L1R_LR | 6 | L1-regularized logistic regression |
| L2R_LR_DUAL | 7 | L2-regularized logistic regression (dual) |
| L2R_L2LOSS_SVR | 11 | L2-loss SVR (primal) |
| L2R_L2LOSS_SVR_DUAL | 12 | L2-loss SVR (dual) |
| L2R_L1LOSS_SVR_DUAL | 13 | L1-loss SVR (dual) |

## Resource management

WASM heap memory is not garbage collected. Call `.dispose()` on every model when done.
A `FinalizationRegistry` safety net warns if you forget, but do not rely on it.

## Build from source

Requires [Emscripten](https://emscripten.org/) (emsdk) activated.

```bash
git clone --recurse-submodules https://github.com/wlearn-org/liblinear-wasm
cd liblinear-wasm
npm run build
npm test
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

## License

BSD-3-Clause (same as upstream LIBLINEAR)
