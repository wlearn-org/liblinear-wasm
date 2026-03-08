const { loadLinear, getWasm } = require('./wasm.js')
const { LinearModel, Solver } = require('./model.js')

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await LinearModel.create(params)
  model.fit(X, y)
  return model
}

// Convenience: load WLRN bundle and predict, auto-disposes model
async function predict(bundleBytes, X) {
  const model = await LinearModel.load(bundleBytes)
  const result = model.predict(X)
  model.dispose()
  return result
}

module.exports = { loadLinear, getWasm, LinearModel, Solver, train, predict }
