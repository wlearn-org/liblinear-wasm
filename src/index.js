const { loadLinear, getWasm } = require('./wasm.js')
const { LinearModel: LinearModelImpl, Solver } = require('./model.js')
const { createModelClass } = require('@wlearn/core')

const LinearModel = createModelClass(LinearModelImpl, LinearModelImpl, { name: 'LinearModel', load: loadLinear })

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await LinearModel.create(params)
  await model.fit(X, y)
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
