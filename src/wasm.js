// WASM loader -- loads the LIBLINEAR WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

async function loadLinear(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    // SINGLE_FILE=1: .wasm is embedded in the .js file, no locateFile needed
    const createLinear = require('../wasm/linear.js')
    wasmModule = await createLinear(options)
    return wasmModule
  })()

  return loading
}

function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadLinear() first')
  return wasmModule
}

module.exports = { loadLinear, getWasm }
