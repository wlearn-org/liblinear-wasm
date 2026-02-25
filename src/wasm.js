// WASM loader -- loads the LIBLINEAR WASM module (singleton, lazy init)

import { createRequire } from 'module'

let wasmModule = null
let loading = null

export async function loadLinear(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    // SINGLE_FILE=1: .wasm is embedded in the .js file, no locateFile needed
    // Emscripten output is CJS, use createRequire for ESM compatibility
    const require = createRequire(import.meta.url)
    const createLinear = require('../wasm/linear.cjs')
    wasmModule = await createLinear(options)
    return wasmModule
  })()

  return loading
}

export function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadLinear() first')
  return wasmModule
}
