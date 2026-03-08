const { getWasm, loadLinear } = require('./wasm.js')
const {
  normalizeX, normalizeY,
  encodeBundle, decodeBundle,
  register,
  DisposedError, NotFittedError
} = require('@wlearn/core')

// FinalizationRegistry safety net -- warns if dispose() was never called
const leakRegistry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ptr, freeFn }) => {
    if (ptr[0]) {
      console.warn('@wlearn/liblinear: Model was not disposed -- calling free() automatically. This is a bug in your code.')
      freeFn(ptr[0])
    }
  })
  : null

// --- Solver constants ---

const Solver = {
  L2R_LR: 0,
  L2R_L2LOSS_SVC_DUAL: 1,
  L2R_L2LOSS_SVC: 2,
  L2R_L1LOSS_SVC_DUAL: 3,
  MCSVM_CS: 4,
  L1R_L2LOSS_SVC: 5,
  L1R_LR: 6,
  L2R_LR_DUAL: 7,
  L2R_L2LOSS_SVR: 11,
  L2R_L2LOSS_SVR_DUAL: 12,
  L2R_L1LOSS_SVR_DUAL: 13
}

const SOLVER_NAMES = Object.fromEntries(
  Object.entries(Solver).map(([k, v]) => [v, k])
)

const LR_SOLVERS = new Set([Solver.L2R_LR, Solver.L1R_LR, Solver.L2R_LR_DUAL])
const SVR_SOLVERS = new Set([Solver.L2R_L2LOSS_SVR, Solver.L2R_L2LOSS_SVR_DUAL, Solver.L2R_L1LOSS_SVR_DUAL])

function resolveSolver(s) {
  if (typeof s === 'number') return s
  if (typeof s === 'string' && s in Solver) return Solver[s]
  return Solver.L2R_LR
}

// --- Helper: write C string to WASM heap ---

function withCString(wasm, str, fn) {
  const bytes = new TextEncoder().encode(str + '\0')
  const ptr = wasm._malloc(bytes.length)
  wasm.HEAPU8.set(bytes, ptr)
  try {
    return fn(ptr)
  } finally {
    wasm._free(ptr)
  }
}

function getLastError() {
  const wasm = getWasm()
  return wasm.ccall('wl_linear_get_last_error', 'string', [], [])
}

// --- Internal sentinel for load path ---
const LOAD_SENTINEL = Symbol('load')

// --- LinearModel ---

class LinearModel {
  #handle = null
  #freed = false
  #ptrRef = null
  #params = {}
  #coerce = 'auto'
  #warned = false
  #fitted = false

  constructor(handle, params, coerce) {
    if (handle === LOAD_SENTINEL) {
      // Internal: created by load()
      this.#handle = params // params holds the handle in this path
      this.#params = coerce || {} // coerce holds params in this path
      this.#coerce = this.#params.coerce || 'auto'
      this.#fitted = true
    } else {
      // Normal construction (from create())
      this.#handle = null
      this.#params = handle || {}
      this.#coerce = this.#params.coerce || 'auto'
    }

    this.#freed = false
    if (this.#handle) {
      this.#ptrRef = [this.#handle]
      if (leakRegistry) {
        leakRegistry.register(this, {
          ptr: this.#ptrRef,
          freeFn: (h) => getWasm()._wl_linear_free_model(h)
        }, this)
      }
    }
  }

  static async create(params = {}) {
    await loadLinear()
    return new LinearModel(params)
  }

  // --- Estimator interface ---

  fit(X, y) {
    this.#ensureFitted(false)
    const wasm = getWasm()

    // Dispose previous model if refitting
    if (this.#handle) {
      wasm._wl_linear_free_model(this.#handle)
      this.#handle = null
      if (this.#ptrRef) this.#ptrRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yNorm = normalizeY(y)
    // WASM boundary requires Float64Array
    const yData = yNorm instanceof Float64Array ? yNorm : new Float64Array(yNorm)

    if (yData.length !== rows) {
      throw new Error(`y length (${yData.length}) does not match X rows (${rows})`)
    }

    // Allocate X on WASM heap (float64)
    const xBytes = xData.length * 8
    const xPtr = wasm._malloc(xBytes)
    wasm.HEAPF64.set(xData, xPtr / 8)

    // Allocate y on WASM heap
    const yBytes = yData.length * 8
    const yPtr = wasm._malloc(yBytes)
    wasm.HEAPF64.set(yData, yPtr / 8)

    const solver = resolveSolver(this.#params.solver)
    const C = this.#params.C ?? 1.0
    const eps = this.#params.eps ?? 0.01
    const bias = this.#params.bias ?? -1
    const p = this.#params.p ?? 0.1

    const modelPtr = wasm._wl_linear_train(
      xPtr, rows, cols,
      yPtr,
      solver, C, eps, bias, p,
      0, 0, 0  // nr_weight, weight_label, weight (none)
    )

    wasm._free(xPtr)
    wasm._free(yPtr)

    if (!modelPtr) {
      throw new Error(`Training failed: ${getLastError()}`)
    }

    this.#handle = modelPtr
    this.#fitted = true

    // Register for leak detection
    this.#ptrRef = [this.#handle]
    if (leakRegistry) {
      leakRegistry.register(this, {
        ptr: this.#ptrRef,
        freeFn: (h) => getWasm()._wl_linear_free_model(h)
      }, this)
    }

    return this
  }

  predict(X) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    const outPtr = wasm._malloc(rows * 8)

    const ret = wasm._wl_linear_predict(this.#handle, xPtr, rows, cols, outPtr)

    if (ret !== 0) {
      wasm._free(xPtr)
      wasm._free(outPtr)
      throw new Error(`Predict failed: ${getLastError()}`)
    }

    const result = new Float64Array(rows)
    for (let i = 0; i < rows; i++) {
      result[i] = wasm.HEAPF64[outPtr / 8 + i]
    }

    wasm._free(xPtr)
    wasm._free(outPtr)
    return result
  }

  predictProba(X) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)
    const nrClass = this.nrClass

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    const outPtr = wasm._malloc(rows * nrClass * 8)

    const ret = wasm._wl_linear_predict_probability(this.#handle, xPtr, rows, cols, outPtr)

    if (ret !== 0) {
      wasm._free(xPtr)
      wasm._free(outPtr)
      throw new Error(`predictProba failed: ${getLastError()}`)
    }

    const total = rows * nrClass
    const result = new Float64Array(total)
    for (let i = 0; i < total; i++) {
      result[i] = wasm.HEAPF64[outPtr / 8 + i]
    }

    wasm._free(xPtr)
    wasm._free(outPtr)
    return result
  }

  decisionFunction(X) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)
    const dim = this.decisionDim

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    const outPtr = wasm._malloc(rows * dim * 8)

    const ret = wasm._wl_linear_predict_values(this.#handle, xPtr, rows, cols, outPtr)

    if (ret !== 0) {
      wasm._free(xPtr)
      wasm._free(outPtr)
      throw new Error(`decisionFunction failed: ${getLastError()}`)
    }

    const total = rows * dim
    const result = new Float64Array(total)
    for (let i = 0; i < total; i++) {
      result[i] = wasm.HEAPF64[outPtr / 8 + i]
    }

    wasm._free(xPtr)
    wasm._free(outPtr)
    return result
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = normalizeY(y)

    if (this.#isRegressor()) {
      // R-squared
      let ssRes = 0, ssTot = 0
      let yMean = 0
      for (let i = 0; i < yArr.length; i++) yMean += yArr[i]
      yMean /= yArr.length
      for (let i = 0; i < yArr.length; i++) {
        ssRes += (yArr[i] - preds[i]) ** 2
        ssTot += (yArr[i] - yMean) ** 2
      }
      return ssTot === 0 ? 0 : 1 - ssRes / ssTot
    } else {
      // Accuracy
      let correct = 0
      for (let i = 0; i < preds.length; i++) {
        if (preds[i] === yArr[i]) correct++
      }
      return correct / preds.length
    }
  }

  // --- Model I/O ---

  save() {
    this.#ensureFitted()
    const rawBytes = this.#saveRaw()
    const solver = resolveSolver(this.#params.solver)
    const typeId = SVR_SOLVERS.has(solver)
      ? 'wlearn.liblinear.regressor@1'
      : 'wlearn.liblinear.classifier@1'
    return encodeBundle(
      { typeId, params: this.getParams() },
      [{ id: 'model', data: rawBytes }]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return LinearModel._fromBundle(manifest, toc, blobs)
  }

  static async _fromBundle(manifest, toc, blobs) {
    await loadLinear()
    const wasm = getWasm()

    const entry = toc.find(e => e.id === 'model')
    if (!entry) throw new Error('Bundle missing "model" artifact')
    const raw = blobs.subarray(entry.offset, entry.offset + entry.length)

    const bufPtr = wasm._malloc(raw.length)
    wasm.HEAPU8.set(raw, bufPtr)
    const modelPtr = wasm._wl_linear_load_model(bufPtr, raw.length)
    wasm._free(bufPtr)

    if (!modelPtr) {
      throw new Error(`load failed: ${getLastError()}`)
    }

    return new LinearModel(LOAD_SENTINEL, modelPtr, manifest.params || {})
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    if (this.#handle) {
      const wasm = getWasm()
      wasm._wl_linear_free_model(this.#handle)
    }

    if (this.#ptrRef) this.#ptrRef[0] = null
    if (leakRegistry) leakRegistry.unregister(this)

    this.#handle = null
    this.#fitted = false
  }

  // --- Params ---

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    if ('coerce' in p) this.#coerce = p.coerce
    return this
  }

  static defaultSearchSpace() {
    return {
      solver: { type: 'categorical', values: ['L2R_LR', 'L2R_L2LOSS_SVC_DUAL', 'L1R_LR'] },
      C: { type: 'log_uniform', low: 1e-4, high: 1e4 },
      eps: { type: 'log_uniform', low: 1e-5, high: 1e-1 }
    }
  }

  // --- Inspection ---

  get nrClass() {
    this.#ensureFitted()
    return getWasm()._wl_linear_get_nr_class(this.#handle)
  }

  get nrFeature() {
    this.#ensureFitted()
    return getWasm()._wl_linear_get_nr_feature(this.#handle)
  }

  get classes() {
    this.#ensureFitted()
    const wasm = getWasm()
    const n = this.nrClass
    const outPtr = wasm._malloc(n * 4)
    wasm._wl_linear_get_labels(this.#handle, outPtr)
    const result = new Int32Array(n)
    for (let i = 0; i < n; i++) {
      result[i] = wasm.getValue(outPtr + i * 4, 'i32')
    }
    wasm._free(outPtr)
    return result
  }

  get isFitted() {
    return this.#fitted && !this.#freed
  }

  get capabilities() {
    const solver = resolveSolver(this.#params.solver)
    const isRegressor = SVR_SOLVERS.has(solver)
    const hasProba = LR_SOLVERS.has(solver)
    return {
      classifier: !isRegressor,
      regressor: isRegressor,
      predictProba: hasProba,
      decisionFunction: true,
      sampleWeight: false,
      csr: false,
      earlyStopping: false
    }
  }

  get probaDim() {
    return this.isFitted ? this.nrClass : 0
  }

  get decisionDim() {
    if (!this.isFitted) return 0
    const n = this.nrClass
    return n === 2 ? 1 : n
  }

  // --- Private helpers ---

  #saveRaw() {
    const wasm = getWasm()

    const outBufPtr = wasm._malloc(4)  // char** (pointer to buffer)
    const outLenPtr = wasm._malloc(4)  // int*

    const ret = wasm._wl_linear_save_model(this.#handle, outBufPtr, outLenPtr)

    if (ret !== 0) {
      wasm._free(outBufPtr)
      wasm._free(outLenPtr)
      throw new Error(`save failed: ${getLastError()}`)
    }

    const bufPtr = wasm.getValue(outBufPtr, 'i32')
    const bufLen = wasm.getValue(outLenPtr, 'i32')

    const result = new Uint8Array(bufLen)
    result.set(wasm.HEAPU8.subarray(bufPtr, bufPtr + bufLen))

    wasm._wl_linear_free_buffer(bufPtr)
    wasm._free(outBufPtr)
    wasm._free(outLenPtr)

    return result
  }

  #normalizeX(X) {
    const coerce = this.#warned ? 'auto' : this.#coerce
    const result = normalizeX(X, coerce)
    if (this.#coerce === 'warn' && !this.#warned && Array.isArray(X)) {
      this.#warned = true
    }
    return result
  }

  #ensureFitted(requireFit = true) {
    if (this.#freed) throw new DisposedError('LinearModel has been disposed.')
    if (requireFit && !this.#fitted) throw new NotFittedError('LinearModel is not fitted. Call fit() first.')
  }

  #isRegressor() {
    const solver = resolveSolver(this.#params.solver)
    return SVR_SOLVERS.has(solver)
  }
}

// --- Register loaders with @wlearn/core ---

register('wlearn.liblinear.classifier@1', (m, t, b) => LinearModel._fromBundle(m, t, b))
register('wlearn.liblinear.regressor@1', (m, t, b) => LinearModel._fromBundle(m, t, b))

module.exports = { LinearModel, Solver }
