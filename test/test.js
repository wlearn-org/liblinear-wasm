const { join } = require('path')
const { readFileSync, existsSync } = require('fs')

let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol, msg) {
  const diff = Math.abs(a - b)
  if (diff > tol) throw new Error(msg || `expected ${a} ~ ${b} (diff=${diff}, tol=${tol})`)
}

async function main() {

// ============================================================
// WASM loading
// ============================================================
console.log('\n=== WASM Loading ===')

const { loadLinear } = require('../src/wasm.js')
const wasm = await loadLinear()

await test('WASM module loads', async () => {
  assert(wasm, 'wasm module is null')
  assert(typeof wasm.ccall === 'function', 'ccall not available')
})

await test('get_last_error returns string', async () => {
  const err = wasm.ccall('wl_linear_get_last_error', 'string', [], [])
  assert(typeof err === 'string', `expected string, got ${typeof err}`)
})

// ============================================================
// LinearModel basics
// ============================================================
console.log('\n=== LinearModel ===')

const { LinearModel, Solver } = require('../src/model.js')

await test('create() returns model', async () => {
  const model = await LinearModel.create({ solver: 'L2R_LR', C: 1.0 })
  assert(model, 'model is null')
  assert(!model.isFitted, 'should not be fitted yet')
  model.dispose()
})

// ============================================================
// Classification
// ============================================================
console.log('\n=== Classification ===')

await test('Binary classification L2R_LR', async () => {
  const model = await LinearModel.create({
    solver: 'L2R_LR',
    C: 10.0,
    eps: 0.01
  })

  // Deterministic, well-separated data in [-1, 1]
  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    // LCG-style deterministic pseudo-random
    const t = ((i * 7 + 3) % 100) / 100
    const s = ((i * 13 + 7) % 100) / 100
    const x1 = t * 2 - 1  // [-1, 1]
    const x2 = s * 2 - 1  // [-1, 1]
    X.push([x1, x2])
    y.push(x1 + x2 > 0 ? 1 : 0)
  }

  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  assert(model.nrClass === 2, `expected 2 classes, got ${model.nrClass}`)
  assert(model.nrFeature === 2, `expected 2 features, got ${model.nrFeature}`)

  const preds = model.predict(X)
  assert(preds instanceof Float64Array, 'predictions should be Float64Array')
  assert(preds.length === 100, `expected 100 predictions, got ${preds.length}`)

  // Accuracy > 80% on linearly separable data
  let correct = 0
  for (let i = 0; i < preds.length; i++) {
    if (preds[i] === y[i]) correct++
  }
  const accuracy = correct / preds.length
  assert(accuracy > 0.8, `accuracy ${accuracy} too low for linearly separable data`)

  model.dispose()
})

await test('Binary classification L2R_L2LOSS_SVC', async () => {
  const model = await LinearModel.create({
    solver: 'L2R_L2LOSS_SVC_DUAL',
    C: 10.0
  })

  // Deterministic, well-separated data in [-1, 1]
  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const t = ((i * 7 + 3) % 100) / 100
    const s = ((i * 13 + 7) % 100) / 100
    const x1 = t * 2 - 1
    const x2 = s * 2 - 1
    X.push([x1, x2])
    y.push(x1 + x2 > 0 ? 1 : -1)  // SVC convention: +1/-1
  }

  model.fit(X, y)
  const preds = model.predict(X)

  let correct = 0
  for (let i = 0; i < preds.length; i++) {
    if (preds[i] === y[i]) correct++
  }
  assert(correct / preds.length > 0.8, 'SVC accuracy too low')

  model.dispose()
})

await test('Multi-class classification', async () => {
  const model = await LinearModel.create({
    solver: 'L2R_LR',
    C: 10.0
  })

  // Deterministic 3-class data
  const X = []
  const y = []
  for (let i = 0; i < 150; i++) {
    const t = ((i * 7 + 3) % 150) / 150
    const s = ((i * 13 + 7) % 150) / 150
    const x1 = t * 2 - 1
    const x2 = s * 2 - 1
    X.push([x1, x2])
    const sum = x1 + x2
    y.push(sum < -0.3 ? 0 : sum < 0.3 ? 1 : 2)
  }

  model.fit(X, y)
  assert(model.nrClass === 3, `expected 3 classes, got ${model.nrClass}`)

  const preds = model.predict(X)
  assert(preds.length === 150, `expected 150 predictions, got ${preds.length}`)

  // All predictions should be valid class labels
  for (let i = 0; i < preds.length; i++) {
    assert(preds[i] === 0 || preds[i] === 1 || preds[i] === 2,
      `invalid prediction: ${preds[i]}`)
  }

  model.dispose()
})

// ============================================================
// Probability
// ============================================================
console.log('\n=== Probability ===')

await test('predictProba returns probabilities', async () => {
  const model = await LinearModel.create({
    solver: 'L2R_LR',
    C: 10.0
  })

  // Deterministic data
  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const t = ((i * 7 + 3) % 100) / 100
    const s = ((i * 13 + 7) % 100) / 100
    X.push([t * 2 - 1, s * 2 - 1])
    y.push(t + s > 1 ? 1 : 0)
  }

  model.fit(X, y)
  const probs = model.predictProba(X)
  const nrClass = model.nrClass

  assert(probs.length === 100 * nrClass,
    `expected ${100 * nrClass} probabilities, got ${probs.length}`)

  // Each row should sum to ~1
  for (let r = 0; r < 100; r++) {
    let sum = 0
    for (let c = 0; c < nrClass; c++) {
      const p = probs[r * nrClass + c]
      assert(p >= 0 && p <= 1, `probability out of [0,1]: ${p}`)
      sum += p
    }
    assertClose(sum, 1.0, 1e-6, `row ${r} probabilities sum to ${sum}`)
  }

  model.dispose()
})

// ============================================================
// Decision function
// ============================================================
console.log('\n=== Decision Function ===')

await test('decisionFunction returns values', async () => {
  const model = await LinearModel.create({
    solver: 'L2R_L2LOSS_SVC_DUAL',
    C: 1.0
  })

  const X = [[1, 2], [3, 4], [5, 6], [7, 8]]
  const y = [-1, -1, 1, 1]

  model.fit(X, y)
  const vals = model.decisionFunction(X)
  assert(vals instanceof Float64Array, 'should be Float64Array')
  // Binary: 1 decision value per row
  assert(vals.length === 4 * model.decisionDim,
    `expected ${4 * model.decisionDim} values, got ${vals.length}`)

  model.dispose()
})

// ============================================================
// Score
// ============================================================
console.log('\n=== Score ===')

await test('score returns accuracy for classification', async () => {
  const model = await LinearModel.create({ solver: 'L2R_LR', C: 10.0 })

  // Deterministic, well-separated data in [-1, 1]
  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const t = ((i * 11 + 5) % 100) / 100
    const s = ((i * 17 + 3) % 100) / 100
    const x1 = t * 2 - 1
    const x2 = s * 2 - 1
    X.push([x1, x2])
    y.push(x1 + x2 > 0 ? 1 : 0)
  }

  model.fit(X, y)
  const acc = model.score(X, y)
  assert(typeof acc === 'number', 'score should be a number')
  assert(acc > 0.8, `accuracy ${acc} too low`)
  assert(acc <= 1.0, `accuracy ${acc} > 1`)

  model.dispose()
})

// ============================================================
// Regression
// ============================================================
console.log('\n=== Regression ===')

await test('SVR regression', async () => {
  const model = await LinearModel.create({
    solver: 'L2R_L2LOSS_SVR_DUAL',
    C: 10.0,
    p: 0.1
  })

  // Deterministic: y = 2*x1 + 3*x2 + small noise
  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const x1 = ((i * 7 + 3) % 100) / 50 - 1  // [-1, 1]
    const x2 = ((i * 13 + 7) % 100) / 50 - 1  // [-1, 1]
    const noise = ((i * 31 + 11) % 100) / 500 - 0.1  // small noise
    X.push([x1, x2])
    y.push(2 * x1 + 3 * x2 + noise)
  }

  model.fit(X, y)
  assert(model.capabilities.regressor, 'should be regressor')

  const preds = model.predict(X)
  assert(preds.length === 100, `expected 100 predictions, got ${preds.length}`)

  // R-squared should be reasonable
  const r2 = model.score(X, y)
  assert(r2 > 0.5, `R-squared ${r2} too low`)

  model.dispose()
})

// ============================================================
// Save / Load (WLRN bundle format)
// ============================================================
console.log('\n=== Save / Load ===')

const { decodeBundle, load: coreLoad } = require('@wlearn/core')

await test('save produces WLRN bundle', async () => {
  const model = await LinearModel.create({ solver: 'L2R_LR', C: 1.0 })
  model.fit([[1, 2], [3, 4], [5, 6], [7, 8]], [0, 0, 1, 1])

  const buf = model.save()
  assert(buf instanceof Uint8Array, 'save should return Uint8Array')
  assert(buf.length > 0, 'saved model should not be empty')

  // Verify WLRN magic
  assert(buf[0] === 0x57, 'bad magic[0]')
  assert(buf[1] === 0x4c, 'bad magic[1]')
  assert(buf[2] === 0x52, 'bad magic[2]')
  assert(buf[3] === 0x4e, 'bad magic[3]')

  // Verify manifest
  const { manifest, toc } = decodeBundle(buf)
  assert(manifest.typeId === 'wlearn.liblinear.classifier@1',
    `expected classifier typeId, got ${manifest.typeId}`)
  assert(manifest.bundleVersion === 1, `expected bundleVersion 1, got ${manifest.bundleVersion}`)
  assert(manifest.params.solver === 'L2R_LR', `expected solver L2R_LR, got ${manifest.params.solver}`)
  assert(manifest.params.C === 1.0, `expected C=1.0, got ${manifest.params.C}`)
  assert(toc.length === 1, `expected 1 TOC entry, got ${toc.length}`)
  assert(toc[0].id === 'model', `expected TOC entry "model", got ${toc[0].id}`)

  model.dispose()
})

await test('save regressor uses regressor typeId', async () => {
  const model = await LinearModel.create({ solver: 'L2R_L2LOSS_SVR_DUAL', C: 1.0 })
  model.fit([[1, 2], [3, 4]], [1.5, 3.5])

  const buf = model.save()
  const { manifest } = decodeBundle(buf)
  assert(manifest.typeId === 'wlearn.liblinear.regressor@1',
    `expected regressor typeId, got ${manifest.typeId}`)

  model.dispose()
})

await test('save and load model round-trip', async () => {
  const model = await LinearModel.create({
    solver: 'L2R_LR',
    C: 1.0
  })

  const X = [[1, 2], [3, 4], [5, 6], [7, 8]]
  const y = [0, 0, 1, 1]
  model.fit(X, y)

  const preds1 = model.predict(X)
  const buf = model.save()

  const model2 = await LinearModel.load(buf)
  assert(model2.isFitted, 'loaded model should be fitted')

  const preds2 = model2.predict(X)

  // Same-runtime round-trip: exact match
  assert(preds1.length === preds2.length, 'prediction length mismatch')
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i],
      `prediction ${i}: ${preds1[i]} !== ${preds2[i]}`)
  }

  // Loaded model preserves params
  const params = model2.getParams()
  assert(params.solver === 'L2R_LR', `loaded params.solver = ${params.solver}`)
  assert(params.C === 1.0, `loaded params.C = ${params.C}`)

  model.dispose()
  model2.dispose()
})

// ============================================================
// core.load() registry dispatch
// ============================================================
console.log('\n=== Registry Dispatch ===')

await test('core.load() dispatches to liblinear loader', async () => {
  const model = await LinearModel.create({ solver: 'L2R_LR', C: 1.0 })
  model.fit([[1, 2], [3, 4], [5, 6], [7, 8]], [0, 0, 1, 1])

  const preds1 = model.predict([[1, 2], [7, 8]])
  const buf = model.save()

  // Load via core registry dispatcher (not LinearModel.load directly)
  const model2 = await coreLoad(buf)
  assert(model2.isFitted, 'registry-loaded model should be fitted')

  const preds2 = model2.predict([[1, 2], [7, 8]])
  assert(preds1.length === preds2.length, 'prediction length mismatch')
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i],
      `core.load prediction ${i}: ${preds1[i]} !== ${preds2[i]}`)
  }

  model.dispose()
  model2.dispose()
})

await test('core.load() works for regressor bundles', async () => {
  const model = await LinearModel.create({ solver: 'L2R_L2LOSS_SVR_DUAL', C: 1.0 })
  model.fit([[1, 2], [3, 4]], [1.5, 3.5])

  const buf = model.save()
  const model2 = await coreLoad(buf)
  assert(model2.isFitted, 'registry-loaded regressor should be fitted')

  const preds = model2.predict([[1, 2]])
  assert(preds.length === 1, `expected 1 prediction, got ${preds.length}`)

  model.dispose()
  model2.dispose()
})

// ============================================================
// Params
// ============================================================
console.log('\n=== Params ===')

await test('getParams / setParams', async () => {
  const model = await LinearModel.create({ solver: 'L2R_LR', C: 2.0 })

  const params = model.getParams()
  assert(params.solver === 'L2R_LR', `expected L2R_LR, got ${params.solver}`)
  assert(params.C === 2.0, `expected C=2.0, got ${params.C}`)

  model.setParams({ C: 5.0 })
  const params2 = model.getParams()
  assert(params2.C === 5.0, `expected C=5.0 after setParams, got ${params2.C}`)

  model.dispose()
})

await test('defaultSearchSpace returns object', async () => {
  const space = LinearModel.defaultSearchSpace()
  assert(space, 'search space is null')
  assert(space.solver, 'missing solver in search space')
  assert(space.C, 'missing C in search space')
})

// ============================================================
// Resource management
// ============================================================
console.log('\n=== Resource Management ===')

await test('dispose is idempotent', async () => {
  const model = await LinearModel.create({ solver: 'L2R_LR' })
  model.fit([[1, 2], [3, 4]], [0, 1])
  model.dispose()
  model.dispose()  // should not throw
})

await test('throws after dispose', async () => {
  const model = await LinearModel.create({ solver: 'L2R_LR' })
  model.fit([[1, 2], [3, 4]], [0, 1])
  model.dispose()

  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict after dispose should throw')
})

await test('throws before fit', async () => {
  const model = await LinearModel.create({ solver: 'L2R_LR' })

  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict before fit should throw')

  model.dispose()
})

// ============================================================
// Input coercion
// ============================================================
console.log('\n=== Input Coercion ===')

await test('typed matrix fast path', async () => {
  const model = await LinearModel.create({ solver: 'L2R_LR', C: 1.0 })

  const X = {
    data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]),
    rows: 4,
    cols: 2
  }
  const y = new Float64Array([0, 0, 1, 1])

  model.fit(X, y)
  const preds = model.predict(X)
  assert(preds.length === 4, `expected 4 predictions, got ${preds.length}`)

  model.dispose()
})

await test('coerce error mode rejects arrays', async () => {
  const model = await LinearModel.create({
    solver: 'L2R_LR',
    coerce: 'error'
  })

  let threw = false
  try {
    model.fit([[1, 2], [3, 4]], [0, 1])
  } catch {
    threw = true
  }
  assert(threw, 'error mode should reject number[][]')

  model.dispose()
})

// ============================================================
// Capabilities
// ============================================================
console.log('\n=== Capabilities ===')

await test('capabilities reflect solver type', async () => {
  const lr = await LinearModel.create({ solver: 'L2R_LR' })
  assert(lr.capabilities.classifier === true, 'LR should be classifier')
  assert(lr.capabilities.predictProba === true, 'LR should support predictProba')
  lr.dispose()

  const svc = await LinearModel.create({ solver: 'L2R_L2LOSS_SVC_DUAL' })
  assert(svc.capabilities.classifier === true, 'SVC should be classifier')
  assert(svc.capabilities.predictProba === false, 'SVC should not support predictProba')
  svc.dispose()

  const svr = await LinearModel.create({ solver: 'L2R_L2LOSS_SVR_DUAL' })
  assert(svr.capabilities.regressor === true, 'SVR should be regressor')
  assert(svr.capabilities.classifier === false, 'SVR should not be classifier')
  svr.dispose()
})

// ============================================================
// Cross-runtime parity (when fixtures exist)
// ============================================================
console.log('\n=== Cross-Runtime Parity ===')

const fixturesDir = join(__dirname, 'fixtures')
const hasFixtures = existsSync(join(fixturesDir, 'classification.data.json'))

if (!hasFixtures) {
  console.log('  SKIP: no fixtures (run: python test/fixtures/generate.py)')
} else {
  function loadFixture(name) {
    return JSON.parse(readFileSync(join(fixturesDir, `${name}.data.json`), 'utf-8'))
  }

  function loadModelFile(name) {
    return readFileSync(join(fixturesDir, name))
  }

  await test('Cross-runtime: classification parity', async () => {
    const fix = loadFixture('classification')
    const modelBuf = loadModelFile('classification.model')

    const model = await LinearModel.load(modelBuf)
    const preds = model.predict(fix.X)

    assert(preds.length === fix.predictions.length,
      `length mismatch: ${preds.length} vs ${fix.predictions.length}`)

    for (let i = 0; i < preds.length; i++) {
      assert(preds[i] === fix.predictions[i],
        `pred[${i}]: JS=${preds[i]} Python=${fix.predictions[i]}`)
    }

    model.dispose()
  })

  await test('Cross-runtime: regression parity', async () => {
    const fix = loadFixture('regression')
    const modelBuf = loadModelFile('regression.model')

    const model = await LinearModel.load(modelBuf)
    const preds = model.predict(fix.X)

    assert(preds.length === fix.predictions.length,
      `length mismatch: ${preds.length} vs ${fix.predictions.length}`)

    for (let i = 0; i < preds.length; i++) {
      const rel = Math.abs(preds[i] - fix.predictions[i]) / (Math.abs(fix.predictions[i]) + 1e-8)
      assert(rel < 1e-4,
        `pred[${i}]: JS=${preds[i]} Python=${fix.predictions[i]} relDiff=${rel}`)
    }

    model.dispose()
  })
}

// ============================================================
// Summary
// ============================================================
console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`)
process.exit(failed > 0 ? 1 : 0)

}

main()
