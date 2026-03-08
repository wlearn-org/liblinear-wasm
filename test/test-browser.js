#!/usr/bin/env node
// Browser smoke test for IIFE + ESM bundles

const { chromium } = require('playwright')
const path = require('path')
const http = require('http')
const fs = require('fs')

const ROOT = path.resolve(__dirname, '..')

const bundles = [
  { name: 'IIFE', file: 'dist/wlearnLiblinear.iife.js', type: 'iife', global: 'wlearnLiblinear' },
  { name: 'ESM',  file: 'dist/wlearnLiblinear.esm.mjs', type: 'esm' },
]

function makeIifeHtml(jsPath, globalName) {
  return `<!DOCTYPE html><html><body>
<script src="${jsPath}"></script>
<script>
async function runTest() {
  try {
    var lib = ${globalName}
    var model = await lib.LinearModel.create({ solver: 0, C: 1.0 })
    model.fit([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1],[0,2],[1,2]], [0,0,0,1,1,1,0,1])
    var preds = model.predict([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1],[0,2],[1,2]])
    var score = model.score([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1],[0,2],[1,2]], [0,0,0,1,1,1,0,1])
    var bundle = model.save()
    var m2 = await lib.LinearModel.load(bundle)
    var p2 = m2.predict([[1,1]])
    model.dispose(); m2.dispose()
    return { ok: true, n: preds.length, score: score, bundleSize: bundle.length }
  } catch(e) { return { ok: false, error: e.message, stack: e.stack } }
}
window.__testResult = runTest()
</script></body></html>`
}

function makeEsmHtml(jsPath) {
  return `<!DOCTYPE html><html><body>
<script type="module">
import { LinearModel } from '${jsPath}'
async function runTest() {
  try {
    var model = await LinearModel.create({ solver: 0, C: 1.0 })
    model.fit([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1],[0,2],[1,2]], [0,0,0,1,1,1,0,1])
    var preds = model.predict([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1],[0,2],[1,2]])
    var score = model.score([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1],[0,2],[1,2]], [0,0,0,1,1,1,0,1])
    var bundle = model.save()
    var m2 = await LinearModel.load(bundle)
    var p2 = m2.predict([[1,1]])
    model.dispose(); m2.dispose()
    return { ok: true, n: preds.length, score: score, bundleSize: bundle.length }
  } catch(e) { return { ok: false, error: e.message, stack: e.stack } }
}
window.__testResult = runTest()
</script></body></html>`
}

async function main() {
  const server = http.createServer((req, res) => {
    const fp = path.join(ROOT, decodeURIComponent(req.url.slice(1)))
    if (!fs.existsSync(fp)) { res.writeHead(404); res.end('Not found: ' + req.url); return }
    const ext = path.extname(fp)
    const ct = ext === '.html' ? 'text/html' : 'application/javascript'
    res.writeHead(200, { 'Content-Type': ct })
    res.end(fs.readFileSync(fp))
  })
  await new Promise(r => server.listen(0, '127.0.0.1', r))
  const port = server.address().port
  const base = `http://127.0.0.1:${port}`

  const browser = await chromium.launch({ headless: true })
  let passed = 0, failed = 0

  for (const b of bundles) {
    const htmlName = `_test_${b.name.replace(/\s+/g, '_')}.html`
    const htmlPath = path.join(ROOT, 'dist', htmlName)
    const jsUrl = '/' + b.file

    if (b.type === 'iife') {
      fs.writeFileSync(htmlPath, makeIifeHtml(jsUrl, b.global))
    } else {
      fs.writeFileSync(htmlPath, makeEsmHtml(jsUrl))
    }

    const page = await browser.newPage()
    const errors = []
    page.on('pageerror', e => errors.push(e.message))

    try {
      await page.goto(`${base}/dist/${htmlName}`, { timeout: 10000 })
      await page.waitForFunction(() => window.__testResult, { timeout: 10000 })
      const result = await page.evaluate(() => window.__testResult)

      if (result && result.ok) {
        console.log(`  PASS: ${b.name} -- score: ${result.score}, preds: ${result.n}, bundle: ${result.bundleSize}b`)
        passed++
      } else {
        console.log(`  FAIL: ${b.name} -- ${result ? result.error : 'no result'}`)
        if (result && result.stack) console.log(`        ${result.stack.split('\n')[1]}`)
        failed++
      }
    } catch (e) {
      console.log(`  FAIL: ${b.name} -- ${e.message}`)
      if (errors.length) console.log(`        page errors: ${errors.join('; ')}`)
      failed++
    }

    await page.close()
    fs.unlinkSync(htmlPath)
  }

  await browser.close()
  server.close()

  console.log(`\n=== ${passed} passed, ${failed} failed ===`)
  process.exit(failed > 0 ? 1 : 0)
}

main().catch(e => { console.error(e); process.exit(1) })
