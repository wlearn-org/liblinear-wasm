#!/usr/bin/env node
// Browser smoke test for IIFE + ESM bundles
// Generic: auto-discovers package name and exports from package.json + src/index.js

const { chromium } = require('playwright')
const path = require('path')
const http = require('http')
const fs = require('fs')

const ROOT = path.resolve(__dirname, '..')
const pkg = require(path.join(ROOT, 'package.json'))
const NAME = pkg.name.split('/').pop()
const EXPORTS = Object.keys(require(path.join(ROOT, 'src', 'index.js')))

const bundles = [
  { name: 'IIFE', file: `dist/${NAME}.js`,  type: 'iife', global: NAME },
  { name: 'ESM',  file: `dist/${NAME}.mjs`, type: 'esm' },
]

function makeIifeHtml(jsPath, globalName, exportKeys) {
  return `<!DOCTYPE html><html><body>
<script src="${jsPath}"></script>
<script>
async function runTest() {
  try {
    var lib = ${globalName}
    var expected = ${JSON.stringify(exportKeys)}
    var missing = expected.filter(function(k) { return !(k in lib) })
    if (missing.length) return { ok: false, error: 'missing exports: ' + missing.join(', ') }
    var types = {}
    expected.forEach(function(k) { types[k] = typeof lib[k] })
    return { ok: true, exports: expected.length, types: types }
  } catch(e) { return { ok: false, error: e.message, stack: e.stack } }
}
window.__testResult = runTest()
</script></body></html>`
}

function makeEsmHtml(jsPath, exportKeys) {
  const imports = exportKeys.join(', ')
  return `<!DOCTYPE html><html><body>
<script type="module">
import { ${imports} } from '${jsPath}'
async function runTest() {
  try {
    var types = {}
    var exports = [${exportKeys.map(k => `['${k}', ${k}]`).join(', ')}]
    exports.forEach(function(e) { types[e[0]] = typeof e[1] })
    return { ok: true, exports: ${exportKeys.length}, types: types }
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
    const htmlName = `_test_${b.name}.html`
    const htmlPath = path.join(ROOT, 'dist', htmlName)
    const jsUrl = '/' + b.file

    if (b.type === 'iife') {
      fs.writeFileSync(htmlPath, makeIifeHtml(jsUrl, b.global, EXPORTS))
    } else {
      fs.writeFileSync(htmlPath, makeEsmHtml(jsUrl, EXPORTS))
    }

    const page = await browser.newPage()
    const errors = []
    page.on('pageerror', e => errors.push(e.message))

    try {
      await page.goto(`${base}/dist/${htmlName}`, { timeout: 30000 })
      await page.waitForFunction(() => window.__testResult, { timeout: 30000 })
      const result = await page.evaluate(() => window.__testResult)

      if (result && result.ok) {
        console.log(`  PASS: ${b.name} -- ${result.exports} exports`)
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
