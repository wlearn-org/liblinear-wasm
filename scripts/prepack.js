'use strict'

const fs = require('fs')
const path = require('path')
const { spawnSync } = require('child_process')

const pkgDir = path.resolve(__dirname, '..')
const pkg = JSON.parse(fs.readFileSync(path.join(pkgDir, 'package.json'), 'utf8'))
const baseName = pkg.name.split('/').pop()
const scripts = pkg.scripts || {}
const files = pkg.files || []

function run(cmd, args, extraEnv = {}) {
  const result = spawnSync(cmd, args, {
    cwd: pkgDir,
    stdio: 'inherit',
    env: { ...process.env, ...extraEnv }
  })
  if (result.status !== 0) {
    process.exit(result.status || 1)
  }
}

const buildEnv = {
  EM_CACHE: process.env.EM_CACHE || path.join(pkgDir, 'build', '.emcache')
}

if (!process.env.EMSDK_PYTHON && fs.existsSync('/usr/bin/python3')) {
  buildEnv.EMSDK_PYTHON = '/usr/bin/python3'
}

const wantsWasm = files.includes('wasm/')
const wantsDist = files.includes('dist/')

if (wantsWasm) {
  const wasmDir = path.join(pkgDir, 'wasm')
  const hasWasmJs = fs.existsSync(wasmDir) && fs.readdirSync(wasmDir).some((name) => name.endsWith('.js'))
  if (!hasWasmJs) {
    if (!scripts.build) {
      console.error('prepack: package declares wasm/ in files but has no build script')
      process.exit(1)
    }
    run('npm', ['run', 'build'], buildEnv)
  }
}

if (wantsDist) {
  const distDir = path.join(pkgDir, 'dist')
  const wantJs = path.join(distDir, baseName + '.js')
  const wantMjs = path.join(distDir, baseName + '.mjs')
  if (!fs.existsSync(wantJs) || !fs.existsSync(wantMjs)) {
    if (!scripts['build:browser']) {
      console.error('prepack: package declares dist/ in files but has no build:browser script')
      process.exit(1)
    }
    run('npm', ['run', 'build:browser'], buildEnv)
  }
}

for (const entry of files) {
  const full = path.join(pkgDir, entry.replace(/\/$/, ''))
  if (!fs.existsSync(full)) {
    console.error('prepack: missing published path ' + entry)
    process.exit(1)
  }
}
