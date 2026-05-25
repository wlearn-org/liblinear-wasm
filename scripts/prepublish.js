'use strict'

const fs = require('fs')
const { spawnSync } = require('child_process')
const path = require('path')

const pkgDir = path.resolve(__dirname, '..')

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

function versionFor(exe, args, pattern) {
  const result = spawnSync(exe, args, { encoding: 'utf8' })
  if (result.status !== 0) return null
  const text = (result.stdout || result.stderr || '').trim()
  const match = text.match(pattern)
  if (!match) return null
  return { major: Number(match[1]), minor: Number(match[2]), patch: Number(match[3] || 0), text }
}

function atLeast(version, major, minor) {
  return version && (version.major > major || (version.major === major && version.minor >= minor))
}

function findEmscriptenPython() {
  if (process.env.EMSDK_PYTHON) return process.env.EMSDK_PYTHON

  const candidates = ['/usr/bin/python3', '/opt/homebrew/bin/python3', '/usr/local/bin/python3', 'python3', 'python']
  for (const exe of candidates) {
    if (exe.startsWith('/') && !fs.existsSync(exe)) continue
    const version = versionFor(exe, ['--version'], /Python\s+(\d+)\.(\d+)(?:\.(\d+))?/)
    if (atLeast(version, 3, 10)) return exe
  }

  console.error('prepublish: Emscripten requires Python >= 3.10. Set EMSDK_PYTHON to a compatible Python.')
  process.exit(1)
}

function findCMakeBin() {
  const candidates = ['/usr/bin/cmake', '/opt/homebrew/bin/cmake', '/usr/local/bin/cmake', 'cmake']
  for (const exe of candidates) {
    if (exe.startsWith('/') && !fs.existsSync(exe)) continue
    const version = versionFor(exe, ['--version'], /cmake version\s+(\d+)\.(\d+)(?:\.(\d+))?/)
    if (atLeast(version, 3, 28)) return exe.startsWith('/') ? path.dirname(exe) : null
  }
  return null
}

const cmakeBin = findCMakeBin()
const buildEnv = {
  EMSDK_PYTHON: findEmscriptenPython()
}

if (cmakeBin) {
  buildEnv.PATH = cmakeBin + path.delimiter + process.env.PATH
}

run('npm', ['run', 'build'], buildEnv)
run('npm', ['test'], buildEnv)
