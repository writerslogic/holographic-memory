// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Client-side demo of bind, bundle and recall by resonance. Dense binary vectors of 4,096 bits,
// bind = XOR, bundle = bitwise majority of an odd number of pairs, similarity = matching bits.
(function () {
  "use strict"

  var D = 4096
  var WORDS = D / 32
  var MAX_PAIRS = 63
  var SYMBOLS = ["amber", "basalt", "cobalt", "dune", "ember", "fjord", "garnet", "harbor",
    "indigo", "jasper", "kelp", "lagoon"]

  function rng(seed) {
    var a = seed >>> 0
    return function () {
      a = (a + 0x6d2b79f5) >>> 0
      var t = a
      t = Math.imul(t ^ (t >>> 15), t | 1)
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
      return ((t ^ (t >>> 14)) >>> 0)
    }
  }

  function randomVector(next) {
    var v = new Uint32Array(WORDS)
    for (var i = 0; i < WORDS; i++) v[i] = next()
    return v
  }

  function xor(a, b) {
    var out = new Uint32Array(WORDS)
    for (var i = 0; i < WORDS; i++) out[i] = a[i] ^ b[i]
    return out
  }

  function popcount(x) {
    x -= (x >>> 1) & 0x55555555
    x = (x & 0x33333333) + ((x >>> 2) & 0x33333333)
    x = (x + (x >>> 4)) & 0x0f0f0f0f
    return Math.imul(x, 0x01010101) >>> 24
  }

  function similarity(a, b) {
    var diff = 0
    for (var i = 0; i < WORDS; i++) diff += popcount((a[i] ^ b[i]) >>> 0)
    return 1 - diff / D
  }

  function bit(v, i) {
    return (v[i >>> 5] >>> (i & 31)) & 1
  }

  function World(seed) {
    var next = rng(seed)
    this.symbols = SYMBOLS.map(function () { return randomVector(next) })
    this.keys = []
    this.values = []
    for (var i = 0; i < MAX_PAIRS; i++) {
      this.keys.push(randomVector(next))
      this.values.push(next() % SYMBOLS.length)
    }
    // A fixed order of bit positions: the first n flip, so more damage only adds flips.
    var order = new Uint16Array(D)
    for (var j = 0; j < D; j++) order[j] = j
    for (var k = D - 1; k > 0; k--) {
      var r = next() % (k + 1)
      var tmp = order[k]
      order[k] = order[r]
      order[r] = tmp
    }
    this.flipOrder = order
  }

  World.prototype.memory = function (pairs) {
    var counts = new Uint8Array(D)
    for (var p = 0; p < pairs; p++) {
      var bound = xor(this.keys[p], this.symbols[this.values[p]])
      for (var i = 0; i < D; i++) counts[i] += bit(bound, i)
    }
    var mem = new Uint32Array(WORDS)
    for (var b = 0; b < D; b++) {
      if (counts[b] * 2 > pairs) mem[b >>> 5] |= 1 << (b & 31)
    }
    return mem
  }

  World.prototype.damage = function (mem, flips) {
    var out = new Uint32Array(mem)
    for (var f = 0; f < flips; f++) {
      var pos = this.flipOrder[f]
      out[pos >>> 5] ^= 1 << (pos & 31)
    }
    return out
  }

  function css(name, fallback) {
    var v = getComputedStyle(document.body).getPropertyValue(name).trim()
    return v || fallback
  }

  function init(root) {
    if (root.getAttribute("data-ready")) return
    root.setAttribute("data-ready", "1")
    var $ = function (sel) { return root.querySelector(sel) }
    var inK = $('[data-in="k"]')
    var inNoise = $('[data-in="noise"]')
    var inKey = $('[data-in="key"]')
    var outK = $('[data-out="k"]')
    var outNoise = $('[data-out="noise"]')
    var canvas = $("[data-canvas]")
    var bars = $("[data-bars]")
    var verdict = $("[data-verdict]")
    var seed = 20261020
    var world = new World(seed)

    var rows = SYMBOLS.map(function (name) {
      var li = document.createElement("li")
      li.innerHTML = '<span class="warm-bars__name"></span><span class="warm-bars__track"><span class="warm-bars__fill"></span><span class="warm-bars__chance"></span></span><span class="warm-bars__val"></span>'
      li.querySelector(".warm-bars__name").textContent = name
      bars.appendChild(li)
      return li
    })

    function fillKeys() {
      var pairs = +inK.value
      var keep = inKey.value
      inKey.innerHTML = ""
      for (var i = 0; i < pairs; i++) {
        var o = document.createElement("option")
        o.value = String(i)
        o.textContent = "key " + (i + 1)
        inKey.appendChild(o)
      }
      inKey.value = keep && +keep < pairs ? keep : "0"
    }

    function draw(mem, damaged) {
      var ctx = canvas.getContext("2d")
      var cell = canvas.width / 64
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = css("--warm-sunken", "#eee")
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      var on = css("--warm-demo-on", "#7a8394")
      var flip = css("--warm-accent", "#0a7b8c")
      for (var i = 0; i < D; i++) {
        var x = (i % 64) * cell
        var y = Math.floor(i / 64) * cell
        var changed = bit(mem, i) !== bit(damaged, i)
        if (changed) {
          ctx.fillStyle = flip
        } else if (bit(damaged, i)) {
          ctx.fillStyle = on
        } else {
          continue
        }
        ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1)
      }
    }

    function render() {
      var pairs = +inK.value
      var noise = +inNoise.value
      outK.textContent = String(pairs)
      outNoise.textContent = noise + "%"
      if (inKey.options.length !== pairs) fillKeys()
      var ask = +inKey.value
      var clean = world.memory(pairs)
      var flips = Math.round(D * noise / 100)
      var mem = world.damage(clean, flips)
      draw(clean, mem)
      var probe = xor(mem, world.keys[ask])
      var sims = world.symbols.map(function (s) { return similarity(probe, s) })
      var best = 0
      for (var i = 1; i < sims.length; i++) if (sims[i] > sims[best]) best = i
      var runnerUp = -1
      for (var j = 0; j < sims.length; j++) {
        if (j !== best && (runnerUp < 0 || sims[j] > sims[runnerUp])) runnerUp = j
      }
      var truth = world.values[ask]
      var sigma = 0.5 / Math.sqrt(D)
      rows.forEach(function (li, idx) {
        var frac = Math.max(0, Math.min(1, (sims[idx] - 0.5) / 0.5))
        li.querySelector(".warm-bars__fill").style.width = (frac * 100).toFixed(1) + "%"
        li.querySelector(".warm-bars__val").textContent = sims[idx].toFixed(3)
        li.classList.toggle("is-best", idx === best)
        li.classList.toggle("is-truth", idx === truth)
      })
      var z = (sims[best] - 0.5) / sigma
      var ok = best === truth
      var text = "Recalled “" + SYMBOLS[best] + "” at similarity " + sims[best].toFixed(3) +
        " (" + z.toFixed(1) + " standard deviations above chance, runner-up " + sims[runnerUp].toFixed(3) + "). "
      text += ok ? "Correct." : "Wrong: the stored value was “" + SYMBOLS[truth] + "”. The signal is lost in the noise."
      verdict.textContent = text
      verdict.className = "warm-demo__verdict " + (ok ? "is-ok" : "is-bad")
    }

    inK.addEventListener("input", render)
    inNoise.addEventListener("input", render)
    inKey.addEventListener("change", render)
    root.querySelector('[data-act="reroll"]').addEventListener("click", function () {
      seed = (seed * 1664525 + 1013904223) >>> 0
      world = new World(seed)
      render()
    })
    new MutationObserver(render).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] })
    fillKeys()
    render()
  }

  function boot() {
    document.querySelectorAll("[data-warm-demo]").forEach(init)
  }

  if (typeof document$ !== "undefined") document$.subscribe(boot)
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot)
  else boot()
})()
