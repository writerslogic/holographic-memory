// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Small accessibility fixes for elements the theme renders without an accessible name.
(function () {
  "use strict"

  function label() {
    var search = document.querySelector(".md-search")
    if (search && !search.getAttribute("aria-label")) search.setAttribute("aria-label", "Search the documentation")
    var progress = document.querySelector(".md-progress")
    if (progress) {
      progress.setAttribute("role", "progressbar")
      progress.setAttribute("aria-label", "Page loading")
      progress.setAttribute("aria-valuemin", "0")
      progress.setAttribute("aria-valuemax", "100")
      if (!progress.hasAttribute("aria-valuenow")) progress.setAttribute("aria-valuenow", "0")
    }
  }

  // The version selector is added after versions.json loads. Its aria-label ("Select version")
  // does not contain the visible text (the version), so the visible text names the button.
  function version() {
    var button = document.querySelector(".md-version__current[aria-label]")
    if (button) button.removeAttribute("aria-label")
  }

  new MutationObserver(version).observe(document.documentElement, { childList: true, subtree: true })

  if (typeof document$ !== "undefined") document$.subscribe(label)
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", label)
  else label()
})()
