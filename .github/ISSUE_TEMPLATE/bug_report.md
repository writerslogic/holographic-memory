name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''

body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us, what did you expect to happen?
      placeholder: Tell us what you see!
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Reproduction Steps
      description: How can we reproduce this bug?
      placeholder: |
        1. Initialize HMS with ...
        2. Call ...
        3. See error
    validations:
      required: true
  - type: textarea
    id: version
    attributes:
      label: Version / Environment
      description: Which version of HMS and what environment (OS, Node version, Rust version)?
      placeholder: HMS v0.2.0, Node v20, macOS
    validations:
      required: true
  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this issue, you agree to follow our Code of Conduct
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
