# PR 363 merge gate

PR https://github.com/writerslogic/warm/pull/363 tested head
`79ce353407fee9df816bf60b5fda2a16dc3589cc` completed 68 successful and four
skipped checks, including the required aggregate `check`. No pending/failed
checks or unresolved review threads remained. Owner-review bypass was used only
after the checks completed. Merge `68e016dba33eee777e3630ce27ec4b9a953995b3`
has GitHub-verified signature and the same tree as the tested head:
`15ed4efc790d38704f41dee04f0fd589defbd051`.

Run 41223183203: Windows job 124601891118 passed 810 tests with 14 skips,
then 170 service tests. Python job 124601891272 passed 59 base tests with four
skips and 46 subtests, 31 fast tests, 17 LangChain tests at each supported
boundary, 42/48 LlamaIndex tests, and 12 Haystack tests at each boundary.
Saved completed job logs and check/review state support these counts. This
validates the metadata-listing and Haystack changes; it does not demonstrate
full framework conformance or resolve the documented Semantic Kernel blocker.
