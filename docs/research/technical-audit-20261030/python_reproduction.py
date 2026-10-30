import importlib.util
import importlib.machinery
import sys

loader = importlib.machinery.ExtensionFileLoader("holographic_memory", sys.argv[1])
spec = importlib.util.spec_from_file_location("holographic_memory", sys.argv[1], loader=loader)
hm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hm)
a = hm.PhaseHVec.random(4, 16, 1)
b = hm.PhaseHVec.random(5, 16, 2)
c = hm.PhaseHVec.random(4, 32, 3)
for name, action in (
    ("bind_dimension_control", lambda: a.bind(b)),
    ("bundle_dimension_mismatch", lambda: hm.PhaseHVec.bundle([a, b])),
    ("bundle_resolution_mismatch", lambda: hm.PhaseHVec.bundle([a, c])),
    ("resonator_empty_codebook", lambda: hm.resonator_factorize(a, [[]])),
):
    try:
        action()
        print(name, "accepted")
    except BaseException as error:
        print(name, type(error).__name__, str(error))
