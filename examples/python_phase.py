from holographic_memory import PhaseHVec, resonator_factorize

left = PhaseHVec.random(1024, 256, 1)
right = PhaseHVec.random(1024, 256, 2)
composite = left.bind(right)

assert composite.unbind(right).similarity(left) == 1.0
print(resonator_factorize(composite, [[left], [right]], max_iter=50))
