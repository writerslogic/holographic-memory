# Build the wheel on the manylinux2014 maturin image (glibc 2.17), then install it into the
# harness image.
FROM ghcr.io/pyo3/maturin:v1.15.0@sha256:1c897d72a79c083cdea8829662f7b28e7657cf5b0d7f44fb4231e55998cf52ed AS build

# holographic-memory commit the wrapper's `code_bits` default was measured at (fetched by id, so it
# need not be on a branch).
ARG HMS_REV=d1aa6097370dad10cded90f0cd62dcd5c560961a

RUN rustup toolchain install 1.96.0 --profile minimal \
    && git clone https://github.com/writerslogic/holographic-memory.git /io/hms \
    && cd /io/hms && git fetch -q origin "${HMS_REV}" && git checkout -q FETCH_HEAD \
    && RUSTUP_TOOLCHAIN=1.96.0 maturin build --release --locked --out /io/dist

FROM ann-benchmarks
COPY --from=build /io/dist /tmp/hms-dist
RUN pip install --no-deps /tmp/hms-dist/*.whl && rm -rf /tmp/hms-dist
RUN python3 -c 'from holographic_memory import VectorIndex; print(VectorIndex())'
