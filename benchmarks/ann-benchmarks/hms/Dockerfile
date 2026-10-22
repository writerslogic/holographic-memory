# Build the wheel on the manylinux2014 maturin image (glibc 2.17), then install it into the
# harness image.
FROM ghcr.io/pyo3/maturin:v1.15.0@sha256:1c897d72a79c083cdea8829662f7b28e7657cf5b0d7f44fb4231e55998cf52ed AS build

# holographic-memory commit on main.
ARG HMS_REV=19f4371781aa7e4ed159e1581d405c8ff95d4533

RUN rustup toolchain install 1.96.0 --profile minimal \
    && git clone https://github.com/writerslogic/holographic-memory.git /io/hms \
    && cd /io/hms && git checkout "${HMS_REV}" \
    && RUSTUP_TOOLCHAIN=1.96.0 maturin build --release --locked --out /io/dist

FROM ann-benchmarks
COPY --from=build /io/dist /tmp/hms-dist
RUN pip install --no-deps /tmp/hms-dist/*.whl && rm -rf /tmp/hms-dist
RUN python3 -c 'from holographic_memory import VectorIndex; print(VectorIndex())'
