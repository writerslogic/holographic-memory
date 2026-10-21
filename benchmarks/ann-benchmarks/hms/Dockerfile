FROM ann-benchmarks

# The holographic-memory revision to build; pin it to a release tag or commit.
ARG HMS_REV=main

RUN apt-get update && apt-get install -y curl git
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.96.0
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install maturin==1.15.0

RUN git clone https://github.com/writerslogic/holographic-memory.git /tmp/hms \
    && cd /tmp/hms && git checkout "${HMS_REV}" \
    && maturin build --release --locked --out /tmp/hms-dist \
    && pip install /tmp/hms-dist/*.whl \
    && rm -rf /tmp/hms /tmp/hms-dist /root/.cargo/registry

RUN python3 -c 'from holographic_memory import VectorIndex; print(VectorIndex())'
