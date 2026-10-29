#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 || $(uname -m) != x86_64 ]]; then
  echo 'Run as root on the dedicated x86_64 AML VM.' >&2
  exit 1
fi
if [[ -e /opt/hms-aml/src || -e /etc/systemd/system/hms-aml.service ]]; then
  echo 'AML installation already exists; inspect it before retrying.' >&2
  exit 1
fi

native_commit=12c80decb094d57ee13ba187a01bf1de8788ca2f
adapter_commit=a25cb4c53731eaa950155dfc1c9f1a8e9184b15c
adapter_sha256=1d5875e28b2b273e5bbe26c00bf60ca0be2b5621313313e82636a2230026f452
root=/opt/hms-aml
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
test -f "$script_dir/runtime-requirements.txt" || {
  echo 'Keep runtime-requirements.txt next to gce_install.sh.' >&2
  exit 1
}

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl git gnupg debian-keyring debian-archive-keyring \
  apt-transport-https build-essential pkg-config libssl-dev python3 python3-venv python3-pip

curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
  | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
  -o /etc/apt/sources.list.d/caddy-stable.list
chmod 0644 /usr/share/keyrings/caddy-stable-archive-keyring.gpg \
  /etc/apt/sources.list.d/caddy-stable.list
apt-get update
apt-get install -y caddy

id -u hms-aml >/dev/null 2>&1 || useradd --system --home-dir /var/lib/hms-aml \
  --create-home --shell /usr/sbin/nologin hms-aml
install -d -m 0755 "$root"
install -d -o hms-aml -g hms-aml -m 0700 /var/lib/hms-aml/stores
install -d -o hms-aml -g hms-aml -m 0700 /etc/hms-aml

git clone --filter=blob:none https://github.com/writerslogic/holographic-memory.git "$root/src"
git -C "$root/src" checkout --detach "$native_commit"
git -C "$root/src" cat-file -e "${adapter_commit}^{commit}"
install -d -m 0755 "$root/src/benchmarks/public/aml_cycle2"
git -C "$root/src" show "${adapter_commit}:benchmarks/public/aml_cycle2/adapter.py" \
  > "$root/src/benchmarks/public/aml_cycle2/adapter.py"
git -C "$root/src" show "${adapter_commit}:benchmarks/public/aml_cycle2/local_host.py" \
  > "$root/src/benchmarks/public/aml_cycle2/local_host.py"
actual_adapter_sha256=$(sha256sum "$root/src/benchmarks/public/aml_cycle2/adapter.py" | cut -d' ' -f1)
if [[ "$actual_adapter_sha256" != "$adapter_sha256" ]]; then
  echo 'Adapter source hash differs from the reviewed repair.' >&2
  exit 1
fi

install -d -m 0755 "$root/cargo" "$root/rustup"
curl -fsSL https://sh.rustup.rs -o "$root/rustup-init.sh"
RUSTUP_HOME="$root/rustup" CARGO_HOME="$root/cargo" \
  sh "$root/rustup-init.sh" -y --default-toolchain 1.96.0 --profile minimal
cd "$root/src"
RUSTUP_HOME="$root/rustup" CARGO_HOME="$root/cargo" \
  "$root/cargo/bin/cargo" build --locked --release --features server --bin hms-server
binary_sha256=$(sha256sum "$root/src/target/release/hms-server" | cut -d' ' -f1)
printf '%s  %s\n' "$binary_sha256" "$root/src/target/release/hms-server" > "$root/native.sha256"

python3 -m venv "$root/uv-bootstrap"
cat > "$root/uv-bootstrap-requirements.txt" <<'HMS_UV_REQUIREMENTS'
uv==0.12.10 \
    --hash=sha256:0334577671a65b621faddfdf5f9e9dd4eb08eaa7231c4405c4a7f4f17d224d9f \
    --hash=sha256:0ac914e9750bb1de2518d444cc1cff084f4eff3eefe9cdd5814b775443a16df1 \
    --hash=sha256:27e8350faffb35d8ecaa5ed39236709809bb9a883f1d11d13c0964d6051b2384 \
    --hash=sha256:2ad71395c31b5db20c56327f62ee0299f75833308d4413284aba87324d092afa \
    --hash=sha256:46ca83d33bf87a67d0c5aa5b64f5576fc5c8e700a0ecf4900d409f8a1a1c9810 \
    --hash=sha256:4f097c6b7f63eceb3faf5102009f9a8cc22fd7d8a341858d2fd540f46a592f7a \
    --hash=sha256:5c0bcf0598742eb417f1aa41ec82f0d89bd7b7f42d0a519bea495a9419d8d8ee \
    --hash=sha256:5dc26c73826d2119292d49d71c5a9d5ba9dd97dd02459d816b8226f47f3dc6bd \
    --hash=sha256:7e3a70d9dff95481afccd7d41e7ec42fdc230d53487b3e7e633f9b35610401b5 \
    --hash=sha256:a34e1be296e6e30fdbfee9b84a45d082ce7df4dd4e588ebcfc44249505c10f02 \
    --hash=sha256:b610e8cf7c6d51dc119d352e699b674534d0de327b4d2a67eb5d71ec364b8970 \
    --hash=sha256:bd0afae6918795c6a61a649e64d735e339f3d62f8310235da6f19eb3f66f323b \
    --hash=sha256:c1fd0fcabc18dccaf541ca44a061c245045c937bec75c0491e94d59cd33ca97f \
    --hash=sha256:c49cd1f6e5ded1bd844119da8b184d5c044ae51441c476af2382fdb8af99edb8 \
    --hash=sha256:de4b06934bd025ceeeb48af5c6f9a82407a2fe2b5b35e456cfb3198ef5029de0 \
    --hash=sha256:efbbc586d40014b9dd3bc37a2ad5803d12919458f0f9d16f122db5d91626f5ab \
    --hash=sha256:f43c06b19137bb05877b0ec59c4932878db5d1aeb6f7b5c33259c22e0359c4fe \
    --hash=sha256:f7d6248ad9f2d282fea795f248da8fa666ab382df50d8385bd98e01049440a0c \
    --hash=sha256:f7f2bc56fbd367b1066e8461a82e51e8886bbab52b1f8d1793af8ce52232348c
HMS_UV_REQUIREMENTS
"$root/uv-bootstrap/bin/pip" install --disable-pip-version-check --require-hashes -r "$root/uv-bootstrap-requirements.txt"
"$root/uv-bootstrap/bin/uv" venv --python 3.12 "$root/venv"
install -m 0644 "$script_dir/runtime-requirements.txt" "$root/runtime-requirements.txt"
"$root/uv-bootstrap/bin/uv" pip install --python "$root/venv/bin/python" \
  --require-hashes -r "$root/runtime-requirements.txt"

cat > /etc/systemd/system/hms-aml.service <<EOF
[Unit]
Description=Holographic Memory AML Add/Search endpoint
After=network-online.target
Wants=network-online.target
ConditionPathExists=/etc/hms-aml/token.env

[Service]
Type=simple
User=hms-aml
Group=hms-aml
WorkingDirectory=$root/src/benchmarks/public/aml_cycle2
Environment=HOME=/var/lib/hms-aml
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStart=$root/venv/bin/python $root/src/benchmarks/public/aml_cycle2/local_host.py --binary $root/src/target/release/hms-server --binary-sha256 $binary_sha256 --data-dir /var/lib/hms-aml/stores --token-file /etc/hms-aml/token.env
Restart=on-failure
RestartSec=5
TimeoutStopSec=1900
UMask=0077
NoNewPrivileges=yes
PrivateTmp=yes
ProtectHome=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/hms-aml

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable hms-aml.service
echo "AML_INSTALL_READY native_commit=$native_commit adapter_sha256=$actual_adapter_sha256 binary_sha256=$binary_sha256"
