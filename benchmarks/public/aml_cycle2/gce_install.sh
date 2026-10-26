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
"$root/uv-bootstrap/bin/pip" install --disable-pip-version-check uv==0.12.10
"$root/uv-bootstrap/bin/uv" venv --python 3.12 "$root/venv"
"$root/uv-bootstrap/bin/uv" pip install --python "$root/venv/bin/python" \
  fastapi==0.119.0 httpx==0.28.1 uvicorn==0.38.0

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
