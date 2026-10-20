# Deploying hms-server

| Path | What |
|---|---|
| `docker/Dockerfile` | Static musl binary on distroless, uid 65532, `/data` volume, self-probing healthcheck |
| `compose/compose.yaml` | Single service, read-only root, no capabilities, key required |
| `helm/hms-server/` | StatefulSet (one replica, PVC), Service, optional Ingress, NetworkPolicy, ServiceMonitor |
| `systemd/hms-server.service` | Sandboxed unit with `ExecReload` |
| `bench/server_bench.py` | Load, scaling and soak harness (`uv run`); `baseline_sqlite_fastapi.py` is its SQLite baseline |

Operations guide: [`docs/server/operations.md`](../docs/server/operations.md). Configuration: [`docs/server/configuration.md`](../docs/server/configuration.md).

```sh
# Container
docker build -f deploy/docker/Dockerfile -t hms-server .
docker run -d -p 8080:8080 -e HMS_API_KEY="$(openssl rand -hex 32)" -v hms-data:/data hms-server

# Kubernetes
helm install hms deploy/helm/hms-server --set auth.apiKey="$(openssl rand -hex 32)"
helm lint deploy/helm/hms-server --set auth.apiKey=x
helm template hms deploy/helm/hms-server --set auth.apiKey=x | kubeconform -strict -kubernetes-version 1.30.0
```
