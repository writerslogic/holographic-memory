# Deploying warm-server

| Path | What |
|---|---|
| `docker/Dockerfile` | Static musl binary on distroless, uid 65532, `/data` volume, self-probing healthcheck |
| `compose/compose.yaml` | Single service, read-only root, no capabilities, key required |
| `helm/warm-server/` | StatefulSet (one replica, PVC), Service, optional Ingress, NetworkPolicy, ServiceMonitor |
| `systemd/warm-server.service` | Sandboxed unit with `ExecReload` |
| `bench/server_bench.py` | Load, scaling and soak harness (`uv run`); `baseline_sqlite_fastapi.py` is its SQLite baseline |

Operations guide: [`docs/server/operations.md`](../docs/server/operations.md). Configuration: [`docs/server/configuration.md`](../docs/server/configuration.md).

```sh
# Container
docker build -f deploy/docker/Dockerfile -t warm-server .
docker run -d -p 8080:8080 -e WARM_API_KEY="$(openssl rand -hex 32)" -v warm-data:/data warm-server

# Kubernetes
helm install warm deploy/helm/warm-server --set auth.apiKey="$(openssl rand -hex 32)"
helm lint deploy/helm/warm-server --set auth.apiKey=x
helm template warm deploy/helm/warm-server --set auth.apiKey=x | kubeconform -strict -kubernetes-version 1.30.0
```
