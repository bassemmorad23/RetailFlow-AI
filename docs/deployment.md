# Deployment

Full setup guide for running StoreFlow AI, from local development to live production on Railway with MongoDB Atlas, Qdrant Cloud, and Sentry.

For a quick-start summary, see the [main README](../README.md).

---

## Overview

The production stack is fully managed cloud services chosen to remove operational burden:

| Layer | Service | Free tier | Cost at v1 scale |
|---|---|---|---|
| API host | Railway | Yes (trial) | ~$5/month |
| Catalog + memory DB | MongoDB Atlas | M0 (512MB, shared) | Free |
| Vector search | Qdrant Cloud | 1GB, shared | Free |
| LLM | OpenRouter | Free-tier models available | Pay per token when swapped to paid |
| Error monitoring | Sentry | 5k errors + 10k transactions/mo | Free |
| Uptime monitoring | UptimeRobot | 50 monitors, 5-min checks | Free |

No self-hosted infrastructure. Zero SSH access needed. Every service is signed up for, configured, and left alone.

---

## Local development

### Prerequisites

- Python 3.12
- Docker Desktop (only if you want to run MongoDB locally instead of Atlas)
- A `.env` file with real credentials (see `.env.example`)

### First-time setup

```bash
# Clone
git clone https://github.com/bassemmorad23/RetailFlow-AI.git
cd RetailFlow-AI

# Virtualenv (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install deps
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env with real values (see "Environment variables" below)
```

### Load a store's data

Every store gets its own data loaded via the migration script, then embeddings built in Qdrant.

```bash
# 1. Load catalog into MongoDB
python -m scripts.migrate_json_to_mongodb \
    --store-id store_001 \
    --currency EGP \
    --source-dir path/to/store_001/json

# 2. Build the store's embeddings in Qdrant Cloud
python -m app.rag.build_embeddings --store-id store_001
```

The migration script is **idempotent** — running it again drops and recreates that store's documents, so re-runs are always safe.

The build script is **destructive per store** — it drops and recreates that store's Qdrant collection. Other stores are untouched.

### Run the API

```bash
uvicorn app.api:app --reload
```

Then open:
- [http://localhost:8000](http://localhost:8000) → redirects to interactive API docs
- [http://localhost:8000/health](http://localhost:8000/health) → liveness check
- [http://localhost:8000/metrics](http://localhost:8000/metrics) → operational metrics

---

## Environment variables

All values live in `.env` locally and in Railway's Variables tab in production.

| Variable | Required | Purpose |
|---|---|---|
| `APP_ENV` | Yes | `prototype` locally, `production` on Railway. Controls log level, Sentry sample rate, etc. |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key for response generation + fact extraction |
| `MONGO_URI` | Yes | MongoDB Atlas connection string with password inline |
| `MONGO_DB` | Yes | Database name inside Atlas (e.g. `storeflow`) |
| `QDRANT_URL` | Yes | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud API key |
| `HF_TOKEN` | Optional | HuggingFace token for higher HF Hub rate limits during model download |
| `SENTRY_DSN` | Optional | Sentry DSN. If empty, Sentry is skipped silently — the app still runs. |

`.env` is gitignored. `.env.example` is the committed template.

---

## Docker

The `Dockerfile` builds a production image:

1. Installs system build tools
2. Installs CPU-only torch first (biggest dependency, cached separately)
3. Installs `requirements.txt`
4. Copies `app/` into the image
5. Pre-downloads local HF models (emotion + intent) at build time, not at first request
6. Runs `uvicorn` on port 8000

Local docker-compose:

```bash
docker-compose up -d --build
```

This also starts a local MongoDB container. In production, MongoDB is Atlas — the app container just uses the `MONGO_URI` env var.

To rebuild after code changes:

```bash
docker-compose up -d --build
```

To stop everything:

```bash
docker-compose down
```

---

## MongoDB Atlas

### Setup (one-time)

1. Sign up at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Create a free M0 cluster in the region closest to your deployment (Frankfurt / `eu-central-1` for Egypt-based projects)
3. **Database Access** → create a user (e.g. `storeflow_app`) with the built-in **Read and write to any database** role. Save the password.
4. **Network Access** → **Add IP Address** → **Allow Access From Anywhere** (`0.0.0.0/0`). Required because Railway containers don't have a fixed IP. The password is what protects the DB.
5. **Clusters** → **Connect** → **Drivers** → copy the connection string. Looks like:
   ```
   mongodb+srv://storeflow_app:<db_password>@cluster0.xxxx.mongodb.net/?appName=Cluster0
   ```
6. Replace `<db_password>` with the real password and paste into `.env` as `MONGO_URI`.

### Later: tightening security

For real customer data, add:
- Network peering with the hosting provider instead of `0.0.0.0/0`
- Row-level access rules
- Rotated credentials

Deferred until real customers exist.

---

## Qdrant Cloud

### Setup (one-time)

1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a free cluster in the same region as MongoDB Atlas
3. Copy the **cluster URL** and **API key**
4. Paste into `.env`:
   ```
   QDRANT_URL=https://your-cluster.qdrant.tech:6333
   QDRANT_API_KEY=your-key
   ```

Collections are created automatically by `build_embeddings.py` — one per `store_id`.

---

## Railway (production host)

### First deploy

1. Sign up at [railway.app](https://railway.app) (GitHub sign-in is easiest — enables auto-deploy).
2. **New Project → Deploy from GitHub repo → RetailFlow-AI**.
3. Railway detects the `Dockerfile` and starts building. First build will fail because env vars aren't set — that's expected.
4. **Variables tab** → add every variable from your `.env` (see [Environment variables](#environment-variables) above). Railway auto-redeploys when variables are saved.
5. **Settings → Networking → Generate Domain**. Port = **8000** (from `EXPOSE 8000` in Dockerfile). You get a URL like `retailflow-ai-production.up.railway.app`.
6. Verify: `https://<your-url>/health` returns `{"status":"ok"}`.

### Auto-deploy on push

Once the GitHub connection is set up, every `git push` to `main` triggers a fresh Railway build. If Railway ever misses a webhook (rare), trigger a rebuild manually from the Deployments tab, or push an empty commit:

```bash
git commit --allow-empty -m "Trigger Railway redeploy"
git push
```

### Rebuilding after Dockerfile changes

Same as auto-deploy above — push to `main` and Railway rebuilds.

### When a deploy fails

- Open **Deployments** tab → click the failed build → **View logs**
- Common causes:
  - Broken Dockerfile line (missing file, wrong path)
  - Missing env var (crashes at import time)
  - Dependency install failure

Railway keeps the previous working deployment running while the new one builds, so a failed deploy never takes production down.

---

## Sentry

### Setup (one-time)

1. Sign up at [sentry.io](https://sentry.io)
2. Create a new project → platform **Python → FastAPI**
3. Copy the DSN — looks like `https://xxx@xxx.ingest.sentry.io/xxx`
4. Add to `.env` (local) AND Railway Variables (production) as `SENTRY_DSN`

The app initializes Sentry at startup only if `SENTRY_DSN` is set. In tests and local dev without a DSN, it silently skips — no crash.

### Verify

Temporary test endpoint:

```python
@app.get("/sentry-debug")
def trigger_error() -> None:
    raise ValueError("Sentry test error")
```

Hit `/sentry-debug` in a browser. Wait ~30 seconds. Error should appear in the Sentry dashboard's Issues tab.

**Remove the endpoint after verifying.**

### Configuration

Set in `app/api.py`:

```python
sentry_sdk.init(
    dsn=settings.SENTRY_DSN,
    environment=settings.APP_ENV,
    send_default_pii=False,        # never send customer message text or user IPs
    traces_sample_rate=0.1 if settings.APP_ENV == "production" else 1.0,
)
```

- `send_default_pii=False` — customer messages and IPs never leave your infra
- `traces_sample_rate` — 100% locally, 10% in production to stay within free tier

---

## UptimeRobot

### Setup (one-time)

1. Sign up at [uptimerobot.com](https://uptimerobot.com)
2. **+ Add New Monitor**
3. Type: **HTTPS**
4. URL: `https://<your-railway-url>/health`
5. Interval: **5 minutes**
6. Alert Contact: your email (added by default)
7. Save

Monitor goes green within a few minutes. If `/health` ever fails, you get an email within 5 minutes.

### Why `/health` and not `/`

`/health` deliberately does not touch MongoDB or any model — it only answers "is the web server responding?" That's exactly what a liveness probe should check. `/` would redirect to `/docs` and inflate response times.

---

## Migration commands (per store)

```bash
# One-time: load catalog JSON into MongoDB Atlas
python -m scripts.migrate_json_to_mongodb \
    --store-id store_002 \
    --currency EGP \
    --source-dir tests/store_002 \
    --yes

# One-time: build embeddings for that store in Qdrant Cloud
python -m app.rag.build_embeddings --store-id store_002
```

The `--yes` flag skips the "are you sure?" confirmation. Handy for scripting.

To onboard a new store later, only these two commands change (`--store-id`, `--source-dir`).

---

## Observability in production

After deploy, verify each layer:

- **Health:** `https://<url>/health` → `{"status":"ok"}`
- **Metrics:** `https://<url>/metrics` → JSON with request counts, latency, error rate
- **Sentry:** Issues tab in Sentry dashboard
- **UptimeRobot:** dashboard shows green
- **File logs:** `data/logs/app.jsonl` inside the container (visible only via `docker exec` locally — on Railway, logs are ephemeral per deploy, use Sentry + `/metrics` for cross-restart visibility)

---

## Troubleshooting

### Zombie processes on port 8000 (local)

Multiple uvicorn processes can hang around, especially with `--reload`. Symptoms: rate limiting looks broken, or old code seems to be running.

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Format-Table -AutoSize
```

If more than one process is listed:

```powershell
Get-Process python | Stop-Process -Force
```

Then start uvicorn fresh.

### Railway deploy uses old commit

Railway occasionally misses a GitHub webhook. Trigger a rebuild:

```bash
git commit --allow-empty -m "Trigger Railway redeploy"
git push
```

Or click **Redeploy** on the newest deployment in the Deployments tab.

### `/chat` returns 502 on Railway

Railway's edge proxy has a request timeout (~30-60s). Free-tier OpenRouter models sometimes take 60-90 seconds, which trips this. Swap to a paid model (`gpt-4o-mini` responds in 1-3s) to fix.

### CORS blocked in browser

Verify `CORSMiddleware` is present in `app/api.py` and applied to the real `app` object (not a duplicated one — a duplicated `app = FastAPI(...)` will wipe out any middleware attached to the first one).

Test:

```javascript
fetch('https://<url>/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({conversation_id: 'test', customer_id: 'u1', store_id: 'store_002', text: 'hi', channel: 'web'})
}).then(r => r.json()).then(console.log).catch(console.error);
```

Run from any random cross-origin site's console (e.g. `example.com`). Should succeed.