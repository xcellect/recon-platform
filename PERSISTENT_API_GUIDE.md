# ReCon Platform API - Persistent Running Guide

🎉 **Your API server is now running persistently!** No more keeping terminals open!

## Current Status
✅ **API is RUNNING** on `http://localhost:5001`  
✅ **Managed by Supervisor** - auto-restart, survives disconnections  
✅ **Logs available** at `/var/log/recon-api.log`

## Quick Commands

### Check API Status
```bash
curl http://localhost:5001/
# Should return: {"message":"ReCon ARC Demo API","version":"0.1.0","networks":1}
```

### Manage the Service
```bash
# Check status
supervisorctl status recon-api

# Stop the service
supervisorctl stop recon-api

# Start the service
supervisorctl start recon-api

# Restart the service
supervisorctl restart recon-api

# View logs
tail -f /var/log/recon-api.log
```

## Available Solutions

### 1. 🏆 Supervisor (CURRENTLY ACTIVE - RECOMMENDED)
**Most robust solution - what you're using now**

- ✅ **Auto-restart** if the process crashes
- ✅ **Survives terminal closure**
- ✅ **Survives SSH disconnections**
- ✅ **Centralized logging**
- ✅ **Easy management commands**

**Usage:**
```bash
# Use the interactive script
cd /workspace/recon-platform
./run_api_persistent.sh
# Choose option 1

# Or manage directly
supervisorctl status recon-api
supervisorctl restart recon-api
```

### 2. 🐳 Docker (Alternative - Most Portable)
**Best for production deployments**

```bash
cd /workspace/recon-platform

# Build and run
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f recon-api

# Stop
docker-compose down
```

### 3. 🔧 Nohup (Simple Background Process)
**Lightweight option**

```bash
cd /workspace/recon-platform
./run_api_persistent.sh
# Choose option 2
```

## File Overview

| File | Purpose |
|------|---------|
| `run_api_persistent.sh` | Interactive script to start/stop services |
| `supervisor-recon-api.conf` | Supervisor configuration |
| `Dockerfile.api` | Docker container definition |
| `docker-compose.yml` | Docker compose configuration |
| `start_api.sh` | Basic startup script |

## Troubleshooting

### API Not Responding?
```bash
# Check if supervisor is running
supervisorctl status

# Check logs for errors
tail -f /var/log/recon-api.log

# Restart the service
supervisorctl restart recon-api
```

### Port Already in Use?
```bash
# Find what's using port 5001
lsof -i :5001

# Kill old processes
pkill -f "python.*api/app.py"

# Restart supervisor service
supervisorctl restart recon-api
```

### Want to Change the Port?
Edit `/workspace/recon-platform/api/app.py` line 597:
```python
uvicorn.run(app, host="0.0.0.0", port=5001)  # Change 5001 to your desired port
```

Then restart: `supervisorctl restart recon-api`

## Migration from Your Old Method

**Before (annoying):**
```bash
cd /workspace/recon-platform && PYTHONPATH=/workspace/recon-platform python api/app.py &
# Had to keep terminal open, process died when terminal closed
```

**Now (robust):**
```bash
# One-time setup (already done!)
# Service is now running automatically

# To check status anytime:
supervisorctl status recon-api

# To restart if needed:
supervisorctl restart recon-api
```

## Benefits of This Solution

1. **🔄 Auto-restart**: If your API crashes, supervisor automatically restarts it
2. **📱 Terminal independence**: Close your terminal, SSH session, whatever - API keeps running
3. **📋 Centralized logging**: All logs go to `/var/log/recon-api.log`
4. **⚡ Fast management**: Simple commands to start/stop/restart
5. **🔧 Multiple options**: Choose between supervisor, Docker, or nohup based on your needs

## Next Steps

Your API is now running persistently! You can:
- Close this terminal
- Disconnect from SSH
- Reboot the system (supervisor will auto-start the service)

The API will keep running at `http://localhost:5001` 🚀
