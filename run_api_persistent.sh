#!/bin/bash

# ReCon API Persistent Runner
# Multiple methods to keep your API running without keeping terminal open

set -e

echo "=== ReCon API Persistent Runner ==="
echo "Choose your preferred method:"
echo "1. Supervisor (Most robust - survives reboots, auto-restart)"
echo "2. Nohup (Simple background process)"
echo "3. Check status of running services"
echo "4. Stop all services"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Setting up Supervisor..."
        
        # Copy supervisor config
        cp /workspace/recon-platform/supervisor-recon-api.conf /etc/supervisor/conf.d/
        
        # Start supervisor if not running
        if ! pgrep supervisord > /dev/null; then
            echo "Starting supervisor daemon..."
            /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
            sleep 2
        fi
        
        # Reload supervisor config
        supervisorctl reread
        supervisorctl update
        
        # Start the API service
        supervisorctl start recon-api
        
        echo "✅ API server started with Supervisor!"
        echo "Commands to manage:"
        echo "  supervisorctl status recon-api    # Check status"
        echo "  supervisorctl stop recon-api      # Stop service"
        echo "  supervisorctl restart recon-api   # Restart service"
        echo "  tail -f /var/log/recon-api.log    # View logs"
        ;;
    2)
        echo "Starting with nohup..."
        cd /workspace/recon-platform
        
        # Kill any existing processes
        pkill -f "python.*api/app.py" || true
        
        # Start with nohup
        nohup bash -c 'export PYTHONPATH=/workspace/recon-platform; python3 api/app.py' > /var/log/recon-api-nohup.log 2>&1 &
        
        echo "✅ API server started with nohup!"
        echo "Process ID: $!"
        echo "Logs: tail -f /var/log/recon-api-nohup.log"
        ;;
    3)
        echo "=== Service Status ==="
        
        # Check supervisor
        if pgrep supervisord > /dev/null; then
            echo "Supervisor is running"
            supervisorctl status recon-api 2>/dev/null || echo "  recon-api not configured in supervisor"
        else
            echo "Supervisor is not running"
        fi
        
        # Check nohup processes
        if pgrep -f "python.*api/app.py" > /dev/null; then
            echo "API processes running:"
            ps aux | grep "python.*api/app.py" | grep -v grep
        else
            echo "No API processes found"
        fi
        
        # Check if API is responding
        echo "Testing API endpoint..."
        if curl -s http://localhost:5001/ > /dev/null 2>&1; then
            echo "✅ API is responding on http://localhost:5001"
        else
            echo "❌ API is not responding"
        fi
        ;;
    4)
        echo "Stopping all services..."
        
        # Stop supervisor service
        supervisorctl stop recon-api 2>/dev/null || true
        
        # Kill nohup processes
        pkill -f "python.*api/app.py" || true
        
        echo "✅ All services stopped"
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-4."
        exit 1
        ;;
esac
