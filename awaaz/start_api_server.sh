#!/bin/bash
# AWAAZ FastAPI Server Startup Script

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 AWAAZ FastAPI Voice Processing Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ ERROR: .env file not found!"
    echo "📋 Creating from .env.example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your API keys and configuration"
    exit 1
fi

# Load environment
source .env

# Check required environment variables
check_env() {
    local required=("GROQ_API_KEY")
    for var in "${required[@]}"; do
        if [ -z "${!var}" ]; then
            echo "❌ MISSING ENV: $var"
            return 1
        else
            echo "✅ ENV: $var is set"
        fi
    done
    return 0
}

echo ""
echo "📋 Checking environment variables..."
if ! check_env; then
    echo ""
    echo "⚠️  Some required environment variables are missing!"
    echo "Please update .env file with your API keys"
    exit 1
fi

echo ""
echo "📦 Checking Python dependencies..."

# Check if FastAPI is installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📥 Installing FastAPI and dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ All checks passed!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Starting AWAAZ API Server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📍 API will be available at:"
echo "   🔷 HTTP:  http://localhost:8000"
echo "   📚 Docs:  http://localhost:8000/docs      (Swagger UI)"
echo "   📖 ReDoc: http://localhost:8000/redoc     (ReDoc)"
echo ""
echo "🏥 Health Check: http://localhost:8000/health"
echo "ℹ️  Pipeline Info: http://localhost:8000/api/v1/pipeline/background-info"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the API server
python3 -m uvicorn api_server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
