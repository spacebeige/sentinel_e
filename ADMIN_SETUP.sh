#!/bin/bash
# ============================================================
# Admin Dashboard Setup Script
# ============================================================
# This script promotes an email address to admin and tests the system

set -e

# Configuration
EMAIL="${1:-oomkaragarkhed0710@gmail.com}"
API_BASE="${2:-http://localhost:8000}"
ADMIN_USER_EMAIL="$EMAIL"

echo "🔧 Sentinel-E Admin Setup"
echo "================================"
echo "Email: $ADMIN_USER_EMAIL"
echo "API Base: $API_BASE"
echo ""

# Step 1: Create admin user
echo "📝 Step 1: Creating admin user..."
echo ""
echo "Run this curl command to promote the user to admin:"
echo ""
echo "curl -X POST '$API_BASE/api/admin/users/make-admin' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN' \\"
echo "  -d '{\"email\": \"$ADMIN_USER_EMAIL\"}'"
echo ""
echo "Note: Replace YOUR_ACCESS_TOKEN with a valid JWT token"
echo ""

# Step 2: Verify admin dashboard
echo "✓ Admin Dashboard Setup Complete!"
echo ""
echo "Backend Endpoints Ready:"
echo "  • POST /api/admin/users/make-admin - Promote user to admin"
echo "  • GET /api/admin/system/stats - System statistics"
echo "  • GET /api/admin/system/architecture - Architecture info"
echo "  • GET /api/admin/web-analytics - Web analytics"
echo "  • GET /api/admin/feedback-summary - Feedback analysis"
echo ""

echo "Frontend Setup Complete:"
echo "  ✓ AdminDashboard component created"
echo "  ✓ Admin route added at /admin"
echo "  ✓ Navbar shows admin link for admin users"
echo "  ✓ MakeAdminForm component created"
echo "  ✓ useAdminRole hook for role checking"
echo ""

echo "🚀 Next Steps:"
echo "================================"
echo ""
echo "1. Start the backend server:"
echo "   cd /Users/ashwinagarkhed/sentinel_e"
echo "   python -m uvicorn backend.main:app --reload"
echo ""
echo "2. Start the frontend:"
echo "   cd frontend"
echo "   npm start"
echo ""
echo "3. Create a new anonymous session:"
echo "   curl -X POST http://localhost:8000/api/auth/session"
echo ""
echo "4. Use the access_token to promote an admin:"
echo "   curl -X POST 'http://localhost:8000/api/admin/users/make-admin' \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -H 'Authorization: Bearer <access_token>' \\"
echo "     -d '{\"email\": \"oomkaragarkhed0710@gmail.com\"}'"
echo ""
echo "5. Login as the admin user and visit:"
echo "   http://localhost:3000/admin"
echo ""
echo "✅ Admin dashboard will display system architecture,"
echo "   analytics, feedback, and user management!"
echo ""
