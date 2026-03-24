#!/bin/bash
# 🚀 Google Cloud Text-to-Speech Setup Script
# Quick setup for premium natural voices

set -e

echo \"\"
echo \"════════════════════════════════════════════════════════════\"
echo \"  🎤 Google Cloud TTS Setup for AWAAZ\"
echo \"════════════════════════════════════════════════════════════\"
echo \"\"

# Check if already got Google SDK
if command -v gcloud &> /dev/null; then
    echo \"✅ Google Cloud SDK already installed\"
else
    echo \"📦 Installing Google Cloud SDK...\"
    # macOS
    if [[ \"$OSTYPE\" == \"darwin\"* ]]; then
        brew install --cask google-cloud-sdk
    # Linux
    elif [[ \"$OSTYPE\" == \"linux-gnu\"* ]]; then
        curl https://sdk.cloud.google.com | bash
    fi
fi

# Install Python library
echo \"\"
echo \"📦 Installing google-cloud-texttospeech library...\"
pip install google-cloud-texttospeech

# Ask for service account
echo \"\"
echo \"════════════════════════════════════════════════════════════\"
echo \"  📋 Service Account Setup\"
echo \"════════════════════════════════════════════════════════════\"
echo \"\"
echo \"You need a Google Cloud service account JSON key.\"
echo \"\"
echo \"Steps:\"
echo \"  1. Go to: https://console.cloud.google.com/\"
echo \"  2. Create a new project or select existing\"
echo \"  3. Enable 'Cloud Text-to-Speech API'\"
echo \"  4. Go to 'Service Accounts' (APIs & Services)\"
echo \"  5. Create a new service account\"
echo \"  6. Create JSON key\"
echo \"  7. Save the JSON file\"
echo \"\"
read -p \"Enter path to your service account JSON file: \" json_path

if [ ! -f \"$json_path\" ]; then
    echo \"❌ File not found: $json_path\"
    exit 1
fi

# Set up credentials
mkdir -p ~/.config/gcloud
cp \"$json_path\" ~/.config/gcloud/application_default_credentials.json
echo \"\"
echo \"✅ Service account copied to: ~/.config/gcloud/application_default_credentials.json\"

# Set environment variable for this session
export GOOGLE_APPLICATION_CREDENTIALS=\"$HOME/.config/gcloud/application_default_credentials.json\"

# Add to shell profile for persistence
shell_profile=\"$HOME/.zshrc\"
if [ -f \"$HOME/.bashrc\" ] && [ ! -f \"$HOME/.zshrc\" ]; then
    shell_profile=\"$HOME/.bashrc\"
fi

if grep -q \"GOOGLE_APPLICATION_CREDENTIALS\" \"$shell_profile\"; then
    echo \"✅ GOOGLE_APPLICATION_CREDENTIALS already in $shell_profile\"
else
    echo \"export GOOGLE_APPLICATION_CREDENTIALS=\\\"$HOME/.config/gcloud/application_default_credentials.json\\\"\" >> \"$shell_profile\"
    echo \"✅ Added GOOGLE_APPLICATION_CREDENTIALS to $shell_profile\"
fi

echo \"\"
echo \"════════════════════════════════════════════════════════════\"
echo \"  ✅ Setup Complete!\"
echo \"════════════════════════════════════════════════════════════\"
echo \"\"
echo \"Free Tier Benefits:\"
echo \"  • 1,000,000 characters per month (FREE)\"
echo \"  • All Indian languages supported\"
echo \"  • Neural2 voices (premium quality)\"
echo \"  • Best pronunciation for Indian languages\"
echo \"\"
echo \"Test it:\"
echo \"  cd /Users/ashwinagarkhed/AVA-AI-Voice-Agent-for-Asterisk/awaaz\"
echo \"  python3 test_live_voice.py --lang hi --output ./google_test.wav\"
echo \"  afplay ./google_test.wav\"
echo \"\"
echo \"📊 Provider Priority (automatic fallback):\"
echo \"  1. Google Cloud Neural (if credentials set) ← BEST\"
echo \"  2. Sarvam (priya, pooja, simran, etc.) ← Now available\"
echo \"  3. ElevenLabs \"
echo \"  4. Groq\"
echo \"  5. gTTS\"
echo \"\"
echo \"✨ Your AWAAZ system now has the best voice quality!\"
echo \"\"
