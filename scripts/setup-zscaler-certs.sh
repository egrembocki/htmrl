#!/bin/bash
# Setup script for Zscaler SSL certificates
# This script creates a combined CA bundle that includes both system certificates
# and Zscaler certificates for corporate proxy environments.

set -e

echo "🔒 Setting up Zscaler SSL certificates..."

# Create certificate directory
CERT_DIR="$HOME/.local/share/ca-certificates"
mkdir -p "$CERT_DIR"

# Extract Zscaler certificate chain
echo "📥 Extracting Zscaler certificate chain..."
echo | openssl s_client -connect pypi.org:443 -servername pypi.org -showcerts 2>/dev/null | \
    awk '/BEGIN CERTIFICATE/,/END CERTIFICATE/ {print}' > "$CERT_DIR/zscaler-chain.pem"

# Create combined bundle
echo "🔗 Creating combined CA bundle..."
cat /etc/ssl/certs/ca-certificates.crt "$CERT_DIR/zscaler-chain.pem" > "$CERT_DIR/combined-ca-bundle.crt"

# Test the bundle
echo "✅ Testing certificate bundle..."
if curl --cacert "$CERT_DIR/combined-ca-bundle.crt" -I https://pypi.org/simple/matplotlib/ >/dev/null 2>&1; then
    echo "✅ Certificate bundle is working!"
else
    echo "❌ Certificate bundle test failed"
    exit 1
fi

# Add environment variables to shell profile
SHELL_RC=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
fi

if [ -n "$SHELL_RC" ]; then
    echo ""
    echo "To make these settings permanent, add the following to your $SHELL_RC:"
    echo ""
    echo "# Zscaler SSL certificate configuration"
    echo "export SSL_CERT_FILE=\"$CERT_DIR/combined-ca-bundle.crt\""
    echo "export REQUESTS_CA_BUNDLE=\"$CERT_DIR/combined-ca-bundle.crt\""
    echo "export CURL_CA_BUNDLE=\"$CERT_DIR/combined-ca-bundle.crt\""
    echo "export UV_NATIVE_TLS=1"
    echo ""
    
    read -p "Would you like to add these lines automatically? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "" >> "$SHELL_RC"
        echo "# Zscaler SSL certificate configuration" >> "$SHELL_RC"
        echo "export SSL_CERT_FILE=\"$CERT_DIR/combined-ca-bundle.crt\"" >> "$SHELL_RC"
        echo "export REQUESTS_CA_BUNDLE=\"$CERT_DIR/combined-ca-bundle.crt\"" >> "$SHELL_RC"
        echo "export CURL_CA_BUNDLE=\"$CERT_DIR/combined-ca-bundle.crt\"" >> "$SHELL_RC"
        echo "export UV_NATIVE_TLS=1" >> "$SHELL_RC"
        echo "✅ Environment variables added to $SHELL_RC"
        echo "Run 'source $SHELL_RC' to apply changes in current shell"
    fi
fi

echo ""
echo "✅ Setup complete!"
echo "The Makefile is already configured to use the certificate bundle."
