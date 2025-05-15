import os
from pathlib import Path

# SSL Configuration
SSL_DIR = Path(__file__).parent
SSL_CERT_PATH = SSL_DIR / 'cert.pem'
SSL_KEY_PATH = SSL_DIR / 'key.pem'

def get_ssl_context():
    """Get SSL context for HTTPS server"""
    import ssl
    
    if not (SSL_CERT_PATH.exists() and SSL_KEY_PATH.exists()):
        raise FileNotFoundError(
            "SSL certificate files not found. Please generate them using OpenSSL:"
            "\nopenssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes"
        )
    
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(SSL_CERT_PATH, SSL_KEY_PATH)
    return ssl_context