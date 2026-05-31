import jwt
secret = b"test_secret"
token = jwt.encode({"iss": "supabase", "sub": "123", "aud": "authenticated"}, secret, algorithm="HS256")
try:
    decoded = jwt.decode(token, options={"verify_signature": False})
    print("SUCCESS:", decoded)
except Exception as e:
    print("ERROR:", type(e).__name__, e)
