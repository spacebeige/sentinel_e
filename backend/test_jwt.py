import jwt
secret = b"test_secret"
token = jwt.encode({"iss": "supabase", "sub": "123"}, secret, algorithm="HS256")
try:
    decoded = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_aud": False})
    print("SUCCESS")
except Exception as e:
    print("ERROR:", type(e).__name__, e)
