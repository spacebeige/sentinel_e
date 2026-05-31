import jwt
secret = b"test_secret"
token = jwt.encode({"iss": "supabase", "sub": "123", "aud": "authenticated"}, secret, algorithm="HS256")
try:
    decoded = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_aud": False})
    print("DECODE WITH VERIFY_AUD=False:", decoded)
except Exception as e:
    print("ERROR 1:", type(e).__name__, e)

try:
    decoded2 = jwt.decode(token, secret, algorithms=["HS256"])
    print("DECODE WITHOUT VERIFY_AUD:", decoded2)
except Exception as e:
    print("ERROR 2:", type(e).__name__, e)

