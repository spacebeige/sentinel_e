import requests

res = requests.get("http://127.0.0.1:8000/api/admin/system/stats", headers={"x-debug-user": "test-user-123", "x-debug-email": "oomkaragarkhed0710@gmail.com"})
print(res.status_code)
print(res.text)
