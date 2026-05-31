import urllib.request
import json

req = urllib.request.Request(
    'http://127.0.0.1:8001/api/v2/analytics/events',
    data=json.dumps({"event_type": "TEST"}).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    with urllib.request.urlopen(req) as f:
        print("Status:", f.status)
        print("Body:", f.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print("Status:", e.code)
    print("Body:", e.read().decode('utf-8'))
except Exception as e:
    print("Error:", e)
