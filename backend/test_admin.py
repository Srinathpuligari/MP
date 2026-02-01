import urllib.request
import json

try:
    data = json.dumps({'code': 'cbit'}).encode('utf-8')
    req = urllib.request.Request(
        'http://localhost:5000/admin/users',
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=10) as res:
        result = json.loads(res.read().decode('utf-8'))
        print(f"Status: success")
        print(f"Total users: {result.get('total', 0)}")
        for u in result.get('users', []):
            print(f"  - {u['name']} ({u['uid']})")
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code} - {e.reason}")
except urllib.error.URLError as e:
    print(f"Connection Error: {e.reason}")
except Exception as e:
    print(f"Error: {e}")
