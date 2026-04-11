import urllib.request
import json

url = 'http://localhost:8000/evaluate/code'
data = {
    "question_id": 1,
    "language": "python",
    "source_code": "x = input()\nprint(int(x) * 3)"
}
req = urllib.request.Request(
    url, 
    data=json.dumps(data).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        print(json.dumps(result, indent=2))
except urllib.error.URLError as e:
    if hasattr(e, 'read'):
        print(e.read().decode())
    else:
        print(e)
