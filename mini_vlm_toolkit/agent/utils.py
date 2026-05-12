import json
import urllib.request
HTTP_TIMEOUT_SEC = 30


def _get_json(host, path):
    url = f"{host}{path}"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SEC) as response:
            payload = response.read()
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None
    
def pretty_print_msgs(msgs):
    for msg in msgs:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # skip for image content for better readability
        print("role:", role)
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    print("content:", item.get("text", ""))
                elif item.get("type") == "image_url":
                    print("content: [image]")
                else:
                    print("content: [unknown content type]")
        else:
            print("content:", content)
