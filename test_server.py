import requests

print(
    requests.post(
        "http://0.0.0.0:10000",
        json={
            "query": "What is the United States?"
        }
    ).json()
)