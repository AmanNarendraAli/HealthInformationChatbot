BASEURL = "http://3.137.168.19:8000"

import requests
import json


def chat(user_input, session_id: str = None):
    url = f"{BASEURL}/med_assist/invoke"
    payload = json.dumps({"input": user_input, "config": {}, "kwargs": {}})
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    print(payload)
    response = requests.request(
        "POST",
        url,
        headers=headers,
        data=payload,
    )
    print

    if response.status_code == 200:
        return response.json()
