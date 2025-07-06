#!/usr/bin/env python3
import requests
import json

def chat_with_local_qwen(system_prompt: str, user_prompt: str, 
                         endpoint: str = "http://localhost:8080/v1/chat/completions"):
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    }
    resp = requests.post(endpoint, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    # Extract the assistant reply
    return data["choices"][0]["message"]["content"]

def main():
    system_prompt = "You are Qwen."
    user_prompt   = "Write hello world in Rust."
    try:
        reply = chat_with_local_qwen(system_prompt, user_prompt)
        print("Assistant reply:\n")
        print(reply)
    except requests.HTTPError as e:
        print(f"Request failed ({e.response.status_code}): {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
