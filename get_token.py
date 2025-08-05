import requests

# 1) Your Egnyte domain (no protocol)
DOMAIN = "citizen.egnyte.com"

# 2) From your Internal App in the Developer Portal
CLIENT_ID     = "6fgq8d9ye7q3fcxj4uv3the9"
CLIENT_SECRET = "Qdzgwdzjbkxncs5QcSWm7CgpTepqguqaa65jG8qdQNHwKhcYU4"

# 3) Your Egnyte domain login
USERNAME = "citpub"
PASSWORD = "Conlen12!@#$%"

# 4) Request an access token
resp = requests.post(
    f"https://{DOMAIN}/puboauth/token",
    data={
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type":    "password",
        "username":      USERNAME,
        "password":      PASSWORD
    }
)

# 5) Print status & body
print("Status code:", resp.status_code)
print("Response body:", resp.text)
