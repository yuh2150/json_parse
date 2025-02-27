
from base64 import b64encode
import requests
import os
def basic_auth(username, password):
    token = b64encode(f"{username}:{password}".encode('utf-8')).decode("ascii")
    return f'Basic {token}'
class GeoCodingAPI:

    def __init__(self):
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': basic_auth(os.getenv("MAP_USERNAME"), os.getenv("MAP_PASSWORD"))
        }

    def get_geocoding(self, address, channel="ai",X="goog"):
        params = {
            "address": address,
            "channel": channel,
            "forceProvider": X
            }
        try:
            response = requests.get(os.getenv("GEOCODING_API"), params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": response.status_code}
        except requests.RequestException as e:
            return {"error": str(e)}
