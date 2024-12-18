import msal
import requests
import time
import urllib.parse
import os
import yaml

# Connect the path with your '.env' file name

config_file = os.path.join(os.getcwd(),"smda_password","smda_api.yaml")
with open(config_file, "r") as file:
    config = yaml.safe_load(file)
    print("CONFIG", config)
    TENANT = config['TENANT']
    CLIENT_ID = config['CLIENT_ID']
    SCOPE = config['SCOPE']
    CLIENT_SECRET = config['CLIENT_SECRET']
    Subscription_Key = config['Subscription_Key']


AUTHORITY = f"https://login.microsoftonline.com/{TENANT}"




class SmdaApiClient:
    def __init__(self) -> None:
        self._token_cache = None
        self._token_expiry = 0
        self._app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
 
    def _get_token(self):
        current_time = time.time()
        if self._token_cache and current_time < self._token_expiry:
            return self._token_cache
 
        result = self._app.acquire_token_silent([SCOPE], account=None)
        if not result:
            result = self._app.acquire_token_interactive(scopes=[SCOPE])
 
        if "access_token" in result:
            self._token_cache = result["access_token"]
            self._token_expiry = current_time + result.get("expires_in", 3600) - 300  # Subtract 5 minutes for safety
            return self._token_cache
        else:
            print(result.get("error"))
            print(result.get("error_description"))
            print(result.get("correlation_id"))
 
    def _authorization_header(self) -> dict[str, str]:
        token = self._get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Ocp-Apim-Subscription-Key": Subscription_Key,
            "Content-Type": "application/json",
        }
 
    def get_wellbore_plan(self, API_ENDPOINT):
        headers = self._authorization_header()
 
        response = requests.get(
            API_ENDPOINT,
            headers=headers,
        )
        if response.status_code == 200:
            return response.json()
        else:
            msg = f"Unable to read data from API. Status code = {response.status_code}."
            if response.text:
                msg += f" Message: {response.text}."
            print(msg)
            return None
def get_wellbore_trajectory(uwi):
    encoded_uwi = urllib.parse.quote(uwi)
    API_ENDPOINT = f"https://api.gateway.equinor.com/smda/v2.0/smda-api/wellbore-plan-survey-samples?_projection=unique_wellbore_identifier,%20md,%20tvd_msl&unique_wellbore_identifier={encoded_uwi}"
    c = SmdaApiClient()
    response = c.get_wellbore_plan(API_ENDPOINT)
    return response
