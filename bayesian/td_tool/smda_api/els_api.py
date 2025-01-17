import msal
import requests
import time
import urllib.parse
import os
import yaml

# Connect the path with your '.env' file name

config_file = os.path.join(os.getcwd(),"smda_password","els_api.yaml")
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
            msg = f"Connection to ELS Logs successful. Status code = {response.status_code}."
            print(msg)
            return response.json(), msg
            
        else:
            msg = f"Unable to read data from API. Status code = {response.status_code}."
            if response.text:
                msg += f" Message: {response.text}."
            print(msg)
            return None, msg
def get_api(uwi, selected_source_well_log):
    encoded_uwi = urllib.parse.quote(uwi)
    print(encoded_uwi)
    #API_ENDPOINT = "https://api.gateway.equinor.com/els/curves/data?source=LFP&unique_wellbore_identifier=NO%201%{2F3-1}&curve_identifier=MD,%20TVDMSL,%20LFP_VP_B,%20LFP_VP_LOG,%20LFP_VP_G,%20LFP_VP_O,%20LFP_VP_V"
    if selected_source_well_log == 'LFP':
        API_ENDPOINT = f"https://api.gateway.equinor.com/els/curves/data?source=LFP&unique_wellbore_identifier={encoded_uwi}&curve_identifier=MD,%20TVDMSL,%20LFP_VP_V,%20LFP_VP_LOG,%20LFP_VP_G,%20LFP_VP_O,%20LFP_VP_B"
        print("encoded_uwi", encoded_uwi)
    elif selected_source_well_log == 'FMB':
        API_ENDPOINT = f"https://api.gateway.equinor.com/els/curves/data?source=LFP&unique_wellbore_identifier={encoded_uwi}&curve_identifier=MD,%20TVDMSL,%20LFP_VP_V,%20LFP_VP_LOG,%20LFP_VP_G,%20LFP_VP_O,%20LFP_VP_B"
    c = SmdaApiClient()
    response, msg = c.get_wellbore_plan(API_ENDPOINT)
    return response, msg
