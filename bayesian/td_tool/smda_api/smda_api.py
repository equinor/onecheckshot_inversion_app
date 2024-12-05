
import urllib.request, json
from dotenv import load_dotenv
import os

########### Python 3.2 #############
load_dotenv('prod.env')
try:
    url = "https://api.gateway.equinor.com/smda/v2.0/smda-api/wellbore-plan-survey-samples?_items=1&unique_wellbore_identifier=NO%201%2F9-1"
    url = os.getenv("API_BASE_URL")
    hdr ={
    # Request headers
    'Cache-Control': 'no-cache',
    'Authorization': os.getenv("AUTHORIZATION"),
    'Ocp-Apim-Subscription-Key': os.getenv("API_SUBSCRIPTION_KEY"),
    }

    req = urllib.request.Request(url, headers=hdr)

    req.get_method = lambda: 'GET'
    response = urllib.request.urlopen(req)
    print(response.getcode())
    print(response.read())
except Exception as e:
    print(e)