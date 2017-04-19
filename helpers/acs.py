'''
Helper Functions
'''
import requests
from requests.auth import HTTPBasicAuth

API_KEY = '9032d3c94c7f4afe905da54f889af02a6b51f63f'
request_url = "http://citysdk.commerce.gov"

variables = ['income','population','poverty','median_contract_rent','education_bachelors','poverty_family']

def acs_data (zip_code):
    request_obj = {
        'level': 'tract',
        'zip': int(zip_code),
        'sublevel': False,
        'api': 'acs5',
        'year': 2014,
        'variables': variables
        }

    data = None

    response = requests.post(request_url, auth=HTTPBasicAuth(API_KEY, None), json=request_obj)
    if response:
        data = response.json()
        data = data['features'][0]['properties']

    return data
