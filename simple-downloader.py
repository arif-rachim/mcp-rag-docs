#!/usr/bin/env python3
import os
import sys
from urllib.parse import quote
import requests
from requests_ntlm import HttpNtlmAuth
import urllib3
from config import DOWNLOADS_FOLDER as OUTPUT_FOLDER

# Configuration (SharePoint-specific)
SITE_URL = 'https://sharepoint.company.com/sites/mysite'
USERNAME = 'username'
PASSWORD = 'password'
DOMAIN = 'DOMAIN'
LIBRARY_NAME = 'Shared Documents'
MAX_RESULTS = 5000
IGNORE_CERT = True

if IGNORE_CERT:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if __name__ == '__main__':
    # Setup session
    session = requests.Session()
    session.auth = HttpNtlmAuth(f"{DOMAIN}\\{USERNAME}", PASSWORD)
    session.verify = not IGNORE_CERT

    # Fetch file list
    api_url = f"{SITE_URL}/_api/web/lists/getbytitle('{quote(LIBRARY_NAME)}')/items"
    query = f"?$filter=FSObjType eq 0&$select=FileRef,FileLeafRef&$top={MAX_RESULTS}"

    print("Fetching files...")
    response = session.get(api_url + query, headers={'Accept': 'application/json;odata=verbose'})
    files = response.json().get('d', {}).get('results', [])
    print(f"Found {len(files)} files")

    # Download files
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    success = 0

    for i, f in enumerate(files, 1):
        name = f.get('FileLeafRef', '')
        path = f.get('FileRef', '')
        if not name or not path:
            continue

        print(f"[{i}/{len(files)}] {name}")
        try:
            content = session.get(SITE_URL + path).content
            with open(os.path.join(OUTPUT_FOLDER, name.translate(str.maketrans('<>:"/\\|?*', '_________'))), 'wb') as file:
                file.write(content)
            success += 1
        except Exception as e:
            print(f"  Failed: {e}")

    print(f"Done! {success}/{len(files)} files downloaded")
