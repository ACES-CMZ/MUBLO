#!/usr/bin/env python3
"""
Download ALMA data for OUS ID uid://A001/X3845/Xa31
Query: https://almascience.nrao.edu/aq?member_ous_id=uid://A001/X3845/Xa31
"""

import requests
import os
import sys
import re
from pathlib import Path
from urllib.parse import urljoin

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: BeautifulSoup not installed. Install with: pip install beautifulsoup4")
    sys.exit(1)

try:
    import keyring
except ImportError:
    keyring = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

USERNAME = "keflavich"
OUS_ID = "uid://A001/X3845/Xa31"

def get_password():
    """Get password from keyring or prompt"""
    if keyring:
        password = keyring.get_password("almascience.nrao.edu", USERNAME)
        if password:
            print(f"Using password from keyring for {USERNAME}")
            return password

    password = input(f"Enter password for ALMA account {USERNAME}: ")

    # Optionally save to keyring
    if keyring:
        try:
            keyring.set_password("almascience.nrao.edu", USERNAME, password)
            print("Password saved to keyring")
        except:
            pass

    return password

def login_to_alma(username, password):
    """Create authenticated session with ALMA archive"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    })

    login_url = "https://almascience.nrao.edu/aq"

    login_data = {
        'username': username,
        'password': password,
    }

    print("Authenticating...")
    response = session.post(login_url, data=login_data, allow_redirects=True, timeout=30)

    if response.status_code != 200:
        print(f"Warning: Login returned status {response.status_code}")

    return session

def fetch_dataset_info(session, ous_id):
    """Fetch dataset information from archive query"""
    query_url = f"https://almascience.nrao.edu/aq?member_ous_id={ous_id}"

    print(f"Fetching: {query_url}")
    response = session.get(query_url, timeout=30)
    response.raise_for_status()

    # Save response for debugging
    with open("query_response.html", "w") as f:
        f.write(response.text)

    print(f"Response size: {len(response.content)} bytes")

    soup = BeautifulSoup(response.content, 'html.parser')

    download_links = []

    # Approach 1: Look for direct dataPortal links
    for link in soup.find_all('a', href=True):
        href = link['href']
        if 'dataPortal' in href:
            full_url = href if href.startswith('http') else urljoin('https://almascience.nrao.edu', href)
            download_links.append({
                'url': full_url,
                'name': link.text.strip() or href.split('/')[-1]
            })

    # Approach 2: Regex search for dataPortal URLs in page
    if not download_links:
        pattern = r'https?://[^\s"<>]*almascience[^\s"<>]*dataPortal[^\s"<>]*\.(tar|tgz|fits|ms|txt)'
        for match in re.finditer(pattern, response.text):
            url = match.group(0)
            download_links.append({
                'url': url,
                'name': url.split('/')[-1]
            })

    # Approach 3: Look for .tar / .tgz files
    if not download_links:
        pattern = r'https?://[^\s"<>]*\.(tar|tgz)'
        for match in re.finditer(pattern, response.text):
            url = match.group(0)
            download_links.append({
                'url': url,
                'name': url.split('/')[-1]
            })

    # Remove duplicates
    seen = set()
    unique_links = []
    for link in download_links:
        if link['url'] not in seen:
            seen.add(link['url'])
            unique_links.append(link)

    return unique_links

def download_file(session, url, filename=None):
    """Download file with resume capability"""
    if filename is None:
        filename = url.split('/')[-1]

    # Check if file exists
    existing_size = os.path.getsize(filename) if os.path.exists(filename) else 0

    headers = {}
    mode = 'ab'
    if existing_size > 0:
        headers['Range'] = f'bytes={existing_size}-'
    else:
        mode = 'wb'

    response = session.get(url, stream=True, headers=headers, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0)) + existing_size

    if tqdm:
        pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=filename, initial=existing_size)
    else:
        print(f"Downloading {filename}...")

    try:
        with open(filename, mode) as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    if tqdm:
                        pbar.update(len(chunk))
    finally:
        if tqdm:
            pbar.close()

    print(f"✓ Downloaded: {filename}")

def main():
    print(f"ALMA Data Download for OUS: {OUS_ID}")
    print("=" * 50)
    print()

    # Get password
    password = get_password()

    # Authenticate
    session = login_to_alma(USERNAME, password)

    # Fetch dataset info
    try:
        links = fetch_dataset_info(session, OUS_ID)
    except Exception as e:
        print(f"Error fetching dataset info: {e}")
        print("Response saved to query_response.html")
        sys.exit(1)

    if not links:
        print("Error: No download links found in archive query.")
        print("Saved response to query_response.html for inspection.")
        print()
        print("Possible solutions:")
        print("  - Check if OUS ID is correct")
        print("  - Verify account has access to this data")
        print("  - Visit https://almascience.nrao.edu/aq?member_ous_id=" + OUS_ID)
        print("  - Download files manually from browser")
        sys.exit(1)

    print(f"Found {len(links)} files:")
    for i, link in enumerate(links, 1):
        print(f"  [{i}] {link['name']}")
    print()

    # Download files
    failed = []
    for link in links:
        try:
            download_file(session, link['url'], link['name'])
        except Exception as e:
            print(f"✗ Error downloading {link['name']}: {e}")
            failed.append(link['name'])

    if failed:
        print()
        print(f"Failed downloads ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
        print("Run script again to retry.")

if __name__ == '__main__':
    main()
