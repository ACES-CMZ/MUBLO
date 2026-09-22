#!/usr/bin/env python3
"""
Download ALMA data for member OUS uid://A001/X3845/Xa2f
(2025.1.00021.S, G0.02467-0.0727, Band 9 TM1 = extended 12m config;
 the companion TM2/compact execution is uid://A001/X3845/Xa31, already on disk).

Query: https://almascience.nrao.edu/aq?member_ous_id=uid://A001/X3845/Xa2f

Uses astroquery's selective (in-tar) download so we can grab the small continuum
products first without pulling the full 200 GB science tar.

Usage:
    python download_ous_Xa2f.py            # continuum + mfs products only (~0.25 GB)
    python download_ous_Xa2f.py --cubes    # also the per-spw cubes (~160 GB)
    python download_ous_Xa2f.py --list     # just print what is available
"""
import argparse
import os
import shutil
import sys

import keyring
import numpy as np
from astroquery.alma import Alma

USERNAME = "keflavich"
OUS_ID = "uid://A001/X3845/Xa2f"

BASE = os.path.dirname(os.path.abspath(__file__))
DESTDIR = os.path.join(
    BASE,
    "2025.1.00021.S",
    "science_goal.uid___A001_X3845_Xa2d",
    "group.uid___A001_X3845_Xa2e",
    "member.uid___A001_X3845_Xa2f",
    "product",
)


def get_alma():
    alma = Alma()
    alma.archive_url = "https://almascience.nrao.edu"
    password = keyring.get_password("almascience.nrao.edu", USERNAME)
    if password is None:
        alma.login(USERNAME)
    else:
        alma.login(USERNAME, store_password=False)
    return alma


def size_gb(row):
    cl = row["content_length"]
    return 0.0 if cl is np.ma.masked else float(cl) / 1e9


def wanted(fn, cubes=False):
    """Science image products for the target (skip calibrator check-source images)."""
    if "_sci" not in fn or not fn.endswith((".fits", ".fits.gz")):
        return False
    if ".cube." in fn:
        return cubes
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubes", action="store_true",
                        help="also download the per-spw cubes (~160 GB)")
    parser.add_argument("--list", action="store_true",
                        help="list available files and exit")
    args = parser.parse_args()

    alma = get_alma()
    print(f"Fetching data info for {OUS_ID} ...")
    info = alma.get_data_info(OUS_ID, expand_tarfiles=True)

    if args.list:
        for row in info:
            print(f"{size_gb(row):9.3f} GB  {row['access_url'].split('/')[-1]}")
        print(f"TOTAL: {sum(size_gb(r) for r in info):.1f} GB")
        return

    urls = [row["access_url"] for row in info
            if wanted(row["access_url"].split("/")[-1], cubes=args.cubes)]
    total = sum(size_gb(row) for row in info
                if wanted(row["access_url"].split("/")[-1], cubes=args.cubes))

    print(f"Selected {len(urls)} files, {total:.2f} GB total:")
    for url in urls:
        print(f"  {url.split('/')[-1]}")

    os.makedirs(DESTDIR, exist_ok=True)
    print(f"\nDownloading to {DESTDIR}")
    downloaded = alma.download_files(urls, savedir=DESTDIR, cache=True)

    # astroquery caches under savedir; make sure the files land flat in product/
    for path in downloaded:
        if os.path.dirname(path) != DESTDIR and os.path.exists(path):
            target = os.path.join(DESTDIR, os.path.basename(path))
            if not os.path.exists(target):
                shutil.move(path, target)

    print(f"\nDone. {len(downloaded)} files in {DESTDIR}")


if __name__ == "__main__":
    sys.exit(main())
