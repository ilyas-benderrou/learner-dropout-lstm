"""
Download the Open University Learning Analytics Dataset (OULAD) into data/raw/.

OULAD - Kuzilek, Hlosta & Zdrahal (2017), Scientific Data 4:170171.
Licence: CC BY 4.0.

The historical download link on the Open University website
(analyse.kmi.open.ac.uk/open_dataset) now redirects to a page whose zip link
returns 404 (checked on 2026-11-12). We therefore use the copy of the same
dataset hosted by the UCI Machine Learning Repository (dataset id 349), which
the Open University dataset page also points to.

Usage:  python src/download_data.py
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
URLS = [
    # Official OU CDN link listed on research.stem.open.ac.uk/ouanalyse/dataset/
    "https://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip",
    # UCI Machine Learning Repository mirror (same files, CC BY 4.0)
    "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip",
]
NEEDED = ["studentInfo.csv", "studentRegistration.csv", "studentVle.csv", "courses.csv"]


def data_is_present() -> bool:
    return all((RAW_DIR / name).exists() for name in NEEDED)


def download() -> None:
    """Try each URL until one gives a valid zip, then extract the CSV files."""
    if data_is_present():
        print(f"OULAD already present in {RAW_DIR}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "oulad.zip"
    for url in URLS:
        try:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, zip_path)
            if not zipfile.is_zipfile(zip_path):
                raise ValueError("not a zip file (probably an error page)")
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    name = Path(member).name
                    if name.endswith(".csv"):  # extract flat, ignore folders
                        (RAW_DIR / name).write_bytes(zf.read(member))
            zip_path.unlink()
            if data_is_present():
                print("Done.")
                return
        except Exception as exc:  # noqa: BLE001 - we just try the next mirror
            print(f"  failed: {exc}")
    raise RuntimeError(
        "Could not download OULAD automatically. Download it manually from "
        "https://archive.ics.uci.edu/dataset/349 and unzip the CSV files into data/raw/."
    )


if __name__ == "__main__":
    download()
