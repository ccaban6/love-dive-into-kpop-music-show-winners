import os
import re
import requests
import logging
import csv
from bs4 import BeautifulSoup
from typing import Dict, List

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def sanitize_filename(name: str) -> str:
    name = name.replace(" ", "_")
    return re.sub(r"[^\w\-_.]", "", name)

def normalize_show_name(name: str) -> str:
    return name.strip().lower().replace("the ", "").title()

def parse_metadata_row(row) -> (str, str):
    cells = row.find_all("td")
    if len(cells) >= 3:
        date = cells[1].get_text(strip=True)
        show = normalize_show_name(cells[2].get_text(strip=True).split(" #")[0])
        return show, date
    return "", ""

def parse_detail_rows(nested_rows, show_name: str, date: str) -> (List[str], List[List[str]]):
    header_cells = nested_rows[0].find_all(["th", "td"])
    header = ["Show", "Date"] + [cell.get_text(strip=True) for cell in header_cells]
    header[2:5] = ["Placement", "Artist", "Song"]  # Rename for consistency

    rows = []
    for detail_row in nested_rows[1:]:
        data_cells = detail_row.find_all("td")
        if data_cells:
            row_data = [cell.get_text(strip=True) for cell in data_cells]
            rows.append([show_name, date] + row_data)
    return header, rows

def scrape_kpopdb_year(year: int, save_dir: str = "data/tables") -> Dict[str, Dict[str, List]]:
    base_url = "https://www.kpopdb.net/en/wins.php"
    url = f"{base_url}?y={year}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to retrieve data for {year}: {e}")
        return {}

    soup = BeautifulSoup(response.text, "html.parser")
    all_rows = soup.select("table.table tbody tr")
    
    data_by_show = {}
    current_show = ""
    current_date = ""

    for row in all_rows:
        row_id = row.get("id", "")

        if row_id.startswith("detail"):
            nested_table = row.find("table")
            if not nested_table:
                continue

            nested_rows = nested_table.find_all("tr")
            if nested_rows:
                header, rows = parse_detail_rows(nested_rows, current_show, current_date)
                if current_show not in data_by_show:
                    data_by_show[current_show] = {"header": header, "rows": []}
                data_by_show[current_show]["rows"].extend(rows)

        else:
            current_show, current_date = parse_metadata_row(row)
            current_date += f"-{year}"  # Append year if missing

    os.makedirs(save_dir, exist_ok=True)

    for show, data in data_by_show.items():
        filename = os.path.join(save_dir, f"{year}_{sanitize_filename(show)}.csv")
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(data["header"])
            writer.writerows(data["rows"])
        logging.info(f"Wrote {filename} with {len(data['rows'])} rows")

    return data_by_show