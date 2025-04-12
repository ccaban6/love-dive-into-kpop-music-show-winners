import re
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify

app = Flask(__name__)

def search_melon_songs(query):
    # Build the search URL
    search_url = f"https://www.melon.com/search/total/index.htm?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    
    # Locate the form with id 'frm_songList'
    form = soup.find("form", id="frm_songList")
    if not form:
        return None
        
    # Locate the table within the form
    table = form.find("table")
    if not table:
        return None
    
    # Get all rows in the table; assuming the first row is the header row
    rows = table.find_all("tr")
    songs = []
    
    for row in rows[1:]:
        columns = row.find_all("td")
        # We expect at least 4 columns based on the headers: NO, Song Name, artist, album, etc.
        if len(columns) >= 4:
            song_name = columns[1].get_text(strip=True)
            artist = columns[2].get_text(strip=True)
            album = columns[3].get_text(strip=True)
            
            # Extract the song id from the <a> tag in the row using regex
            a_tag = row.find("a", class_="btn btn_icon_detail")
            song_id = None
            if a_tag and a_tag.get("href"):
                href = a_tag["href"]
                match = re.search(r"melon\.link\.goSongDetail\('(\d+)'\)", href)
                if match:
                    song_id = match.group(1)
            
            songs.append({
                "song_name": song_name,
                "artist": artist,
                "album": album,
                "song_id": song_id
            })
    
    return songs

## http://127.0.0.1:5000/api/v1/search?q=
@app.route('/api/v1/search', methods=['GET'])
def search_endpoint():
    query = request.args.get('q')
    if not query:
        return jsonify({"error": "Missing search query parameter 'q'"}), 400

    results = search_melon_songs(query)
    if results is None:
        return jsonify({"error": "Failed to retrieve search results"}), 500

    return jsonify(results)
if __name__ == '__main__':
    app.run(debug=True)