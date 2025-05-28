import os
import requests
import hashlib
import pandas as pd
from ast import literal_eval
from urllib.parse import urlparse, parse_qs, unquote
from tqdm import tqdm
from pathlib import Path
import ast

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}



def get_extension_from_url(url):
    parsed = urlparse(url)
    path = parsed.path

    if "_next/image" in path:
        qs = parse_qs(parsed.query)
        real_url = qs.get("url")
        if real_url:
            real_url = unquote(real_url[0])
            return get_extension_from_url(real_url)

    # Інакше — працюємо зі звичайним шляхом
    ext = os.path.splitext(path)[1].lower()
    return ext[1:] if ext.startswith('.') else ext

def download_media(media_urls, media_folder="data/media"):
    os.makedirs(media_folder, exist_ok=True)
    local_paths = []

    for url in media_urls:
        if not isinstance(url, str):
            continue

        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                ext = get_extension_from_url(url)
                if not ext:
                    print(f"⚠️ No extension found in URL: {url}")
                    continue

                filename = hashlib.md5(url.encode()).hexdigest() + "." + ext
                filepath = Path(media_folder) / filename
                filepath_str = filepath.as_posix()

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                local_paths.append(filepath_str)

        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            continue

    return local_paths



def process_dataframe(df):
    updated_media = []

    for raw_urls in tqdm(df["media_urls"], desc="📥 Processing media"):
        try:
            urls = literal_eval(raw_urls) if isinstance(raw_urls, str) else raw_urls
        except Exception as e:
            print(f"⚠️ Invalid format: {raw_urls} — {e}")
            urls = []

        flat_urls = []
        for u in urls:
            if isinstance(u, list):
                flat_urls.extend(u)
            else:
                flat_urls.append(u)

        local_paths = download_media(flat_urls)
        updated_media.append(local_paths)

    df["media_urls"] = updated_media
    return df

if __name__ == "__main__":
    df=pd.read_csv('data/the_batch_articles.csv', converters={'media_urls': ast.literal_eval})
    df=process_dataframe(df)
    df['content'] = df['content'].str.replace(
        "✨ New course! Enroll in Reinforcement Fine-Tuning LLMs with GRPO\n",
        "",
        regex=False
    )
    df.to_csv("data/articles_with_local_images.csv")
    print("Media downloading complete")