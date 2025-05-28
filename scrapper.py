import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import os
import time
from tqdm import tqdm
import time

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}


def fetch_all_tag_urls():
    base_url = "https://www.deeplearning.ai/the-batch/"
    all_tag_urls = set()
    page_num = 1
    while True:
        url = f"{base_url}page/{page_num}/"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            break
        soup = BeautifulSoup(response.text, 'html.parser')
        links_found = 0
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/the-batch/tag/' in href and not any(skip in href for skip in [
                '/tag/letters/', '/tag/data-points/', '/tag/research/',
                '/tag/business/', '/tag/science/', '/tag/culture/',
                '/tag/hardware/', '/tag/ai-careers/'
            ]):
                full_url = urljoin(base_url, href)
                if full_url not in all_tag_urls:
                    all_tag_urls.add(full_url)
                    links_found += 1
        if links_found == 0:
            break
        page_num += 1
        time.sleep(0.25)
    return list(all_tag_urls)


def get_valid_article_links():
    article_links = set()
    tag_urls = fetch_all_tag_urls()
    for tag_url in tqdm(tag_urls, desc="Scanning tag pages"):
        try:
            response = requests.get(tag_url, headers=HEADERS)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/the-batch/' in href and '/tag/' not in href and '/issue-' not in href:
                    full_url = urljoin(tag_url, href)
                    article_links.add(full_url)
        except Exception as e:
            print(f"Error fetching {tag_url}: {e}")
        time.sleep(0.25)
    return list(article_links)


def is_unwanted_image(url: str) -> bool:
    unwanted_keywords = [
        'logo', 'banner', 'ads', 'ad-', 'side', 'sidebar',
        'dlai-batch-logo', 'sprite', 'icon'
    ]
    return (
            url.startswith("data:image/gif;base64,R0lGODlhAQABAIAAAAAA") or
            (url.startswith("data:image/svg+xml") and "%3csvg" in url and "/%3e" in url) or
            any(keyword in url.lower() for keyword in unwanted_keywords)
    )


def extract_article_data(article_url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = requests.get(article_url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                break
            else:
                print(f"Attempt {attempt + 1}/{retries}: Status code {response.status_code} for {article_url}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{retries}: Error fetching {article_url}: {e}")

        if attempt < retries - 1:
            time.sleep(delay)
    else:
        print(f"❌ Could not fetch {article_url} after {retries} attempts.")
        return None

    try:
        soup = BeautifulSoup(response.text, 'html.parser')

        title_tag = soup.find('h1')
        title = title_tag.text.strip() if title_tag else "No title"

        paragraphs = [p.text.strip() for p in soup.find_all('p') if p.text.strip()]
        content = "\n".join(paragraphs)

        media_urls = []
        for tag in soup.find_all(['img', 'video']):
            src = tag.get('src')
            if src:
                full_url = urljoin(article_url, src)
                if not is_unwanted_image(full_url):
                    media_urls.append(full_url)

        return {
            'url': article_url,
            'title': title,
            'content': content,
            'media_urls': media_urls
        }

    except Exception as e:
        print(f"❌ Error parsing {article_url}: {e}")
        return None


def scrape_the_batch_articles(limit=None):
    article_links = get_valid_article_links()
    if limit:
        article_links = article_links[:limit]
    articles = []
    skipped = 0
    for url in tqdm(article_links, desc="Scraping articles"):
        data = extract_article_data(url)
        if data:
            articles.append(data)
        else:
            skipped += 1
    print(f"✅ Зібрано {len(articles)} статей, пропущено {skipped}")
    return pd.DataFrame(articles)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = scrape_the_batch_articles()
    df.to_csv("data/the_batch_articles.csv", index=False)
    print("Scraping complete. Articles saved to data/the_batch_articles.csv")
