import os

import requests
from bs4 import BeautifulSoup
from datetime import date
from tqdm import tqdm
import re
from selenium import webdriver


DATA_FOLDER = "/home/m1nhd3n/Works/SideProjects/Document2Graph/data"
URL = "https://edition.cnn.com/politics"
BASE_URL = "https://edition.cnn.com/"


def get_urls_today():
    html = requests.get(URL).content
    soup = BeautifulSoup(html, "html.parser")

    links = soup.find_all("a", class_="container__link")

    return list(set([link.get("href") for link in links if "https" not in link]))


def get_article(url: str):
    html = requests.get(url).content

    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1", id="maincontent")
    paragraphs = soup.find_all("p", class_="paragraph")
    content = "".join([p.get_text() for p in paragraphs])
    content = re.sub(r"\n\s+", "\n", content)
    content = re.sub(r"\s+\n", "\n", content)
    content = re.sub(r"^\n", "", content)
    return title.get_text(strip=True), content


# option = webdriver.FirefoxOptions()
# option.add_argument("--headless")
# drv = webdriver.Firefox(options=option)

urls = get_urls_today()
today = str(date.today()).replace('-', '_')
for u in tqdm(urls):
    try:
        t, c = get_article(BASE_URL + u)
        if t is None or c is None:
            print(URL + u)
            continue
        with open(os.path.join(DATA_FOLDER, "_".join([ti for ti in t.split()]) + ".txt"), 'w') as f:
            f.write(c)
    except Exception:
        continue
