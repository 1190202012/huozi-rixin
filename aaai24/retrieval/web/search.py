import json
from typing import List

import requests

# SERPAPI_KEY = "b221e975b82bdeb07f3a81708227451a41c2ff296700648fcee86f0fae85644e"
# SERPAPI_KEY = "e71bdb1b52b858a85a698003bae10266f769bcb37e2640d60c73d1b8eda7fdbb"
SERPAPI_KEY = "4bfeb6d7403f6e50d878018b66054e79334622b37da7c76f36df29073396b134"


SERPER_URL = "https://google.serper.dev/search"
SERPER_HEADERS = {
    'X-API-KEY': '05faa66bfe24b1afe114c2c22add07b114de0e3a',
    'Content-Type': 'application/json'
}


def filter_urls(urls):
    urls_new = []
    for url in urls:
        if url in urls_new or f"http://{url}" in urls_new or f"https://{url}" in urls_new:
            continue
        if not url.startswith("http"):
            url = f"http://{url}"
        urls_new.append(url)
    return urls_new


def serp_api(query: str, language: str):
    params = {"engine": "bing", "q": query, "api_key": SERPAPI_KEY}
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        raise Exception("serpapi returned %d\n%s" % (response.status_code, response.text))
    result = response.json()
    urls, info = [], []
    if "organic_results" not in result:
        return [], []
    
    for item in result['organic_results']:
        if "title" not in item or "link" not in item or "snippet" not in item:
            continue
        info.append({"title": item['title'], "url": item['link'], "snip": item['snippet']})
        urls.append(item["link"])
        
    return urls, info


def serper_api(query: str, language: str):
    # gl代表地域
    if language == "zh":
        payload = json.dumps({"q": query, "gl": "cn", "hl": "zh-cn"})
    else:
        payload = json.dumps({"q": query, "gl": "us", "hl": "en"})

    try:
        response = requests.request("POST", SERPER_URL, headers=SERPER_HEADERS, data=payload, verify=False, timeout=12)
    except:
        try:
            response = requests.request("POST", SERPER_URL, headers=SERPER_HEADERS, data=payload, verify=False, timeout=12)
        except:
            return [], {"organic": [], "peopleAlsoAsk": [], "relatedSearches":[]}
        
    if response.status_code != 200:
        raise Exception("serper api returned %d\n%s" % (response.status_code, response.text))

    result = response.json()
    urls = []
    info = {"organic": [], "peopleAlsoAsk": result.get("peopleAlsoAsk", []),
            "relatedSearches": result.get("relatedSearches", [])}

    for item in result['organic']:
        if "title" not in item or "link" not in item or "snippet" not in item:
            continue
        urls.append(item["link"])
        info["organic"].append(item)

    return urls, info


def serper_batch_api(queries: List[str], language_list: List[str]):
    payloads = json.dumps([{"q": query, "gl": "cn", "hl": "zh-cn" if language == "zh" else "en"}
                           for query, language in zip(queries, language_list)])
    response = requests.request("POST", SERPER_URL, headers=SERPER_HEADERS, data=payloads)

    if response.status_code != 200:
        raise Exception("serper api returned %d\n%s" % (response.status_code, response.text))

    results = response.json()
    urls_list, infos = [], []

    for i in range(len(queries)):
        result = results[i]
        infos.append({"organic": [], "peopleAlsoAsk": result.get("peopleAlsoAsk", []),
                      "relatedSearches": result.get("relatedSearches", [])})
        urls_list.append([])

        for item in result['organic']:
            if "title" not in item or "link" not in item or "snippet" not in item:
                continue
            urls_list[i].append(item["link"])
            infos[i]["organic"].append(item)

    return urls_list, infos
