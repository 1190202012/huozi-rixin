import asyncio
import warnings

import aiohttp
import async_timeout
from loguru import logger
from urllib3.exceptions import InsecureRequestWarning

headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 "
                  "Safari/537.36",
    'accept-language': "zh-CN,zh;q=0.9,en-CN;q=0.8,en;q=0.7"
}

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

proxy = "http://127.0.0.1:7895"


async def request_one_url(session, url):
    try:
        async with async_timeout.timeout(10):
            response = await session.get(url, headers=headers, proxy=proxy)
            html = await response.text(errors="ignore")
            return html
    except Exception as aiohttp_exception:
        logger.warning(f"aiohttp fetch {url} error: {aiohttp_exception}")
        html = ""
        # pass

    # cmd = f"curl --insecure --connect-timeout 5 {url}"
    # proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    # # stdout, stderr = await proc.communicate()
    
    # try:
    #     stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
    # except asyncio.TimeoutError:
    #     proc.kill()
    #     await proc.wait()
    #     stdout, stderr = "", "timeout"
    
    # if proc.returncode != 0:
    #     logger.warning(f"curl {url} return {proc.returncode}")
    
    # # if stderr:
    # #     print(f'[stderr]\n{stderr.decode(encoding="UTF-8", errors="ignore")}')
    
    # if stdout:
    #     html = stdout.decode(encoding="UTF-8", errors="ignore")
    # else:
    #     html = ""

    return html


async def main(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [request_one_url(session, url) for url in urls]
        html_texts = await asyncio.gather(*tasks)
        return html_texts

def fetch(urls):
    # asyncio.run() 通常是为单进程的事件循环设计的，适合demo
    html_texts = asyncio.run(main(urls))
    
    # 适合多进程爬虫，适合eval
    # loop = asyncio.get_event_loop()
    # html_texts = loop.run_until_complete(main(urls))
    return {urls[i]: html_texts[i] for i in range(len(urls))}

if __name__ == "__main__":
    urls = ["https://news.un.org/zh/story/2023/11/1123617", "https://www.bbc.com/zhongwen/simp/world-67044823", "https://www.mfa.gov.cn/fyrbt_673021/202310/t20231023_11166298.shtml", "https://www.bbc.com/zhongwen/trad/world-67038483","https://news.cctv.com/zhibo/tuwen/202310zjbyxylct/index.shtml", "https://zh.wikipedia.org/zh-hans/%E4%BB%A5%E5%B7%B4%E5%86%B2%E7%AA%81", "http://hk.ocmfa.gov.cn/chn/gjlc/202310/t20231030_11170661.htm", "https://www.fmprc.gov.cn/fyrbt_673021/202311/t20231107_11175342.shtml", "http://bata.china-consulate.gov.cn/fyrth/202310/t20231009_11158229.htm", "http://world.people.com.cn/n1/2023/1107/c1002-40112630.html"]
    ret = fetch(urls)
    print(ret)
    
# import asyncio
# import subprocess
# import warnings

# import aiohttp
# import async_timeout
# from loguru import logger
# from urllib3.exceptions import InsecureRequestWarning

# headers = {
#     'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 "
#                   "Safari/537.36",
#     'accept-language': "zh-CN,zh;q=0.9,en-CN;q=0.8,en;q=0.7"
# }

# warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# proxy = "http://127.0.0.1:7890"


# async def request_one_url(session, url):
#     try:
#         async with async_timeout.timeout(15):
#             response = await session.get(url, headers=headers, proxy=proxy)
#             html = await response.text(errors="ignore")
#             return html
#     except Exception as aiohttp_exception:
#         # logger.warning(f"aiohttp fetch {url} error: {aiohttp_exception}")
#         pass

#     # try:
#     #     response = requests.get(url, headers=headers, verify=False, timeout=5, proxies={"http": proxy, "https": proxy})
#     #     response.encoding = response.apparent_encoding
#     #     html = response.text
#     #     return html
#     # except Exception as request_exception:
#     #     logger.warning(f"requests fetch {url} error: {request_exception}")
    
#     cmd = f"curl --insecure --connect-timeout 10 {url}"
#     proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
#     # stdout, stderr = await proc.communicate()
    
#     try:
#         stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
#     except asyncio.TimeoutError:
#         proc.kill()
#         await proc.wait()
#         stdout, stderr = "", "timeout"
    
#     if proc.returncode != 0:
#         logger.warning(f"curl {url} return {proc.returncode}")
    
#     # if stderr:
#     #     print(f'[stderr]\n{stderr.decode(encoding="UTF-8", errors="ignore")}')
    
#     if stdout:
#         html = stdout.decode(encoding="UTF-8", errors="ignore")
#     else:
#         html = ""
    
#     # process = subprocess.Popen(["curl", "--insecure", "--connect-timeout", "10", url], stdout=subprocess.PIPE,
#     #                            stderr=subprocess.DEVNULL, text=True)
#     # html, error = process.communicate()

#     # if error is not None:
#     #     logger.warning(f"unable to fetch {url}. curl error: {error}")

#     # if html is None:
#     #     html = ""

#     return html


# html_texts = []
# async def main(urls):
#     global html_texts
#     async with aiohttp.ClientSession() as session:
#         tasks = [request_one_url(session, url) for url in urls]
#         html_texts = await asyncio.gather(*tasks)

# def fetch(urls):
#     # asyncio.run() 通常是为单进程的事件循环设计的
#     # html_texts = asyncio.run(main(urls)) 
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(main(urls))
#     return {urls[i]: html_texts[i] for i in range(len(urls))}
