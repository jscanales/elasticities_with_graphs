# Authentication
# https://disc.gsfc.nasa.gov/datasets/M2T1NXFLX_5.12.4/summary


class Nasa:
    user = "YOUR_USERNAME"
    password = "YOUR_PASSWORD"
    header = {
        "Host": "search.earthdata.nasa.gov",
        "User-Agent": "python-requests/2.28.1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en,de-DE;q=0.9,de;q=0.8,en-US;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": r"https://search.earthdata.nasa.gov/auth_callback?redirect=https%3A%2F%2Fsearch.earthdata.nasa.gov%2Fsearch%3Fq%3Dwind%2520speed%26ee%3Dprod&jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6Mjg4Nzk5LCJ1c2VybmFtZSI6InJhZmZhZWxlLnNnIiwicHJlZmVyZW5jZXMiOnt9LCJ1cnNQcm9maWxlIjp7ImZpcnN0X25hbWUiOiJSYWZmYWVsZSJ9LCJlYXJ0aGRhdGFFbnZpcm9ubWVudCI6InByb2QiLCJpYXQiOjE2NjcxNDk5Mjh9.ghPa8ztmYuh6eozpVPkVQwskeYxe0u0ioxc8OKzQ7PQ",
        "Cookie": "YOUR_COOKIE",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",   
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "TE": "trailers",
    }


# ENTSOE
class Entsoe:
    token = "YOUR_TOKEN"
