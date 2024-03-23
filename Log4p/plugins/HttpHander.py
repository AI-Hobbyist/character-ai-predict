import logging
import requests
import httpx

class HTTPhandler(logging.Handler):
    def __init__(self, url):
        super().__init__()
        self.url = url

    def emit(self, record):
        log_entry = self.format(record)
        payload = {'log': log_entry}
        try:
            response = requests.post(self.url, json=payload)
            if not response.ok:
                raise ValueError(response.text)
        except Exception as e:
            logging.error("Failed to send log to %s: %s", self.url, e)

class AsyncHTTPhandler(logging.Handler):
    def __init__(self, url):
        super().__init__()
        self.url = url

    async def emit(self, record):
        log_entry = self.format(record)
        payload = {'log': log_entry}
        try:
            async with httpx.AsyncClient(timeout=120,max_redirects=5) as client:
                response = await client.post(self.url, json=payload)
                if not response.is_success:
                    raise ValueError(await response.text())
        except Exception as e:
            logging.error("Failed to send log to %s: %s", self.url, e)

