from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from ..guardrails import check_url

USER_AGENT = "my-agents/0.1 (+https://github.com/Kp759/my-agents)"


class HTTPStatusError(RuntimeError):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} from {url}")
        self.status = status


def get(url: str, params: dict | None = None, timeout: int = 20) -> bytes:
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    check_url(url)  # egress allowlist: https and known hosts only
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise HTTPStatusError(e.code, url.split("?")[0]) from e


def get_json(url: str, params: dict | None = None, timeout: int = 20):
    return json.loads(get(url, params, timeout))
