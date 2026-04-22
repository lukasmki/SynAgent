import os
import time
import httpx
import json
import pathlib
import tempfile


class ChemspaceTokenManager:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.chem-space.com/",
    ):
        self.api_key = api_key or os.environ.get("CHEMSPACE_API_KEY")
        self.base_url = base_url
        self.auth_url = base_url + "auth/token"

        if not self.api_key:
            raise ValueError(
                "CHEMSPACE_API_KEY is missing. Set it in your environment or pass api_key explicitly."
            )

        self.access_token = None
        self.expires_at = 0

        self.token_cache = (
            pathlib.Path(tempfile.gettempdir()) / ".token_cache"
        ).resolve()
        if self.token_cache.exists():
            with open(self.token_cache, "r") as fp:
                data = json.load(fp)
            self.access_token = data["access_token"]
            self.expires_at = float(data["expires_at"])

    async def refresh_token(self):
        async with httpx.AsyncClient() as client:
            r = await client.get(
                self.auth_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                timeout=10,
            )

        r.raise_for_status()
        data = r.json()

        self.access_token = data["access_token"]
        self.expires_at = time.time() + data["expires_in"] - 30

        with open(self.token_cache, "w") as fp:
            json.dump(
                {"access_token": self.access_token, "expires_at": self.expires_at},
                fp,
            )

    async def get_token(self):
        if time.time() >= self.expires_at:
            await self.refresh_token()
        return self.access_token