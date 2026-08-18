import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from groq import AsyncGroq

async def main():
    client = AsyncGroq()
    models = await client.models.list()
    for m in models.data:
        print(m.id)

if __name__ == "__main__":
    asyncio.run(main())
