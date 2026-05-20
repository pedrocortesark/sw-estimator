import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

async def main():
    client = AsyncOpenAI()
    try:
        stream = await client.responses.create(
            model="gpt-4o-mini",
            instructions="You are a bot",
            input="Say hello",
            stream=True
        )
        async for chunk in stream:
            print(chunk)
    except Exception as e:
        print("Error:", e)

asyncio.run(main())
