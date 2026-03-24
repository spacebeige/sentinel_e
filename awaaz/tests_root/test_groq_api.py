import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv("awaaz/.env")

async def test():
    api_key = os.getenv("GROQ_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}
    
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        with open("test_audio.wav", "rb") as f:
            audio_bytes = f.read()
        data.add_field('file', audio_bytes, filename='test_audio.wav', content_type='audio/wav')
        data.add_field('model', 'whisper-large-v3')
        data.add_field('response_format', 'verbose_json')
        
        async with session.post(url, headers=headers, data=data) as response:
            result = await response.json()
            print(result)

asyncio.run(test())
