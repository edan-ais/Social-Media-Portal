import asyncio
from app import process_videos

if __name__ == "__main__":
    print("Running montage builder via GitHub Actions...")
    asyncio.run(process_videos())
