from supadata import Supadata
import os
from dotenv import load_dotenv

load_dotenv()

supadata = Supadata(api_key=os.getenv("SUPADATA_API_KEY"))

class YTLoader:
    def __init__(self, video_url: str):
        self.video_url = video_url
        self._transcript = None

    def load_transcript(self):
        if self._transcript is None:
            self._transcript = supadata.transcript(url=self.video_url, text=True)
        return self._transcript

if __name__ ==  "main":
    print("Transcripts succesfully  extracted !")
