from langchain_community.document_loaders import YoutubeLoader

class YoutubeScrapper:
    @staticmethod
    def transcript(video_id: str):
        """Fetch and return the transcript documents for a YouTube video."""
        return YoutubeLoader(video_id=video_id).load()
