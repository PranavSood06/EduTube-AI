from langchain_community.document_loaders import YoutubeLoader


class YoutubeScrapper:

    @staticmethod
    def transcript(video_id: str):

        languages = ["en", "hi", "bn"]

        for language in languages:
            try:
                loader = YoutubeLoader(
                    video_id=video_id,
                    language=language
                )

                return loader.load()

            except Exception:
                continue

        raise ValueError(
            f"No supported transcript found for video: {video_id}"
        )