from pydantic import BaseModel, field_validator

class VideoRequest(BaseModel):
    video_id: str

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value: str):
        value = value.strip()

        if not value:
            raise ValueError("Video ID should be present")

        return value
