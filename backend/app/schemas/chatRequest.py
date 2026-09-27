from pydantic import BaseModel, field_validator

class ChatRequest(BaseModel):
    video_id: str
    query: str

    @field_validator("video_id", "query")
    @classmethod
    def validate_fields(cls, value: str):
        value = value.strip()

        if not value:
            raise ValueError("Field cannot be empty")

        return value
