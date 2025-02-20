from typing import Any

from pydantic import BaseModel

class Tool(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
