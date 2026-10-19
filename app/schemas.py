from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PassengerIn(BaseModel):
    """Input for one passenger. Every field is validated before it reaches the model."""
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0,
        "fare": 80.0, "embarked": "S", "has_cabin": True}]})

    pclass: Literal[1, 2, 3] = Field(description="Ticket class: 1, 2 or 3")
    sex: Literal["male", "female"]
    age: Optional[float] = Field(default=None, ge=0, le=100, description="Age in years (optional, a typical value is used if missing)")
    sibsp: int = Field(ge=0, le=8, description="Siblings or spouses on board")
    parch: int = Field(ge=0, le=6, description="Parents or children on board")
    fare: float = Field(ge=0, le=600, description="Ticket price in pounds")
    embarked: Literal["C", "Q", "S"] = Field(description="Port: C = Cherbourg, Q = Queenstown, S = Southampton")
    has_cabin: bool = Field(default=False, description="Was a cabin number recorded?")

    @field_validator("sex", "embarked", mode="before")
    @classmethod
    def _normalise(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v.lower() if v.lower() in ("male", "female") else v.upper()
        return v


class PredictionOut(BaseModel):
    survived: bool
    survival_probability: float = Field(ge=0, le=1)
    prediction: Literal["Survived", "Did not survive"]
    confidence: Literal["low", "medium", "high"]
    model: str


class BatchIn(BaseModel):
    passengers: list[PassengerIn] = Field(min_length=1, max_length=100)


class BatchOut(BaseModel):
    count: int
    predictions: list[PredictionOut]
