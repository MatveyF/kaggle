from enum import Enum

from pydantic import BaseModel, model_validator


class ParameterType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"


class Parameter(BaseModel):
    name: str
    type: ParameterType
    space: list[int | float | str]

    @model_validator(mode="after")
    def check_space_length(self):
        if self.type == ParameterType.INTEGER:
            if not all(isinstance(value, int) for value in self.space):
                raise ValueError("Parameter space must contain only integers.")

        return self


class NotFittedError(Exception):
    pass
