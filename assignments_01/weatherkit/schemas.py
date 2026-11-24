from pydantic import BaseModel, Field, model_validator


class HourlyBlock(BaseModel):
    """The hourly measurements from the API.

    These are three parallel lists: item i of each list describes hour i.
    """
    time: list[str]
    temperature_2m: list[float]   # a null here fails validation
    precipitation: list[float]

    @model_validator(mode="after")
    def check_same_length(self):
        """Reject the data if the three lists are not the same length."""
        n_time = len(self.time)
        n_temp = len(self.temperature_2m)
        n_precip = len(self.precipitation)
        if not (n_time == n_temp == n_precip):
            raise ValueError(
                f"lists must be the same length, got time={n_time}, "
                f"temperature_2m={n_temp}, precipitation={n_precip}"
            )
        return self


class WeatherResponse(BaseModel):
    """The top level of an Open-Meteo hourly response."""
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock