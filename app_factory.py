"""Creates the FastAPI application. Kept separate from main.py so that tests can
build an app around a temporary database and a stand-in model."""
from typing import Dict, Optional

from fastapi import FastAPI, Header, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import service
from errors import ApiError


class NoCacheStaticFiles(StaticFiles):
    """Serves the interface with revalidation forced on every request. Without
    this the browser can keep using a cached stylesheet or script and go on
    showing an older interface after the files on disk have changed.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


class ObservationIn(BaseModel):
    date: str
    storage_mcm: float
    rainfall: Dict[str, float]
    confirm_large_change: bool = False
    source: Optional[str] = "manual"


class PredictIn(BaseModel):
    target_date: Optional[str] = None
    rainfall_forecast: Dict[str, float]


class AcceptPredictionIn(BaseModel):
    storage_mcm: float
    rainfall: Dict[str, float]


def _observation_dict(body):
    return {
        "date": body.date,
        "storage_mcm": body.storage_mcm,
        "rainfall": body.rainfall,
        "confirm_large_change": body.confirm_large_change,
        "source": body.source,
    }


def create_app(state=None, serve_frontend=True):
    if state is None:
        state = service.build_default_state()

    app = FastAPI(
        title="Kotmale Reservoir Watch API",
        description="Predicts next-day storage change for the Kotmale Reservoir from "
                    "forecasted multi-catchment rainfall, and stores the daily records it relies on.",
        version="1.0",
    )
    app.state.kotmale = state

    @app.exception_handler(ApiError)
    async def api_error_handler(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.to_detail()})

    @app.get("/health", tags=["system"])
    def health():
        return {"status": "ok"}

    @app.get("/status", tags=["system"])
    def status():
        return service.get_status(state)

    @app.get("/history", tags=["data"])
    def history(days: int = Query(90, ge=1, le=3650)):
        return service.list_history(state, days)

    @app.get("/observations", tags=["data"])
    def observations(days: int = Query(14, ge=1, le=365)):
        return service.list_observations(state, days)

    @app.post("/observations", tags=["data"])
    def add_observation(body: ObservationIn, x_entry_pin: Optional[str] = Header(default=None)):
        return service.add_observation(state, _observation_dict(body), x_entry_pin)

    @app.put("/observations/latest", tags=["data"])
    def correct_latest(body: ObservationIn, x_entry_pin: Optional[str] = Header(default=None)):
        return service.correct_latest(state, _observation_dict(body), x_entry_pin)

    @app.post("/predict", tags=["prediction"])
    def predict(body: PredictIn):
        return service.predict(state, {
            "target_date": body.target_date,
            "rainfall_forecast": body.rainfall_forecast,
        })

    @app.post("/demo/accept-prediction", tags=["demo"])
    def accept_prediction(body: AcceptPredictionIn):
        return service.accept_prediction_as_real(state, {"storage_mcm": body.storage_mcm, "rainfall": body.rainfall})

    @app.get("/openmeteo/rainfall", tags=["data"])
    def openmeteo_rainfall(date: Optional[str] = None):
        return service.fetch_rainfall(state, date)

    # The web interface is served by the same application, so no cross-origin setup is needed.
    if serve_frontend and config.STATIC_DIR.exists():
        app.mount("/", NoCacheStaticFiles(directory=str(config.STATIC_DIR), html=True), name="static")

    return app
