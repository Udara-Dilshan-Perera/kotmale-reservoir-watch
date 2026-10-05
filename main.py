"""Entry point:  python -m uvicorn main:app --port 8000"""
from app_factory import create_app

app = create_app()
