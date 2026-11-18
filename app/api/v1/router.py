from fastapi import APIRouter

from app.api.v1.evals import router as evals_router
from app.api.v1.rag import router as rag_router

api_v1_router = APIRouter()
api_v1_router.include_router(rag_router)
api_v1_router.include_router(evals_router)
