from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predict

app = FastAPI(title="Exoplanet Detection API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/api", tags=["predictions"])


@app.get("/")
async def root():
    return {"message": "Exoplanet Detection API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

