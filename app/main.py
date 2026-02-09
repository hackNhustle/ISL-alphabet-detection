from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from .predictor import AlphabetPredictor
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    try:
        app.state.predictor = AlphabetPredictor()
        logger.info("Alphabet predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize alphabet predictor: {e}")
        app.state.predictor = None
    yield
    # Clean up the ML model and release resources
    if hasattr(app.state, 'predictor') and app.state.predictor:
        # If predictor has a cleanup method, call it here
        pass

app = FastAPI(
    title="ISL Alphabet Recognition Service",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "alphabet-recognition"}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    predictor = getattr(request.app.state, 'predictor', None)
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        content = await file.read()
        result, error = predictor.predict(content)
        
        if error:
            return {"error": error}
            
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
