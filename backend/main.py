import os
import shutil
import threading
import time
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from apps.core.logger import logging, log_queue
from apps.training.train_model import TrainModel
from apps.prediction.predict_model import PredictModel
from apps.core.config import Config

# ============================================================
# Constants
# ============================================================
MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# ============================================================
# App Setup
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://employee-retention-system-frontend.onrender.com",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Rate Limiter (simple in-memory per-IP)
# ============================================================
from collections import defaultdict

rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_MAX_REQUESTS = 30  # max requests per window
RATE_LIMIT_WINDOW_SECONDS = 60  # window size


def check_rate_limit(client_ip: str):
    """Returns True if request is allowed, False if rate limited."""
    now = time.time()
    # Remove old entries outside the window
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip]
        if now - t < RATE_LIMIT_WINDOW_SECONDS
    ]
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return False
    rate_limit_store[client_ip].append(now)
    return True


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        return JSONResponse(
            {"error": "Rate limit exceeded. Try again later."},
            status_code=429
        )
    response = await call_next(request)
    return response


# ============================================================
# Pydantic Models for Input Validation (#1)
# ============================================================
class PredictionRequest(BaseModel):
    satisfaction_level: float = Field(ge=0.0, le=1.0, description="Employee satisfaction level (0-1)")
    last_evaluation: float = Field(ge=0.0, le=1.0, description="Last evaluation score (0-1)")
    number_project: int = Field(ge=1, le=10, description="Number of projects (1-10)")
    average_monthly_hours: int = Field(ge=50, le=400, description="Average monthly hours (50-400)")
    time_spend_company: int = Field(ge=1, le=20, description="Years at company (1-20)")
    work_accident: int = Field(ge=0, le=1, description="Work accident (0 or 1)")
    promotion_last_5years: int = Field(ge=0, le=1, description="Promoted in last 5 years (0 or 1)")
    salary: str = Field(pattern=r'^(low|medium|high)$', description="Salary level: low, medium, or high")


# ============================================================
# Global State for Model Registry
# ============================================================
app.state.active_model_path = None
app.state.active_model_version = None


# Auto-load latest model on startup
@app.on_event("startup")
def auto_load_latest_model():
    try:
        from apps.core.hf_uploader import HFUploader
        uploader = HFUploader(logger=logging.getLogger('HFUploader'))
        versions = uploader.list_models_versions()
        if versions:
            latest = versions[0]['version']
            snapshot_path = uploader.get_model_snapshot(tag_name=latest)
            if snapshot_path:
                app.state.active_model_path = snapshot_path
                app.state.active_model_version = latest
                logging.getLogger('Main').info(f'Auto-loaded model {latest} on startup')
    except Exception as e:
        logging.getLogger('Main').info(f'No model to auto-load: {e}')


def get_inference_guard():
    if not app.state.active_model_path:
        return JSONResponse(
            {"error": "No model loaded. Please train the model first or select a version from the Model Registry."},
            status_code=400
        )
    return None


# ============================================================
# Health Check (#9)
# ============================================================
@app.get('/health')
def health_check():
    return JSONResponse({
        "status": "healthy",
        "model_loaded": app.state.active_model_path is not None,
        "active_version": app.state.active_model_version,
        "training_running": training_state['is_running']
    })


# ============================================================
# Training
# ============================================================
training_state = {
    'is_running': False,
    'run_id': None,
    'log_file': None
}


def run_training_thread(run_id, data_path):
    """Background thread function to run the model training"""
    global training_state
    training_state['is_running'] = True
    try:
        trainModel = TrainModel(run_id, data_path)
        trainModel.training_model()

        # Auto-load the latest model after successful training
        try:
            from apps.core.hf_uploader import HFUploader
            uploader = HFUploader(logger=logging.getLogger('HFUploader'))
            versions = uploader.list_models_versions()
            if versions:
                latest = versions[0]['version']
                snapshot_path = uploader.get_model_snapshot(tag_name=latest)
                if snapshot_path:
                    app.state.active_model_path = snapshot_path
                    app.state.active_model_version = latest
        except Exception:
            pass

        log_queue.put("TRAINING_COMPLETE")
    except Exception as e:
        log_queue.put(f"TRAINING_FAILED: {str(e)}")
    finally:
        training_state['is_running'] = False


@app.post('/training')
async def training_route_client(file: UploadFile = File(...)):
    """Start training with uploaded CSV file."""
    global training_state
    try:
        # Validate file type (#6)
        if not file.filename.endswith('.csv'):
            return JSONResponse({"error": "Only CSV files are allowed."}, status_code=400)

        # Read file with size limit (#6)
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            return JSONResponse(
                {"error": f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB."},
                status_code=413
            )

        # Clear the queue from any previous runs
        while not log_queue.empty():
            log_queue.get_nowait()

        config = Config()
        run_id = config.get_run_id()
        data_path = config.training_data_path
        from apps.core.logger import log_file_path
        log_file = log_file_path

        # Store state for frontend reconnect
        training_state['run_id'] = run_id
        training_state['log_file'] = log_file

        # Clear existing training_data folder and save new file
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
        os.makedirs(data_path, exist_ok=True)

        safe_filename = os.path.basename(file.filename)
        file_location = f"{data_path}/{safe_filename}"
        with open(file_location, "wb+") as f:
            f.write(contents)

        # Start training in background thread
        thread = threading.Thread(target=run_training_thread, args=(run_id, data_path))
        thread.start()

        return JSONResponse({"message": "Training started", "run_id": run_id})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get('/training_status')
def get_training_status():
    """Returns current training status and run_id for frontend reconnect."""
    return JSONResponse({
        'is_running': training_state['is_running'],
        'run_id': training_state['run_id'],
        'has_logs': training_state['log_file'] is not None
    })


@app.get('/training_logs')
def get_training_logs():
    """Returns all log lines from the current training log file."""
    log_file = training_state.get('log_file')
    if not log_file or not os.path.exists(log_file):
        return JSONResponse({'logs': [], 'is_running': training_state['is_running']})
    try:
        with open(log_file, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines() if line.strip()]
        return JSONResponse({'logs': lines, 'is_running': training_state['is_running']})
    except Exception as e:
        return JSONResponse({'logs': [f'Error reading logs: {str(e)}'], 'is_running': False})


@app.get('/training_stream')
def training_stream():
    """SSE generator that yields logs from the Logger queue."""
    def event_generator():
        while True:
            try:
                message = log_queue.get(timeout=1.0)
                yield f"data: {message}\n\n"
                if message == "TRAINING_COMPLETE" or message.startswith("TRAINING_FAILED"):
                    break
            except Exception:
                pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================
# Batch Prediction
# ============================================================
@app.post('/batch_predict_file')
async def batch_predict_file_route(file: UploadFile = File(...)):
    """Batch prediction from uploaded CSV file."""
    try:
        # Validate file type (#6)
        if not file.filename.endswith('.csv'):
            return JSONResponse({"error": "Only CSV files are allowed."}, status_code=400)

        # Read file with size limit (#6)
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            return JSONResponse(
                {"error": f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB."},
                status_code=413
            )

        config = Config()
        run_id = config.get_run_id()
        data_path = config.prediction_data_path

        # Clear existing contents in prediction_data
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
        os.makedirs(data_path, exist_ok=True)

        # Save uploaded file
        safe_filename = os.path.basename(file.filename)
        file_location = f"{data_path}/{safe_filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(contents)

        # Ensure results directory exists and delete any existing Predictions.csv
        results_dir = data_path + '_results/'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, 'Predictions.csv')
        if os.path.exists(results_file):
            os.remove(results_file)

        guard = get_inference_guard()
        if guard:
            return guard

        # Initialize and run prediction
        predictModel = PredictModel(run_id, data_path, base_path=app.state.active_model_path)
        predictModel.batch_predict_from_model()

        # Check if predictions were generated
        if os.path.exists(results_file):
            return FileResponse(
                path=results_file,
                filename="Predictions.csv",
                media_type="text/csv"
            )
        else:
            return JSONResponse({"error": "Prediction failed: No output file generated."}, status_code=500)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# Single Prediction (with Pydantic validation #1)
# ============================================================
@app.post('/prediction')
async def single_prediction_route_client(req: PredictionRequest):
    """Single employee prediction with validated input."""
    try:
        config = Config()
        run_id = config.get_run_id()
        data_path = config.prediction_data_path

        data = pd.DataFrame(data=[[
            0,
            req.satisfaction_level,
            req.last_evaluation,
            req.number_project,
            req.average_monthly_hours,
            req.time_spend_company,
            req.work_accident,
            req.promotion_last_5years,
            req.salary
        ]], columns=[
            'empid', 'satisfaction_level', 'last_evaluation', 'number_project',
            'average_monthly_hours', 'time_spend_company', 'Work_accident',
            'promotion_last_5years', 'salary'
        ])

        convert_dict = {
            'empid': int,
            'satisfaction_level': float,
            'last_evaluation': float,
            'number_project': int,
            'average_monthly_hours': int,
            'time_spend_company': int,
            'Work_accident': int,
            'promotion_last_5years': int,
            'salary': object
        }
        data = data.astype(convert_dict)

        guard = get_inference_guard()
        if guard:
            return guard

        predictModel = PredictModel(run_id, data_path, base_path=app.state.active_model_path)
        output = predictModel.single_predict_from_model(data)
        return JSONResponse({
            "prediction": int(output),
            "result": "Will Leave" if output == 1 else "Will Stay"
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# Model Registry
# ============================================================
@app.get('/models')
def list_models():
    """Returns a list of available model versions from Hugging Face Hub."""
    from apps.core.hf_uploader import HFUploader
    try:
        uploader = HFUploader(logger=logging.getLogger('HFUploader'))
        versions = uploader.list_models_versions()

        # Try to resolve metrics for the latest version
        if versions:
            latest_v = versions[0]["version"]
            snap_path = uploader.get_model_snapshot(tag_name=latest_v)
            if snap_path:
                res_path = os.path.join(snap_path, "results.csv")
                if os.path.exists(res_path):
                    try:
                        df = pd.read_csv(res_path)
                        versions[0]["metrics"] = {
                            "accuracy": f"{df['Accuracy'].iloc[0]*100:.2f}%",
                            "recall": f"{df['Recall'].iloc[0]*100:.2f}%",
                            "auc_roc": f"{df['AUC_ROC'].iloc[0]*100:.2f}%",
                            "best_model": df['Model'].iloc[0]
                        }
                    except Exception:
                        pass

        return JSONResponse({"models": versions})
    except Exception as e:
        return JSONResponse({"error": str(e), "models": []}, status_code=500)


@app.post('/models/load/{version}')
def load_model(version: str):
    """Downloads a specific model version tag from Hugging Face Hub."""
    from apps.core.hf_uploader import HFUploader
    try:
        uploader = HFUploader(logger=logging.getLogger('HFUploader'))
        snapshot_path = uploader.get_model_snapshot(tag_name=version)

        if snapshot_path:
            app.state.active_model_path = snapshot_path
            app.state.active_model_version = version

            # Load metrics for the loaded version
            metrics = None
            results_path = os.path.join(snapshot_path, "results.csv")
            if os.path.exists(results_path):
                try:
                    df = pd.read_csv(results_path)
                    metrics = {
                        "accuracy": f"{df['Accuracy'].iloc[0]*100:.2f}%",
                        "recall": f"{df['Recall'].iloc[0]*100:.2f}%",
                        "auc_roc": f"{df['AUC_ROC'].iloc[0]*100:.2f}%",
                        "best_model": df['Model'].iloc[0]
                    }
                except Exception:
                    pass
            return JSONResponse({"message": f"Successfully loaded version {version} from Hugging Face Hub", "evaluation": metrics})
        else:
            return JSONResponse({"error": "Failed to resolve model snapshot from Hub"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# Logs Viewer
# ============================================================
@app.get('/logs')
def get_all_logs():
    """Returns all training and prediction log entries sorted by timestamp."""
    import glob as glob_mod
    log_entries = []
    logs_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

    for log_type, folder in [("training", "training_logs"), ("prediction", "prediction_logs")]:
        folder_path = os.path.join(logs_base, folder)
        if not os.path.isdir(folder_path):
            continue
        for log_file in glob_mod.glob(os.path.join(folder_path, "*.log")):
            filename = os.path.basename(log_file)
            try:
                with open(log_file, 'r') as f:
                    lines = [line.rstrip('\n') for line in f.readlines() if line.strip()]
                # Extract timestamp from first line for sorting
                timestamp = lines[0].split(' : ')[0].strip() if lines else ''
                log_entries.append({
                    "type": log_type,
                    "filename": filename,
                    "timestamp": timestamp,
                    "lines": lines
                })
            except Exception:
                continue

    # Sort by timestamp descending (newest first)
    log_entries.sort(key=lambda x: x['timestamp'], reverse=True)
    return JSONResponse({"logs": log_entries})


# ============================================================
# Entry Point (#5 - removed reload=True for production)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
