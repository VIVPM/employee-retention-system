import os
import shutil
import threading
import time
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import PlainTextResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
# import flask_monitoringdashboard as dashboard
import pandas as pd
from apps.core.logger import Logger
from apps.training.train_model import TrainModel
from apps.prediction.predict_model import PredictModel
from apps.core.config import Config

app = FastAPI()
# dashboard.bind(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://employee-retention-system.onrender.com",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State for Model Registry
app.state.active_model_path = None
app.state.active_model_version = None

def get_inference_guard():
    if not app.state.active_model_path:
        return PlainTextResponse("No model loaded. Please train the model first or select a version from the Model Registry.", status_code=400)
    return None



# Track current training run state
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
        Logger.log_queue.put("TRAINING_COMPLETE")
    except Exception as e:
        Logger.log_queue.put(f"TRAINING_FAILED: {str(e)}")
    finally:
        training_state['is_running'] = False

@app.post('/training')
async def training_route_client(file: UploadFile = File(...)):
    """
    * method: training_route_client
    * description: method to start training route with uploaded CSV
    """
    global training_state
    try:
        # Clear the queue from any previous runs
        while not Logger.log_queue.empty():
            Logger.log_queue.get_nowait()

        config = Config()
        run_id = config.get_run_id()
        data_path = config.training_data_path
        log_file = f'logs/training_logs/train_log_{run_id}.log'

        # Store state for frontend reconnect
        training_state['run_id'] = run_id
        training_state['log_file'] = log_file

        # Clear existing training_data folder and save new file
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
        os.makedirs(data_path, exist_ok=True)

        file_location = f"{data_path}/{file.filename}"
        with open(file_location, "wb+") as f:
            f.write(file.file.read())

        # Start training in background thread
        thread = threading.Thread(target=run_training_thread, args=(run_id, data_path))
        thread.start()

        return PlainTextResponse("Training started", status_code=200)
    except Exception as e:
        return PlainTextResponse(f"Error Occurred! {str(e)}", status_code=500)

@app.get('/training_status')
def get_training_status():
    """Returns current training status and run_id for frontend reconnect."""
    from fastapi.responses import JSONResponse
    return JSONResponse({
        'is_running': training_state['is_running'],
        'run_id': training_state['run_id'],
        'has_logs': training_state['log_file'] is not None
    })

@app.get('/training_logs')
def get_training_logs():
    """Returns all log lines from the current training log file."""
    from fastapi.responses import JSONResponse
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
    """
    * method: training_stream
    * description: SSE generator that yields logs from the Logger queue
    """
    def event_generator():
        while True:
            # Check the queue for new logs with a small timeout
            try:
                message = Logger.log_queue.get(timeout=1.0)
                # Yield the message in SSE format
                yield f"data: {message}\n\n"
                
                # Stop the generator if we hit the completion or failure flags
                if message == "TRAINING_COMPLETE" or message.startswith("TRAINING_FAILED"):
                    break
            except Exception:
                # Queue empty, just continue waiting
                pass
                
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post('/batchprediction')
def batch_prediction_route_client():
    """
    * method: batch_prediction_route_client
    * description: method to call batch prediction route
    * return: none
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * bcheekati       05-MAY-2020    1.0      initial creation
    *
    * Parameters
    *   None
    """
    try:
        config = Config()
        #get run id
        run_id = config.get_run_id()
        data_path = config.prediction_data_path

        guard = get_inference_guard()
        if guard: return guard

        #prediction object initialization
        predictModel=PredictModel(run_id, data_path, base_path=app.state.active_model_path)
        #prediction the model
        predictModel.batch_predict_from_model()
        return PlainTextResponse("Prediction successfull! and its RunID is : "+str(run_id))
    except ValueError as e:
        return PlainTextResponse("Error Occurred! %s" % e, status_code=400)
    except KeyError as e:
        return PlainTextResponse("Error Occurred! %s" % e, status_code=400)
    except Exception as e:
        return PlainTextResponse("Error Occurred! %s" % e, status_code=500)


@app.post('/batch_predict_file')
async def batch_predict_file_route(file: UploadFile = File(...)):
    """
    * method: batch_predict_file_route
    * description: method to handle batch prediction from uploaded file
    """
    try:
        config = Config()
        run_id = config.get_run_id()
        data_path = config.prediction_data_path
        
        # Clear existing contents in prediction_data
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
        os.makedirs(data_path, exist_ok=True)
        
        # Save uploaded file
        file_location = f"{data_path}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
            
        # Ensure results directory exists and delete any existing Predictions.csv
        results_dir = data_path + '_results/'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, 'Predictions.csv')
        if os.path.exists(results_file):
            os.remove(results_file)

        guard = get_inference_guard()
        if guard: return guard

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
            return PlainTextResponse("Prediction failed: No output file generated.", status_code=500)

    except Exception as e:
        return PlainTextResponse(f"Error Occurred! {str(e)}", status_code=500)


@app.post('/prediction')
async def single_prediction_route_client(request: Request):
    """
    * method: prediction_route_client
    * description: method to call prediction route
    * return: none
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * bcheekati       05-MAY-2020    1.0      initial creation
    *
    * Parameters
    *   request: Request
    """
    try:
        config = Config()
        #get run id
        run_id = config.get_run_id()
        data_path = config.prediction_data_path
        print('Test')

        form_data = await request.form()
        satisfaction_level = form_data.get('satisfaction_level')
        last_evaluation = form_data.get("last_evaluation")
        number_project = form_data.get("number_project")
        average_monthly_hours = form_data.get("average_monthly_hours")
        time_spend_company = form_data.get("time_spend_company")
        work_accident = form_data.get("work_accident")
        promotion_last_5years = form_data.get("promotion_last_5years")
        salary = form_data.get("salary")

        data = pd.DataFrame(data=[[0 ,satisfaction_level, last_evaluation, number_project,average_monthly_hours,time_spend_company,work_accident,promotion_last_5years,salary]],
                          columns=['empid','satisfaction_level', 'last_evaluation', 'number_project','average_monthly_hours','time_spend_company','Work_accident','promotion_last_5years','salary'])
        convert_dict = {'empid': int,
                        'satisfaction_level': float,
                        'last_evaluation': float,
                        'number_project': int,
                        'average_monthly_hours': int,
                        'time_spend_company': int,
                        'Work_accident': int,
                        'promotion_last_5years': int,
                        'salary': object}

        data = data.astype(convert_dict)

        guard = get_inference_guard()
        if guard: return guard

        # object initialization
        predictModel = PredictModel(run_id, data_path, base_path=app.state.active_model_path)
        # prediction the model
        output = predictModel.single_predict_from_model(data)
        print('output : '+str(output))
        return PlainTextResponse("Predicted Output is : "+str(output))
    except ValueError as e:
        return PlainTextResponse("Error Occurred! %s" % e, status_code=400)
    except KeyError as e:
        return PlainTextResponse("Error Occurred! %s" % e, status_code=400)
    except Exception as e:
        return PlainTextResponse("Error Occurred! %s" % e, status_code=500)

@app.get('/models')
def list_models():
    """Returns a list of available model versions from Hugging Face Hub"""
    from fastapi.responses import JSONResponse
    from apps.core.hf_uploader import HFUploader
    import os, pandas as pd
    try:
        uploader = HFUploader(logger=Logger("system", "Main", "api"))
        versions = uploader.list_models_versions()
        
        # Try to resolve metrics for the latest version from Hub instead of just local
        if versions:
            latest_v = versions[0]["version"]
            # Get snapshot path for metrics (cached or downloaded)
            snap_path = uploader.get_model_snapshot(tag_name=latest_v)
            if snap_path:
                res_path = os.path.join(snap_path, "results.csv")
                if os.path.exists(res_path):
                    try:
                        df = pd.read_csv(res_path)
                        selected = df[df['Selected'] == 'Yes'] if 'Selected' in df.columns else df
                        best_row = selected.loc[selected['Best Score AUC'].idxmax()]
                        versions[0]["metrics"] = {
                            "accuracy": f"{best_row['Best Score AUC']*100:.2f}%",
                            "best_model": best_row['Model Name'],
                            "cluster": str(best_row['Cluster'])
                        }
                    except:
                        pass

        return JSONResponse({"models": versions})
    except Exception as e:
        return JSONResponse({"error": str(e), "models": []}, status_code=500)

@app.post('/models/load/{version}')
def load_model(version: str):
    """Downloads a specific model version tag from Hugging Face Hub"""
    from fastapi.responses import JSONResponse
    from apps.core.hf_uploader import HFUploader
    import os, pandas as pd
    try:
        uploader = HFUploader(logger=Logger("system", "Main", "api"))
        
        # version will be a tag name like 'v1.0'
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
                    selected = df[df['Selected'] == 'Yes'] if 'Selected' in df.columns else df
                    best_row = selected.loc[selected['Best Score AUC'].idxmax()]
                    metrics = {
                        "accuracy": f"{best_row['Best Score AUC']*100:.2f}%",
                        "best_model": best_row['Model Name']
                    }
                except:
                    pass
            return JSONResponse({"message": f"Successfully loaded version {version} from Hugging Face Hub", "evaluation": metrics})
        else:
            return JSONResponse({"error": "Failed to resolve model snapshot from Hub"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)