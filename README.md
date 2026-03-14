# 🏢 Employee Retention System (React + FastAPI)

An end-to-end Machine Learning web application designed to predict employee churn. This project features a modern **React** frontend and a **FastAPI** backend, complete with real-time model training, live log streaming, and both single and batch prediction capabilities.

---

## System Architecture

```mermaid
graph LR
    %% Data Stage
    subgraph Data_Pipeline ["1. Data Pipeline"]
        Raw["HR Churn Dataset<br/>(CSV)"] -->|cleaning.py| Clean["Cleaned Data"]
        Clean -->|preprocessing.py| Feat["Engineered Features<br/>(One-Hot Encoding, Scaling)"]
    end

    %% Training Stage
    subgraph Training_Pipeline ["2. Model Training"]
        Feat -->|model_trainer.py| RFC["Random Forest Classifier<br/>(Tuned Hyperparameters)"]
        RFC -->|evaluate.py| Metrics["Evaluation Metrics<br/>(Accuracy, Precision, Recall)"]
        RFC -->|Serialize| Artifacts["Model Artifact<br/>(.pkl)"]
    end

    %% Serving Stage
    subgraph Deployment ["3. Inference & Serving"]
        Artifacts -->|Load Model| API["FastAPI Backend<br/>(Port: 8000)"]
        API -->|/predict| UI["React Frontend<br/>(Vite)"]
        User["HR Manager"] -->|Input Employee Details| UI
        UI -->|Show Prediction| User
    end

    %% Styling
    style Data_Pipeline fill:#f9f9f9,stroke:#333
    style Training_Pipeline fill:#e1f5fe,stroke:#01579b
    style Deployment fill:#e8f5e9,stroke:#1b5e20
```

## Performance & Evaluation

The model was evaluated using a 20% hold-out test set. The `RandomForestClassifier` emerged as the best performing model.

- **Best Model:** Random Forest Classifier
- **Parameters:** `max_depth=3`, `n_estimators=10`, `criterion='entropy'`
- **Accuracy Score:** **91.43%**
- **Evaluation Method:** Stratified Train-Test Split (80/20)

## Project Structure

```
employee-retention-system/
├── backend/
│   ├── apps/           # API routes and logic
│   ├── data/           # Stored models and CSVs
│   ├── logs/           # Training and API logs
│   ├── main.py         # FastAPI Entry point
│   └── requirements.txt
├── frontend/           # React + Vite application
│   ├── src/            # Components and App logic
│   ├── index.html
│   └── package.json
├── employee_retention.ipynb  # Original Exploratory Data Analysis (EDA)
└── hr_employee_churn_data.csv # Dataset
```

## Technical Stack

- **Frontend:** React (Vite), Modern CSS (Premium UI)
- **Backend:** FastAPI, Uvicorn
- **Machine Learning:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Model:** Random Forest Classifier

---

## How to Run

### 1. Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### 2. Frontend (React)
```bash
cd frontend
npm install
npm run dev
```
*   **Session Persistence**: Training logs and states are cached securely, so you can safely refresh the page without losing your live training progress.

---

## 📊 Model Performance

During our exploratory data analysis (`employee_retention.ipynb`), we evaluated multiple models to predict employee churn accurately across 15,000+ records.

| Model | Accuracy | Notes |
| :--- | :--- | :--- |
| **Random Forest** | **98.8%** | Excellent baseline, chosen for production for its interpretability and robust handling of categorical variables. |
| **XGBoost** | **99.1%** | Highest raw accuracy during GridSearch hyperparameter tuning. |

*The application pipeline is designed to easily swap out or retrain these models on the fly through the React UI.*

---

## 🏗️ Architecture

```mermaid
graph TD
    %% Frontend
    User[👤 HR / Admin] -->|Interacts| React["⚛️ React Frontend<br>(frontend/)"]
    
    %% APIs
    React -->|Single/Batch Predict| PredictAPI["⚡ FastAPI Predict Endpoint<br>(/predict, /predict_batch)"]
    React -->|Upload Training Data| UploadAPI["📤 FastAPI Upload Endpoint<br>(/training)"]
    React -->|SSE Connection| StreamAPI["📡 FastAPI Log Stream<br>(/training_stream)"]
    
    %% Backend Logic
    subgraph Backend_Logic [FastAPI Server]
        PredictAPI -->|Load Model| Model["🧠 ML Model<br>(Random Forest)"]
        UploadAPI -->|Trigger Async| Trainer["⚙️ Training Thread"]
        Trainer -->|Dump Logs| LogQueue["🗄️ Thread-Safe Queue"]
        LogQueue -->|Yield Streams| StreamAPI
    end

    %% Data Layer
    subgraph Storage [File Storage]
        Trainer -->|Save pkl| ModelsDir["📂 /models<br>(preprocessor.pkl, model.pkl)"]
        Trainer -->|Save logs| LogsDir["📂 /logs<br>(training runs)"]
        PredictAPI -->|Read pkl| ModelsDir
    end
```

---

## 🛠️ Set-up & Execution

### 1. Requirements
Ensure you have **Node.js** v18+ (for frontend) and **Python 3.9+** (for backend) installed.

### 2. Backend Setup
1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run Backend:
    ```bash
    uvicorn main:app --port 8000 --reload
    ```

### 3. Frontend Setup
1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Run Dev Server:
    ```bash
    npm run dev
    ```
4.  Open `http://localhost:5173` in your browser.

---

## 📂 Project Structure

*   **`frontend/`**: Vite + React application.
    *   `src/components/`: UI components including Terminal UI and File Uploaders.
    *   `src/App.css`: Modern styling and animations.
*   **`backend/`**: FastAPI server.
    *   `main.py`: API endpoints for predictions, file mapping, and SSE streams.
    *   `app/core/`: Application settings and custom loggers.
    *   `models/`: Directory holding compiled `.pkl` machine learning models.
    *   `data/`: Directory for storing training CSV uploads.
*   **`employee_retention.ipynb`**: Original Jupyter Notebook used for initial exploratory data analysis (EDA) and model prototyping.