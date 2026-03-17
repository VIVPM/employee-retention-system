import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'https://employee-retention-system.onrender.com' || 'http://localhost:8000'

function App() {
  const [activeTab, setActiveTab] = useState('prediction')

  // Toast notification state
  const [toast, setToast] = useState(null)
  const toastTimer = useRef(null)

  const showToast = (message, type = 'success') => {
    if (toastTimer.current) clearTimeout(toastTimer.current)
    setToast({ message, type })
    toastTimer.current = setTimeout(() => setToast(null), 4000)
  }

  const [formData, setFormData] = useState({
    satisfaction_level: '',
    last_evaluation: '',
    number_project: '',
    average_monthly_hours: '',
    time_spend_company: '',
    work_accident: '',
    promotion_last_5years: '',
    salary: ''
  })

  const [batchFile, setBatchFile] = useState(null)
  const [batchResults, setBatchResults] = useState(null)
  const [resultBlob, setResultBlob] = useState(null)

  
  const [trainingLogs, setTrainingLogs] = useState([])
  const [isTraining, setIsTraining] = useState(false)
  const [trainingFile, setTrainingFile] = useState(null)
  const eventSourceRef = useRef(null)

  // Model Versioning State
  const [modelVersions, setModelVersions] = useState([])
  const [selectedVersion, setSelectedVersion] = useState('')
  const [loadedVersion, setLoadedVersion] = useState(null)
  const [loadingModels, setLoadingModels] = useState(true)
  const [loadingSpecificModel, setLoadingSpecificModel] = useState(false)

  const fetchModels = async (justTrained = false) => {
    setLoadingModels(true)
    try {
      const res = await fetch(`${API_URL}/models`)
      const data = await res.json()
      const versions = data.models || []
      setModelVersions(versions)
      if (versions.length > 0) {
        setSelectedVersion(versions[0].version)
        if (justTrained) {
          setLoadedVersion(versions[0].version)
        }
      }
    } catch (e) {
      console.error("Could not fetch models", e)
    } finally {
      setLoadingModels(false)
    }
  }

  useEffect(() => {
    fetchModels()
  }, [])

  const handleLoadModel = async () => {
    if (!selectedVersion) return
    setLoadingSpecificModel(true)
    try {
      const res = await fetch(`${API_URL}/models/load/${selectedVersion}`, { method: 'POST' })
      if (!res.ok) throw new Error('Failed to download model from Hugging Face')
      setLoadedVersion(selectedVersion)
      showToast(`Successfully loaded ${selectedVersion} for predictions!`, 'success')
    } catch (e) {
      showToast(`Error loading model: ${e.message}`, 'error')
    } finally {
      setLoadingSpecificModel(false)
    }
  }

  // On page load: check if training was running and restore logs
  useEffect(() => {
    const checkAndRestoreLogs = async () => {
      try {
        const statusRes = await fetch(`${API_URL}/training_status`)
        const status = await statusRes.json()

        if (status.has_logs) {
          const logsRes = await fetch(`${API_URL}/training_logs`)
          const data = await logsRes.json()
          if (data.logs && data.logs.length > 0) {
            setTrainingLogs(data.logs)
          }
          if (status.is_running) {
            setIsTraining(true)
            // Reconnect to SSE stream for new live logs
            connectToStream()
          }
        }
      } catch (e) {
        // Backend not up yet, ignore silently
      }
    }
    checkAndRestoreLogs()
  }, [])

  const connectToStream = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }
    const eventSource = new EventSource(`${API_URL}/training_stream`)
    eventSourceRef.current = eventSource

    eventSource.onmessage = (event) => {
      const message = event.data
      if (message === 'TRAINING_COMPLETE') {
        setTrainingLogs(prev => [...prev, '==================================', 'SUCCESS: Model Training Complete!', '=================================='])
        setIsTraining(false)
        fetchModels(true)
        eventSource.close()
      } else if (message.startsWith('TRAINING_FAILED')) {
        setTrainingLogs(prev => [...prev, `CRITICAL ERROR: ${message}`])
        setIsTraining(false)
        eventSource.close()
      } else {
        setTrainingLogs(prev => [...prev, message])
      }
    }

    eventSource.onerror = () => {
      setTrainingLogs(prev => [...prev, 'Connection lost. Stopping stream reader.'])
      setIsTraining(false)
      eventSource.close()
    }
  }
  
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setBatchFile(e.target.files[0])
    }
  }

  const handleTrainingFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setTrainingFile(e.target.files[0])
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Build form data representation
      const formBody = new URLSearchParams()
      for (const key in formData) {
        formBody.append(key, formData[key])
      }

      const response = await fetch(`${API_URL}/prediction`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formBody.toString()
      })

      if (!response.ok) {
        let errMsg = `Server returned ${response.status}`
        try {
          const errText = await response.text()
          if (errText) {
            // If it's HTML (likely a proxy error), just use the status code
            if (errText.trim().startsWith('<!DOCTYPE')) {
              errMsg = `Server Error (${response.status})`
            } else {
              errMsg = errText.replace(/"/g, '') // Remove quotes if JSON string
            }
          }
        } catch (e) {}
        throw new Error(errMsg)
      }

      const data = await response.text()
      setResult(data)
    } catch (err) {
      setError(err.message || 'An error occurred while fetching the prediction.')
    } finally {
      setLoading(false)
    }
  }

  const handleBatchSubmit = async (e) => {
    e.preventDefault()
    if (!batchFile) {
      setError('Please select a CSV file first.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', batchFile)

      const response = await fetch(`${API_URL}/batch_predict_file`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        let errMsg = `Server returned ${response.status}`
        try {
          const errText = await response.text()
          if (errText) {
             if (errText.trim().startsWith('<!DOCTYPE')) {
              errMsg = `Server Error (${response.status})`
            } else {
              errMsg = errText.replace(/"/g, '')
            }
          }
        } catch (e) {}
        throw new Error(errMsg)
      }

      // Read as blob for downloading later
      const blob = await response.blob()
      setResultBlob(blob)
      
      // Parse CSV text for UI display
      const text = await blob.text()
      const lines = text.split('\n').filter(line => line.trim() !== '')
      if (lines.length > 0) {
        const headers = lines[0].split(',')
        const rows = lines.slice(1).map(line => line.split(','))
        setBatchResults({ headers, rows })
      }

      setResult('Batch prediction completed successfully! View results below.')
      setBatchFile(null)
      // Reset the file input
      const fileInput = document.getElementById('batch_file')
      if (fileInput) fileInput.value = ''

    } catch (err) {
      setError(err.message || 'An error occurred during batch prediction.')
    } finally {
      setLoading(false)
    }
  }

  const handleTrainingSubmit = async () => {
    if (!trainingFile) {
      setError('Please select a training CSV file first.')
      return
    }
    setIsTraining(true)
    setTrainingLogs(['Uploading training file and initializing training sequence...'])
    setError(null)

    try {
      // 1. Upload file and trigger background training
      const form = new FormData()
      form.append('file', trainingFile)

      const response = await fetch(`${API_URL}/training`, {
        method: 'POST',
        body: form
      })

      if (!response.ok) {
        let errMsg = `Server returned ${response.status}`
        try {
          const errText = await response.text()
          if (errText) {
             if (errText.trim().startsWith('<!DOCTYPE')) {
              errMsg = `Server Error (${response.status})`
            } else {
              errMsg = errText.replace(/"/g, '')
            }
          }
        } catch (e) {}
        throw new Error(errMsg)
      }

      // Connect SSE stream for live logs
      connectToStream()

    } catch (err) {
      setError(err.message || 'Failed to start training process.')
      setIsTraining(false)
    }
  }

  const handleReconnect = async () => {
    try {
      const logsRes = await fetch(`${API_URL}/training_logs`)
      const data = await logsRes.json()
      if (data.logs && data.logs.length > 0) {
        setTrainingLogs(data.logs)
      }
      if (data.is_running) {
        setIsTraining(true)
        connectToStream()
      }
    } catch (e) {
      setError('Could not reconnect to training stream.')
    }
  }

  const handleDownload = () => {
    if (!resultBlob) return
    const url = window.URL.createObjectURL(resultBlob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'Predictions.csv'
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(url)
    document.body.removeChild(a)
  }

  return (
    <div className="app-container">
      <header className="hero-section">
        <div className="brand">
          <div className="logo-placeholder">ER</div>
          <h1>Employee Retention System</h1>
        </div>
        <p className="subtitle">
          AI-powered insights to help you understand HR dynamics and employee churn.
        </p>
      </header>

      <main className="main-layout" style={{ display: 'flex', gap: '20px', maxWidth: '1200px', margin: '0 auto', alignItems: 'flex-start' }}>
        
        {/* Model Registry Sidebar */}
        <aside className="models-sidebar" style={{ flex: '0 0 300px', backgroundColor: 'var(--surface-color)', padding: '20px', borderRadius: '12px', border: '1px solid var(--border-color)', marginTop: '20px' }}>
          <h2 style={{ fontSize: '1.2rem', marginTop: 0, color: 'var(--primary-color)' }}>Model Registry</h2>
          <p style={{ fontSize: '0.9rem', color: '#888', marginBottom: '20px' }}>Select a Hugging Face model version to download and use for inference.</p>

          {loadingModels ? (
            <div className="alert alert-loading" style={{ margin: 0, padding: '10px' }}>Loading versions...</div>
          ) : modelVersions.length === 0 ? (
            <div className="alert alert-error" style={{ margin: 0, padding: '10px' }}>No models found on Hub.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
              <select
                className="input-field"
                value={selectedVersion}
                onChange={e => setSelectedVersion(e.target.value)}
                style={{ padding: '10px', borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-color)', color: 'var(--text-color)' }}
              >
                {modelVersions.map(m => {
                  const displayStatus = m.status && m.status !== 'None' ? ` (${m.status})` : ''
                  return (
                    <option key={m.version} value={m.version}>
                      Version {m.version}{displayStatus}
                    </option>
                  )
                })}
              </select>
              

              {/* Metrics Display */}
              {modelVersions.find(m => m.version === selectedVersion)?.metrics && Object.keys(modelVersions.find(m => m.version === selectedVersion).metrics).length > 0 && (
                <div style={{ fontSize: '0.85rem', backgroundColor: 'rgba(59,130,246,0.1)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(59,130,246,0.3)', lineHeight: '1.6' }}>
                  <strong>🏆 Best Model:</strong> <span style={{color: 'var(--primary-color)'}}>{modelVersions.find(m => m.version === selectedVersion).metrics.best_model}</span> <br />
                  <strong>📈 Top AUC:</strong> <span style={{ color: 'var(--success-color)', fontWeight: 'bold' }}>{modelVersions.find(m => m.version === selectedVersion).metrics.accuracy}</span>
                </div>
              )}

              {selectedVersion !== loadedVersion && loadedVersion !== null && (
                <div style={{ fontSize: '0.85rem', color: '#ff9800', lineHeight: '1.4' }}>
                  ⚠️ To use this version for <strong>Inference</strong>, click Load below.
                </div>
              )}
              
              <button
                className="submit-btn"
                onClick={handleLoadModel}
                disabled={loadingSpecificModel || isTraining}
                style={{ padding: '10px', backgroundColor: loadedVersion === selectedVersion ? 'var(--success-color)' : 'var(--primary-color)' }}
              >
                {loadingSpecificModel ? <span className="loader" style={{width: '16px', height: '16px'}}></span> : (loadedVersion === selectedVersion ? '✅ Loaded Active' : '📥 Load Selected Model')}
              </button>
            </div>
          )}
        </aside>

        <div className="form-wrapper" style={{ flex: 1, margin: '20px 0' }}>
          <div className="nav-bar">
          <button 
            className={`nav-btn ${activeTab === 'training' ? 'active' : ''}`}
            onClick={() => setActiveTab('training')}
          >
            Training
          </button>
          <button 
            className={`nav-btn ${activeTab === 'prediction' ? 'active' : ''}`}
            onClick={() => setActiveTab('prediction')}
          >
            Prediction
          </button>
          <button
            className={`nav-btn ${activeTab === 'batch_prediction' ? 'active' : ''}`}
            onClick={() => setActiveTab('batch_prediction')}
          >
            Batch Prediction
          </button>
        </div>

        <div className="form-card">
          {error && <div className="alert alert-error">{error}</div>}
          {result && <div className="alert alert-success">{result}</div>}

          {activeTab === 'training' && (
            <div className="training-view">
              <h3>Model Training</h3>
              <p className="subtitle">Upload your training dataset and re-train the Retention Model.</p>

              <form onSubmit={(e) => { e.preventDefault(); handleTrainingSubmit(); }} className="batch-form">
                <div className="form-group file-upload-group">
                  <label htmlFor="training_file">Select Training CSV File</label>
                  <input
                    type="file"
                    id="training_file"
                    accept=".csv"
                    onChange={handleTrainingFileChange}
                    className="file-input"
                  />
                  {trainingFile && <p className="file-name">Selected: {trainingFile.name}</p>}
                </div>

                <div className="submit-row" style={{ marginTop: '1.5rem', marginBottom: '2rem' }}>
                  <button
                    type="submit"
                    className="submit-btn"
                    disabled={isTraining || !trainingFile}
                  >
                    {isTraining ? <span className="loader"></span> : 'Start ML Training'}
                  </button>
                </div>
              </form>

              <div className="reconnect-row">
                <span className="reconnect-label">Already training?</span>
                <button
                  type="button"
                  className="download-btn"
                  onClick={handleReconnect}
                  disabled={isTraining}
                >
                  Reconnect to Stream
                </button>
              </div>

              <div className="terminal-window">
                <div className="terminal-header">
                  <div className="mac-buttons">
                    <span className="mac-btn close"></span>
                    <span className="mac-btn min"></span>
                    <span className="mac-btn max"></span>
                  </div>
                  <div className="terminal-title">training_stream — bash — 80x24</div>
                </div>
                <div className="terminal-body">
                  {trainingLogs.length === 0 ? (
                    <div className="log-line empty">System Ready. Awaiting start command...</div>
                  ) : (
                    trainingLogs.map((log, index) => (
                      <div key={index} className={`log-line ${log.includes('INFO') ? 'info' : log.includes('ERROR') ? 'error' : log.includes('SUCCESS') ? 'success' : ''}`}>
                        <span className="prompt">{'> '}</span>{log}
                      </div>
                    ))
                  )}
                  {/* Invisible element to help scroll to bottom if needed */}
                  <div style={{ float:"left", clear: "both" }}></div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'batch_prediction' && (
            <div className="batch-view">
              <h3>Batch Prediction</h3>
              <p className="subtitle">Upload a CSV file containing employee records to predict retention.</p>
              
              <form onSubmit={handleBatchSubmit} className="batch-form">
                <div className="form-group file-upload-group">
                  <label htmlFor="batch_file">Select CSV File</label>
                  <input
                    type="file"
                    id="batch_file"
                    accept=".csv"
                    onChange={handleFileChange}
                    className="file-input"
                  />
                  {batchFile && <p className="file-name">Selected: {batchFile.name}</p>}
                </div>

                <div className="submit-row">
                  <button type="submit" className="submit-btn" disabled={loading || !batchFile}>
                    {loading ? <span className="loader"></span> : 'Run Batch Prediction'}
                  </button>
                </div>
              </form>

              {batchResults && (
                <div className="results-container">
                  <div className="results-header">
                    <h4>Prediction Results</h4>
                    <button type="button" onClick={handleDownload} className="download-btn">
                      Download CSV
                    </button>
                  </div>
                  <div className="table-wrapper">
                    <table className="results-table">
                      <thead>
                        <tr>
                          {batchResults.headers.map((h, i) => <th key={i}>{h}</th>)}
                        </tr>
                      </thead>
                      <tbody>
                        {batchResults.rows.map((row, i) => (
                          <tr key={i}>
                            {row.map((cell, j) => {
                              // If it's the prediction column (index 1)
                              if (j === 1) {
                                const isLeft = cell.trim() === '1' || cell.trim().toLowerCase() === 'left';
                                return (
                                  <td key={j}>
                                    <span className={`status-badge ${isLeft ? 'status-leave' : 'status-stay'}`}>
                                      {isLeft ? 'Left' : 'Stayed'}
                                    </span>
                                  </td>
                                )
                              }
                              // Otherwise, just raw cell data (format EmpId floats to ints if applicable)
                              let textContent = cell;
                              if (j === 0 && !isNaN(cell)) {
                                textContent = parseInt(cell).toString(); // Convert 1.0 to 1
                              }
                              return <td key={j}>{textContent}</td>;
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'prediction' && (
            <>
              <h3>Employee Details</h3>

          <form onSubmit={handleSubmit}>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="satisfaction_level">Satisfaction Level</label>
                <input
                  type="number"
                  step="0.01"
                  autoComplete="off"
                  id="satisfaction_level"
                  name="satisfaction_level"
                  value={formData.satisfaction_level}
                  onChange={handleChange}
                  required
                  placeholder="e.g. 0.5"
                />
              </div>
              <div className="form-group">
                <label htmlFor="last_evaluation">Last Evaluation</label>
                <input
                  type="number"
                  step="0.01"
                  autoComplete="off"
                  id="last_evaluation"
                  name="last_evaluation"
                  value={formData.last_evaluation}
                  onChange={handleChange}
                  required
                  placeholder="e.g. 0.8"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="number_project">Number of Projects</label>
                <input
                  type="number"
                  autoComplete="off"
                  id="number_project"
                  name="number_project"
                  value={formData.number_project}
                  onChange={handleChange}
                  required
                  placeholder="e.g. 4"
                />
              </div>
              <div className="form-group">
                <label htmlFor="average_monthly_hours">Avg Monthly Hours</label>
                <input
                  type="number"
                  autoComplete="off"
                  id="average_monthly_hours"
                  name="average_monthly_hours"
                  value={formData.average_monthly_hours}
                  onChange={handleChange}
                  required
                  placeholder="e.g. 200"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="time_spend_company">Years at Company</label>
                <input
                  type="number"
                  autoComplete="off"
                  id="time_spend_company"
                  name="time_spend_company"
                  value={formData.time_spend_company}
                  onChange={handleChange}
                  required
                  placeholder="e.g. 3"
                />
              </div>
              <div className="form-group">
                <label htmlFor="work_accident">Work Accident</label>
                <input
                  type="number"
                  autoComplete="off"
                  id="work_accident"
                  name="work_accident"
                  value={formData.work_accident}
                  onChange={handleChange}
                  required
                  placeholder="0 or 1"
                  min="0"
                  max="1"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="promotion_last_5years">Promotions Last 5 Yrs</label>
                <input
                  type="number"
                  autoComplete="off"
                  id="promotion_last_5years"
                  name="promotion_last_5years"
                  value={formData.promotion_last_5years}
                  onChange={handleChange}
                  required
                  placeholder="e.g. 2"
                  min="0"
                />
              </div>
              <div className="form-group">
                <label htmlFor="salary">Salary Bracket</label>
                <select
                  id="salary"
                  name="salary"
                  value={formData.salary}
                  onChange={handleChange}
                  required
                >
                  <option value="" disabled>Select Salary Bracket</option>
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
            </div>

            <div className="submit-row">
              <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? <span className="loader"></span> : 'Predict Retention'}
              </button>
            </div>
          </form>
          </>
          )}
          </div>
        </div>
      </main>

      {toast && (
        <div className={`toast toast-${toast.type}`} onClick={() => setToast(null)}>
          {toast.message}
        </div>
      )}
    </div>
  )
}

export default App
