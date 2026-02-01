import { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { 
  Search, 
  KeyRound, 
  Camera, 
  CheckCircle, 
  XCircle, 
  Loader,
  User,
  Fingerprint,
  X
} from 'lucide-react';
import './Verify.css';

const API_URL = 'http://localhost:5000';
const IMAGES_REQUIRED = 5;

function Verify() {
  const [mode, setMode] = useState(null); // 'uid' or 'search'
  const [uid, setUid] = useState('');
  const [capturedImages, setCapturedImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const webcamRef = useRef(null);

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user"
  };

  // Focus box dimensions (same as CameraCapture)
  const focusBox = {
    width: 200,
    height: 150
  };

  const capture = useCallback(() => {
    if (capturedImages.length >= IMAGES_REQUIRED) return;
    
    const video = webcamRef.current?.video;
    if (!video) return;

    // Create canvas for cropping
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size to focus box size
    canvas.width = focusBox.width;
    canvas.height = focusBox.height;

    // Calculate center crop coordinates
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    
    // Scale factors
    const scaleX = videoWidth / 640;
    const scaleY = videoHeight / 480;
    
    // Focus box position (centered)
    const boxX = ((640 - focusBox.width) / 2) * scaleX;
    const boxY = ((480 - focusBox.height) / 2) * scaleY;
    const boxW = focusBox.width * scaleX;
    const boxH = focusBox.height * scaleY;

    // Draw only the focus box region
    ctx.drawImage(
      video,
      boxX, boxY, boxW, boxH,  // Source (crop area)
      0, 0, focusBox.width, focusBox.height  // Destination
    );

    const croppedImage = canvas.toDataURL('image/jpeg', 0.9);
    setCapturedImages(prev => [...prev, croppedImage]);
  }, [capturedImages.length]);

  const removeImage = (index) => {
    setCapturedImages(prev => prev.filter((_, i) => i !== index));
  };

  const resetImages = () => {
    setCapturedImages([]);
    setResult(null);
    setError(null);
  };

  // Helper function to convert base64 to File
  const base64ToFile = (base64, filename) => {
    const arr = base64.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  };

  const handleVerifyWithUID = async () => {
    if (!uid.trim() || capturedImages.length < IMAGES_REQUIRED) return;
    
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('uid', uid.trim());
      
      // Add all 5 images for DGCNN processing
      for (let i = 0; i < capturedImages.length; i++) {
        const file = base64ToFile(capturedImages[i], `verify_${i}.jpg`);
        formData.append('images', file);
      }

      const res = await axios.post(`${API_URL}/verify`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setResult({
        type: 'verify',
        match: res.data.match,
        score: res.data.score,
        name: res.data.name
      });
    } catch (err) {
      setError(err.response?.data?.message || 'Verification failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (capturedImages.length < IMAGES_REQUIRED) return;
    
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      
      // Add all 5 images for DGCNN processing
      for (let i = 0; i < capturedImages.length; i++) {
        const file = base64ToFile(capturedImages[i], `search_${i}.jpg`);
        formData.append('images', file);
      }

      const res = await axios.post(`${API_URL}/identify`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setResult({
        type: 'search',
        found: res.data.found,
        name: res.data.name,
        score: res.data.score
      });
    } catch (err) {
      setError(err.response?.data?.message || 'Search failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetAll = () => {
    setMode(null);
    setUid('');
    setCapturedImages([]);
    setResult(null);
    setError(null);
  };

  // Mode Selection Screen
  if (!mode) {
    return (
      <div className="verify-page">
        <div className="mode-selection">
          <h1>Verification Mode</h1>
          <p>Choose how you want to verify the knuckle pattern</p>
          
          <div className="mode-cards">
            <div className="mode-card" onClick={() => setMode('uid')}>
              <KeyRound size={48} />
              <h3>Verify with UID</h3>
              <p>Enter your UID and capture 5 images for verification</p>
            </div>
            
            <div className="mode-card" onClick={() => setMode('search')}>
              <Search size={48} />
              <h3>Search (1:N)</h3>
              <p>Capture 5 images to search across all registered users</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="verify-page">
      <div className="verify-container">
        {/* Header */}
        <div className="verify-header">
          {mode === 'uid' ? (
            <>
              <KeyRound size={40} className="header-icon" />
              <h2>Verify with UID</h2>
              <p>Enter your UID and capture {IMAGES_REQUIRED} knuckle images</p>
            </>
          ) : (
            <>
              <Search size={40} className="header-icon" />
              <h2>Search Identity (1:N)</h2>
              <p>Capture {IMAGES_REQUIRED} knuckle images to search in database</p>
            </>
          )}
        </div>

        {/* UID Input (only for UID mode) */}
        {mode === 'uid' && (
          <div className="uid-input-section">
            <div className="input-group">
              <KeyRound size={20} className="input-icon" />
              <input
                type="text"
                placeholder="Enter your 12-digit UID"
                value={uid}
                onChange={(e) => setUid(e.target.value.replace(/\D/g, ''))}
                maxLength={12}
              />
            </div>
          </div>
        )}

        {/* Image Counter */}
        <div className="image-counter">
          <span>Images Captured: {capturedImages.length} / {IMAGES_REQUIRED}</span>
        </div>

        {/* Camera Section */}
        <div className="camera-section">
          <div className="camera-container">
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              videoConstraints={videoConstraints}
              className="webcam"
            />
            
            {/* Focus Box */}
            <div className="focus-overlay">
              <div className="focus-box">
                <span className="focus-label">Place Knuckle Here</span>
                <div className="corner tl"></div>
                <div className="corner tr"></div>
                <div className="corner bl"></div>
                <div className="corner br"></div>
              </div>
            </div>

            <button 
              className="capture-btn" 
              onClick={capture}
              disabled={capturedImages.length >= IMAGES_REQUIRED}
            >
              <Camera size={24} />
              <span>{capturedImages.length >= IMAGES_REQUIRED ? 'All Images Captured' : `Capture (${capturedImages.length}/${IMAGES_REQUIRED})`}</span>
            </button>
          </div>
        </div>

        {/* Captured Images Preview */}
        {capturedImages.length > 0 && (
          <div className="captured-preview">
            <h4>Captured Images:</h4>
            <div className="images-grid">
              {capturedImages.map((img, index) => (
                <div key={index} className="preview-item">
                  <img src={img} alt={`Capture ${index + 1}`} />
                  <button className="remove-btn" onClick={() => removeImage(index)}>
                    <X size={16} />
                  </button>
                </div>
              ))}
            </div>
            <button className="reset-btn" onClick={resetImages}>
              Clear All Images
            </button>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="error-message">
            <XCircle size={20} />
            <span>{error}</span>
          </div>
        )}

        {/* Result Display */}
        {result && (
          <div className={`result-display ${result.match || result.found ? 'success' : 'failure'}`}>
            {result.type === 'verify' ? (
              result.match ? (
                <>
                  <CheckCircle size={48} />
                  <h3>Identity Verified!</h3>
                  <p>Match Score: {(result.score * 100).toFixed(1)}%</p>
                  {result.name && <p className="name">Name: {result.name}</p>}
                </>
              ) : (
                <>
                  <XCircle size={48} />
                  <h3>Verification Failed</h3>
                  <p>Knuckle pattern does not match the UID</p>
                  <p className="score">Similarity: {(result.score * 100).toFixed(1)}%</p>
                </>
              )
            ) : (
              result.found ? (
                <>
                  <CheckCircle size={48} />
                  <h3>Identity Found!</h3>
                  <div className="found-info">
                    <User size={24} />
                    <span className="found-name">{result.name}</span>
                  </div>
                  <p>Match Score: {(result.score * 100).toFixed(1)}%</p>
                </>
              ) : (
                <>
                  <XCircle size={48} />
                  <h3>Person Not Found</h3>
                  <p>This knuckle pattern is not registered in the database</p>
                </>
              )
            )}
          </div>
        )}

        {/* Action Buttons */}
        <div className="action-buttons">
          {capturedImages.length >= IMAGES_REQUIRED && !loading && !result && (
            <button 
              className="verify-btn"
              onClick={mode === 'uid' ? handleVerifyWithUID : handleSearch}
              disabled={mode === 'uid' && !uid.trim()}
            >
              {loading ? (
                <Loader size={20} className="spin" />
              ) : (
                <>
                  <Fingerprint size={20} />
                  <span>{mode === 'uid' ? 'Verify Identity' : 'Search Database'}</span>
                </>
              )}
            </button>
          )}

          {result && (
            <button className="try-again-btn" onClick={resetImages}>
              Try Again
            </button>
          )}

          <button className="back-btn" onClick={resetAll}>
            ← Change Mode
          </button>
        </div>
      </div>
    </div>
  );
}

export default Verify;