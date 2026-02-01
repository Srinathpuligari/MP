import { useRef, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Camera, X, Check } from 'lucide-react';
import './CameraCapture.css';

function CameraCapture({ onCapture, maxImages = 20, minImages = 5 }) {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [images, setImages] = useState([]);
  const [isCameraReady, setIsCameraReady] = useState(false);

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user"
  };

  // Focus box dimensions (relative to video)
  const focusBox = {
    width: 200,
    height: 150
  };

  const capture = useCallback(() => {
    if (images.length >= maxImages) return;
    
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
    setImages(prev => [...prev, croppedImage]);
  }, [images.length, maxImages]);

  const removeImage = (index) => {
    setImages(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = () => {
    if (images.length >= minImages) {
      onCapture(images);
    }
  };

  return (
    <div className="camera-capture">
      <div className="camera-container">
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          onUserMedia={() => setIsCameraReady(true)}
          className="webcam"
        />
        
        {/* Focus Box Overlay */}
        <div className="focus-overlay">
          <div className="focus-box">
            <span className="focus-label">Place Knuckle Here</span>
            <div className="corner tl"></div>
            <div className="corner tr"></div>
            <div className="corner bl"></div>
            <div className="corner br"></div>
          </div>
        </div>
        
        {/* Capture Button */}
        <button 
          className="capture-btn"
          onClick={capture}
          disabled={!isCameraReady || images.length >= maxImages}
        >
          <Camera size={24} />
          <span>Capture ({images.length}/{maxImages})</span>
        </button>
      </div>

      {/* Image Counter & Progress */}
      <div className="capture-progress">
        <div className="progress-bar">
          <div 
            className="progress-fill"
            style={{ width: `${(images.length / minImages) * 100}%` }}
          ></div>
        </div>
        <span className="progress-text">
          {images.length < minImages 
            ? `Capture at least ${minImages - images.length} more images`
            : `Ready to submit! (${images.length} images captured)`
          }
        </span>
      </div>

      {/* Captured Images Preview */}
      {images.length > 0 && (
        <div className="captured-images">
          <h3>Captured Images</h3>
          <div className="images-grid">
            {images.map((img, index) => (
              <div key={index} className="image-preview">
                <img src={img} alt={`Capture ${index + 1}`} />
                <button 
                  className="remove-btn"
                  onClick={() => removeImage(index)}
                >
                  <X size={14} />
                </button>
                <span className="image-number">{index + 1}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Submit Button */}
      <button 
        className={`submit-btn ${images.length >= minImages ? 'ready' : 'disabled'}`}
        onClick={handleSubmit}
        disabled={images.length < minImages}
      >
        <Check size={20} />
        <span>Submit {images.length} Images</span>
      </button>
    </div>
  );
}

export default CameraCapture;
