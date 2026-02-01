import { Link } from 'react-router-dom';
import { Fingerprint, UserPlus, Search, Shield, Cpu, Database } from 'lucide-react';
import AdminDatabase from './AdminDatabase';
import './Home.css';

function Home() {
  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <div className="hero-badge">
            <Shield size={16} />
            <span>UID-Based Biometric System</span>
          </div>
          <h1>
            3D Finger Knuckle
            <span className="gradient-text"> Pattern Recognition</span>
          </h1>
          <p>
            Efficiently Matching Contactless 3D Finger Knuckle Patterns Using a 
            UID-Based Retrieval System. Advanced DGCNN-powered biometric identification 
            with unprecedented accuracy.
          </p>
          <div className="hero-buttons">
            <Link to="/register" className="btn primary">
              <UserPlus size={20} />
              <span>Register Now</span>
            </Link>
            <Link to="/verify" className="btn secondary">
              <Search size={20} />
              <span>Verify Identity</span>
            </Link>
          </div>
        </div>
        
        <div className="hero-visual">
          <div className="fingerprint-animation">
            <Fingerprint size={200} />
            <div className="scan-line"></div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features">
        <h2>System Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">
              <Fingerprint size={32} />
            </div>
            <h3>3D Reconstruction</h3>
            <p>Convert 2D knuckle images into detailed 3D point clouds using advanced photogrammetry</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">
              <Cpu size={32} />
            </div>
            <h3>DGCNN Features</h3>
            <p>Deep Graph Convolutional Neural Network for robust feature extraction from 3D models</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">
              <Database size={32} />
            </div>
            <h3>UID System</h3>
            <p>Unique ID based storage and retrieval system like Aadhar for efficient matching</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">
              <Shield size={32} />
            </div>
            <h3>Dual Verification</h3>
            <p>Both 1:1 UID verification and 1:N database search modes available</p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="how-it-works">
        <h2>How It Works</h2>
        <div className="steps">
          <div className="step">
            <div className="step-number">1</div>
            <h3>Register</h3>
            <p>Enter your details and capture 5+ knuckle images using the camera</p>
          </div>
          <div className="step-arrow">→</div>
          <div className="step">
            <div className="step-number">2</div>
            <h3>Process</h3>
            <p>Images are preprocessed, converted to 3D, and features are extracted via DGCNN</p>
          </div>
          <div className="step-arrow">→</div>
          <div className="step">
            <div className="step-number">3</div>
            <h3>Store</h3>
            <p>Your unique 12-digit UID is generated and linked to your biometric data</p>
          </div>
          <div className="step-arrow">→</div>
          <div className="step">
            <div className="step-number">4</div>
            <h3>Verify</h3>
            <p>Capture 5 images and use your UID or search the database to verify</p>
          </div>
        </div>
      </section>

      {/* Admin Database Section */}
      <section className="admin-section-wrapper">
        <AdminDatabase />
      </section>

      {/* Footer */}
      <footer className="footer">
        <p>Major Project - 3D Knuckle Biometric System</p>
        <p className="sub">Based on: "Efficiently Matching Contactless 3D Finger Knuckle Patterns Using a UID-Based Retrieval System"</p>
      </footer>
    </div>
  );
}

export default Home;
