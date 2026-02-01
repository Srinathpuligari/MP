from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
import json
from database import init_db, save_user, get_user_by_uid, get_all_users
from ai_engine import KnuckleAI

app = Flask(__name__)
CORS(app)

# Initialize
init_db()
ai = KnuckleAI()

# Storage paths
STORAGE_PATH = os.path.join(os.path.dirname(__file__), 'storage')
IMAGES_PATH = os.path.join(STORAGE_PATH, 'images')
MODELS_PATH = os.path.join(STORAGE_PATH, 'models')

os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)


def generate_uid():
    """Generate a 12-digit UID like Aadhar number"""
    # Format: XXXX-XXXX-XXXX (12 digits)
    part1 = str(random.randint(1000, 9999))
    part2 = str(random.randint(1000, 9999))
    part3 = str(random.randint(1000, 9999))
    return f"{part1}{part2}{part3}"


@app.route('/register', methods=['POST'])
def register():
    try:
        # Get form data
        name = request.form.get('name', '').strip()
        phone = request.form.get('phone', '').strip()
        email = request.form.get('email', '').strip()
        dob = request.form.get('dob', '').strip()
        gender = request.form.get('gender', '').strip()
        address = request.form.get('address', '').strip()
        
        if not name:
            return jsonify({"status": "error", "message": "Name required"}), 400
        if not phone or len(phone) < 10:
            return jsonify({"status": "error", "message": "Valid phone number required"}), 400

        files = request.files.getlist('images')
        if len(files) < 5:
            return jsonify({"status": "error", "message": f"Need at least 5 images, got {len(files)}"}), 400

        # Generate 12-digit UID
        uid = generate_uid()
        user_folder = os.path.join(IMAGES_PATH, uid)
        os.makedirs(user_folder, exist_ok=True)

        # Save cropped knuckle images
        saved_paths = []
        for i, f in enumerate(files):
            path = os.path.join(user_folder, f"{i}.jpg")
            f.save(path)
            saved_paths.append(path)

        # Process with AI - DGCNN based 3D feature extraction
        processed = ai.preprocess_images(saved_paths)
        point_cloud = ai.generate_3d_model(processed)
        features = ai.extract_features(point_cloud)

        # Save features
        feat_path = os.path.join(MODELS_PATH, f"{uid}.json")
        with open(feat_path, 'w') as fp:
            json.dump({
                "name": name,
                "phone": phone,
                "features": features,
                "point_cloud_stats": {
                    "num_points": len(point_cloud),
                    "feature_dim": len(features)
                }
            }, fp)

        # Save to DB with all details
        save_user(uid, name, phone, email, dob, gender, address, feat_path)

        return jsonify({
            "status": "success", 
            "uid": uid, 
            "message": f"Registered {name}",
            "images_processed": len(saved_paths)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/verify', methods=['POST'])
def verify():
    try:
        uid = request.form.get('uid', '').strip()
        if not uid:
            return jsonify({"status": "error", "message": "UID required"}), 400

        user = get_user_by_uid(uid)
        if not user:
            return jsonify({"status": "error", "message": "UID not found in database"}), 404

        # Support multiple images for better accuracy
        files = request.files.getlist('images')
        
        # Filter out empty files
        files = [f for f in files if f and f.filename]
        
        if not files:
            # Fallback to single image
            if 'image' in request.files and request.files['image'].filename:
                files = [request.files['image']]
            else:
                print(f"[DEBUG] No files received. Request files: {list(request.files.keys())}")
                return jsonify({"status": "error", "message": "At least 1 image required"}), 400
        
        print(f"[DEBUG] Received {len(files)} images for verification")

        # Save temp images
        temp_paths = []
        for i, file in enumerate(files[:5]):  # Max 5 images for verification
            temp_path = os.path.join(STORAGE_PATH, f"temp_{uid}_{i}.jpg")
            file.save(temp_path)
            temp_paths.append(temp_path)

        # Process with DGCNN
        processed = ai.preprocess_images(temp_paths)
        point_cloud = ai.generate_3d_model(processed)
        input_features = ai.extract_features(point_cloud)

        # Load stored features
        with open(user[2], 'r') as fp:
            data = json.load(fp)
            stored_features = data.get('features', data)

        score = ai.compare_features(input_features, stored_features)
        # Threshold for matching - DGCNN features should be discriminative
        match = score > 0.60

        # Cleanup
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return jsonify({
            "status": "success",
            "match": match,
            "score": round(score, 4),
            "name": user[1],
            "uid": uid
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/identify', methods=['POST'])
def identify():
    try:
        # Support multiple images for better accuracy
        files = request.files.getlist('images')
        
        # Filter out empty files
        files = [f for f in files if f and f.filename]
        
        if not files:
            # Fallback to single image
            if 'image' in request.files and request.files['image'].filename:
                files = [request.files['image']]
            else:
                print(f"[DEBUG] No files received. Request files: {list(request.files.keys())}")
                return jsonify({"status": "error", "message": "At least 1 image required"}), 400
        
        print(f"[DEBUG] Received {len(files)} images for identification")

        # Save temp images
        temp_paths = []
        for i, file in enumerate(files[:5]):  # Max 5 images
            temp_path = os.path.join(STORAGE_PATH, f"temp_identify_{i}.jpg")
            file.save(temp_path)
            temp_paths.append(temp_path)

        # Process with DGCNN
        processed = ai.preprocess_images(temp_paths)
        point_cloud = ai.generate_3d_model(processed)
        input_features = ai.extract_features(point_cloud)

        # Search all users
        users = get_all_users()
        best_name = None
        best_uid = None
        best_score = -1

        for u in users:
            try:
                with open(u[2], 'r') as fp:
                    data = json.load(fp)
                    stored = data.get('features', data)
                score = ai.compare_features(input_features, stored)
                if score > best_score:
                    best_score = score
                    best_name = u[1]
                    best_uid = u[0]
            except:
                continue

        # Cleanup
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Threshold for identification (DGCNN)
        if best_score > 0.60:
            return jsonify({"found": True, "name": best_name, "uid": best_uid, "score": round(best_score, 4)})
        else:
            return jsonify({"found": False, "message": "Person not found in database", "best_score": round(best_score, 4) if best_score > 0 else 0})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/users', methods=['GET'])
def list_users():
    users = get_all_users()
    return jsonify({"users": [{"uid": u[0], "name": u[1]} for u in users]})


@app.route('/admin/users', methods=['POST'])
def admin_list_users():
    """Protected endpoint - requires access code"""
    try:
        code = request.json.get('code', '') if request.is_json else request.form.get('code', '')
        
        if code != 'cbit':
            return jsonify({"status": "error", "message": "Invalid access code"}), 403
        
        conn = __import__('sqlite3').connect(os.path.join(os.path.dirname(__file__), 'knuckle.db'))
        c = conn.cursor()
        c.execute("SELECT uid, name, phone, email, dob, gender, address, created_at FROM users ORDER BY created_at DESC")
        users = c.fetchall()
        conn.close()
        
        user_list = []
        for u in users:
            user_list.append({
                "uid": u[0],
                "name": u[1],
                "phone": u[2],
                "email": u[3],
                "dob": u[4],
                "gender": u[5],
                "address": u[6],
                "registered_at": u[7]
            })
        
        return jsonify({"status": "success", "users": user_list, "total": len(user_list)})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/admin/delete/<uid>', methods=['POST'])
def admin_delete_user(uid):
    """Delete a user - requires access code"""
    try:
        code = request.json.get('code', '') if request.is_json else request.form.get('code', '')
        
        if code != 'cbit':
            return jsonify({"status": "error", "message": "Invalid access code"}), 403
        
        conn = __import__('sqlite3').connect(os.path.join(os.path.dirname(__file__), 'knuckle.db'))
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE uid = ?", (uid,))
        conn.commit()
        conn.close()
        
        # Delete user files
        user_folder = os.path.join(IMAGES_PATH, uid)
        feat_path = os.path.join(MODELS_PATH, f"{uid}.json")
        
        if os.path.exists(user_folder):
            import shutil
            shutil.rmtree(user_folder)
        if os.path.exists(feat_path):
            os.remove(feat_path)
        
        return jsonify({"status": "success", "message": f"User {uid} deleted"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    print("\n=== 3D Knuckle Biometric System ===")
    print("Backend running on http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
