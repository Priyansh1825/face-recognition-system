from flask import Flask, render_template, request, jsonify
import os
from config import BASE_DIR

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'project': 'Face Recognition System',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🚀 Starting Face Recognition System...")
    print(f"📁 Project Directory: {BASE_DIR}")
    print("🌐 Starting web server on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)