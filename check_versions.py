# check_versions.py
import pkg_resources

packages = [
    'opencv-python', 'face-recognition', 'dlib', 'numpy', 'pillow',
    'flask', 'flask-cors', 'scikit-learn', 'matplotlib', 'seaborn',
    'jupyter', 'pandas', 'cmake', 'scipy', 'joblib', 'requests'
]

print("📦 Package Versions Check:")
print("=" * 40)

for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"✅ {package}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"❌ {package}: Not installed")