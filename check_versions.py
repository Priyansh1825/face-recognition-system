# check_versions.py
"""Print installed package versions.

This script prefers the stdlib importlib.metadata (Python 3.8+). If that's
not available it will try the importlib_metadata backport. As a last resort
it will use pkg_resources (setuptools) if present.
"""
import sys

try:
    # Python 3.8+
    from importlib.metadata import version as _get_version, PackageNotFoundError
except Exception:
    try:
        # Backport for older Pythons: pip install importlib-metadata
        from importlib_metadata import version as _get_version, PackageNotFoundError  # type: ignore
    except Exception:
        _get_version = None
        PackageNotFoundError = Exception

packages = [
    'opencv-python', 'face-recognition', 'dlib', 'numpy', 'pillow',
    'flask', 'flask-cors', 'scikit-learn', 'matplotlib', 'seaborn',
    'jupyter', 'pandas', 'cmake', 'scipy', 'joblib', 'requests'
]

print("📦 Package Versions Check:")
print("=" * 40)

for package in packages:
    if _get_version is not None:
        try:
            v = _get_version(package)
            print(f"✅ {package}: {v}")
        except PackageNotFoundError:
            print(f"❌ {package}: Not installed")
    else:
        # Last-resort fallback: pkg_resources from setuptools
        try:
            import pkg_resources

            v = pkg_resources.get_distribution(package).version
            print(f"✅ {package}: {v}")
        except Exception:
            print(f"❌ {package}: Not installed")