"""Build and upload malaysian-manglish-nlp to PyPI.

Usage:
    python release_pypi.py          # Build + upload to PyPI
    python release_pypi.py --test   # Build + upload to TestPyPI
    python release_pypi.py --build  # Build only (no upload)
"""
import sys
import os
import shutil
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Clean old builds
for d in ['build', 'dist', 'malaysian_manglish_nlp.egg-info']:
    if os.path.exists(d):
        print(f"Cleaning {d}/")
        shutil.rmtree(d)

# Build
print("\n=== Building sdist + wheel ===")
subprocess.run([sys.executable, '-m', 'build'], check=True)

# Check files
print("\n=== Built files ===")
for f in os.listdir('dist'):
    size = os.path.getsize(os.path.join('dist', f))
    print(f"  {f} ({size/1024:.1f} KB)")

# Upload
if '--build' in sys.argv:
    print("\n=== Build only, skipping upload ===")
elif '--test' in sys.argv:
    print("\n=== Uploading to TestPyPI ===")
    subprocess.run([
        sys.executable, '-m', 'twine', 'upload',
        '--repository', 'testpypi',
        'dist/*'
    ], check=True)
    print("\nTest install:")
    print("  pip install --index-url https://test.pypi.org/simple/ malaysian-manglish-nlp")
else:
    print("\n=== Uploading to PyPI ===")
    subprocess.run([
        sys.executable, '-m', 'twine', 'upload',
        'dist/*'
    ], check=True)
    print("\nInstall:")
    print("  pip install malaysian-manglish-nlp")
