#!/usr/bin/env python
"""
Installation verification script.

Run this script to verify that the COVID-19 Detection package
is correctly installed and all dependencies are available.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need ≥3.8)")
        return False


def check_dependencies():
    """Check required packages."""
    print("\nChecking dependencies...")

    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'openpyxl': 'openpyxl',
        'joblib': 'joblib'
    }

    all_ok = True
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} (MISSING)")
            all_ok = False

    return all_ok


def check_project_structure():
    """Check project directories."""
    print("\nChecking project structure...")

    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'data/results',
        'src',
        'src/data',
        'src/features',
        'src/models',
        'src/visualization',
        'scripts',
        'notebooks',
        'tests'
    ]

    all_ok = True
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (MISSING)")
            all_ok = False

    return all_ok


def check_data_files():
    """Check if data files exist."""
    print("\nChecking data files...")

    dataset_path = PROJECT_ROOT / 'data' / 'raw' / 'dataset.xlsx'

    if dataset_path.exists():
        print(f"✓ dataset.xlsx ({dataset_path.stat().st_size / 1024:.1f} KB)")
        return True
    else:
        print(f"✗ dataset.xlsx (MISSING)")
        print(f"  Expected location: {dataset_path}")
        return False


def check_imports():
    """Check if project modules can be imported."""
    print("\nChecking project modules...")

    modules_to_check = [
        ('src.config', 'Configuration'),
        ('src.data.preprocessing', 'Data preprocessing'),
        ('src.features.engineering', 'Feature engineering'),
        ('src.models.train', 'Model training'),
        ('src.models.evaluate', 'Model evaluation'),
        ('src.visualization.plots', 'Visualization')
    ]

    all_ok = True
    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            print(f"✓ {description}")
        except ImportError as e:
            print(f"✗ {description} (IMPORT ERROR)")
            print(f"  Error: {e}")
            all_ok = False

    return all_ok


def test_basic_functionality():
    """Test basic package functionality."""
    print("\nTesting basic functionality...")

    try:
        # Test config import
        from src.config import TARGET_FEATURE, RANDOM_STATE
        print(f"✓ Config loaded (target: {TARGET_FEATURE})")

        # Test preprocessing functions
        from src.data.preprocessing import encodage
        import pandas as pd
        test_df = pd.DataFrame({'col': ['positive', 'negative']})
        result = encodage(test_df)
        assert result['col'].tolist() == [1, 0]
        print("✓ Preprocessing functions work")

        # Test feature engineering
        from src.features.engineering import feature_engineering
        test_df = pd.DataFrame({
            'viral1': [1, 0],
            'viral2': [0, 1]
        })
        result = feature_engineering(test_df, ['viral1', 'viral2'])
        assert 'est malade' in result.columns
        print("✓ Feature engineering works")

        # Test model building
        from src.models.train import build_models
        models = build_models()
        assert len(models) > 0
        print(f"✓ Model building works ({len(models)} models)")

        return True

    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 70)
    print("COVID-19 DETECTION PACKAGE - INSTALLATION CHECK")
    print("=" * 70)

    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project structure", check_project_structure),
        ("Data files", check_data_files),
        ("Module imports", check_imports),
        ("Basic functionality", test_basic_functionality)
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error during {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {name}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("=" * 70)
        print("\nYou're ready to use the COVID-19 Detection package!")
        print("\nNext steps:")
        print("  1. Read QUICKSTART.md for usage examples")
        print("  2. Run: python scripts/train_model.py")
        print("  3. Or open: notebooks/00_quick_start.ipynb")
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("=" * 70)
        print("\nPlease fix the issues above before using the package.")
        print("\nCommon solutions:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Install package: pip install -e .")
        print("  - Ensure dataset.xlsx is in data/raw/")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
