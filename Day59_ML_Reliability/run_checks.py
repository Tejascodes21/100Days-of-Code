from tests.test_data import validate_data
from tests.test_features import validate_features
from tests.test_model import validate_model

print("\n Running ML Reliability Checks...\n")

validate_data()
validate_features()
validate_model()

print("\n All checks passed — system ready for deployment")
