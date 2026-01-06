def validate_input(features):
    if not isinstance(features, list):
        return False, "Input is not a list"

    if len(features) != 5:
        return False, "Invalid feature length"

    for value in features:
        if not isinstance(value, (int, float)):
            return False, "Non-numeric value detected"

        if value < 0 or value > 100:
            return False, "Feature value out of safe range"

    return True, "Valid input"
