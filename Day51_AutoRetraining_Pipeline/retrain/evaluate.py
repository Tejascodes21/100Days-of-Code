def compare_models(old_acc, new_acc, threshold=0.02):
    print(f"Champion Accuracy: {old_acc:.2f}")
    print(f"Challenger Accuracy: {new_acc:.2f}")

    if new_acc > old_acc + threshold:
        print("Challenger outperforms champion")
        return True

    print("Challenger rejected")
    return False
