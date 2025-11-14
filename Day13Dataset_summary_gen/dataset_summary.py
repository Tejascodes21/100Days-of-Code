from pathlib import Path
import pandas as pd

def get_csv_files(folder_path):
    """Return all CSV files inside the data folder."""
    path = Path(folder_path)
    return list(path.glob("*.csv"))


def summarize_dataset(file_path):
    """Generate basic summary for a dataset."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Error reading {file_path.name}: {e}"

    summary = []
    summary.append(f"File Name     : {file_path.name}")
    summary.append(f"Rows          : {len(df)}")
    summary.append(f"Columns       : {len(df.columns)}")
    summary.append(f"Column Names  : {', '.join(df.columns)}")

    missing = df.isnull().sum().to_dict()
    summary.append("Missing Values:")
    for col, val in missing.items():
        summary.append(f"  - {col}: {val}")

    return "\n".join(summary)


def save_report(content, report_name):
    """Save summary to a text file."""
    reports_folder = Path("reports")
    reports_folder.mkdir(exist_ok=True)

    report_path = reports_folder / f"{report_name}.txt"
    report_path.write_text(content, encoding="utf-8")


def main():
    csv_files = get_csv_files("data")

    if not csv_files:
        print("No CSV files found in /data folder.")
        return

    for file in csv_files:
        print(f"Processing: {file.name}")
        report_text = summarize_dataset(file)
        save_report(report_text, file.stem)

    print("Reports generated successfully in /reports folder.")


if __name__ == "__main__":
    main()
