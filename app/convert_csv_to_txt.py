# app/convert_csv_to_txt.py
import os
import pandas as pd

def convert_csv_to_txt(csv_folder="data/source_documents"):
    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            csv_path = os.path.join(csv_folder, file)
            df = pd.read_csv(csv_path)

            txt_content = df.to_string(index=False)
            txt_filename = file.replace(".csv", ".txt")

            with open(os.path.join(csv_folder, txt_filename), "w", encoding="utf-8") as f:
                f.write(txt_content)
            print(f"Converted {file} to {txt_filename}")

if __name__ == "__main__":
    convert_csv_to_txt()
