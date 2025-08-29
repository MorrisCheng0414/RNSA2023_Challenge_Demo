import pandas as pd
import numpy as np

def main():
    train_df = pd.read_csv("./train.csv")
    meta_df = pd.read_csv("./train_series_meta.csv")
    
    merge_df = train_df.merge(meta_df, how = "inner", on = "patient_id") # Merge patient_id, series_id, and label information
    merge_df.drop(["aortic_hu", "incomplete_organ"], axis = 1, inplace = True) # Drop unused column
    merge_df[["patient_id", "series_id"]] = merge_df[["patient_id", "series_id"]].astype(str) # Convert ID from int to str
    # Move column
    series_id_col = merge_df.pop("series_id")
    merge_df.insert(1, series_id_col.name, series_id_col)
    print(merge_df.head(5))

    row = merge_df.loc[(merge_df["patient_id"] == "10004") & (merge_df["series_id"] == "21057")]
    print(row.values)
    bowel, extra, kidney, liver, spleen = row.iloc[:, 2 : 4], row.iloc[:, 4 : 6], row.iloc[:, 6 : 9], row.iloc[:, 9 : 12], row.iloc[:, 12 : 15]
    
    print(np.argmax(bowel), np.argmax(extra), np.argmax(kidney), np.argmax(liver), np.argmax(spleen))

if __name__ == "__main__":
    main()