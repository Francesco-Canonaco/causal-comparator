import glob
import pandas as pd
import os
NODES = 25
def concatenate_nested_csvs():
    # The '**/' tells Python to look inside 0.15, 0.20, 0.25, etc.
    file_pattern = f"**/grid_results_{NODES}nodes_*.csv"
    
    print(f"🔍 Searching in: {os.getcwd()}")
    
    # recursive=True is what allows it to dive into the subfolders
    files = glob.glob(file_pattern, recursive=True)

    if not files:
        print(f"❌ Error: No files found. Make sure this script is saved directly in the '15_nodes' folder.")
        return

    print(f"✅ Found {len(files)} files to combine!")
        
    try:
        dataframes = []
        for f in files:
            # Read the CSV
            temp_df = pd.read_csv(f)
            # Create a new column with the folder/file name so you know exactly where the data came from
            temp_df['source_file'] = f 
            dataframes.append(temp_df)
            
        # Mash them all together
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Save the master file right next to the script
        output_filename = f"MASTER_grid_results_{NODES}nodes.csv"
        combined_df.to_csv(output_filename, index=False)
        
        print(f"\n🎉 Success! All files concatenated.")
        print(f"💾 Saved as: '{output_filename}'")
        print(f"📊 Total rows in master file: {len(combined_df)}")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    concatenate_nested_csvs()