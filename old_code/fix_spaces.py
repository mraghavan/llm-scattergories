import os

if __name__ == '__main__':
    folder = './samples/'
    all_files = os.listdir(folder)
    for fname in all_files:
        if fname.endswith('_verified.pkl') and ' ' in fname:
            full_path = os.path.join(folder, fname)
            new_full_path = full_path.replace(' ', '_')
            print(full_path)
            print(new_full_path)
            # move file from full_path to new_full_path
            os.rename(full_path, new_full_path)
