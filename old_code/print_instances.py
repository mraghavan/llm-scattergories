from file_manager import FileManager
import pandas as pd

if __name__ == '__main__':
    fm = FileManager.from_base('.')
    all_samples = fm.get_all_samples()
    instances = all_samples[['letter', 'category']].drop_duplicates()
    # sort by letter, then category
    instances = instances.sort_values(by=['letter', 'category'])
    # print as latex table
    # replace '_' with ' '
    instances['category'] = instances['category'].str.replace('_', ' ')
    # fix ReptilesAmphibians issue
    instances['category'] = instances['category'].str.replace('ReptilesAmphibians', 'Reptiles/Amphibians')
    print(instances.to_latex(index=False))

