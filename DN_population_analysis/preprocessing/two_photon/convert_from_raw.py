import os.path
import utils2p
import DN_population_analysis.utils as utils


SKIP_EXISTING = True

def convert_from_raw(directory):
    metadata_file = utils2p.find_metadata_file(directory)
    metadata = utils2p.Metadata(metadata_file)
    raw_file = utils2p.find_raw_file(directory)
    green, red = utils2p.load_raw(raw_file, metadata)
    utils2p.save_img(os.path.join(directory, "green.tif"), green)
    utils2p.save_img(os.path.join(directory, "red.tif"), red)

if __name__ == "__main__":
    
    f = "../../../recordings.txt"
    
    directories = utils.load_exp_dirs(f)

    for directory in directories:
        print("\n" + directory + "\n" + len(directory) * "#")
        if SKIP_EXISTING and os.path.isfile(os.path.join(directory, "2p/green.tif")) and os.path.isfile(os.path.join(directory, "2p/red.tif")):
            print("Skipping because files already exist.")
            continue
        convert_from_raw(os.path.join(directory, "2p"))
