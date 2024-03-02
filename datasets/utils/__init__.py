import requests
import tqdm

CHUNK_SIZE = 1 * 1024 * 1024


# From https://github.com/learnables/learn2learn/blob/master/learn2learn/data/utils.py
def download_file(source, destination, size=None):
    if size is None:
        size = 0
    req = requests.get(source, stream=True)
    with open(destination, "wb") as archive:
        for chunk in tqdm.tqdm(
            req.iter_content(chunk_size=CHUNK_SIZE),
            total=size // CHUNK_SIZE,
            leave=False,
        ):
            if chunk:
                archive.write(chunk)
