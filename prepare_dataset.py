import os
from urllib.request import urlretrieve
from zipfile import ZipFile

# Download and save dataset
datasets = ['nq', 
            'msmarco', 
            'hotpotqa']
for dataset in datasets:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, dataset)
    if not os.path.exists(data_path):
        os.makedirs(out_dir, exist_ok=True)
        zip_path = os.path.join(out_dir, f"{dataset}.zip")
        urlretrieve(url, zip_path)
        with ZipFile(zip_path) as dataset_zip:
            dataset_zip.extractall(out_dir)
        os.remove(zip_path)
