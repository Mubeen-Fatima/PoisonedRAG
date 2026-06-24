import csv
import json
import os


class GenericDataLoader:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def load(self, split="test"):
        corpus = self._load_jsonl(os.path.join(self.data_folder, "corpus.jsonl"), id_key="_id")
        queries = {
            entry["_id"]: entry["text"]
            for entry in self._iter_jsonl(os.path.join(self.data_folder, "queries.jsonl"))
        }
        qrels = self._load_qrels(os.path.join(self.data_folder, "qrels", f"{split}.tsv"))
        return corpus, queries, qrels

    def _iter_jsonl(self, path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _load_jsonl(self, path, id_key="_id"):
        return {entry[id_key]: entry for entry in self._iter_jsonl(path)}

    def _load_qrels(self, path):
        qrels = {}
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                qid, did, score = row["query-id"], row["corpus-id"], int(row["score"])
                qrels.setdefault(qid, {})[did] = score
        return qrels
