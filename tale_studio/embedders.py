from sentence_transformers import SentenceTransformer


class EmbeddersStorage:
    embedders = dict()

    @classmethod
    def get_embedder(cls, embedder_name: str):
        if embedder_name not in cls.embedders:
            cls.embedders[embedder_name] = SentenceTransformer(embedder_name)
        return cls.embedders[embedder_name]


EMBEDDER_LIST = [
    "embaas/sentence-transformers-multilingual-e5-base",
    "sentence-transformers/multi-qa-mpnet-base-cos-v1"
]
