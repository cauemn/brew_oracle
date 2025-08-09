class QdrantClient:
    def __init__(self, *args, **kwargs):
        pass

    def collection_exists(self, *args, **kwargs):
        return False

    def create_collection(self, *args, **kwargs):
        pass

    def count(self, *args, **kwargs):
        class R:
            count = 0
        return R()
