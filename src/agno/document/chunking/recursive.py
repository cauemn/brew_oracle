class RecursiveChunking:
    def __init__(self, chunk_size: int = 0, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap
