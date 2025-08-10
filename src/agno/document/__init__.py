class Document:
    def __init__(self, content: str = "", meta_data: dict | None = None):
        self.content = content
        self.meta_data = meta_data or {}
