from langchain.schema import Document

class TextProcessor:
    """Handles advanced text processing, like contextualization."""

    @staticmethod
    def contextualize_documents(docs: list[Document], left: int = 1, right: int = 1):
        """
        For each center chunk, embed a window of neighbors.
        Returns a new list[Document] whose page_content = window(text),
        but whose metadata points to the *center* chunk.
        """
        by_file: dict[str, list[Document]] = {}
        for d in docs:
            by_file.setdefault(d.metadata.get("source_file", ""), []).append(d)

        ctx_docs: list[Document] = []
        for file_name, seq in by_file.items():
            seq.sort(key=lambda d: (d.metadata.get("page", 0), d.metadata.get("chunk", 0)))
            n = len(seq)
            for i, center in enumerate(seq):
                s = max(0, i - left)
                e = min(n, i + right + 1)
                window_text = "\n\n".join(d.page_content for d in seq[s:e])

                window_pages = sorted({d.metadata.get("page") for d in seq[s:e]})
                md = dict(center.metadata)
                md.update({
                    "center_id": f'{file_name}::p{center.metadata.get("page")}::c{center.metadata.get("chunk")}',
                    "window_left": left,
                    "window_right": right,
                    "window_pages": window_pages,
                })
                ctx_docs.append(Document(page_content=window_text, metadata=md))
        return ctx_docs