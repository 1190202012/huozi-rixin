def truncate_en_doc(doc: str, max_doc_len: int = None, min_doc_len: int = None):
    """
    truncate doc if doc_len >= max_length, doc will be truncated and discard last uncompleted sentence. if doc_len <= min_length, return will be None. doc_len is defined len(doc.split(" ")) * 2.

    Args:
        doc (str): _description_
        max_doc_len (int, optional): _description_. Defaults to None.
        min_doc_len (int, optional): _description_. Defaults to None.
    """

    if min_doc_len is not None and len(doc.split(" ")) * 2 <= min_doc_len:
        return None

    if max_doc_len is None or len(doc.split(" ")) * 2 <= max_doc_len:
        return doc

    doc = " ".join(doc.split(" ")[:max_doc_len // 2])

    index = len(doc) - 1
    while index >= 0:
        if doc[index] in ".!?\n":
            return doc[:index + 1]
        else:
            index -= 1

    return doc


def truncate_zh_doc(doc: str, max_doc_len: int = None, min_doc_len: int = None):
    if min_doc_len is not None and len(doc) <= min_doc_len:
        return None

    if max_doc_len is None or len(doc) <= max_doc_len:
        return doc

    doc = doc[:max_doc_len]

    index = len(doc) - 1
    while index >= 0:
        if doc[index] in "。！？\n":
            return doc[:index + 1]
        else:
            index -= 1

    return doc
