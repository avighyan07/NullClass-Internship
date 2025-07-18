# medquad_parser.py

import os
import glob
import pandas as pd
from lxml import etree

def parse_medquad(folder_path):
    data = []

    xml_files = glob.glob(os.path.join(folder_path, "**/*.xml"), recursive=True)

    for file in xml_files:
        tree = etree.parse(file)
        root = tree.getroot()

        for qa in root.findall(".//question"):
            question_text = qa.text.strip() if qa.text else ""
            next_elem = qa.getnext()
            answer = next_elem.text.strip() if next_elem is not None else ""

            data.append({
                "question": question_text,
                "answer": answer
            })

    df = pd.DataFrame(data)
    return df
