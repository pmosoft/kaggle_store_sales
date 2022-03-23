import pandas as pd
import glob
import os
import json


# path내 json파일 모두 읽어 하나의 df로 리턴
# int가 string으로 오변환 주의
def read_jsons_to_pandas(path):
    json_pattern = os.path.join(path, '*.json')
    file_list = glob.glob(json_pattern)
    dfs = []
    for file in file_list:
        with open(file) as f:
            for line in f.readlines():
                json_data = pd.json_normalize(json.loads(line))
                dfs.append(json_data)

    return pd.concat(dfs, sort=False)


