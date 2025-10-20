import pandas as pd

mapping_1_old = {
    'NKETCJF0EC': 0,
    'NKETCJF0FC': 1,
    'NKETCJF0M0': 2,
    'NKETCJF0M6': 3,
    'NKETCJF0R3': 4,
    'NKETCJF0R8': 5,
}

mapping_1 = {
    'NKETCJF0NW': 0,
    'NKETCJF0EA': 1,
    'NKETCJF0EC': 2,
    'NKEUCKP0QB': 3,
    'NKETCJF0NY': 4,
    'NKETCJF0JY': 5,
    'NKETCJF0JA': 6,
    'NHEUCKG067': 7,
    'NKETCJF0JD': 8,
    'NKETCJF0JC': 9,
    'NKETCJF0NL': 10,
    'NKETCJF0J4': 11,
    'NKETCJF0QZ': 12,
    'NKETCJF0J3': 13,
    'NKETCJF0J5': 14,
    'NKETCJF0J8': 15,
    'NKETCJF0J7': 16,
    'NKETCJF0RD': 17,
    'NKETCJF0RC': 18,
    'NKETCJF0MD': 19,
    'NKETCJF0MG': 20,
    'NKETCJF0R2': 21,
    'NKETCJF0R1': 22,
    'NKETCJF0R4': 23,
    'NKETCJF0V7': 24,
    'NKETCJF0R3': 25,
    'NKETCJF0M0': 26,
    'NKETCJF0PV': 27,
    'NKETCJF0LR': 28,
    'NKETCJF0M3': 29,
    'NKETCJF0Q6': 30,
    'NKETCJF0Q0': 31,
    'NKETCJF0Q3': 32,
    'NKETCJF0NX': 33,
    'MNK0CDN0C7': 34,
    'NKETCJF0FC': 35,
    'NKETCJF0NM': 36,
    'NKETCJF0NP': 37,
    'NKETCJF0R6': 38,
    'NKETCJF0EK': 39,
    'NKETCJF0R8': 40,
    'NKETCJF0R9': 41,
    'NKETCJF0M2': 42,
    'NKETCJF0M5': 43,
    'NKETCJF0PZ': 44,
    'NKETCJF0M7': 45,
    'NKETCJF0LU': 46,
    'NKETCJF0M6': 47,
    'NKETCJF0M9': 48,
    'NKETCJF0M8': 49,
    'NKETCJF0RE': 50,
    'NKETCJF0PJ': 51,
    'NKETCJF0Q1': 52,
    'NKETCJF0LM': 53,
    'NKETCJF0P7': 54,
    'NKETCJF0P2': 55,
    'NKETCJF0NQ': 56,
    'NKETCJF0NR': 57,
    'NKETCJF0NT': 58,
    'NKETCJF0NS': 59,
    'NHEUCKG08G': 60
}


mapping_2 = {
    "6T2559050609": 0,
    "6T2559050616": 1,
    "ES2530190098": 2,
    "ES2550147530": 3,
    "ES2550147561": 4,
    "GR2549064221": 5,
    "GR2549064254": 6,
    "GR2549064272": 7,
    "GR2549064278": 8,
    "GR2549064280": 9,
    "GR2549064281": 10,
    "GR2549064283": 11,
    "GR2549064288": 12
}

mapping_3 = {
    "GR24C9052308": 0,
    "GR24C9052206": 1,
    "GR24C9052296": 2,
    "GR24C9052323": 3,
    "GR24C9052197": 4,
    "GR24C9052270": 5,
    "GR24C9052317": 6,
    "GR24C9052328": 7,
    "GR24C9052205": 8,
    "GR24C9052324": 9,
    "GR24C9052301": 10,
    "GR24C9052265": 11,
    "GR24C9052224": 12,
    "GR24C9052329": 13
}

mapping_4 = {
    "ES2540029169": 0,
    "GR2549032153": 1,
    "6T2549037566": 2,
    "ES2540028834": 3,
    "6T2549037573": 4,
    "ES2540097345": 5,
    "ES2540027303": 6,
    "ES2540043352": 7,
    "ES2540028841": 8,
    "ES2540097358": 9,
    "ES2540097367": 10,
    "ES2540097354": 11,
    "GR2549032171": 12
}

# 5电站映射
mapping_5 = {
    "ES2530073550": 0,
    "ES2530073554": 1,
    "ES2540028715": 2,
    "ES2540028725": 3,
    "ES2540028727": 4,
    "ES2540028762": 5,
    "ES2540096963": 6,
    "GR2549064166": 7,
    "GR2549064175": 8
}

# 惠州电站映射
mapping_CS01= {
    "30090549A00248H06969": 0,
    "30081527C002G001A01253H26763": 1,
    "30081527C002G001A01253H26771": 2,
    "30081527C002G001A01253H10655": 3,
    "30100205F00253H36034": 4,
    "30081661AC002A01254H41684": 5,
    "30081661AC002A01254H41526": 6,
    "30081661AC002A01254H41685": 7,
    "30081661AC002A01254H41571": 8,
    "30081661AC002A01254H41678": 9,
    "30081661AC002A01254H41596": 10,
    "30081661AC002A01254H41557": 11,
    "30081527C002G001A01253H26774": 12,
    "30100205F00253H36009": 13
}

# 深圳塘头学校
mapping_SZTT01= {
    "GR2539079574": 0,
    "6T2549037641": 1,
    "6T2549037586": 2
}

# 深圳兴围学校
mapping_SZXW01= {
    '2549101064': 0,
    '2549101063': 1
}

# 汤西敬老院
mapping_CS02= {
    "PFKQE5E05U": 0,
    "SHP0E330E3": 1,
    "PFKQE5E05Y": 2
}

DEFAULT_MAPPINGS = {
    1941310812125962333: mapping_1,
    1941310480876609537: mapping_2,
    1923271358890020865: mapping_3,
    1941310656240459778: mapping_4,
    1941310812125962241: mapping_5,
    1963133351044857857: mapping_SZTT01,
    1963133607132282882: mapping_SZXW01,
    1: mapping_CS01,
    1923271358890020888: mapping_CS02
}

def inverter_features(codes, station_id):

    sta_mappings = DEFAULT_MAPPINGS
    if station_id not in sta_mappings:
        raise KeyError(f"未找到 station_id={station_id} 的逆变器映射，可在 DEFAULT_MAPPINGS 中补充。")

    mapping = dict(sta_mappings[station_id])

    n_class = len(mapping)
    inv_id = pd.Series(codes).map(mapping)

    # 检查是否有未映射的值
    if inv_id.isna().any():
        missing_codes = codes[inv_id.isna()]
        raise ValueError(f"以下逆变器型号未在映射表中找到: {missing_codes}")

    feats = inv_id.to_numpy().astype(int)
    feats = feats / (n_class - 1) - 0.5
    return feats.reshape(-1, 1)