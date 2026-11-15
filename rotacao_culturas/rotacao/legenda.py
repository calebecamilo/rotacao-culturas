"""Legenda comum de culturas e tabelas de conversão de cada fonte.

Este arquivo é a fonte da verdade da harmonização. Para mudar uma decisão (ex.: mandar
"sweet corn" para hortaliças em vez de milho), edite aqui e rode o pipeline de novo.
`python pipeline.py legendas` exporta tudo em CSV para conferência.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------------------
# Legenda comum: cultura -> (grupo, é lavoura temporária/entra na rotação?)
# --------------------------------------------------------------------------------------
LEGENDA: dict[str, tuple[str, bool]] = {
    # cereais
    "milho": ("cereal", True), "trigo": ("cereal", True), "cevada": ("cereal", True),
    "aveia": ("cereal", True), "centeio": ("cereal", True), "triticale": ("cereal", True),
    "arroz": ("cereal", True), "sorgo": ("cereal", True), "milheto": ("cereal", True),
    "outros_cereais": ("cereal", True),
    # leguminosas de grão (fixadoras de N)
    "soja": ("leguminosa_grao", True), "feijao": ("leguminosa_grao", True),
    "ervilha": ("leguminosa_grao", True), "lentilha": ("leguminosa_grao", True),
    "grao_de_bico": ("leguminosa_grao", True), "amendoim": ("leguminosa_grao", True),
    "outras_leguminosas_graos": ("leguminosa_grao", True),
    # oleaginosas
    "girassol": ("oleaginosa", True), "canola_colza": ("oleaginosa", True),
    "linho": ("oleaginosa", True), "outras_oleaginosas": ("oleaginosa", True),
    # raízes e tubérculos
    "batata": ("raiz_tuberculo", True), "batata_doce": ("raiz_tuberculo", True),
    "beterraba_acucareira": ("raiz_tuberculo", True),
    # industriais
    "algodao": ("industrial", True), "cana_de_acucar": ("industrial", True),
    "tabaco": ("industrial", True), "lupulo": ("industrial", True),
    "outras_industriais": ("industrial", True),
    # forrageiras
    "alfafa": ("forrageira", True), "forrageiras_leguminosas": ("forrageira", True),
    "pastagem_temporaria_feno": ("forrageira", True),
    # outras temporárias
    "hortalicas": ("hortalica", True), "pousio": ("pousio", True),
    "outras_temporarias": ("outra_temporaria", True),
    # perenes (não entram na rotação anual)
    "fruticultura": ("perene", False), "citros": ("perene", False), "vinhedos": ("perene", False),
    "oliveiras": ("perene", False), "nozes_castanhas": ("perene", False),
    "frutas_vermelhas": ("perene", False), "cafe": ("perene", False), "dende": ("perene", False),
    "outras_perenes": ("perene", False),
    # outros usos
    "pastagem_campo": ("pastagem", False), "silvicultura": ("silvicultura", False),
    "nao_agricola": ("nao_agricola", False), "desconhecido": ("desconhecido", False),
}


def info_cultura(cultura: str) -> tuple[str, bool]:
    return LEGENDA.get(cultura, LEGENDA["desconhecido"])


def anexar_grupo(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche `grupo` e `lavoura_temporaria` a partir da coluna `cultura`."""
    df["cultura"] = df["cultura"].fillna("desconhecido")
    df["grupo"] = df["cultura"].map(lambda c: info_cultura(c)[0])
    df["lavoura_temporaria"] = df["cultura"].map(lambda c: info_cultura(c)[1])
    return df


# --------------------------------------------------------------------------------------
# USDA Cropland Data Layer (códigos usados nas colunas CDLyyyy do CSB)
# código -> (nome original, cultura, cultura_2 [safra dupla], estação)
# --------------------------------------------------------------------------------------
_V = "hortalicas"
CDL: dict[int, tuple[str, str, str | None, str | None]] = {
    0: ("Background", "desconhecido", None, None),
    1: ("Corn", "milho", None, None), 2: ("Cotton", "algodao", None, None),
    3: ("Rice", "arroz", None, None), 4: ("Sorghum", "sorgo", None, None),
    5: ("Soybeans", "soja", None, None), 6: ("Sunflower", "girassol", None, None),
    10: ("Peanuts", "amendoim", None, None), 11: ("Tobacco", "tabaco", None, None),
    12: ("Sweet Corn", "milho", None, None), 13: ("Pop or Orn Corn", "milho", None, None),
    14: ("Mint", "outras_industriais", None, None), 21: ("Barley", "cevada", None, None),
    22: ("Durum Wheat", "trigo", None, None), 23: ("Spring Wheat", "trigo", None, "primavera"),
    24: ("Winter Wheat", "trigo", None, "inverno"), 25: ("Other Small Grains", "outros_cereais", None, None),
    26: ("Dbl Crop WinWht/Soybeans", "trigo", "soja", "inverno"), 27: ("Rye", "centeio", None, None),
    28: ("Oats", "aveia", None, None), 29: ("Millet", "milheto", None, None),
    30: ("Speltz", "trigo", None, None), 31: ("Canola", "canola_colza", None, None),
    32: ("Flaxseed", "linho", None, None), 33: ("Safflower", "outras_oleaginosas", None, None),
    34: ("Rape Seed", "canola_colza", None, None), 35: ("Mustard", "outras_oleaginosas", None, None),
    36: ("Alfalfa", "alfafa", None, None), 37: ("Other Hay/Non Alfalfa", "pastagem_temporaria_feno", None, None),
    38: ("Camelina", "outras_oleaginosas", None, None), 39: ("Buckwheat", "outros_cereais", None, None),
    41: ("Sugarbeets", "beterraba_acucareira", None, None), 42: ("Dry Beans", "feijao", None, None),
    43: ("Potatoes", "batata", None, None), 44: ("Other Crops", "outras_temporarias", None, None),
    45: ("Sugarcane", "cana_de_acucar", None, None), 46: ("Sweet Potatoes", "batata_doce", None, None),
    47: ("Misc Vegs & Fruits", _V, None, None), 48: ("Watermelons", _V, None, None),
    49: ("Onions", _V, None, None), 50: ("Cucumbers", _V, None, None),
    51: ("Chick Peas", "grao_de_bico", None, None), 52: ("Lentils", "lentilha", None, None),
    53: ("Peas", "ervilha", None, None), 54: ("Tomatoes", _V, None, None),
    55: ("Caneberries", "frutas_vermelhas", None, None), 56: ("Hops", "lupulo", None, None),
    57: ("Herbs", "outras_industriais", None, None), 58: ("Clover/Wildflowers", "forrageiras_leguminosas", None, None),
    59: ("Sod/Grass Seed", "pastagem_temporaria_feno", None, None), 60: ("Switchgrass", "outras_industriais", None, None),
    61: ("Fallow/Idle Cropland", "pousio", None, None),
    63: ("Forest", "nao_agricola", None, None), 64: ("Shrubland", "nao_agricola", None, None),
    65: ("Barren", "nao_agricola", None, None), 66: ("Cherries", "fruticultura", None, None),
    67: ("Peaches", "fruticultura", None, None), 68: ("Apples", "fruticultura", None, None),
    69: ("Grapes", "vinhedos", None, None), 70: ("Christmas Trees", "silvicultura", None, None),
    71: ("Other Tree Crops", "outras_perenes", None, None), 72: ("Citrus", "citros", None, None),
    74: ("Pecans", "nozes_castanhas", None, None), 75: ("Almonds", "nozes_castanhas", None, None),
    76: ("Walnuts", "nozes_castanhas", None, None), 77: ("Pears", "fruticultura", None, None),
    81: ("Clouds/No Data", "desconhecido", None, None), 82: ("Developed", "nao_agricola", None, None),
    83: ("Water", "nao_agricola", None, None), 87: ("Wetlands", "nao_agricola", None, None),
    88: ("Nonag/Undefined", "nao_agricola", None, None), 92: ("Aquaculture", "nao_agricola", None, None),
    111: ("Open Water", "nao_agricola", None, None), 112: ("Perennial Ice/Snow", "nao_agricola", None, None),
    121: ("Developed/Open Space", "nao_agricola", None, None), 122: ("Developed/Low Intensity", "nao_agricola", None, None),
    123: ("Developed/Med Intensity", "nao_agricola", None, None), 124: ("Developed/High Intensity", "nao_agricola", None, None),
    131: ("Barren", "nao_agricola", None, None), 141: ("Deciduous Forest", "nao_agricola", None, None),
    142: ("Evergreen Forest", "nao_agricola", None, None), 143: ("Mixed Forest", "nao_agricola", None, None),
    152: ("Shrubland", "nao_agricola", None, None), 176: ("Grass/Pasture", "pastagem_campo", None, None),
    190: ("Woody Wetlands", "nao_agricola", None, None), 195: ("Herbaceous Wetlands", "nao_agricola", None, None),
    204: ("Pistachios", "nozes_castanhas", None, None), 205: ("Triticale", "triticale", None, None),
    206: ("Carrots", _V, None, None), 207: ("Asparagus", _V, None, None), 208: ("Garlic", _V, None, None),
    209: ("Cantaloupes", _V, None, None), 210: ("Prunes", "fruticultura", None, None),
    211: ("Olives", "oliveiras", None, None), 212: ("Oranges", "citros", None, None),
    213: ("Honeydew Melons", _V, None, None), 214: ("Broccoli", _V, None, None),
    215: ("Avocados", "fruticultura", None, None), 216: ("Peppers", _V, None, None),
    217: ("Pomegranates", "fruticultura", None, None), 218: ("Nectarines", "fruticultura", None, None),
    219: ("Greens", _V, None, None), 220: ("Plums", "fruticultura", None, None),
    221: ("Strawberries", _V, None, None), 222: ("Squash", _V, None, None),
    223: ("Apricots", "fruticultura", None, None), 224: ("Vetch", "forrageiras_leguminosas", None, None),
    225: ("Dbl Crop WinWht/Corn", "trigo", "milho", "inverno"),
    226: ("Dbl Crop Oats/Corn", "aveia", "milho", None), 227: ("Lettuce", _V, None, None),
    228: ("Dbl Crop Triticale/Corn", "triticale", "milho", None), 229: ("Pumpkins", _V, None, None),
    230: ("Dbl Crop Lettuce/Durum Wht", _V, "trigo", None),
    231: ("Dbl Crop Lettuce/Cantaloupe", _V, _V, None),
    232: ("Dbl Crop Lettuce/Cotton", _V, "algodao", None),
    233: ("Dbl Crop Lettuce/Barley", _V, "cevada", None),
    234: ("Dbl Crop Durum Wht/Sorghum", "trigo", "sorgo", None),
    235: ("Dbl Crop Barley/Sorghum", "cevada", "sorgo", None),
    236: ("Dbl Crop WinWht/Sorghum", "trigo", "sorgo", "inverno"),
    237: ("Dbl Crop Barley/Corn", "cevada", "milho", None),
    238: ("Dbl Crop WinWht/Cotton", "trigo", "algodao", "inverno"),
    239: ("Dbl Crop Soybeans/Cotton", "soja", "algodao", None),
    240: ("Dbl Crop Soybeans/Oats", "soja", "aveia", None),
    241: ("Dbl Crop Corn/Soybeans", "milho", "soja", None),
    242: ("Blueberries", "frutas_vermelhas", None, None), 243: ("Cabbage", _V, None, None),
    244: ("Cauliflower", _V, None, None), 245: ("Celery", _V, None, None),
    246: ("Radishes", _V, None, None), 247: ("Turnips", _V, None, None),
    248: ("Eggplants", _V, None, None), 249: ("Gourds", _V, None, None),
    250: ("Cranberries", "frutas_vermelhas", None, None),
    254: ("Dbl Crop Barley/Soybeans", "cevada", "soja", None),
}

# --------------------------------------------------------------------------------------
# MapBiomas Brasil – Coleção 10 (banda "classification")
# código -> (nome original, cultura)
# --------------------------------------------------------------------------------------
MAPBIOMAS: dict[int, tuple[str, str]] = {
    0: ("Não observado", "desconhecido"),
    1: ("Floresta", "nao_agricola"), 3: ("Formação Florestal", "nao_agricola"),
    4: ("Formação Savânica", "nao_agricola"), 5: ("Mangue", "nao_agricola"),
    6: ("Floresta Alagável", "nao_agricola"), 9: ("Silvicultura", "silvicultura"),
    10: ("Vegetação Herbácea e Arbustiva", "nao_agricola"), 11: ("Campo Alagado e Área Pantanosa", "nao_agricola"),
    12: ("Formação Campestre", "nao_agricola"), 13: ("Outras Formações não Florestais", "nao_agricola"),
    14: ("Agropecuária", "desconhecido"), 15: ("Pastagem", "pastagem_campo"),
    18: ("Agricultura", "outras_temporarias"), 19: ("Lavoura Temporária", "outras_temporarias"),
    20: ("Cana", "cana_de_acucar"), 21: ("Mosaico de Usos", "desconhecido"),
    22: ("Área não Vegetada", "nao_agricola"), 23: ("Praia, Duna e Areal", "nao_agricola"),
    24: ("Área Urbanizada", "nao_agricola"), 25: ("Outras Áreas não Vegetadas", "nao_agricola"),
    26: ("Corpo D'água", "nao_agricola"), 29: ("Afloramento Rochoso", "nao_agricola"),
    30: ("Mineração", "nao_agricola"), 31: ("Aquicultura", "nao_agricola"),
    32: ("Apicum", "nao_agricola"), 33: ("Rio, Lago e Oceano", "nao_agricola"),
    35: ("Dendê", "dende"), 36: ("Lavoura Perene", "outras_perenes"),
    39: ("Soja", "soja"), 40: ("Arroz", "arroz"), 41: ("Outras Lavouras Temporárias", "outras_temporarias"),
    46: ("Café", "cafe"), 47: ("Citrus", "citros"), 48: ("Outras Lavouras Perenes", "outras_perenes"),
    49: ("Restinga Arbórea", "nao_agricola"), 50: ("Restinga Herbácea", "nao_agricola"),
    62: ("Algodão (beta)", "algodao"), 75: ("Usina Fotovoltaica (beta)", "nao_agricola"),
}

# --------------------------------------------------------------------------------------
# HCAT4 (EuroCrops v2.0): regras por prefixo do código de 10 dígitos.
# O prefixo mais longo que casar vence.
# --------------------------------------------------------------------------------------
REGRAS_HCAT4: list[tuple[str, str]] = [
    ("3310101", "trigo"), ("3310102", "trigo"), ("3310103", "centeio"), ("3310104", "cevada"),
    ("3310105", "aveia"), ("3310106", "milho"), ("3310107", "arroz"), ("3310108", "triticale"),
    ("3310109", "milheto"), ("3310110", "trigo"),  # espelta
    ("3310111", "outros_cereais"), ("3310112", "trigo"), ("3310113", "trigo"),  # emmer, einkorn
    ("3310115", "outros_cereais"), ("3310116", "sorgo"), ("33101", "outros_cereais"),
    ("331020101", "feijao"), ("331020102", "amendoim"), ("331020103", "soja"),
    ("331020104", "grao_de_bico"), ("331020107", "lentilha"), ("331020108", "ervilha"),
    ("3310201", "outras_leguminosas_graos"), ("331020201", "alfafa"),
    ("3310202", "forrageiras_leguminosas"), ("33102", "outras_leguminosas_graos"),
    ("33103", "batata"), ("33104", "batata_doce"),
    ("3310601", "tabaco"), ("3310602", "lupulo"), ("3310603", "algodao"),
    ("331060401", "canola_colza"), ("331060402", "girassol"), ("3310604", "outras_oleaginosas"),
    ("3310607", "linho"), ("3310615", "outras_oleaginosas"), ("33106", "outras_industriais"),
    ("331071901", "beterraba_acucareira"), ("331071902", "outras_temporarias"),  # beterraba forrageira
    ("33107", "hortalicas"),
    ("33108", "outras_temporarias"),  # flores/ornamentais
    ("33110", "outras_temporarias"),  # sementes e mudas
    ("33111", "pousio"), ("33112", "hortalicas"), ("33113", "outras_temporarias"),  # facélia (cobertura)
    ("33199", "outras_temporarias"), ("3310000000", "outras_temporarias"),
    ("33202", "pastagem_temporaria_feno"), ("3320", "pastagem_campo"),
    ("33301", "fruticultura"), ("33302", "frutas_vermelhas"), ("33303", "nozes_castanhas"),
    ("33304", "citros"), ("33305", "oliveiras"), ("33306", "vinhedos"), ("333", "outras_perenes"),
    ("334", "outras_perenes"),  # cogumelos/trufas
    ("335", "outras_temporarias"),  # estufas
    ("336", "silvicultura"), ("337", "nao_agricola"), ("339", "desconhecido"),
]
_REGRAS_ORD = sorted(REGRAS_HCAT4, key=lambda r: len(r[0]), reverse=True)


def hcat4_para_cultura(codigo: str | None) -> str:
    if not codigo or not isinstance(codigo, str):
        return "desconhecido"
    for prefixo, cultura in _REGRAS_ORD:
        if codigo.startswith(prefixo):
            return cultura
    return "desconhecido"


ESTACAO_HCAT4 = {"winter": "inverno", "spring": "primavera"}


def exportar_legendas(destino: Path, hcat4_csv: Path | None = None) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [(c, g, t) for c, (g, t) in LEGENDA.items()],
        columns=["cultura", "grupo", "lavoura_temporaria"],
    ).to_csv(destino / "legenda_comum.csv", index=False)
    pd.DataFrame(
        [(k, *v) for k, v in CDL.items()],
        columns=["codigo_cdl", "nome_cdl", "cultura", "cultura_2", "estacao"],
    ).to_csv(destino / "cdl_para_comum.csv", index=False)
    pd.DataFrame(
        [(k, *v) for k, v in MAPBIOMAS.items()],
        columns=["codigo_mapbiomas", "nome_mapbiomas", "cultura"],
    ).to_csv(destino / "mapbiomas_para_comum.csv", index=False)
    if hcat4_csv and Path(hcat4_csv).exists():
        h = pd.read_csv(hcat4_csv, dtype=str)
        h["cultura"] = h["hcat4_code"].map(hcat4_para_cultura)
        h.to_csv(destino / "hcat4_para_comum.csv", index=False)
    print(f"[ok] legendas exportadas em {destino}")
