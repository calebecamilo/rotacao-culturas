"""USDA NASS Crop Sequence Boundaries (CSB): talhões dos EUA com 8 anos de culturas do CDL.

Cada polígono tem colunas CDL<ano> (código do Cropland Data Layer em cada ano), CSBID, CSBACRES,
STATEFIPS, CNTYFIPS, INSIDE_X/INSIDE_Y (ponto interno, na projeção do arquivo).
Distribuído como .zip com um File Geodatabase (.gdb) nacional. Domínio público.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
from pyproj import Transformer

from . import util
from .legenda import CDL, anexar_grupo

URL = ("https://www.nass.usda.gov/sites/default/files/Research_and_Science/"
       "Crop-Sequence-Boundaries/datasets/NationalCSB_{ini}-{fim}_rev23.zip")
JANELAS = [(a, a + 7) for a in range(2008, 2019)]  # 2008-2015 ... 2018-2025
LICENCA = "Domínio público (USDA NASS)"

FIPS = {
    "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "DC": "11",
    "FL": "12", "GA": "13", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29",
    "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37",
    "ND": "38", "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56",
}
_RE_ANO = re.compile(r"^(?:CDL|R)(\d{4}|\d{2})$", re.IGNORECASE)


def baixar(ini: int = 2018, fim: int = 2025) -> Path:
    if (ini, fim) not in JANELAS:
        raise ValueError(f"Janela {ini}-{fim} inválida. Opções: {[f'{a}-{b}' for a, b in JANELAS]}")
    url = URL.format(ini=ini, fim=fim)
    return util.baixar(url, util.pasta("raw", "usda_csb") / Path(url).name,
                       fonte="usda_csb", licenca=LICENCA)


def _caminho_gdb(zip_path: Path) -> str:
    """Devolve o caminho GDAL /vsizip/ para o .gdb de dentro do zip (sem precisar descompactar)."""
    with zipfile.ZipFile(zip_path) as z:
        for nome in z.namelist():
            partes = nome.split("/")
            for i, p in enumerate(partes):
                if p.lower().endswith(".gdb"):
                    return f"/vsizip/{zip_path.resolve()}/{'/'.join(partes[:i + 1])}"
    raise FileNotFoundError(f"Nenhum .gdb encontrado em {zip_path}")


def _escolher_camada(fonte: str) -> str:
    camadas = pyogrio.list_layers(fonte)  # array [[nome, tipo_geom], ...]
    poligonais = [c[0] for c in camadas if c[1] and "polygon" in str(c[1]).lower()]
    if not poligonais:
        poligonais = [c[0] for c in camadas]
    return max(poligonais, key=lambda n: pyogrio.read_info(fonte, layer=n)["features"])


def _normalizar(df: pd.DataFrame, colunas_ano: dict[str, int], transf: Transformer) -> pd.DataFrame:
    lon, lat = transf.transform(df["INSIDE_X"].to_numpy(), df["INSIDE_Y"].to_numpy())
    base = pd.DataFrame({
        "unidade_id": "usa_csb_" + df["CSBID"].astype(str),
        "regiao": df["STATEFIPS"].astype(str).str.zfill(2),
        "area_ha": df["CSBACRES"].astype(float) * 0.40468564224,
        "lon": np.round(lon, 6), "lat": np.round(lat, 6),
    })
    anos = df[list(colunas_ano)].rename(columns=colunas_ano)
    longo = pd.concat([base, anos], axis=1).melt(
        id_vars=list(base.columns), var_name="ano", value_name="codigo")
    longo = longo.dropna(subset=["codigo"])
    codigo = longo["codigo"].astype(int)
    tab = pd.DataFrame.from_dict(CDL, orient="index", columns=["_nome", "cultura", "cultura_2", "estacao"])
    longo = longo.assign(
        cultura=codigo.map(tab["cultura"]).fillna("desconhecido"),
        cultura_2=codigo.map(tab["cultura_2"]),
        estacao=codigo.map(tab["estacao"]),
    )
    longo["codigo_original"] = "cdl:" + codigo.astype(str)
    longo["ano"] = longo["ano"].astype(int)
    longo["fonte"] = "usda_csb"
    longo["pais_iso3"] = "USA"
    return anexar_grupo(longo.drop(columns=["codigo"]))


def harmonizar(zip_path: Path, estados: list[str] | None = None, salvar_geometria: bool = False,
               lote: int = 1_000_000) -> None:
    fonte = _caminho_gdb(Path(zip_path))
    camada = _escolher_camada(fonte)
    info = pyogrio.read_info(fonte, layer=camada)
    campos = list(info["fields"])
    colunas_ano = {}
    for c in campos:
        m = _RE_ANO.match(c)
        if m:
            g = m.group(1)
            colunas_ano[c] = int(g) if len(g) == 4 else 2000 + int(g)
    if not colunas_ano:
        raise ValueError(f"Não achei colunas de ano (CDLyyyy) na camada {camada}. Campos: {campos}")
    print(f"[info] camada {camada}: {info['features']:,} talhões; anos {sorted(colunas_ano.values())}")
    transf = Transformer.from_crs(info["crs"], "EPSG:4326", always_xy=True)
    colunas = ["CSBID", "CSBACRES", "STATEFIPS", "INSIDE_X", "INSIDE_Y", *colunas_ano]

    def processar(df, lote_id=None):
        obs = _normalizar(df, colunas_ano, transf)
        util.escrever_observacoes(obs, lote_id=lote_id)
        if salvar_geometria and "geometry" in df:
            import geopandas as gpd
            g = gpd.GeoDataFrame({"unidade_id": "usa_csb_" + df["CSBID"].astype(str)},
                                 geometry=df.geometry, crs=info["crs"])
            for reg, parte in g.groupby(df["STATEFIPS"].astype(str).str.zfill(2)):
                util.escrever_unidades(parte, "usda_csb", reg)

    if estados:
        for uf in estados:
            fips = FIPS.get(uf.upper(), uf.zfill(2))
            print(f"[info] lendo estado {uf} (FIPS {fips})")
            df = pyogrio.read_dataframe(fonte, layer=camada, columns=colunas,
                                        where=f"STATEFIPS = '{fips}'", read_geometry=salvar_geometria)
            if len(df):
                processar(df)
            else:
                print(f"[aviso] nenhum talhão para {uf}")
    else:
        # País inteiro, em lotes, para não estourar a memória. Cada lote vira um arquivo
        # dentro da partição do estado; a pasta da fonte é limpa uma vez antes de começar.
        util.limpar_fonte("usda_csb")
        for n, ini in enumerate(range(0, info["features"], lote)):
            df = pyogrio.read_dataframe(fonte, layer=camada, columns=colunas, skip_features=ini,
                                        max_features=lote, read_geometry=False)
            processar(df, lote_id=n)
            print(f"[info] processados {min(ini + lote, info['features']):,}/{info['features']:,}")
        if salvar_geometria:
            print("[aviso] geometria do país inteiro não foi salva; use --estados para salvá-la por estado")
