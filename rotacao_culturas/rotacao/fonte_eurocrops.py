"""EuroCrops v2.0 (JRC): declarações de cultura por talhão (GSA/PAC) em 18 regiões da UE, 2008–2023.

Arquivos no FTP do JRC (GeoParquet, EPSG:3035):
  <nuts>_<ano>.parquet  -> colunas cropfield, original_code, area_ha, geometry
  <nuts>_stack.parquet  -> interseção espacial dos anos: cf<ano> (id do talhão no ano) e
                           c<ano> (código original no ano), area_ha, geometry
A tradução dos códigos nacionais para a taxonomia HCAT4 vem de `eurocrops.csv`
(junção por nuts + original_code), no repositório GitHub Martincccc/EuroCropsV2.
Licença: CC BY 4.0.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

from . import util
from .legenda import ESTACAO_HCAT4, anexar_grupo, hcat4_para_cultura

BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/EuroCropsV2/"
VERSOES = ["gpqtv202", "gpqtv201", "gpqt"]  # da mais nova para a mais antiga
MAPEAMENTO = "https://raw.githubusercontent.com/Martincccc/EuroCropsV2/main/data/cropcodemapping/"
LICENCA = "CC BY 4.0 (JRC / Comissão Europeia)"

NUTS_ISO3 = {
    "at": "AUT", "be2": "BEL", "be3": "BEL", "bg": "BGR", "cz": "CZE", "de4": "DEU", "dea": "DEU",
    "dk": "DNK", "ee": "EST", "es": "ESP", "fi": "FIN", "fr": "FRA", "ie": "IRL", "iti1": "ITA",
    "nl": "NLD", "pt": "PRT", "si": "SVN", "sk": "SVK",
}
# Anos disponíveis por região (de all_gpqt_files.csv no FTP).
ANOS = {
    "at": range(2015, 2024), "be2": range(2008, 2024), "be3": range(2015, 2024),
    "bg": range(2014, 2024), "cz": range(2021, 2024), "de4": range(2010, 2024),
    "dea": range(2019, 2024), "dk": range(2010, 2024), "ee": range(2015, 2024),
    "es": range(2021, 2024), "fi": range(2020, 2024), "fr": range(2015, 2024),
    "ie": range(2017, 2024), "iti1": range(2016, 2024), "nl": range(2009, 2024),
    "pt": range(2017, 2023), "si": range(2019, 2024), "sk": range(2018, 2024),
}
_RE_C = re.compile(r"^c(\d{4})$")
_RE_CF = re.compile(r"^cf(\d{4})$")


def baixar_mapeamento() -> tuple[Path, Path]:
    destino = util.pasta("raw", "eurocrops", "mapeamento")
    m = util.baixar(MAPEAMENTO + "eurocrops.csv", destino / "eurocrops.csv", "eurocrops", LICENCA)
    h = util.baixar(MAPEAMENTO + "hcat4.csv", destino / "hcat4.csv", "eurocrops", LICENCA)
    return m, h


def _baixar_arquivo(nome: str, versao: str | None) -> Path:
    versoes = [versao] if versao else VERSOES
    for v in versoes:
        destino = util.pasta("raw", "eurocrops", v) / nome
        if destino.exists():
            return destino
        try:
            return util.baixar(BASE + f"{v}/{nome}", destino, "eurocrops", LICENCA)
        except FileNotFoundError:
            print(f"[info] {nome} não existe em {v}/")
    raise FileNotFoundError(f"{nome} não encontrado em {versoes}")


def baixar(regiao: str, versao: str | None = None, anuais: bool = False) -> dict:
    """Baixa o stack da região (e, se pedido ou se o stack não existir, os arquivos anuais)."""
    regiao = regiao.lower()
    if regiao not in NUTS_ISO3:
        raise ValueError(f"Região {regiao} inválida. Opções: {sorted(NUTS_ISO3)}")
    saida = {"stack": None, "anuais": {}}
    try:
        saida["stack"] = _baixar_arquivo(f"{regiao}_stack.parquet", versao)
    except FileNotFoundError:
        print(f"[aviso] sem stack para {regiao}; vou ligar os anos por pontos internos")
        anuais = True
    if anuais:
        for ano in ANOS[regiao]:
            try:
                saida["anuais"][ano] = _baixar_arquivo(f"{regiao}_{ano}.parquet", versao)
            except FileNotFoundError:
                pass
    return saida


def _limpar_codigo(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.str.replace(r"\.0$", "", regex=True).replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def _stack_por_pontos(anuais: dict[int, Path]) -> gpd.GeoDataFrame:
    """Alternativa ao stack oficial: usa pontos internos dos talhões do ano mais recente e
    descobre, por junção espacial, o código declarado em cada ano anterior."""
    ano_ref = max(anuais)
    ref = gpd.read_parquet(anuais[ano_ref], columns=["cropfield", "area_ha", "geometry"])
    pts = gpd.GeoDataFrame(ref[["cropfield", "area_ha"]], geometry=ref.representative_point(), crs=ref.crs)
    for ano, arq in sorted(anuais.items()):
        g = gpd.read_parquet(arq, columns=["original_code", "geometry"])
        j = gpd.sjoin(pts[["geometry"]], g, how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")]
        pts[f"c{ano}"] = j["original_code"]
    return pts


def _crs_parquet(caminho: Path) -> str:
    meta = pq.read_schema(caminho).metadata or {}
    geo = json.loads(meta.get(b"geo", b"{}") or b"{}")
    col = geo.get("primary_column", "geometry")
    crs = geo.get("columns", {}).get(col, {}).get("crs")
    if isinstance(crs, dict) and crs.get("id"):
        return f"{crs['id'].get('authority', 'EPSG')}:{crs['id'].get('code', 3035)}"
    return "EPSG:3035"  # CRS documentado do EuroCrops v2.0


def _para_longo(ids, area, geoms, anos: dict[int, pd.Series], regiao, mapa, estacoes, crs) -> pd.DataFrame:
    pts = gpd.GeoSeries(geoms, crs=crs).representative_point().to_crs("EPSG:4326")
    base = pd.DataFrame({
        "unidade_id": f"euc_{regiao}_" + pd.Series(ids).astype(str).values,
        "area_ha": np.asarray(area, dtype=float),
        "lon": np.round(pts.x.values, 6), "lat": np.round(pts.y.values, 6),
    })
    tab = pd.DataFrame({ano: _limpar_codigo(pd.Series(v)).values for ano, v in anos.items()})
    longo = pd.concat([base, tab], axis=1).melt(
        id_vars=list(base.columns), var_name="ano", value_name="original_code")
    longo = longo.dropna(subset=["original_code"]).merge(mapa, on="original_code", how="left")
    longo["cultura"] = longo["hcat4_code"].map(hcat4_para_cultura)
    longo["estacao"] = longo["hcat4_code"].map(estacoes).map(ESTACAO_HCAT4)
    longo["cultura_2"] = pd.NA
    longo["codigo_original"] = ("euc:" + longo["original_code"].astype(str)
                                + "|hcat4:" + longo["hcat4_code"].fillna(""))
    longo["ano"] = longo["ano"].astype(int)
    longo["fonte"] = "eurocrops"
    longo["pais_iso3"] = NUTS_ISO3[regiao]
    longo["regiao"] = regiao
    return anexar_grupo(longo)


def harmonizar(regiao: str, arquivos: dict, salvar_geometria: bool = False, lote: int = 500_000) -> None:
    regiao = regiao.lower()
    mapa_csv, hcat4_csv = baixar_mapeamento()
    mapa = pd.read_csv(mapa_csv, dtype=str)
    mapa["nuts"] = mapa["nuts"].str.lower()
    mapa = mapa[mapa["nuts"] == regiao][["original_code", "hcat4_code"]].copy()
    mapa["original_code"] = _limpar_codigo(mapa["original_code"])
    mapa = mapa.drop_duplicates("original_code")
    estacoes = pd.read_csv(hcat4_csv, dtype=str).set_index("hcat4_code")["seasonality_name"]
    util.limpar_fonte_regiao("eurocrops", regiao)

    stack = arquivos.get("stack")
    nomes = pq.read_schema(stack).names if stack else []
    cols_c = {c: int(_RE_C.match(c).group(1)) for c in nomes if _RE_C.match(c)}
    total, sem_mapa = 0, 0

    def gravar(longo, n, ids, geoms, crs):
        nonlocal total, sem_mapa
        total += len(longo)
        sem_mapa += int(longo["hcat4_code"].isna().sum())
        util.escrever_observacoes(longo, lote_id=n)
        if salvar_geometria:
            g = gpd.GeoDataFrame({"unidade_id": f"euc_{regiao}_" + pd.Series(ids).astype(str).values},
                                 geometry=gpd.GeoSeries(geoms, crs=crs))
            util.escrever_unidades(g, "eurocrops", f"{regiao}__{n:04d}")

    if stack and cols_c:
        # Caminho principal: stack oficial com c<ano>, lido em lotes.
        crs = _crs_parquet(stack)
        id_col = next((c for c in ("cropfield", "id", "fid") if c in nomes), None)
        geom_col = "geometry" if "geometry" in nomes else "geom"
        ler = [c for c in [id_col, "area_ha", geom_col, *cols_c] if c and c in nomes]
        arq = pq.ParquetFile(stack)
        inicio = 0
        for n, lote_pa in enumerate(arq.iter_batches(batch_size=lote, columns=ler)):
            df = lote_pa.to_pandas()
            geoms = shapely.from_wkb(df[geom_col].values)
            ids = df[id_col] if id_col else np.arange(inicio, inicio + len(df))
            area = df["area_ha"] if "area_ha" in df else shapely.area(geoms) / 1e4
            anos = {ano: df[c] for c, ano in cols_c.items()}
            gravar(_para_longo(ids, area, geoms, anos, regiao, mapa, estacoes, crs), n, ids, geoms, crs)
            inicio += len(df)
            print(f"[info] {regiao}: {inicio:,} unidades processadas")
    else:
        # Alternativas: stack só com cf<ano> (busca o código no arquivo anual) ou sem stack.
        if stack:
            gdf = gpd.read_parquet(stack)
            for c in [c for c in gdf.columns if _RE_CF.match(c)]:
                ano = int(_RE_CF.match(c).group(1))
                arq_ano = arquivos["anuais"].get(ano) or _baixar_arquivo(f"{regiao}_{ano}.parquet", None)
                a = pd.read_parquet(arq_ano, columns=["cropfield", "original_code"]).set_index("cropfield")
                gdf[f"c{ano}"] = gdf[c].map(a["original_code"])
        else:
            gdf = _stack_por_pontos(arquivos["anuais"])
        cols = {c: int(_RE_C.match(c).group(1)) for c in gdf.columns if _RE_C.match(c)}
        ids = gdf["cropfield"] if "cropfield" in gdf else np.arange(len(gdf))
        area = gdf["area_ha"] if "area_ha" in gdf else gdf.geometry.area / 1e4
        anos = {ano: gdf[c] for c, ano in cols.items()}
        geoms = gdf.geometry.values
        if not stack:  # geometria do talhão do ano de referência, não do ponto
            geoms = gpd.read_parquet(arquivos["anuais"][max(arquivos["anuais"])],
                                     columns=["geometry"]).geometry.values
        gravar(_para_longo(ids, area, geoms, anos, regiao, mapa, estacoes, gdf.crs), 0, ids, geoms, gdf.crs)

    if total and sem_mapa:
        print(f"[aviso] {sem_mapa / total:.1%} das observações de {regiao} sem código HCAT4 "
              f"(ficam como 'desconhecido')")
