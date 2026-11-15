"""MapBiomas Brasil – Coleção 10: mapas anuais de uso e cobertura (30 m, 1985–2024).

Um GeoTIFF nacional por ano (EPSG:4326, banda única com o código da classe). Todos os anos
compartilham a mesma grade, então o mesmo pixel (linha, coluna) é a mesma área em todos os anos.

Como não há talhões, a unidade de rotação é o pixel: sorteamos pontos dentro da área de interesse,
mantemos os que tiveram lavoura temporária em pelo menos N anos e lemos a classe em cada ano.
Licença: CC BY 4.0 (citar "Projeto MapBiomas – Coleção 10").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from . import util
from .legenda import MAPBIOMAS, anexar_grupo, info_cultura

URL = ("https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/"
       "collection_10/lulc/coverage/brazil_coverage_{ano}.tif")
ANOS_DISPONIVEIS = range(1985, 2025)
LICENCA = "CC BY 4.0 (Projeto MapBiomas – Coleção 10)"

# Retângulos aproximados (lon_min, lat_min, lon_max, lat_max). Pegam pedaços de estados vizinhos;
# para recorte exato passe --limite com o polígono do estado (ex.: malha do IBGE).
UF_BBOX = {
    "MT": (-61.7, -18.1, -50.2, -7.3), "MS": (-58.2, -24.1, -50.9, -17.1),
    "GO": (-53.3, -19.5, -45.9, -12.4), "PR": (-54.7, -26.8, -48.0, -22.5),
    "RS": (-57.7, -33.8, -49.6, -27.0), "SC": (-53.9, -29.4, -48.3, -25.9),
    "SP": (-53.2, -25.4, -44.1, -19.7), "MG": (-51.1, -23.0, -39.8, -14.2),
    "BA": (-46.7, -18.4, -37.3, -8.5), "TO": (-50.8, -13.5, -45.7, -5.1),
    "MA": (-48.8, -10.3, -41.8, -1.0), "PI": (-45.1, -10.95, -40.3, -2.7),
}

# Classes que contam como lavoura temporária (entram na rotação).
_TEMPORARIAS = np.array([k for k, (_, c) in MAPBIOMAS.items() if info_cultura(c)[1]], dtype=np.int32)


def baixar(anos) -> dict[int, Path]:
    saida = {}
    for ano in anos:
        if ano not in ANOS_DISPONIVEIS:
            raise ValueError(f"Ano {ano} fora de {ANOS_DISPONIVEIS.start}-{ANOS_DISPONIVEIS.stop - 1}")
        url = URL.format(ano=ano)
        saida[ano] = util.baixar(url, util.pasta("raw", "mapbiomas", "colecao10") / Path(url).name,
                                 fonte="mapbiomas", licenca=LICENCA)
    return saida


def _ler_pixels(src, linhas: np.ndarray, colunas: np.ndarray) -> np.ndarray:
    """Lê os valores de muitos pixels agrupando por bloco interno do GeoTIFF (uma leitura por bloco)."""
    bh, bw = src.block_shapes[0]
    vals = np.zeros(len(linhas), dtype=np.int32)
    chave = (linhas // bh) * (src.width // bw + 1) + (colunas // bw)
    ordem = np.argsort(chave, kind="stable")
    k_ord = chave[ordem]
    cortes = np.flatnonzero(np.diff(k_ord)) + 1
    for grupo in np.split(ordem, cortes):
        r0 = (linhas[grupo[0]] // bh) * bh
        c0 = (colunas[grupo[0]] // bw) * bw
        h = min(bh, src.height - r0)
        w = min(bw, src.width - c0)
        bloco = src.read(1, window=Window(c0, r0, w, h))
        vals[grupo] = bloco[linhas[grupo] - r0, colunas[grupo] - c0]
    return vals


def _area_pixel_ha(transform, crs, lat: np.ndarray) -> np.ndarray:
    dx, dy = abs(transform.a), abs(transform.e)
    if crs and crs.is_geographic:
        return (dx * 111_320 * np.cos(np.radians(lat))) * (dy * 110_574) / 1e4
    return np.full(len(lat), dx * dy / 1e4)


def harmonizar(arquivos: dict[int, Path], regiao: str, bbox=None, limite: Path | None = None,
               n_pontos: int = 50_000, min_anos_lavoura: int = 2, semente: int = 42,
               salvar_geometria: bool = False) -> None:
    anos = sorted(arquivos)
    if bbox is None:
        if regiao.upper() not in UF_BBOX and limite is None:
            raise ValueError("Informe --bbox, --limite ou uma UF com retângulo pré-definido "
                             f"({sorted(UF_BBOX)})")
        bbox = UF_BBOX.get(regiao.upper())
    poligono = None
    if limite is not None:
        import geopandas as gpd
        poligono = gpd.read_file(limite).to_crs("EPSG:4326").union_all()
        if bbox is None:
            bbox = poligono.bounds

    fontes = {ano: rasterio.open(arquivos[ano]) for ano in anos}
    ref = fontes[anos[-1]]
    for ano, s in fontes.items():
        if s.transform != ref.transform or s.shape != ref.shape:
            raise ValueError(f"O raster de {ano} não está na mesma grade do de {anos[-1]}")

    rng = np.random.default_rng(semente)
    lon0, lat0, lon1, lat1 = bbox
    aceitos: list[pd.DataFrame] = []
    n_aceitos, tentativas = 0, 0
    vistos: set[int] = set()
    while n_aceitos < n_pontos and tentativas < 50:
        tentativas += 1
        lote = max(20_000, (n_pontos - n_aceitos) * 5)
        lon = rng.uniform(lon0, lon1, lote)
        lat = rng.uniform(lat0, lat1, lote)
        if poligono is not None:
            import shapely
            dentro = shapely.contains_xy(poligono, lon, lat)
            lon, lat = lon[dentro], lat[dentro]
        linhas, colunas = rasterio.transform.rowcol(ref.transform, lon, lat)
        linhas, colunas = np.asarray(linhas), np.asarray(colunas)
        ok = (linhas >= 0) & (linhas < ref.height) & (colunas >= 0) & (colunas < ref.width)
        linhas, colunas = linhas[ok], colunas[ok]
        pix = linhas.astype(np.int64) * ref.width + colunas
        pix, idx = np.unique(pix, return_index=True)
        novo = np.array([p not in vistos for p in pix], dtype=bool)
        linhas, colunas, pix = linhas[idx][novo], colunas[idx][novo], pix[novo]
        if len(pix) == 0:
            continue
        valores = np.column_stack([_ler_pixels(fontes[a], linhas, colunas) for a in anos])
        n_lav = np.isin(valores, _TEMPORARIAS).sum(axis=1)
        manter = n_lav >= min_anos_lavoura
        vistos.update(pix.tolist())
        if not manter.any():
            continue
        linhas, colunas, valores = linhas[manter], colunas[manter], valores[manter]
        xs, ys = rasterio.transform.xy(ref.transform, linhas, colunas)
        df = pd.DataFrame(valores, columns=anos)
        df["unidade_id"] = [f"bra_mb_{r}_{c}" for r, c in zip(linhas, colunas)]
        df["lon"], df["lat"] = np.round(xs, 6), np.round(ys, 6)
        aceitos.append(df)
        n_aceitos += len(df)
        print(f"[info] {n_aceitos:,}/{n_pontos:,} pixels agrícolas amostrados")
    transform, crs = ref.transform, ref.crs
    for s in fontes.values():
        s.close()
    if not aceitos:
        print("[aviso] nenhum pixel com lavoura temporária encontrado na área")
        return

    largo = pd.concat(aceitos, ignore_index=True).head(n_pontos)
    largo["area_ha"] = _area_pixel_ha(transform, crs, largo["lat"].to_numpy())
    longo = largo.melt(id_vars=["unidade_id", "lon", "lat", "area_ha"], var_name="ano", value_name="codigo")
    tab = pd.Series({k: c for k, (_, c) in MAPBIOMAS.items()})
    longo["cultura"] = longo["codigo"].map(tab).fillna("desconhecido")
    longo["codigo_original"] = "mb:" + longo["codigo"].astype(str)
    longo["cultura_2"] = pd.NA
    longo["estacao"] = pd.NA
    longo["ano"] = longo["ano"].astype(int)
    longo["fonte"] = "mapbiomas"
    longo["pais_iso3"] = "BRA"
    longo["regiao"] = regiao.upper()
    util.escrever_observacoes(anexar_grupo(longo))

    if salvar_geometria:
        import geopandas as gpd
        from shapely.geometry import box
        dx, dy = abs(transform.a) / 2, abs(transform.e) / 2
        g = gpd.GeoDataFrame(
            {"unidade_id": largo["unidade_id"]},
            geometry=[box(x - dx, y - dy, x + dx, y + dy) for x, y in zip(largo["lon"], largo["lat"])],
            crs=crs)
        util.escrever_unidades(g, "mapbiomas", regiao.upper())
