"""Gera arquivos sintéticos no formato das fontes reais, para testar o pipeline offline."""
import shutil
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

raiz = Path(sys.argv[1])
rng = np.random.default_rng(0)

# ---------------- USDA CSB: FileGDB zipado ----------------
n = 3000
x0, y0 = 200_000, 1_900_000  # Albers (EPSG:5070), região de Iowa aprox.
geoms = [box(x0 + (i % 60) * 500, y0 + (i // 60) * 500, x0 + (i % 60) * 500 + 400, y0 + (i // 60) * 500 + 400)
         for i in range(n)]
anos = range(2018, 2026)
padroes = [[1, 5], [1, 1, 5], [5, 1], [24, 26], [36, 36, 36, 1], [1], [61, 24]]
dados = {"CSBID": [f"19{2018}{i:09d}" for i in range(n)], "CSBYEARS": "1825",
         "CSBACRES": rng.uniform(10, 200, n),
         "STATEFIPS": np.where(np.arange(n) < 2000, "19", "17"),
         "CNTYFIPS": "001"}
for i, ano in enumerate(anos):
    dados[f"CDL{ano}"] = [padroes[k % len(padroes)][(i + k) % len(padroes[k % len(padroes)])] for k in range(n)]
g = gpd.GeoDataFrame(dados, geometry=geoms, crs="EPSG:5070")
g["INSIDE_X"] = g.geometry.representative_point().x
g["INSIDE_Y"] = g.geometry.representative_point().y
tmp = raiz / "_tmp_csb"
shutil.rmtree(tmp, ignore_errors=True)
(tmp / "NationalCSB_2018-2025_rev23").mkdir(parents=True)
gdb = tmp / "NationalCSB_2018-2025_rev23" / "CSB1825.gdb"
pyogrio.write_dataframe(g, gdb, layer="national1825", driver="OpenFileGDB")
destino = raiz / "raw" / "usda_csb" / "NationalCSB_2018-2025_rev23.zip"
destino.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
    for f in gdb.rglob("*"):
        z.write(f, f.relative_to(tmp))
shutil.rmtree(tmp)
print("CSB ok", destino)

# ---------------- EuroCrops: stack (dk) e anuais (at) ----------------
mapa = pd.read_csv(raiz / "raw" / "eurocrops" / "mapeamento" / "eurocrops.csv", dtype=str)
cod_dk = mapa[mapa.nuts == "dk"].original_code.sample(25, random_state=1).tolist()
m = 2000
geoms = [box(4_300_000 + (i % 50) * 300, 3_600_000 + (i // 50) * 300,
             4_300_000 + (i % 50) * 300 + 250, 3_600_000 + (i // 50) * 300 + 250) for i in range(m)]
st = {"cropfield": np.arange(m), "area_ha": 6.25}
for j, ano in enumerate(range(2019, 2024)):
    col = [cod_dk[(k + j * (k % 3 + 1)) % len(cod_dk)] for k in range(m)]
    col = [None if (k % 97 == 0 and j == 2) else c for k, c in enumerate(col)]
    col = ["99999" if k % 211 == 0 else c for k, c in enumerate(col)]  # código sem mapeamento
    st[f"cf{ano}"] = np.arange(m) + j * 10_000
    st[f"c{ano}"] = col
d = raiz / "raw" / "eurocrops" / "gpqtv202"
d.mkdir(parents=True, exist_ok=True)
gpd.GeoDataFrame(st, geometry=geoms, crs="EPSG:3035").to_parquet(d / "dk_stack.parquet")

cod_at = mapa[mapa.nuts == "at"].original_code.sample(10, random_state=2).tolist()
for j, ano in enumerate(range(2015, 2024)):
    k = 500
    gg = [box(4_700_000 + (i % 25) * 200, 2_800_000 + (i // 25) * 200,
              4_700_000 + (i % 25) * 200 + 180, 2_800_000 + (i // 25) * 200 + 180) for i in range(k)]
    gpd.GeoDataFrame({"cropfield": np.arange(k) + j * 1000,
                      "original_code": [cod_at[(i + j) % len(cod_at)] for i in range(k)],
                      "area_ha": 3.24}, geometry=gg, crs="EPSG:3035").to_parquet(d / f"at_{ano}.parquet")
print("EuroCrops ok")

# ---------------- MapBiomas: GeoTIFFs na mesma grade ----------------
res = 0.00026949458523585647  # ~30 m
w = h = 2048
transf = from_origin(-56.0, -12.0, res, res)
base = rng.choice([3, 15, 39, 41], size=(h, w), p=[0.3, 0.2, 0.35, 0.15]).astype(np.uint8)
d = raiz / "raw" / "mapbiomas" / "colecao10"
d.mkdir(parents=True, exist_ok=True)
for ano in range(2015, 2025):
    arr = base.copy()
    lav = (base == 39) | (base == 41)
    alterna = lav & (((np.arange(w)[None, :] + ano) % 3) == 0)
    arr[alterna] = np.where(base[alterna] == 39, 41, 39)
    arr[(base == 15) & (ano % 4 == 0)] = 39  # integração lavoura-pecuária
    with rasterio.open(d / f"brazil_coverage_{ano}.tif", "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="uint8", crs="EPSG:4326", transform=transf, tiled=True,
                       blockxsize=256, blockysize=256, compress="lzw") as dst:
        dst.write(arr, 1)
print("MapBiomas ok")
