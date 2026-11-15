#!/usr/bin/env python3
"""Pipeline de dados de rotação de culturas: USDA CSB (EUA), EuroCrops v2.0 (UE) e MapBiomas (Brasil).

Exemplos:
  python pipeline.py usda --janela 2018-2025 --estados IA IL NE
  python pipeline.py eurocrops --regioes fr dk nl
  python pipeline.py mapbiomas --anos 2015-2024 --uf MT --pontos 50000
  python pipeline.py features --lags 3
  python pipeline.py legendas
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rotacao import features, util
from rotacao import fonte_eurocrops as euc
from rotacao import fonte_mapbiomas as mb
from rotacao import fonte_usda as usda
from rotacao.legenda import exportar_legendas


def _intervalo(txt: str) -> list[int]:
    if "-" in txt:
        a, b = (int(x) for x in txt.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in txt.split(",")]


def cmd_usda(a):
    ini, fim = _intervalo(a.janela)[0], _intervalo(a.janela)[-1]
    zip_path = Path(a.arquivo) if a.arquivo else usda.baixar(ini, fim)
    if not a.so_baixar:
        usda.harmonizar(zip_path, estados=a.estados, salvar_geometria=a.salvar_geometria)


def cmd_eurocrops(a):
    for r in a.regioes:
        arquivos = euc.baixar(r, versao=a.versao, anuais=a.anuais)
        if not a.so_baixar:
            euc.harmonizar(r, arquivos, salvar_geometria=a.salvar_geometria)


def cmd_mapbiomas(a):
    anos = _intervalo(a.anos)
    if a.pasta_tifs:
        arquivos = {ano: Path(a.pasta_tifs) / f"brazil_coverage_{ano}.tif" for ano in anos}
    else:
        arquivos = mb.baixar(anos)
    if not a.so_baixar:
        mb.harmonizar(arquivos, regiao=a.uf or a.nome_regiao, bbox=a.bbox,
                      limite=Path(a.limite) if a.limite else None, n_pontos=a.pontos,
                      min_anos_lavoura=a.min_anos_lavoura, semente=a.semente,
                      salvar_geometria=a.salvar_geometria)


def cmd_features(a):
    features.gerar(n_lags=a.lags, min_anos_lavoura=a.min_anos_lavoura, fontes=a.fontes)


def cmd_legendas(a):
    hcat = util.RAIZ / "raw" / "eurocrops" / "mapeamento" / "hcat4.csv"
    exportar_legendas(Path(a.destino), hcat if hcat.exists() else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dados", default="dados", help="pasta raiz dos dados (padrão: ./dados)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("usda", help="USDA Crop Sequence Boundaries (EUA)")
    s.add_argument("--janela", default="2018-2025", help="janela de 8 anos, ex. 2018-2025")
    s.add_argument("--estados", nargs="*", help="siglas (IA IL) ou FIPS (19 17); vazio = país todo")
    s.add_argument("--arquivo", help="usar um .zip já baixado em vez de baixar")
    s.set_defaults(func=cmd_usda)

    s = sub.add_parser("eurocrops", help="EuroCrops v2.0 (UE)")
    s.add_argument("--regioes", nargs="+", required=True, choices=sorted(euc.NUTS_ISO3))
    s.add_argument("--versao", choices=euc.VERSOES, help="pasta do FTP (padrão: mais nova disponível)")
    s.add_argument("--anuais", action="store_true", help="baixar também os arquivos anuais")
    s.set_defaults(func=cmd_eurocrops)

    s = sub.add_parser("mapbiomas", help="MapBiomas Coleção 10 (Brasil)")
    s.add_argument("--anos", default="2015-2024", help="ex. 2015-2024 ou 2020,2022,2024")
    s.add_argument("--uf", help=f"UF com retângulo pré-definido: {sorted(mb.UF_BBOX)}")
    s.add_argument("--bbox", nargs=4, type=float, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    s.add_argument("--limite", help="arquivo vetorial (gpkg/shp/geojson) com o polígono da área")
    s.add_argument("--nome-regiao", default="AOI", help="rótulo da região quando não usar --uf")
    s.add_argument("--pontos", type=int, default=50_000, help="nº de pixels agrícolas a amostrar")
    s.add_argument("--min-anos-lavoura", type=int, default=2)
    s.add_argument("--semente", type=int, default=42)
    s.add_argument("--pasta-tifs", help="usar GeoTIFFs já baixados (brazil_coverage_<ano>.tif)")
    s.set_defaults(func=cmd_mapbiomas)

    for nome in ("usda", "eurocrops", "mapbiomas"):
        sp = sub.choices[nome]
        sp.add_argument("--so-baixar", action="store_true", help="só baixar, sem harmonizar")
        sp.add_argument("--salvar-geometria", action="store_true", help="gravar polígonos em GeoParquet")

    s = sub.add_parser("features", help="gera features de rotação e o dataset de ML")
    s.add_argument("--lags", type=int, default=3)
    s.add_argument("--min-anos-lavoura", type=int, default=2)
    s.add_argument("--fontes", nargs="*", choices=["usda_csb", "eurocrops", "mapbiomas"])
    s.set_defaults(func=cmd_features)

    s = sub.add_parser("legendas", help="exporta as tabelas de conversão em CSV")
    s.add_argument("--destino", default="legendas")
    s.set_defaults(func=cmd_legendas)

    a = p.parse_args()
    util.definir_raiz(a.dados)
    a.func(a)


if __name__ == "__main__":
    main()
