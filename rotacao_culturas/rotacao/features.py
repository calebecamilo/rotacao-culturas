"""Transforma a tabela longa harmonizada em (1) features de rotação por unidade e
(2) um dataset de ML no formato "prever a cultura do ano t a partir dos anos anteriores".

Processa uma partição (fonte/país/região) por vez para caber na memória.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from . import util

LEGUMINOSAS = {"leguminosa_grao"}
LEGUMINOSAS_CULTURAS = {"alfafa", "forrageiras_leguminosas"}


def particoes() -> list[tuple[dict, Path]]:
    base = util.RAIZ / "interim" / "observacoes"
    saida = []
    for d in sorted(base.glob("fonte=*/pais_iso3=*/regiao=*")):
        chaves = dict(p.split("=", 1) for p in d.relative_to(base).parts)
        saida.append((chaves, d))
    return saida


def carregar(d: Path, chaves: dict) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))], ignore_index=True)
    for k, v in chaves.items():
        df[k] = v
    df["ano"] = df["ano"].astype(int)
    return df.sort_values(["unidade_id", "ano"], kind="stable").reset_index(drop=True)


def _e_leguminosa(df: pd.DataFrame) -> pd.Series:
    return df["grupo"].isin(LEGUMINOSAS) | df["cultura"].isin(LEGUMINOSAS_CULTURAS)


def features_por_unidade(obs: pd.DataFrame) -> pd.DataFrame:
    df = obs.copy()
    df["rotulo"] = df["cultura"].where(df["cultura_2"].isna(), df["cultura"] + "+" + df["cultura_2"].astype(str))
    df["leg"] = _e_leguminosa(df)
    g = df.groupby("unidade_id", sort=False)

    anterior = g["cultura"].shift(1)
    consecutivo = (df["ano"] - g["ano"].shift(1)) == 1
    df["troca"] = (consecutivo & (df["cultura"] != anterior)).fillna(False).astype(bool)

    tmp = df[df["lavoura_temporaria"]]
    cont = tmp.groupby(["unidade_id", "cultura"]).size().rename("n").reset_index()
    cont["p"] = cont["n"] / cont.groupby("unidade_id")["n"].transform("sum")
    cont["h"] = -cont["p"] * np.log(cont["p"])
    div = cont.groupby("unidade_id").agg(
        n_culturas_distintas=("cultura", "nunique"),
        shannon=("h", "sum"),
        prop_monocultura=("p", "max"),
    )
    mais_comum = cont.sort_values(["unidade_id", "n"], ascending=[True, False]).drop_duplicates("unidade_id")
    div["cultura_mais_comum"] = mais_comum.set_index("unidade_id")["cultura"]

    feat = g.agg(
        fonte=("fonte", "first"), pais_iso3=("pais_iso3", "first"), regiao=("regiao", "first"),
        lon=("lon", "first"), lat=("lat", "first"), area_ha=("area_ha", "first"),
        ano_ini=("ano", "min"), ano_fim=("ano", "max"), n_anos=("ano", "size"),
        anos_lavoura=("lavoura_temporaria", "sum"),
        n_trocas=("troca", "sum"),
        anos_safra_dupla=("cultura_2", lambda s: int(s.notna().sum())),
        anos_pousio=("cultura", lambda s: int((s == "pousio").sum())),
        prop_cereal=("grupo", lambda s: float((s == "cereal").mean())),
        prop_leguminosa=("leg", "mean"),
        sequencia=("rotulo", ">".join),
    )
    feat = feat.join(div, how="left")
    feat["n_culturas_distintas"] = feat["n_culturas_distintas"].fillna(0).astype(int)
    return feat.reset_index()


def dataset_ml(obs: pd.DataFrame, n_lags: int = 3, min_anos_lavoura: int = 2) -> pd.DataFrame:
    """Uma linha por (unidade, ano t): alvo = cultura em t; entradas = culturas em t-1..t-n_lags
    e resumos só do passado (sem vazamento do ano t)."""
    df = obs.copy()
    df["leg"] = _e_leguminosa(df)
    lav = df.groupby("unidade_id")["lavoura_temporaria"].transform("sum")
    df = df[lav >= min_anos_lavoura].reset_index(drop=True)
    if df.empty:
        return df
    g = df.groupby("unidade_id", sort=False)

    saida = df[["fonte", "pais_iso3", "regiao", "unidade_id", "ano", "lon", "lat", "area_ha"]].copy()
    saida["alvo_cultura"] = df["cultura"]
    saida["alvo_grupo"] = df["grupo"]
    for k in range(1, n_lags + 1):
        ok = (df["ano"] - g["ano"].shift(k)) == k  # só vale se os anos forem consecutivos
        for col in ("cultura", "grupo"):
            saida[f"{col}_t-{k}"] = g[col].shift(k).where(ok)
        saida[f"safra_dupla_t-{k}"] = g["cultura_2"].shift(k).notna().where(ok)
        saida[f"leguminosa_t-{k}"] = g["leg"].shift(k).where(ok)

    # Quantos anos seguidos, até t-1, repetiram a cultura de t-1 (monocultura recente).
    seq = pd.Series(np.where(saida["cultura_t-1"].notna(), 1, 0), index=saida.index)
    vivo = saida["cultura_t-1"].notna()
    for k in range(2, n_lags + 1):
        vivo = (vivo & (saida[f"cultura_t-{k}"] == saida["cultura_t-1"])).fillna(False).astype(bool)
        seq = seq + vivo.astype(int)
    saida["anos_seguidos_mesma_cultura"] = seq
    saida["leguminosas_ultimos_anos"] = sum(
        saida[f"leguminosa_t-{k}"].fillna(False).astype(bool).astype(int) for k in range(1, n_lags + 1))
    return saida[saida["cultura_t-1"].notna()].reset_index(drop=True)


def _gravar(df: pd.DataFrame, nome: str) -> None:
    if df.empty:
        return
    tabela = pa.Table.from_pandas(df, preserve_index=False)
    ds.write_dataset(
        tabela, base_dir=str(util.pasta("processed", nome)), format="parquet",
        partitioning=ds.partitioning(
            pa.schema([("fonte", pa.string()), ("pais_iso3", pa.string()), ("regiao", pa.string())]),
            flavor="hive"),
        existing_data_behavior="delete_matching",
        file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
        basename_template="parte-{i}.parquet",
    )


def gerar(n_lags: int = 3, min_anos_lavoura: int = 2, fontes: list[str] | None = None) -> None:
    parts = particoes()
    if fontes:
        parts = [(c, d) for c, d in parts if c["fonte"] in fontes]
    if not parts:
        print("[aviso] nenhuma observação em interim/observacoes; rode as fontes primeiro")
        return
    for chaves, d in parts:
        obs = carregar(d, chaves)
        feat = features_por_unidade(obs)
        ml = dataset_ml(obs, n_lags=n_lags, min_anos_lavoura=min_anos_lavoura)
        _gravar(feat, "features_rotacao")
        _gravar(ml, "dataset_ml")
        print(f"[ok] {chaves}: {len(feat):,} unidades, {len(ml):,} linhas de ML")
