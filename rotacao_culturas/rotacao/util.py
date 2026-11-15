"""Utilitários comuns: pastas, download com retomada, catálogo de metadados e escrita em Parquet."""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import requests
from tqdm import tqdm

# Raiz dos dados. Pode ser trocada com --dados na linha de comando ou com a variável ROTACAO_DADOS.
RAIZ = Path(os.environ.get("ROTACAO_DADOS", "dados"))

# Esquema da tabela longa harmonizada (uma linha por unidade x ano).
COLUNAS_OBS = [
    "fonte", "pais_iso3", "regiao", "unidade_id", "ano",
    "codigo_original", "cultura", "grupo", "lavoura_temporaria",
    "cultura_2", "estacao", "area_ha", "lon", "lat",
]


def definir_raiz(caminho: str | Path) -> None:
    global RAIZ
    RAIZ = Path(caminho)


def pasta(*partes: str) -> Path:
    p = RAIZ.joinpath(*partes)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256(caminho: Path, bloco: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while chunk := f.read(bloco):
            h.update(chunk)
    return h.hexdigest()


def registrar_download(fonte: str, url: str, arquivo: Path, licenca: str) -> None:
    """Acrescenta uma linha em metadata/catalogo_downloads.csv (procedência de cada arquivo bruto)."""
    cat = pasta("metadata") / "catalogo_downloads.csv"
    linha = pd.DataFrame([{
        "fonte": fonte,
        "url": url,
        "arquivo": str(arquivo),
        "bytes": arquivo.stat().st_size,
        "sha256": _sha256(arquivo),
        "baixado_em": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "licenca": licenca,
    }])
    linha.to_csv(cat, mode="a", header=not cat.exists(), index=False)


def baixar(url: str, destino: Path, fonte: str = "", licenca: str = "",
           tentativas: int = 6, timeout: int = 60) -> Path:
    """Baixa `url` para `destino` com retomada (HTTP Range), arquivo .part e novas tentativas.

    Se o arquivo final já existe, não baixa de novo. Levanta FileNotFoundError em 404.
    """
    destino = Path(destino)
    if destino.exists():
        print(f"[ok] já baixado: {destino}")
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".part")

    for tentativa in range(1, tentativas + 1):
        inicio = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "rotacao-culturas/1.0", "Accept-Encoding": "identity"}
        if inicio:
            headers["Range"] = f"bytes={inicio}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                if r.status_code == 416:  # Range além do fim: o .part já está completo
                    break
                r.raise_for_status()
                anexar = bool(inicio) and r.status_code == 206
                total = int(r.headers.get("Content-Length", 0)) + (inicio if anexar else 0)
                with open(tmp, "ab" if anexar else "wb") as f, tqdm(
                    total=total or None, initial=inicio if anexar else 0,
                    unit="B", unit_scale=True, desc=destino.name,
                ) as barra:
                    for bloco in r.iter_content(chunk_size=1 << 20):
                        f.write(bloco)
                        barra.update(len(bloco))
            break
        except FileNotFoundError:
            raise
        except (requests.RequestException, OSError) as e:
            espera = min(60, 2 ** tentativa)
            print(f"[aviso] tentativa {tentativa}/{tentativas} falhou ({e}); nova tentativa em {espera}s")
            time.sleep(espera)
    else:
        raise RuntimeError(f"Não foi possível baixar {url}")

    tmp.replace(destino)
    registrar_download(fonte, url, destino, licenca)
    return destino


ESQUEMA_OBS = pa.schema([
    ("fonte", pa.string()), ("pais_iso3", pa.string()), ("regiao", pa.string()),
    ("unidade_id", pa.string()), ("ano", pa.int16()), ("codigo_original", pa.string()),
    ("cultura", pa.string()), ("grupo", pa.string()), ("lavoura_temporaria", pa.bool_()),
    ("cultura_2", pa.string()), ("estacao", pa.string()), ("area_ha", pa.float32()),
    ("lon", pa.float64()), ("lat", pa.float64()),
])


def limpar_fonte(fonte: str) -> None:
    """Apaga as observações já gravadas de uma fonte (antes de regravar em lotes)."""
    import shutil
    alvo = RAIZ / "interim" / "observacoes" / f"fonte={fonte}"
    if alvo.exists():
        shutil.rmtree(alvo)


def limpar_fonte_regiao(fonte: str, regiao: str) -> None:
    """Apaga as observações de uma região de uma fonte (qualquer país)."""
    import shutil
    base = RAIZ / "interim" / "observacoes" / f"fonte={fonte}"
    for alvo in base.glob(f"pais_iso3=*/regiao={regiao}"):
        shutil.rmtree(alvo)


def escrever_observacoes(df: pd.DataFrame, lote_id: int | None = None) -> None:
    """Grava a tabela longa em interim/observacoes, particionada por fonte/pais_iso3/regiao (Hive).

    Sem `lote_id`, substitui as partições tocadas (rodar de novo não duplica dados).
    Com `lote_id`, acrescenta um arquivo por lote (use `limpar_fonte` antes do primeiro lote).
    """
    if df.empty:
        print("[aviso] nada para gravar")
        return
    df = df[COLUNAS_OBS].copy()
    for c in ("fonte", "pais_iso3", "regiao", "unidade_id", "codigo_original",
              "cultura", "grupo", "cultura_2", "estacao"):
        df[c] = df[c].astype("string")
    tabela = pa.Table.from_pandas(df, schema=ESQUEMA_OBS, preserve_index=False)
    ds.write_dataset(
        tabela,
        base_dir=str(pasta("interim", "observacoes")),
        format="parquet",
        partitioning=ds.partitioning(
            pa.schema([("fonte", pa.string()), ("pais_iso3", pa.string()), ("regiao", pa.string())]),
            flavor="hive"),
        existing_data_behavior="delete_matching" if lote_id is None else "overwrite_or_ignore",
        file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
        basename_template=("parte-{i}.parquet" if lote_id is None else f"lote{lote_id:04d}-{{i}}.parquet"),
    )
    print(f"[ok] {len(df):,} observações gravadas ({df['unidade_id'].nunique():,} unidades)")


def escrever_unidades(gdf, fonte: str, regiao: str) -> Path:
    """Grava as geometrias das unidades (talhões/pixels) em GeoParquet, uma vez por unidade."""
    destino = pasta("interim", "unidades") / f"{fonte}__{regiao}.parquet"
    gdf.to_parquet(destino, compression="zstd", index=False)
    print(f"[ok] geometrias gravadas em {destino}")
    return destino


def ler(nome: str = "dataset_ml", filtros=None) -> pd.DataFrame:
    """Lê uma tabela particionada mantendo fonte/pais_iso3/regiao como texto (FIPS '01' não vira 1).

    nome: "observacoes" (interim) ou "features_rotacao" / "dataset_ml" (processed).
    filtros: lista no formato pyarrow, ex. [("fonte", "=", "mapbiomas")].
    """
    base = RAIZ / ("interim" if nome == "observacoes" else "processed") / nome
    part = ds.partitioning(
        pa.schema([("fonte", pa.string()), ("pais_iso3", pa.string()), ("regiao", pa.string())]),
        flavor="hive")
    dataset = ds.dataset(str(base), format="parquet", partitioning=part)
    filtro = None
    for col, op, val in (filtros or []):
        expr = {"=": ds.field(col) == val, "in": ds.field(col).isin(val)}[op]
        filtro = expr if filtro is None else filtro & expr
    return dataset.to_table(filter=filtro).to_pandas()
