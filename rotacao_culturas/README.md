# Rotação de culturas: USDA CSB + EuroCrops v2.0 + MapBiomas

Este pipeline baixa as três fontes, converte as culturas de cada uma para uma **legenda comum** e monta duas tabelas prontas para machine learning:

| Fonte | Cobertura | Unidade | Anos | Licença |
|---|---|---|---|---|
| USDA NASS Crop Sequence Boundaries | EUA (48 estados + DC) | talhão | janelas de 8 anos, 2008–2025 | domínio público |
| EuroCrops v2.0 (JRC) | 18 regiões de 16 países da UE | talhão (declaração da PAC) | 2008–2023 (varia por região) | CC BY 4.0 |
| MapBiomas Coleção 10 | Brasil | pixel de 30 m (amostrado) | 1985–2024 | CC BY 4.0 |

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# EUA: Corn Belt, janela 2018–2025 (sem --estados = país inteiro, processado em lotes)
python pipeline.py usda --janela 2018-2025 --estados IA IL NE MN

# Europa: França, Dinamarca e Holanda
python pipeline.py eurocrops --regioes fr dk nl

# Brasil: 50 mil pixels agrícolas no MT, 2015–2024
python pipeline.py mapbiomas --anos 2015-2024 --uf MT --pontos 50000
#   recorte exato com o polígono do estado (ex.: malha do IBGE):
python pipeline.py mapbiomas --anos 2015-2024 --limite MT.gpkg --nome-regiao MT

# Features de rotação + dataset de ML (todas as fontes já harmonizadas)
python pipeline.py features --lags 3 --min-anos-lavoura 2

# Exportar as tabelas de conversão em CSV para revisar
python pipeline.py legendas
```

Opções úteis em todas as fontes:
- `--so-baixar` baixa sem harmonizar.
- `--salvar-geometria` grava os polígonos em GeoParquet.
- `--dados <pasta>` muda a raiz dos dados.

Os downloads retomam de onde pararam se a conexão cair, e um arquivo já baixado não é baixado de novo.

**Tamanho dos arquivos:** o zip nacional do CSB e cada GeoTIFF anual do MapBiomas passam de 1 GB. Os stacks grandes do EuroCrops (fr, es) também. Comece por poucos estados, regiões ou anos.

## Onde os dados ficam

```
dados/
├── raw/                          # arquivos originais, nunca editados
│   ├── usda_csb/NationalCSB_2018-2025_rev23.zip
│   ├── eurocrops/gpqtv202/fr_stack.parquet
│   ├── eurocrops/mapeamento/eurocrops.csv, hcat4.csv
│   └── mapbiomas/colecao10/brazil_coverage_2024.tif
├── interim/
│   ├── observacoes/fonte=…/pais_iso3=…/regiao=…/*.parquet   # tabela longa harmonizada
│   └── unidades/*.parquet                                  # geometrias (com --salvar-geometria)
├── processed/
│   ├── features_rotacao/fonte=…/pais_iso3=…/regiao=…/      # 1 linha por talhão/pixel
│   └── dataset_ml/fonte=…/pais_iso3=…/regiao=…/            # 1 linha por talhão/pixel × ano
└── metadata/catalogo_downloads.csv   # URL, sha256, data e licença de cada arquivo baixado
```

Leia as tabelas com `util.ler`. Ela mantém `regiao` como texto, para que o FIPS "01" não vire o número 1:

```python
from rotacao import util
util.definir_raiz("dados")
df = util.ler("dataset_ml", [("fonte", "in", ["usda_csb", "mapbiomas"])])
```

## Esquemas

**`interim/observacoes`**: uma linha por unidade e ano.

| coluna | descrição |
|---|---|
| `fonte`, `pais_iso3`, `regiao` | usda_csb / eurocrops / mapbiomas; ISO3; FIPS do estado, NUTS ou UF |
| `unidade_id` | `usa_csb_<CSBID>`, `euc_<nuts>_<cropfield>`, `bra_mb_<linha>_<coluna>` |
| `ano`, `codigo_original` | ano e código na fonte (`cdl:5`, `euc:105\|hcat4:3310106000`, `mb:39`) |
| `cultura`, `grupo`, `lavoura_temporaria` | legenda comum (ver `legendas/legenda_comum.csv`) |
| `cultura_2` | 2ª cultura em safra dupla (só o CDL informa, ex. trigo de inverno → soja) |
| `estacao` | inverno / primavera, quando a fonte informa |
| `area_ha`, `lon`, `lat` | área da unidade e ponto interno em WGS84 |

**`processed/features_rotacao`**: a sequência completa (`soja>milho>soja…`), `n_culturas_distintas`, `n_trocas` (só entre anos consecutivos), `prop_monocultura`, `shannon`, `prop_cereal`, `prop_leguminosa`, `anos_safra_dupla`, `anos_pousio` e `cultura_mais_comum`. A `prop_leguminosa` inclui as leguminosas de grão, a alfafa e as forrageiras leguminosas.

**`processed/dataset_ml`**: alvo `alvo_cultura` / `alvo_grupo` no ano *t*. As entradas são `cultura_t-k`, `grupo_t-k`, `safra_dupla_t-k` e `leguminosa_t-k` para k = 1…`--lags`, mais `anos_seguidos_mesma_cultura` e `leguminosas_ultimos_anos`. Tudo vem só do passado, então o ano *t* não vaza para as entradas. Um lag fica vazio quando falta um ano no meio da série.

## Decisões de harmonização (edite em `rotacao/legenda.py`)

- **CDL:** o milho doce e o milho de pipoca contam como `milho`. Espelta, emmer e einkorn contam como `trigo`. As safras duplas viram `cultura` + `cultura_2`.
- **HCAT4:** a conversão usa regras por prefixo do código de 10 dígitos e vale o prefixo mais longo que casar. Ela cobre todas as 353 classes do `hcat4.csv` atual.
- **MapBiomas:** a Coleção 10 tem poucas classes de lavoura: soja, arroz, algodão (beta), cana e "outras lavouras temporárias". O milho safrinha não aparece no mapa anual, e "outras temporárias" mistura várias culturas. Os pixels de pastagem entram na sequência, o que permite ver a integração lavoura-pecuária.

## Cuidados para o modelo

- **Validação espacial:** talhões vizinhos são muito parecidos. Separe treino e teste por `regiao` (ou por blocos espaciais de `lon`/`lat`) com `GroupKFold`; uma divisão aleatória superestima a acurácia.
- **Unidades diferentes:** o CSB e o EuroCrops são talhões, e o MapBiomas é um pixel amostrado. Use `fonte` como feature ou avalie cada fonte separadamente.
- **EuroCrops sem stack:** se uma região não tiver o arquivo `_stack`, os anos são ligados pelo ponto interno dos talhões do ano mais recente. Talhões que mudaram de forma podem herdar a cultura do vizinho.
- **Classes raras:** o "desconhecido" e as perenes inflam a contagem de classes. Filtre `lavoura_temporaria == True` no alvo se o foco for a rotação anual.

## Como citar

- USDA NASS, *Crop Sequence Boundaries*, https://www.nass.usda.gov/research-and-science/crop-sequence-boundaries
- EuroCrops v2.0, JRC, DOI 10.2905/b9fb9e67-78a9-4327-9d59-39a928d812d3 (ESSD 18, 4075, 2026)
- Projeto MapBiomas – Coleção 10 da Série Anual de Mapas de Cobertura e Uso da Terra do Brasil, acessado em [data] em https://brasil.mapbiomas.org
