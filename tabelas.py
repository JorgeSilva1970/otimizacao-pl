"""
tabelas.py - Conversão entre o Modelo e tabelas (pandas / Excel).

Formato das tabelas editáveis:
  Variáveis:  Nome | Legenda | Tipo | Limite inferior | Limite superior | Coef. objetivo
  Restrições: Nome | Legenda | <uma coluna por variável> | Sinal | Lado direito
"""
from __future__ import annotations

import io

import pandas as pd

from lp_core import (NOMES_RESERVADOS, Modelo, Restricao, Resultado, Variavel,
                     formulacao_texto, limite, normalizar_sinal, normalizar_tipo, numero)

COLS_VAR = ["Nome", "Legenda", "Tipo", "Limite inferior", "Limite superior", "Coef. objetivo"]
NUM_VAR = ["Limite inferior", "Limite superior", "Coef. objetivo"]
RES_INICIO = ["Nome", "Legenda"]
RES_FIM = ["Sinal", "Lado direito"]


# --------------------------------------------------------------------------- #
# Utilitários de células
# --------------------------------------------------------------------------- #
def vazio(x) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == ""
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def celula(x):
    return None if vazio(x) else x


def texto(x) -> str:
    return "" if vazio(x) else str(x).strip()


# --------------------------------------------------------------------------- #
# Modelo -> DataFrames
# --------------------------------------------------------------------------- #
def df_variaveis(m: Modelo) -> pd.DataFrame:
    df = pd.DataFrame(
        [[v.nome, v.legenda, v.tipo, v.lb, v.ub, v.coef] for v in m.variaveis],
        columns=COLS_VAR,
    )
    for c in NUM_VAR:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in ("Nome", "Legenda", "Tipo"):
        df[c] = df[c].astype(str)
    return df


def df_restricoes(m: Modelo) -> pd.DataFrame:
    nomes = [v.nome for v in m.variaveis]
    linhas = [[r.nome, r.legenda, *[r.coefs.get(n, 0.0) for n in nomes], r.sinal, r.rhs]
              for r in m.restricoes]
    df = pd.DataFrame(linhas, columns=RES_INICIO + nomes + RES_FIM)
    for c in nomes + ["Lado direito"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    for c in ("Nome", "Legenda", "Sinal"):
        df[c] = df[c].astype(str)
    return df


# --------------------------------------------------------------------------- #
# Sincronização das colunas das restrições com a lista de variáveis
# --------------------------------------------------------------------------- #
def nomes_variaveis(df_var: pd.DataFrame) -> list[str]:
    """Nomes válidos, únicos e pela ordem da tabela."""
    vistos, nomes = set(), []
    for n in df_var.get("Nome", []):
        n = texto(n)
        if n and n not in vistos and n.lower() not in NOMES_RESERVADOS:
            vistos.add(n)
            nomes.append(n)
    return nomes


def alinhar_restricoes(df_res: pd.DataFrame, novos: list[str], antigos: list[str] | None = None) -> pd.DataFrame:
    """
    Garante uma coluna de coeficientes por variável.
    - Variável nova: coluna preenchida com 0.
    - Variável removida: coluna eliminada.
    - Variável renomeada (mesmo número de variáveis): a coluna muda de nome e mantém os valores.
    """
    df = df_res.copy()
    if antigos is not None and len(antigos) == len(novos):
        mapa = {a: n for a, n in zip(antigos, novos) if a != n and a in df.columns}
        df = df.rename(columns=mapa)
    for c in RES_INICIO + RES_FIM:
        if c not in df.columns:
            df[c] = "<=" if c == "Sinal" else (0.0 if c == "Lado direito" else "")
    for n in novos:
        if n not in df.columns:
            df[n] = 0.0
        df[n] = pd.to_numeric(df[n], errors="coerce").fillna(0.0).astype(float)
    return df[RES_INICIO + novos + RES_FIM].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# DataFrames -> Modelo
# --------------------------------------------------------------------------- #
def modelo_de_tabelas(nome: str, sentido: str, df_var: pd.DataFrame, df_res: pd.DataFrame) -> Modelo:
    variaveis = []
    for i, row in df_var.reset_index(drop=True).iterrows():
        nome_v, leg = texto(row.get("Nome")), texto(row.get("Legenda"))
        try:
            coef = numero(celula(row.get("Coef. objetivo")), 0.0)
            lb = limite(celula(row.get("Limite inferior")))
            ub = limite(celula(row.get("Limite superior")))
        except ValueError as e:
            raise ValueError(f"Variável na linha {i + 1}: {e}")
        if not nome_v and not leg and coef == 0:
            continue  # linha em branco
        variaveis.append(Variavel(nome=nome_v, coef=coef, legenda=leg,
                                  tipo=normalizar_tipo(row.get("Tipo")), lb=lb, ub=ub))

    nomes = [v.nome for v in variaveis if v.nome]
    restricoes = []
    for j, row in df_res.reset_index(drop=True).iterrows():
        nome_r, leg = texto(row.get("Nome")), texto(row.get("Legenda"))
        try:
            coefs = {n: numero(celula(row.get(n)), 0.0) for n in nomes if n in row.index}
            rhs = numero(celula(row.get("Lado direito")), 0.0)
        except ValueError as e:
            raise ValueError(f"Restrição na linha {j + 1}: {e}")
        if not nome_r and not leg and all(c == 0 for c in coefs.values()) and rhs == 0:
            continue  # linha em branco
        restricoes.append(Restricao(nome=nome_r, coefs=coefs, sinal=normalizar_sinal(celula(row.get("Sinal"))),
                                    rhs=rhs, legenda=leg))

    return Modelo(nome=(nome or "Problema").strip(), sentido=sentido, variaveis=variaveis, restricoes=restricoes)


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def excel_de_modelo(m: Modelo) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"Parâmetro": ["Nome", "Objetivo"],
                      "Valor": [m.nome, m.sentido]}).to_excel(xw, sheet_name="Modelo", index=False)
        df_variaveis(m).to_excel(xw, sheet_name="Variaveis", index=False)
        df_restricoes(m).to_excel(xw, sheet_name="Restricoes", index=False)
    return buf.getvalue()


def modelo_de_excel(ficheiro) -> Modelo:
    folhas = pd.read_excel(ficheiro, sheet_name=None)
    chaves = {k.lower().replace("ç", "c").replace("õ", "o").replace("á", "a"): k for k in folhas}
    if "variaveis" not in chaves or "restricoes" not in chaves:
        raise ValueError("O Excel tem de ter as folhas 'Variaveis' e 'Restricoes'.")
    nome, sentido = "Problema", "max"
    if "modelo" in chaves:
        cfg = folhas[chaves["modelo"]]
        pares = {texto(a).lower(): texto(b) for a, b in cfg.iloc[:, :2].itertuples(index=False)}
        nome = pares.get("nome") or nome
        s = pares.get("objetivo", "max").lower()[:3]
        sentido = "min" if s == "min" else "max"
    df_var = folhas[chaves["variaveis"]]
    df_var.columns = [str(c).strip() for c in df_var.columns]
    falta = [c for c in ("Nome", "Coef. objetivo") if c not in df_var.columns]
    if falta:
        raise ValueError("Folha 'Variaveis' sem as colunas: " + ", ".join(falta))
    df_res = folhas[chaves["restricoes"]]
    df_res.columns = [str(c).strip() for c in df_res.columns]
    falta = [c for c in ("Nome", "Sinal", "Lado direito") if c not in df_res.columns]
    if falta:
        raise ValueError("Folha 'Restricoes' sem as colunas: " + ", ".join(falta))
    return modelo_de_tabelas(nome, sentido, df_var, df_res)


def excel_de_resultado(m: Modelo, r: Resultado) -> bytes:
    buf = io.BytesIO()
    resumo = pd.DataFrame({
        "Campo": ["Problema", "Objetivo", "Estado", "Valor ótimo (Z)", "Tempo (s)"],
        "Valor": [m.nome, m.sentido, r.estado,
                  r.valor_objetivo if r.valor_objetivo is not None else "", round(r.tempo, 4)],
    })
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        resumo.to_excel(xw, sheet_name="Resumo", index=False)
        pd.DataFrame(r.variaveis).to_excel(xw, sheet_name="Variaveis", index=False)
        pd.DataFrame(r.restricoes).to_excel(xw, sheet_name="Restricoes", index=False)
        pd.DataFrame({"Formulação": formulacao_texto(m).splitlines()}).to_excel(
            xw, sheet_name="Formulacao", index=False)
    return buf.getvalue()
