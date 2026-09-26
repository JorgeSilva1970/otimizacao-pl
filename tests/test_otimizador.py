import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grafico import desenhar_2d  # noqa: E402
from lp_core import EXEMPLOS, Modelo, Restricao, Variavel, formulacao_latex, resolver  # noqa: E402
from tabelas import (alinhar_restricoes, df_restricoes, df_variaveis, excel_de_modelo,  # noqa: E402
                     modelo_de_excel, modelo_de_tabelas)


def exemplo(chave_parcial: str) -> Modelo:
    chave = next(k for k in EXEMPLOS if chave_parcial in k)
    return Modelo.from_dict(EXEMPLOS[chave])


def test_producao_otimo_e_precos_sombra():
    r = resolver(exemplo("mesas"))
    assert r.estado == "Ótimo"
    assert r.valor_objetivo == pytest.approx(4100)
    val = {d["Nome"]: d["Valor"] for d in r.variaveis}
    assert val == pytest.approx({"mesas": 30, "cadeiras": 40})
    precos = {d["Nome"]: d["Preço sombra"] for d in r.restricoes}
    assert precos == pytest.approx({"Madeira": 15, "Acabamento": 5})


def test_minimizacao_com_maior_ou_igual():
    r = resolver(exemplo("Dieta"))
    assert r.estado == "Ótimo"
    assert r.valor_objetivo == pytest.approx(3.56)
    energia = next(d for d in r.restricoes if d["Nome"] == "Energia")
    assert energia["Estado"].startswith("Ativa")
    assert energia["Preço sombra"] == pytest.approx(0.002)


def test_binarias():
    r = resolver(exemplo("projetos"))
    assert r.estado == "Ótimo"
    assert r.valor_objetivo == pytest.approx(47)
    assert not r.com_sensibilidade


def test_impossivel():
    m = Modelo(sentido="max", variaveis=[Variavel("x", 1)],
               restricoes=[Restricao("a", {"x": 1}, ">=", 2), Restricao("b", {"x": 1}, "<=", 1)])
    assert resolver(m).estado == "Impossível"


def test_ilimitado():
    m = Modelo(sentido="max", variaveis=[Variavel("x", 1)], restricoes=[Restricao("a", {"x": 1}, ">=", 1)])
    assert resolver(m).estado == "Ilimitado"


def test_igualdade_e_variavel_livre():
    m = Modelo(sentido="min", variaveis=[Variavel("x", 1, lb=None), Variavel("y", 1)],
               restricoes=[Restricao("soma", {"x": 1, "y": 1}, "=", 5), Restricao("lim", {"x": 1}, ">=", -3)])
    r = resolver(m)
    assert r.estado == "Ótimo" and r.valor_objetivo == pytest.approx(5)


def test_validacao():
    m = Modelo(variaveis=[Variavel("x"), Variavel("x"), Variavel("")],
               restricoes=[Restricao("r", {"x": 0}, "<=", -1)])
    erros, _ = m.validar()
    assert any("repetidos" in e for e in erros)
    assert any("falta o nome" in e for e in erros)
    assert any("impossível" in e for e in erros)


def test_json_ida_e_volta():
    m = exemplo("Dieta")
    assert Modelo.from_json(m.to_json()) == m


def _sem_zeros(m: Modelo) -> Modelo:
    for r in m.restricoes:
        r.coefs = {k: v for k, v in r.coefs.items() if v != 0}
    return m


def test_tabelas_e_excel_ida_e_volta(tmp_path):
    m = exemplo("Dieta")
    m2 = modelo_de_tabelas(m.nome, m.sentido, df_variaveis(m), df_restricoes(m))
    assert _sem_zeros(m2) == m
    ficheiro = tmp_path / "m.xlsx"
    ficheiro.write_bytes(excel_de_modelo(m))
    assert _sem_zeros(modelo_de_excel(ficheiro)) == m


def test_alinhar_renomeia_e_acrescenta():
    m = exemplo("mesas")
    df = df_restricoes(m)
    df2 = alinhar_restricoes(df, ["mesas", "bancos"], ["mesas", "cadeiras"])
    assert list(df2["bancos"]) == [3, 1]          # renomeada, mantém valores
    df3 = alinhar_restricoes(df2, ["mesas", "bancos", "sofas"], ["mesas", "bancos"])
    assert list(df3["sofas"]) == [0, 0]           # nova, a zeros


def test_linhas_em_branco_ignoradas():
    m = exemplo("mesas")
    dv = pd.concat([df_variaveis(m), pd.DataFrame([{"Nome": None, "Coef. objetivo": 0}])], ignore_index=True)
    assert len(modelo_de_tabelas("t", "max", dv, df_restricoes(m)).variaveis) == 2


def test_grafico_vertices():
    m = exemplo("mesas")
    fig, vertices = desenhar_2d(m, resolver(m))
    zs = sorted(round(v["Z"]) for v in vertices)
    assert zs == [0, 3500, 4000, 4100]


def test_latex_gera_texto():
    assert r"\max" in formulacao_latex(exemplo("mesas"))
