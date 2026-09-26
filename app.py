"""
app.py - Otimizador de Programação Linear (interface web em Streamlit)

Executar localmente:
    streamlit run app.py
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from grafico import desenhar_2d
from lp_core import (EXEMPLOS, SENTIDOS, SINAIS, TIPOS, Modelo, fmt, formulacao_latex,
                     formulacao_texto, modelo_em_branco, resolver)
from tabelas import (alinhar_restricoes, df_restricoes, df_variaveis, excel_de_modelo,
                     excel_de_resultado, modelo_de_excel, modelo_de_tabelas, nomes_variaveis,
                     texto)

st.set_page_config(page_title="Otimizador de Programação Linear", page_icon="📐", layout="wide")
ss = st.session_state


# --------------------------------------------------------------------------- #
# Estado da sessão
# --------------------------------------------------------------------------- #
def carregar_modelo(m: Modelo) -> None:
    """Coloca um modelo nas tabelas editáveis (chamado em callbacks)."""
    ss.nome = m.nome
    ss.sentido = m.sentido
    ss.vars_base = df_variaveis(m)
    ss.cons_base = df_restricoes(m)
    ss.cons_atual = ss.cons_base.copy()
    ss.cons_sig = tuple(v.nome for v in m.variaveis)
    ss.versao = ss.get("versao", 0) + 1
    ss.cons_ver = 0
    ss.resultado = None
    ss.resultado_hash = None


def carregar_exemplo() -> None:
    carregar_modelo(Modelo.from_dict(EXEMPLOS[ss.exemplo]))


def abrir_ficheiro() -> None:
    f = ss.get("upload")
    if f is None:
        return
    try:
        if f.name.lower().endswith(".json"):
            m = Modelo.from_json(f.getvalue().decode("utf-8"))
        else:
            m = modelo_de_excel(f)
        carregar_modelo(m)
        ss.msg_import = ("ok", f"Modelo '{m.nome}' aberto com {len(m.variaveis)} variáveis "
                               f"e {len(m.restricoes)} restrições.")
    except Exception as e:  # noqa: BLE001
        ss.msg_import = ("erro", f"Não foi possível abrir o ficheiro: {e}")


def hash_modelo(m: Modelo) -> str:
    return hashlib.md5(m.to_json().encode("utf-8")).hexdigest()


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower() or "modelo"


if "versao" not in ss:
    ss.exemplo = next(iter(EXEMPLOS))
    carregar_modelo(Modelo.from_dict(EXEMPLOS[ss.exemplo]))


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Problema")
    st.text_input("Nome do problema", key="nome")
    st.radio("Objetivo", options=list(SENTIDOS), format_func=SENTIDOS.get, key="sentido", horizontal=True)
    st.number_input("Tempo máximo do solver (segundos)", min_value=1, max_value=3600, value=60, key="tempo")

    st.divider()
    st.subheader("Começar a partir de")
    st.selectbox("Exemplo", list(EXEMPLOS), key="exemplo")
    st.button("Carregar exemplo", on_click=carregar_exemplo)
    st.button("Problema em branco", on_click=lambda: carregar_modelo(modelo_em_branco()))

    st.divider()
    st.caption(
        "Motor: HiGHS (via SciPy). Resolve programação linear contínua, inteira e binária. "
        "A análise de sensibilidade aparece quando todas as variáveis são contínuas."
    )


# --------------------------------------------------------------------------- #
# Cabeçalho
# --------------------------------------------------------------------------- #
st.title("Otimizador de Programação Linear")
st.write(
    "Define as variáveis de decisão, a função objetivo e as restrições. "
    "Acrescenta linhas com o **+** no fundo de cada tabela (ou cola diretamente do Excel) "
    "e remove-as selecionando a linha e carregando em **Delete**."
)

tab_modelo, tab_sol, tab_graf, tab_fich = st.tabs(
    ["Modelo", "Solução", "Método gráfico", "Importar e exportar"])


# --------------------------------------------------------------------------- #
# Separador: Modelo
# --------------------------------------------------------------------------- #
with tab_modelo:
    st.subheader("Variáveis de decisão e função objetivo")
    df_var = st.data_editor(
        ss.vars_base,
        key=f"ed_vars_{ss.versao}",
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "Nome": st.column_config.TextColumn("Nome", required=True,
                                                help="Identificador curto e único, ex.: x1, mesas, P1"),
            "Legenda": st.column_config.TextColumn("Legenda", width="large",
                                                   help="O que a variável representa, com unidades"),
            "Tipo": st.column_config.SelectboxColumn("Tipo", options=list(TIPOS), default="Contínua",
                                                     required=True),
            "Limite inferior": st.column_config.NumberColumn("Limite inferior", default=0.0,
                                                             help="Vazio = sem limite inferior (−∞)"),
            "Limite superior": st.column_config.NumberColumn("Limite superior",
                                                             help="Vazio = sem limite superior (+∞)"),
            "Coef. objetivo": st.column_config.NumberColumn("Coef. objetivo", default=0.0, required=True,
                                                            help="Contribuição unitária para Z"),
        },
    )

    # Mantém uma coluna de coeficientes por variável na tabela das restrições
    nomes = nomes_variaveis(df_var)
    if tuple(nomes) != ss.cons_sig:
        ss.cons_base = alinhar_restricoes(ss.cons_atual, nomes, list(ss.cons_sig))
        ss.cons_sig = tuple(nomes)
        ss.cons_ver += 1

    st.subheader("Restrições")
    legendas = {texto(r.get("Nome")): texto(r.get("Legenda")) for _, r in df_var.iterrows()}
    cfg = {
        "Nome": st.column_config.TextColumn("Nome", required=True, help="Identificador da restrição"),
        "Legenda": st.column_config.TextColumn("Legenda", width="medium",
                                               help="O que a restrição representa, ex.: horas de máquina"),
        "Sinal": st.column_config.SelectboxColumn("Sinal", options=list(SINAIS), default="<=", required=True),
        "Lado direito": st.column_config.NumberColumn("Lado direito", default=0.0, required=True,
                                                      help="Valor do lado direito (recurso, procura, etc.)"),
    }
    for n in nomes:
        cfg[n] = st.column_config.NumberColumn(n, default=0.0, help=legendas.get(n) or f"Coeficiente de {n}")
    df_res = st.data_editor(
        ss.cons_base,
        key=f"ed_cons_{ss.versao}_{ss.cons_ver}",
        num_rows="dynamic",
        hide_index=True,
        column_config=cfg,
    )
    ss.cons_atual = df_res

    # Construção e validação do modelo
    modelo, erros, avisos = None, [], []
    try:
        modelo = modelo_de_tabelas(ss.nome, ss.sentido, df_var, df_res)
        erros, avisos = modelo.validar()
    except ValueError as e:
        erros = [str(e)]

    st.subheader("Formulação")
    if modelo is not None and modelo.variaveis:
        st.latex(formulacao_latex(modelo))
    for e in erros:
        st.error(e)
    for a in avisos:
        st.warning(a)

    if st.button("Resolver", type="primary", disabled=bool(erros) or modelo is None):
        with st.spinner("A resolver…"):
            ss.resultado = resolver(modelo, float(ss.tempo))
            ss.resultado_hash = hash_modelo(modelo)

    r = ss.get("resultado")
    if r is not None and modelo is not None and ss.resultado_hash == hash_modelo(modelo):
        if r.tem_solucao:
            st.success(f"{r.estado}: Z = {fmt(r.valor_objetivo)}. Os detalhes estão no separador Solução.")
        else:
            st.error(f"{r.estado}. {r.mensagem}")


# --------------------------------------------------------------------------- #
# Separador: Solução
# --------------------------------------------------------------------------- #
with tab_sol:
    r = ss.get("resultado")
    if r is None:
        st.info("Ainda não resolveste o modelo. Carrega em Resolver no separador Modelo.")
    else:
        if modelo is None or ss.resultado_hash != hash_modelo(modelo):
            st.warning("O modelo foi alterado depois da última resolução. Carrega outra vez em Resolver.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Estado", r.estado)
        c2.metric("Valor ótimo (Z)", fmt(r.valor_objetivo) if r.valor_objetivo is not None else "-")
        c3.metric("Tempo de resolução", f"{r.tempo:.3f} s")
        st.caption(r.mensagem)

        if r.tem_solucao:
            dfv = pd.DataFrame(r.variaveis)
            dfr = pd.DataFrame(r.restricoes)

            st.subheader("Valores das variáveis")
            st.dataframe(dfv, hide_index=True)
            graf = px.bar(dfv, x="Nome", y="Valor", hover_data=["Legenda", "Contribuição para Z"],
                          color_discrete_sequence=["#1F5F8B"])
            graf.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(graf)

            st.subheader("Restrições no ótimo")
            st.dataframe(dfr, hide_index=True)

            if not r.com_sensibilidade:
                st.caption("Os preços sombra e custos reduzidos só são calculados quando todas as "
                           "variáveis são contínuas (em PLI não têm a mesma interpretação).")

            with st.expander("Como interpretar estes resultados"):
                st.markdown(
                    "- **Folga / excesso**: quanto do recurso sobra (restrição <=) ou quanto se "
                    "ultrapassa o mínimo (restrição >=). Zero significa que a restrição está **ativa**.\n"
                    "- **Restrição ativa (vinculativa)**: limita a solução ótima; alterar o lado direito "
                    "altera o ótimo.\n"
                    "- **Preço sombra**: quanto Z varia por cada unidade adicional no lado direito da "
                    "restrição, mantendo-se a mesma base ótima. É o valor máximo que valeria a pena "
                    "pagar por mais uma unidade desse recurso.\n"
                    "- **Custo reduzido**: quanto Z varia por cada unidade que se force uma variável "
                    "que está no seu limite (normalmente 0) a entrar na solução. Nas variáveis que "
                    "estão na solução é 0."
                )


# --------------------------------------------------------------------------- #
# Separador: Método gráfico
# --------------------------------------------------------------------------- #
with tab_graf:
    if modelo is None or erros:
        st.info("Corrige os erros do modelo para ver o gráfico.")
    elif len(modelo.variaveis) != 2:
        st.info(f"O método gráfico só se aplica a problemas com 2 variáveis. "
                f"Este modelo tem {len(modelo.variaveis)}.")
    else:
        r = ss.get("resultado")
        r_ok = r if (r is not None and ss.resultado_hash == hash_modelo(modelo)) else None
        fig, vertices = desenhar_2d(modelo, r_ok)
        st.plotly_chart(fig)
        if r_ok is None:
            st.caption("Resolve o modelo para ver o ponto ótimo e a reta da função objetivo.")
        if vertices:
            st.subheader("Vértices da região admissível")
            dfv = pd.DataFrame(vertices)
            reais = dfv[dfv["Observação"] == ""]
            if not reais.empty:
                melhor = reais["Z"].idxmax() if modelo.sentido == "max" else reais["Z"].idxmin()
                dfv["Melhor vértice"] = ["sim" if i == melhor else "" for i in dfv.index]
            st.dataframe(dfv, hide_index=True)
            st.caption("Pelo teorema fundamental da PL, se existe ótimo finito, ele está num vértice "
                       "da região admissível (em problemas contínuos).")


# --------------------------------------------------------------------------- #
# Separador: Importar e exportar
# --------------------------------------------------------------------------- #
with tab_fich:
    st.subheader("Guardar o modelo")
    if modelo is not None:
        c1, c2 = st.columns(2)
        c1.download_button("Descarregar modelo (JSON)", modelo.to_json(),
                           file_name=f"{slug(modelo.nome)}.json", mime="application/json")
        c2.download_button("Descarregar modelo (Excel)", excel_de_modelo(modelo),
                           file_name=f"{slug(modelo.nome)}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Descarregar formulação (texto)", formulacao_texto(modelo),
                           file_name=f"{slug(modelo.nome)}_formulacao.txt", mime="text/plain")

        r = ss.get("resultado")
        if r is not None and r.tem_solucao and ss.resultado_hash == hash_modelo(modelo):
            st.subheader("Guardar a solução")
            st.download_button("Descarregar solução (Excel)", excel_de_resultado(modelo, r),
                               file_name=f"{slug(modelo.nome)}_solucao.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.subheader("Abrir um modelo")
    st.file_uploader("Ficheiro .json ou .xlsx", type=["json", "xlsx"], key="upload")
    st.button("Abrir modelo", on_click=abrir_ficheiro, disabled=ss.get("upload") is None)
    msg = ss.get("msg_import")
    if msg:
        (st.success if msg[0] == "ok" else st.error)(msg[1])
    st.caption(
        "O Excel deve ter as folhas **Variaveis** (Nome, Legenda, Tipo, Limite inferior, "
        "Limite superior, Coef. objetivo) e **Restricoes** (Nome, Legenda, uma coluna por variável, "
        "Sinal, Lado direito). A folha **Modelo** (Nome, Objetivo) é opcional. "
        "Descarrega um modelo em Excel para teres um ficheiro-base."
    )
