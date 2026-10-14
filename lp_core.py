"""
lp_core.py - Núcleo de Programação Linear (PL) e Programação Linear Inteira (PLI)

Contém o modelo de dados, a validação, a resolução com o solver HiGHS (via SciPy)
e a análise de sensibilidade (preços sombra e custos reduzidos).

Pode ser usado pela app Streamlit (app.py) ou diretamente na linha de comandos:

    python lp_core.py exemplos/producao.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
TIPOS = ("Contínua", "Inteira", "Binária")
SINAIS = ("<=", ">=", "=")
SENTIDOS = {"max": "Maximizar", "min": "Minimizar"}
NOMES_RESERVADOS = {"nome", "legenda", "tipo", "sinal", "lado direito", "rhs"}

EXPLICACAO_ESTADO = {
    "Ótimo": "Foi encontrada a solução ótima.",
    "Admissível": (
        "O solver parou no limite de tempo com uma solução admissível, "
        "mas sem garantia de que é a ótima. Aumenta o tempo máximo."
    ),
    "Impossível": (
        "Não existe nenhuma solução que cumpra todas as restrições em simultâneo. "
        "Revê os sinais (<=, >=, =), os valores do lado direito e os limites das variáveis."
    ),
    "Ilimitado": (
        "A função objetivo pode melhorar sem limite. Falta provavelmente uma "
        "restrição que limite alguma variável."
    ),
    "Limite atingido": "O solver parou no limite de tempo/iterações sem encontrar solução.",
    "Erro": "O solver terminou com erro numérico. Verifica os coeficientes do modelo.",
}


# --------------------------------------------------------------------------- #
# Utilitários de conversão
# --------------------------------------------------------------------------- #
def _sem_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def numero(x, default: Optional[float] = None) -> Optional[float]:
    """Converte para float. Vazio/NaN devolve `default`. Aceita vírgula decimal."""
    if x is None:
        return default
    if isinstance(x, str):
        x = x.strip().replace(",", ".")
        if x == "":
            return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"Valor numérico inválido: {x!r}")
    if math.isnan(v):
        return default
    return v


def limite(x) -> Optional[float]:
    """Limite de variável: vazio ou infinito significa 'sem limite' (None)."""
    v = numero(x, None)
    if v is None or math.isinf(v):
        return None
    return v


def normalizar_sinal(s) -> str:
    s = str(s or "<=").strip()
    return {"≤": "<=", "=<": "<=", "<": "<=", "≥": ">=", "=>": ">=", ">": ">=", "==": "="}.get(s, s)


def normalizar_tipo(t) -> str:
    t = _sem_acentos(str(t or "").strip().lower())
    if t.startswith("bin"):
        return "Binária"
    if t.startswith("int"):
        return "Inteira"
    return "Contínua"


def fmt(x: float) -> str:
    """Formata um número de forma compacta (sem zeros inúteis)."""
    if x is None:
        return "-"
    if abs(x) < 1e-12:
        x = 0.0
    return f"{x:.10g}"


def _limpo(x: float) -> float:
    """Remove ruído numérico (ex.: -0.0 ou 1e-15)."""
    if x is None:
        return None
    x = float(x)
    if abs(x) < 1e-7:
        return 0.0
    return round(x, 8)


def parsear_expressao(expr: str) -> list[tuple[float, str]]:
    """
    Converte uma expressão linear textual em [(coef, nome_variável), ...].

    Exemplos aceites:
        "5 x1 + 3*x2 - x3"
        "Z = 5x1 + 3x2"
        "-mesas + 2 cadeiras"
        "0.5*P1 - 1.2 P2 + P3"

    Lança ValueError se não encontrar nenhuma variável.
    """
    expr = re.sub(r'\s+', ' ', expr.strip())
    expr = re.sub(r'^[Zz]\s*=\s*', '', expr).strip()
    if not expr:
        raise ValueError("Expressão vazia.")
    if expr[0] not in ('+', '-'):
        expr = '+' + expr
    padrao = r'([+-])\s*(\d+(?:\.\d*)?|\.\d+)?\s*\*?\s*([a-zA-Z_][a-zA-Z0-9_]*)'
    termos = [
        ((1.0 if s == '+' else -1.0) * (float(c) if c else 1.0), v)
        for s, c, v in re.findall(padrao, expr)
    ]
    if not termos:
        raise ValueError("Não foi possível reconhecer variáveis. Usa o formato: 5 x1 + 3 x2.")
    return termos


# --------------------------------------------------------------------------- #
# Modelo de dados
# --------------------------------------------------------------------------- #
@dataclass
class Variavel:
    nome: str
    coef: float = 0.0          # coeficiente na função objetivo
    legenda: str = ""          # significado da variável (ex.: "n.º de mesas por semana")
    tipo: str = "Contínua"     # Contínua | Inteira | Binária
    lb: Optional[float] = 0.0  # limite inferior (None = -infinito)
    ub: Optional[float] = None # limite superior (None = +infinito)

    def limites(self) -> tuple[Optional[float], Optional[float]]:
        if self.tipo == "Binária":
            return 0.0, 1.0
        return self.lb, self.ub


@dataclass
class Restricao:
    nome: str
    coefs: dict                # {nome_variavel: coeficiente}
    sinal: str = "<="          # <= | >= | =
    rhs: float = 0.0           # lado direito
    legenda: str = ""          # significado (ex.: "horas de máquina disponíveis")


@dataclass
class Modelo:
    nome: str = "Problema"
    sentido: str = "max"       # max | min
    variaveis: list = field(default_factory=list)
    restricoes: list = field(default_factory=list)

    # ---------------- serialização ----------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Modelo":
        variaveis = [
            Variavel(
                nome=str(v.get("nome", "")).strip(),
                coef=numero(v.get("coef"), 0.0),
                legenda=str(v.get("legenda") or "").strip(),
                tipo=normalizar_tipo(v.get("tipo")),
                lb=limite(v.get("lb", 0.0)),
                ub=limite(v.get("ub")),
            )
            for v in d.get("variaveis", [])
        ]
        restricoes = [
            Restricao(
                nome=str(r.get("nome", "")).strip(),
                coefs={str(k).strip(): numero(c, 0.0) for k, c in (r.get("coefs") or {}).items()},
                sinal=normalizar_sinal(r.get("sinal")),
                rhs=numero(r.get("rhs"), 0.0),
                legenda=str(r.get("legenda") or "").strip(),
            )
            for r in d.get("restricoes", [])
        ]
        sentido = str(d.get("sentido", "max")).strip().lower()[:3]
        return cls(
            nome=str(d.get("nome") or "Problema"),
            sentido="min" if sentido == "min" else "max",
            variaveis=variaveis,
            restricoes=restricoes,
        )

    @classmethod
    def from_json(cls, texto: str) -> "Modelo":
        return cls.from_dict(json.loads(texto))

    # ---------------- validação ----------------
    def validar(self) -> tuple[list[str], list[str]]:
        """Devolve (erros, avisos). Com erros o modelo não pode ser resolvido."""
        erros, avisos = [], []
        if self.sentido not in SENTIDOS:
            erros.append("O objetivo tem de ser 'max' ou 'min'.")
        if not self.variaveis:
            erros.append("Define pelo menos uma variável de decisão.")

        nomes = [v.nome for v in self.variaveis]
        for i, v in enumerate(self.variaveis, 1):
            rot = f"'{v.nome}'" if v.nome else f"na linha {i}"
            if not v.nome:
                erros.append(f"Variável na linha {i}: falta o nome.")
            elif v.nome.lower() in NOMES_RESERVADOS:
                erros.append(f"Variável {rot}: esse nome está reservado; escolhe outro (ex.: x{i}).")
            if v.tipo not in TIPOS:
                erros.append(f"Variável {rot}: tipo '{v.tipo}' desconhecido.")
            if not math.isfinite(v.coef):
                erros.append(f"Variável {rot}: o coeficiente da função objetivo tem de ser finito.")
            if v.tipo != "Binária" and v.lb is not None and v.ub is not None and v.lb > v.ub:
                erros.append(f"Variável {rot}: o limite inferior ({fmt(v.lb)}) é maior que o superior ({fmt(v.ub)}).")

        repetidos = sorted({n for n in nomes if n and nomes.count(n) > 1})
        if repetidos:
            erros.append("Nomes de variáveis repetidos: " + ", ".join(repetidos) + ".")

        if self.variaveis and all(v.coef == 0 for v in self.variaveis):
            avisos.append("A função objetivo tem todos os coeficientes a zero: qualquer solução admissível é ótima.")

        rnomes = [r.nome for r in self.restricoes]
        rep_r = sorted({n for n in rnomes if n and rnomes.count(n) > 1})
        if rep_r:
            erros.append("Nomes de restrições repetidos: " + ", ".join(rep_r) + ".")

        conjunto = set(nomes)
        for j, r in enumerate(self.restricoes, 1):
            rot = f"'{r.nome}'" if r.nome else f"na linha {j}"
            if not r.nome:
                erros.append(f"Restrição na linha {j}: falta o nome.")
            if r.sinal not in SINAIS:
                erros.append(f"Restrição {rot}: sinal '{r.sinal}' inválido (usa <=, >= ou =).")
            if not math.isfinite(r.rhs):
                erros.append(f"Restrição {rot}: o lado direito tem de ser um número finito.")
            if any(not math.isfinite(c) for c in r.coefs.values()):
                erros.append(f"Restrição {rot}: há coeficientes não finitos.")
            ativos = {n: c for n, c in r.coefs.items() if n in conjunto and c != 0}
            if not ativos and r.sinal in SINAIS and math.isfinite(r.rhs):
                ok = {"<=": 0 <= r.rhs, ">=": 0 >= r.rhs, "=": r.rhs == 0}[r.sinal]
                if ok:
                    avisos.append(f"Restrição {rot}: todos os coeficientes são zero; é sempre cumprida e será ignorada.")
                else:
                    erros.append(
                        f"Restrição {rot}: todos os coeficientes são zero e 0 {r.sinal} {fmt(r.rhs)} é falso; "
                        "o problema seria impossível."
                    )

        if self.variaveis and not self.restricoes:
            avisos.append("O modelo não tem restrições: pode ser ilimitado.")
        return erros, avisos


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #
@dataclass
class Resultado:
    estado: str
    mensagem: str
    valor_objetivo: Optional[float]
    variaveis: list          # lista de dicionários (uma linha por variável)
    restricoes: list         # lista de dicionários (uma linha por restrição)
    com_sensibilidade: bool  # True só para PL contínua com solução ótima
    tempo: float

    @property
    def tem_solucao(self) -> bool:
        return self.estado in ("Ótimo", "Admissível")


# --------------------------------------------------------------------------- #
# Resolução
# --------------------------------------------------------------------------- #
def resolver(modelo: Modelo, tempo_limite: float = 60.0) -> Resultado:
    """Resolve o modelo com HiGHS. Usa linprog (PL) ou milp (PLI/binárias)."""
    erros, _ = modelo.validar()
    if erros:
        raise ValueError("\n".join(erros))

    vs = modelo.variaveis
    idx = {v.nome: i for i, v in enumerate(vs)}
    n = len(vs)
    s = -1.0 if modelo.sentido == "max" else 1.0      # SciPy minimiza sempre
    c = np.array([s * v.coef for v in vs], dtype=float)

    # Matrizes das restrições (>= é convertida em <= multiplicando por -1)
    A_ub, b_ub, ref_ub = [], [], []  # ref: (índice da restrição, fator de sinal)
    A_eq, b_eq, ref_eq = [], [], []
    for j, r in enumerate(modelo.restricoes):
        linha = np.zeros(n)
        for nome, coef in r.coefs.items():
            if nome in idx:
                linha[idx[nome]] = coef
        if not np.any(linha):
            continue  # restrição trivial (validada antes)
        if r.sinal == "<=":
            A_ub.append(linha); b_ub.append(r.rhs); ref_ub.append((j, 1.0))
        elif r.sinal == ">=":
            A_ub.append(-linha); b_ub.append(-r.rhs); ref_ub.append((j, -1.0))
        else:
            A_eq.append(linha); b_eq.append(r.rhs); ref_eq.append((j, 1.0))

    limites = [v.limites() for v in vs]
    inteira = any(v.tipo != "Contínua" for v in vs)
    t0 = time.perf_counter()

    if not inteira:
        res = linprog(
            c,
            A_ub=np.array(A_ub) if A_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq) if A_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=limites,
            method="highs",
            options={"time_limit": float(tempo_limite)},
        )
        estado = {0: "Ótimo", 1: "Limite atingido", 2: "Impossível", 3: "Ilimitado"}.get(res.status, "Erro")
    else:
        restr = []
        if A_ub:
            restr.append(LinearConstraint(np.array(A_ub), -np.inf, np.array(b_ub)))
        if A_eq:
            restr.append(LinearConstraint(np.array(A_eq), np.array(b_eq), np.array(b_eq)))
        lb = np.array([-np.inf if a is None else a for a, _ in limites])
        ub = np.array([np.inf if b is None else b for _, b in limites])
        integ = np.array([0 if v.tipo == "Contínua" else 1 for v in vs])
        res = milp(c, constraints=restr, integrality=integ, bounds=Bounds(lb, ub),
                   options={"time_limit": float(tempo_limite)})
        if res.status == 0:
            estado = "Ótimo"
        elif res.status == 1:
            estado = "Admissível" if res.x is not None else "Limite atingido"
        else:
            estado = {2: "Impossível", 3: "Ilimitado"}.get(res.status, "Erro")

    dt = time.perf_counter() - t0
    mensagem = EXPLICACAO_ESTADO.get(estado, str(res.message))

    if estado not in ("Ótimo", "Admissível") or res.x is None:
        return Resultado(estado, mensagem, None, [], [], False, dt)

    x = np.array([_limpo(v) for v in res.x])
    valores = {v.nome: float(x[i]) for i, v in enumerate(vs)}
    z = _limpo(sum(v.coef * valores[v.nome] for v in vs))
    sens = (not inteira) and estado == "Ótimo"

    # Custos reduzidos: variação de Z por unidade que a variável é forçada a mudar
    custos = [None] * n
    if sens:
        custos = [_limpo(s * (lo + up)) for lo, up in zip(res.lower.marginals, res.upper.marginals)]

    linhas_var = []
    for i, v in enumerate(vs):
        linha = {
            "Nome": v.nome,
            "Legenda": v.legenda,
            "Tipo": v.tipo,
            "Valor": valores[v.nome],
            "Coef. objetivo": v.coef,
            "Contribuição para Z": _limpo(v.coef * valores[v.nome]),
        }
        if sens:
            linha["Custo reduzido"] = custos[i]
        linhas_var.append(linha)

    # Preços sombra: variação de Z por unidade adicional no lado direito
    precos = {}
    if sens:
        for (j, fator), m in zip(ref_ub, res.ineqlin.marginals):
            precos[j] = _limpo(s * fator * m)
        for (j, fator), m in zip(ref_eq, res.eqlin.marginals):
            precos[j] = _limpo(s * fator * m)

    linhas_res = []
    for j, r in enumerate(modelo.restricoes):
        lhs = _limpo(sum(coef * valores.get(nome, 0.0) for nome, coef in r.coefs.items() if nome in idx))
        if r.sinal == "<=":
            folga = r.rhs - lhs
        elif r.sinal == ">=":
            folga = lhs - r.rhs
        else:
            folga = abs(lhs - r.rhs)
        folga = _limpo(folga)
        ativa = abs(folga) <= 1e-6 * max(1.0, abs(r.rhs))
        linha = {
            "Nome": r.nome,
            "Legenda": r.legenda,
            "Lado esquerdo": lhs,
            "Sinal": r.sinal,
            "Lado direito": r.rhs,
            "Folga / excesso": folga,
            "Estado": "Ativa (vinculativa)" if ativa else "Não ativa",
        }
        if sens:
            linha["Preço sombra"] = precos.get(j, 0.0)
        linhas_res.append(linha)

    return Resultado(estado, mensagem, z, linhas_var, linhas_res, sens, dt)


# --------------------------------------------------------------------------- #
# Formulação matemática (texto e LaTeX)
# --------------------------------------------------------------------------- #
def _escapar_latex(s: str) -> str:
    for a, b in (("\\", r"\backslash "), ("{", r"\{"), ("}", r"\}"), ("_", r"\_"),
                 ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("^", ""), ("~", "")):
        s = s.replace(a, b)
    return s


def expressao(pares: list[tuple[float, str]], latex: bool = False) -> str:
    """Escreve uma expressão linear, ex.: [(3,'x1'),(-1,'x2')] -> '3 x1 - x2'."""
    nome_fmt = (lambda n: r"\text{" + _escapar_latex(n) + "}") if latex else (lambda n: n)
    sep = r"\," if latex else " "
    partes = []
    for coef, nome in pares:
        if coef == 0:
            continue
        a = abs(coef)
        termo = nome_fmt(nome) if math.isclose(a, 1.0) else f"{fmt(a)}{sep}{nome_fmt(nome)}"
        partes.append(("-" if coef < 0 else "+", termo))
    if not partes:
        return "0"
    texto = ("-" if partes[0][0] == "-" else "") + partes[0][1]
    for sinal, termo in partes[1:]:
        texto += f" {sinal} {termo}"
    return texto


def _dominio(v: Variavel, latex: bool) -> str:
    nome = r"\text{" + _escapar_latex(v.nome) + "}" if latex else v.nome
    le, ge = (r"\le", r"\ge") if latex else ("<=", ">=")
    if v.tipo == "Binária":
        return f"{nome} " + (r"\in \{0,1\}" if latex else "∈ {0,1}")
    lb, ub = v.lb, v.ub
    if lb is not None and ub is not None:
        base = f"{fmt(lb)} {le} {nome} {le} {fmt(ub)}"
    elif lb is not None:
        base = f"{nome} {ge} {fmt(lb)}"
    elif ub is not None:
        base = f"{nome} {le} {fmt(ub)}"
    else:
        base = f"{nome}" + (r"\ \text{livre}" if latex else " livre")
    if v.tipo == "Inteira":
        base += r",\ \text{inteira}" if latex else " (inteira)"
    return base


def formulacao_texto(m: Modelo) -> str:
    nomes = [v.nome for v in m.variaveis]
    linhas = [f"{SENTIDOS[m.sentido]} Z = {expressao([(v.coef, v.nome) for v in m.variaveis])}", "Sujeito a:"]
    for r in m.restricoes:
        expr = expressao([(r.coefs.get(n, 0.0), n) for n in nomes])
        leg = f"   [{r.legenda}]" if r.legenda else ""
        linhas.append(f"  {r.nome}: {expr} {r.sinal} {fmt(r.rhs)}{leg}")
    linhas.append("  " + ", ".join(_dominio(v, False) for v in m.variaveis))
    legendas = [f"  {v.nome}: {v.legenda}" for v in m.variaveis if v.legenda]
    if legendas:
        linhas += ["Legenda das variáveis:"] + legendas
    return "\n".join(linhas)


def formulacao_latex(m: Modelo) -> str:
    nomes = [v.nome for v in m.variaveis]
    sentido = r"\max" if m.sentido == "max" else r"\min"
    linhas = [sentido + r"\; Z = " + expressao([(v.coef, v.nome) for v in m.variaveis], latex=True),
              r"\text{sujeito a:}"]
    simb = {"<=": r"\le", ">=": r"\ge", "=": "="}
    for r in m.restricoes:
        expr = expressao([(r.coefs.get(n, 0.0), n) for n in nomes], latex=True)
        linhas.append(rf"\quad {expr} {simb.get(r.sinal, r.sinal)} {fmt(r.rhs)}"
                      rf"\qquad (\text{{{_escapar_latex(r.nome)}}})")
    linhas.append(r"\quad " + r",\ ".join(_dominio(v, True) for v in m.variaveis))
    return r"\begin{array}{l}" + r" \\ ".join(linhas) + r"\end{array}"


# --------------------------------------------------------------------------- #
# Exemplos
# --------------------------------------------------------------------------- #
EXEMPLOS = {
    "Produção de mesas e cadeiras (2 variáveis)": {
        "nome": "Produção de mesas e cadeiras",
        "sentido": "max",
        "variaveis": [
            {"nome": "mesas", "legenda": "Mesas produzidas por semana", "coef": 70, "lb": 0},
            {"nome": "cadeiras", "legenda": "Cadeiras produzidas por semana", "coef": 50, "lb": 0},
        ],
        "restricoes": [
            {"nome": "Madeira", "legenda": "Metros de madeira disponíveis por semana",
             "coefs": {"mesas": 4, "cadeiras": 3}, "sinal": "<=", "rhs": 240},
            {"nome": "Acabamento", "legenda": "Horas de acabamento disponíveis por semana",
             "coefs": {"mesas": 2, "cadeiras": 1}, "sinal": "<=", "rhs": 100},
        ],
    },
    "Dieta de custo mínimo (4 variáveis)": {
        "nome": "Dieta de custo mínimo",
        "sentido": "min",
        "variaveis": [
            {"nome": "aveia", "legenda": "Porções de aveia por dia", "coef": 0.30, "lb": 0},
            {"nome": "leite", "legenda": "Copos de leite por dia", "coef": 0.25, "lb": 0, "ub": 4},
            {"nome": "ovos", "legenda": "Ovos por dia", "coef": 0.20, "lb": 0, "ub": 3},
            {"nome": "feijao", "legenda": "Porções de feijão por dia", "coef": 0.35, "lb": 0, "ub": 4},
        ],
        "restricoes": [
            {"nome": "Proteína", "legenda": "Gramas de proteína (mínimo diário)",
             "coefs": {"aveia": 5, "leite": 8, "ovos": 6, "feijao": 9}, "sinal": ">=", "rhs": 60},
            {"nome": "Energia", "legenda": "Quilocalorias (mínimo diário)",
             "coefs": {"aveia": 150, "leite": 120, "ovos": 70, "feijao": 230}, "sinal": ">=", "rhs": 2000},
            {"nome": "Fibra", "legenda": "Gramas de fibra (mínimo diário)",
             "coefs": {"aveia": 4, "feijao": 8}, "sinal": ">=", "rhs": 25},
        ],
    },
    "Seleção de projetos (variáveis binárias)": {
        "nome": "Seleção de projetos de investimento",
        "sentido": "max",
        "variaveis": [
            {"nome": "P1", "legenda": "Projeto 1 (1 = aprovar)", "coef": 20, "tipo": "Binária"},
            {"nome": "P2", "legenda": "Projeto 2 (1 = aprovar)", "coef": 15, "tipo": "Binária"},
            {"nome": "P3", "legenda": "Projeto 3 (1 = aprovar)", "coef": 30, "tipo": "Binária"},
            {"nome": "P4", "legenda": "Projeto 4 (1 = aprovar)", "coef": 12, "tipo": "Binária"},
        ],
        "restricoes": [
            {"nome": "Orçamento", "legenda": "Milhares de euros disponíveis",
             "coefs": {"P1": 8, "P2": 5, "P3": 12, "P4": 4}, "sinal": "<=", "rhs": 20},
            {"nome": "Equipa", "legenda": "Pessoas disponíveis",
             "coefs": {"P1": 3, "P2": 2, "P3": 4, "P4": 1}, "sinal": "<=", "rhs": 6},
        ],
    },
}


def modelo_em_branco() -> Modelo:
    return Modelo(
        nome="Novo problema",
        sentido="max",
        variaveis=[
            Variavel("x1", coef=1.0, legenda="Primeira variável de decisão"),
            Variavel("x2", coef=1.0, legenda="Segunda variável de decisão"),
        ],
        restricoes=[
            Restricao("R1", {"x1": 1.0, "x2": 1.0}, "<=", 10.0, "Primeira restrição"),
        ],
    )


# --------------------------------------------------------------------------- #
# Linha de comandos
# --------------------------------------------------------------------------- #
def _tabela(linhas: list[dict]) -> str:
    if not linhas:
        return "(vazio)"
    cols = list(linhas[0])
    txt = [[str(fmt(l[c]) if isinstance(l[c], float) else l[c]) for c in cols] for l in linhas]
    larg = [max(len(c), *(len(t[i]) for t in txt)) for i, c in enumerate(cols)]
    out = ["  ".join(c.ljust(w) for c, w in zip(cols, larg)), "  ".join("-" * w for w in larg)]
    out += ["  ".join(v.ljust(w) for v, w in zip(t, larg)) for t in txt]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Resolve um modelo de Programação Linear guardado em JSON.")
    p.add_argument("ficheiro", help="Caminho para o ficheiro .json do modelo")
    p.add_argument("--tempo", type=float, default=60.0, help="Tempo máximo do solver em segundos")
    a = p.parse_args(argv)

    with open(a.ficheiro, encoding="utf-8") as f:
        m = Modelo.from_json(f.read())
    erros, avisos = m.validar()
    for av in avisos:
        print("Aviso:", av)
    if erros:
        for e in erros:
            print("Erro:", e)
        return 1

    print(formulacao_texto(m), "\n")
    r = resolver(m, a.tempo)
    print(f"Estado: {r.estado} - {r.mensagem}")
    if r.tem_solucao:
        print(f"Z = {fmt(r.valor_objetivo)}\n")
        print(_tabela(r.variaveis), "\n")
        print(_tabela(r.restricoes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
