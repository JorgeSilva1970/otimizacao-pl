"""
grafico.py - Método gráfico para programação linear.

Para 2 variáveis mostra a região admissível completa.
Para n > 2 variáveis projeta no plano das duas variáveis selecionadas,
fixando as restantes nos seus valores ótimos (ou no limite inferior) e
absorvendo a sua contribuição no lado direito de cada restrição.
"""
from __future__ import annotations

import math

import plotly.graph_objects as go

from lp_core import Modelo, Resultado, expressao, fmt

PALETA = ["#1F5F8B", "#B03A2E", "#1E8449", "#7D3C98", "#B9770E", "#148F77", "#2E4053", "#A04000"]
TOL = 1e-7


def _cumpre(x, y, hp):
    return all(a * x + b * y <= c + TOL * max(1.0, abs(c)) for a, b, c in hp)


def _vertices(hp):
    pts = []
    for i in range(len(hp)):
        for j in range(i + 1, len(hp)):
            a1, b1, c1 = hp[i]
            a2, b2, c2 = hp[j]
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-12:
                continue
            x = (c1 * b2 - c2 * b1) / det
            y = (a1 * c2 - a2 * c1) / det
            if _cumpre(x, y, hp):
                pts.append((round(x, 9) + 0.0, round(y, 9) + 0.0))
    pts = sorted(set(pts))
    if len(pts) >= 3:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return pts


def _projetar(m: Modelo, ix1: int, ix2: int, fixos: dict[str, float]):
    """
    Projeta restrições e limites de variáveis no plano (variaveis[ix1], variaveis[ix2]).
    As variáveis em `fixos` são absorvidas no lado direito de cada restrição.
    Devolve (hp, retas) onde hp é a lista de semiplanos a*x + b*y <= c.
    """
    v1, v2 = m.variaveis[ix1], m.variaveis[ix2]
    hp, retas = [], []
    for r in m.restricoes:
        a = float(r.coefs.get(v1.nome, 0.0))
        b = float(r.coefs.get(v2.nome, 0.0))
        # Absorver a contribuição das variáveis fixas no RHS
        c = float(r.rhs) - sum(float(r.coefs.get(vn, 0.0)) * fv for vn, fv in fixos.items())
        if a == 0 and b == 0:
            continue
        foi_projetada = any(r.coefs.get(vn, 0.0) != 0 for vn in fixos)
        retas.append((a, b, c, r, foi_projetada))
        if r.sinal in ("<=", "="):
            hp.append((a, b, c))
        if r.sinal in (">=", "="):
            hp.append((-a, -b, -c))
    for k, v in enumerate((v1, v2)):
        lb, ub = v.limites()
        e = (1.0, 0.0) if k == 0 else (0.0, 1.0)
        if lb is not None:
            hp.append((-e[0], -e[1], -lb))
        if ub is not None:
            hp.append((e[0], e[1], ub))
    return hp, retas


def desenhar_2d(m: Modelo, r: Resultado | None = None, ix1: int = 0, ix2: int = 1):
    """
    Devolve (figura plotly, lista de vértices com Z).

    ix1, ix2 — índices das variáveis a colocar nos eixos x e y.
    As restantes são fixadas no valor ótimo (se disponível) ou no limite inferior.
    Para o caso clássico de 2 variáveis ix1=0, ix2=1 e o comportamento é idêntico
    ao método gráfico tradicional (sem fixos, sem projeção).
    """
    v1, v2 = m.variaveis[ix1], m.variaveis[ix2]

    # Valores da solução (se existir)
    sol: dict[str, float] = {}
    if r is not None and r.tem_solucao:
        sol = {d["Nome"]: d["Valor"] for d in r.variaveis}

    # Valores fixos das variáveis não selecionadas
    fixos: dict[str, float] = {}
    for i, v in enumerate(m.variaveis):
        if i not in (ix1, ix2):
            val = sol.get(v.nome)
            fixos[v.nome] = val if val is not None else (v.lb if v.lb is not None else 0.0)

    # Contribuição de Z das variáveis fixas
    z_fixo = sum(v.coef * fixos[v.nome] for v in m.variaveis if v.nome in fixos)

    hp, retas = _projetar(m, ix1, ix2, fixos)

    # Escala da janela a partir dos interceptos, limites e ótimo
    valores = []
    for a, b, c, *_ in retas:
        if a:
            valores.append(abs(c / a))
        if b:
            valores.append(abs(c / b))
    for v in (v1, v2):
        valores += [abs(x) for x in v.limites() if x is not None]
    otimo = None
    if r is not None and r.tem_solucao:
        val = {d["Nome"]: d["Valor"] for d in r.variaveis}
        otimo = (val[v1.nome], val[v2.nome])
        valores += [abs(otimo[0]), abs(otimo[1])]
    M = max([x for x in valores if math.isfinite(x)], default=0.0)
    M = M * 1.25 if M > 0 else 10.0

    caixa = [(1.0, 0.0, M), (-1.0, 0.0, M), (0.0, 1.0, M), (0.0, -1.0, M)]
    verts = _vertices(hp + caixa)

    def inicio(v):
        lb = v.limites()[0]
        return (min(0.0, lb) - 0.05 * M) if lb is not None else -M

    xlo, xhi = inicio(v1), M
    ylo, yhi = inicio(v2), M
    fig = go.Figure()

    # Região admissível
    if len(verts) >= 3:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in verts] + [verts[0][0]],
            y=[p[1] for p in verts] + [verts[0][1]],
            fill="toself", fillcolor="rgba(31,95,139,0.16)",
            line=dict(color="rgba(31,95,139,0.45)", width=1),
            name="Região admissível", hoverinfo="skip"))
    elif verts:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in verts], y=[p[1] for p in verts], mode="lines+markers",
            line=dict(color="rgba(31,95,139,0.8)", width=4),
            name="Região admissível (degenerada)"))
    else:
        fig.add_annotation(
            x=(xlo + xhi) / 2, y=(ylo + yhi) / 2,
            text="Não existe região admissível",
            showarrow=False, font=dict(size=16, color="#B03A2E"))

    # Pontos inteiros admissíveis (quando as variáveis selecionadas são inteiras/binárias)
    if all(v.tipo != "Contínua" for v in (v1, v2)):
        xi = range(math.ceil(max(xlo, -M)), math.floor(xhi) + 1)
        yi = range(math.ceil(max(ylo, -M)), math.floor(yhi) + 1)
        if len(xi) * len(yi) <= 4000:
            pts = [(x, y) for x in xi for y in yi if _cumpre(x, y, hp)]
            if pts:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers",
                    marker=dict(size=5, color="#566573"),
                    name="Pontos inteiros admissíveis"))

    # Retas das restrições (com RHS já projetado)
    for i, (a, b, c, rr, foi_proj) in enumerate(retas):
        if abs(b) > 1e-12:
            xs = [xlo, xhi]
            ys = [(c - a * x) / b for x in xs]
        else:
            xs, ys = [c / a, c / a], [ylo, yhi]
        sufixo = " *" if foi_proj else ""
        rotulo = f"{rr.nome}{sufixo}: {expressao([(a, v1.nome), (b, v2.nome)])} {rr.sinal} {fmt(c)}"
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=rotulo,
            line=dict(color=PALETA[i % len(PALETA)], width=2),
            hovertemplate=f"{rotulo}<extra></extra>"))

    # Vértices e valor de Z (incluindo a contribuição das variáveis fixas)
    tabela = []
    for x, y in verts:
        corte = abs(abs(x) - M) < 1e-6 * M or abs(abs(y) - M) < 1e-6 * M
        z_total = v1.coef * x + v2.coef * y + z_fixo
        tabela.append({
            v1.nome: x, v2.nome: y, "Z": z_total,
            "Observação": "limite do gráfico (região ilimitada)" if corte else "",
        })
    reais = [t for t in tabela if not t["Observação"]]
    if reais:
        fig.add_trace(go.Scatter(
            x=[t[v1.nome] for t in reais], y=[t[v2.nome] for t in reais], mode="markers",
            marker=dict(size=9, color="#1B2631", line=dict(color="white", width=1)),
            name="Vértices",
            customdata=[t["Z"] for t in reais],
            hovertemplate=(
                f"{v1.nome} = %{{x:.4g}}<br>{v2.nome} = %{{y:.4g}}<br>"
                f"Z = %{{customdata:.4g}}<extra></extra>"
            )))

    # Ótimo e reta da função objetivo no plano projetado
    if otimo is not None:
        # z_proj é a parte de Z que depende de v1 e v2 (excluindo z_fixo)
        z_proj = v1.coef * otimo[0] + v2.coef * otimo[1]
        if abs(v2.coef) > 1e-12:
            xs = [xlo, xhi]
            ys = [(z_proj - v1.coef * x) / v2.coef for x in xs]
        elif abs(v1.coef) > 1e-12:
            xs, ys = [z_proj / v1.coef] * 2, [ylo, yhi]
        else:
            xs, ys = [], []
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color="#1B2631", width=2, dash="dash"),
                name=f"Função objetivo no ótimo (Z = {fmt(round(r.valor_objetivo, 6))})"))
        fig.add_trace(go.Scatter(
            x=[otimo[0]], y=[otimo[1]], mode="markers",
            marker=dict(symbol="star", size=18, color="#D4AC0D", line=dict(color="#1B2631", width=1)),
            name=f"Ótimo ({fmt(otimo[0])}; {fmt(otimo[1])})"))

    def titulo_eixo(v):
        return f"{v.nome} ({v.legenda})" if v.legenda else v.nome

    titulo_grafico = None
    if fixos:
        partes = [f"{vn} = {fmt(fv)}" for vn, fv in fixos.items()]
        titulo_grafico = "Projeção com " + ", ".join(partes)

    fig.update_layout(
        template="plotly_white",
        height=580,
        margin=dict(l=10, r=10, t=45 if titulo_grafico else 30, b=10),
        title=dict(text=titulo_grafico, font=dict(size=13, color="#566573")) if titulo_grafico else None,
        xaxis=dict(title=titulo_eixo(v1), range=[xlo, xhi], zeroline=True, zerolinecolor="#AAB7B8"),
        yaxis=dict(title=titulo_eixo(v2), range=[ylo, yhi], zeroline=True, zerolinecolor="#AAB7B8"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0),
    )
    return fig, tabela
