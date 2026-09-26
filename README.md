# Otimizador de Programação Linear

Aplicação web para formular e resolver problemas de **Programação Linear (PL)**, **Programação Linear Inteira (PLI)** e **binária**, com número ilimitado de variáveis e restrições, legendas para cada elemento, análise de sensibilidade e método gráfico para problemas com 2 variáveis.

Construída em Python com [Streamlit](https://streamlit.io) (interface) e o solver [HiGHS](https://highs.dev) (via SciPy), que vem incluído na instalação, sem binários externos.

## Funcionalidades

- Tabela de **variáveis de decisão**: nome, legenda, tipo (contínua, inteira, binária), limites inferior e superior e coeficiente na função objetivo.
- Tabela de **restrições**: nome, legenda, um coeficiente por variável, sinal (`<=`, `>=`, `=`) e lado direito. As colunas dos coeficientes acompanham automaticamente as variáveis (acrescentar, remover ou renomear).
- Linhas ilimitadas nas duas tabelas (botão **+**, ou colar diretamente do Excel).
- **Formulação matemática** apresentada em notação LaTeX e validação com mensagens claras (nomes em falta ou repetidos, limites incoerentes, restrições triviais…).
- **Solução**: estado (ótimo, impossível, ilimitado), valor de Z, valores das variáveis, contribuição de cada uma para Z, folgas e restrições ativas.
- **Análise de sensibilidade** (PL contínua): preços sombra e custos reduzidos.
- **Método gráfico** (2 variáveis): retas das restrições, região admissível, vértices com o valor de Z, reta da função objetivo no ótimo e pontos inteiros admissíveis quando as variáveis são inteiras.
- **Importar/exportar** modelos em JSON e Excel; exportar a solução em Excel e a formulação em texto.
- **Pipeline por linha de comandos** para resolver ficheiros JSON sem interface.

## Estrutura

```
otimizador-pl/
├── app.py                  # interface web (Streamlit)
├── lp_core.py              # modelo, validação, solver, formulação e CLI
├── tabelas.py              # conversão modelo <-> tabelas <-> Excel
├── grafico.py              # método gráfico (2 variáveis)
├── exemplos/               # modelos prontos (JSON) e ficheiro-base Excel
├── tests/                  # testes automáticos (pytest)
├── .streamlit/config.toml  # tema da app
├── .github/workflows/      # testes automáticos no GitHub
└── requirements.txt
```

## 1. Correr no VS Code (local)

Requisitos: Python 3.10 ou superior, Git e VS Code com a extensão **Python**.

```bash
# 1. Abrir a pasta do projeto no VS Code e abrir o terminal (Ctrl+`)

# 2. Criar e ativar um ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Arrancar a app (abre no browser em http://localhost:8501)
streamlit run app.py
```

No VS Code, escolhe o interpretador da `.venv` com `Ctrl+Shift+P` → *Python: Select Interpreter*.

## 2. Usar a app

1. Na barra lateral, dá um nome ao problema e escolhe **Maximizar** ou **Minimizar** (ou carrega um exemplo).
2. Na tabela **Variáveis**, cria uma linha por variável. Na coluna *Legenda* escreve o significado com unidades (ex.: *Mesas produzidas por semana*).
3. Na tabela **Restrições**, cria uma linha por restrição, preenche os coeficientes, o sinal e o lado direito.
4. Confere a **Formulação** e carrega em **Resolver**.
5. Consulta os separadores **Solução** e **Método gráfico**, e guarda o modelo ou a solução em **Importar e exportar**.

Notas:
- Limite inferior vazio = sem limite (−∞); limite superior vazio = sem limite (+∞). Por omissão as variáveis novas são `>= 0`.
- Aceita vírgula ou ponto como separador decimal nos ficheiros importados.
- Renomear uma variável mantém os coeficientes já escritos nas restrições.

## 3. Pipeline pela linha de comandos

```bash
python lp_core.py exemplos/producao.json
python lp_core.py exemplos/dieta.json --tempo 30
```

Também podes usar o núcleo noutros scripts ou num notebook (Colab/Jupyter):

```python
from lp_core import Modelo, resolver, formulacao_texto

m = Modelo.from_json(open("exemplos/producao.json", encoding="utf-8").read())
print(formulacao_texto(m))
r = resolver(m)
print(r.estado, r.valor_objetivo)
for linha in r.variaveis:
    print(linha["Nome"], linha["Valor"])
```

## 4. Formato dos ficheiros

### JSON

```json
{
  "nome": "Produção de mesas e cadeiras",
  "sentido": "max",
  "variaveis": [
    {"nome": "mesas", "legenda": "Mesas por semana", "tipo": "Contínua", "coef": 70, "lb": 0, "ub": null},
    {"nome": "cadeiras", "legenda": "Cadeiras por semana", "tipo": "Contínua", "coef": 50, "lb": 0, "ub": null}
  ],
  "restricoes": [
    {"nome": "Madeira", "legenda": "Metros de madeira", "coefs": {"mesas": 4, "cadeiras": 3}, "sinal": "<=", "rhs": 240},
    {"nome": "Acabamento", "legenda": "Horas de acabamento", "coefs": {"mesas": 2, "cadeiras": 1}, "sinal": "<=", "rhs": 100}
  ]
}
```

`tipo` pode ser `Contínua`, `Inteira` ou `Binária`; `sentido` pode ser `max` ou `min`; `lb`/`ub` a `null` significam sem limite.

### Excel

Usa `exemplos/modelo_base.xlsx` como ponto de partida. Folhas:

| Folha | Colunas |
|---|---|
| `Modelo` (opcional) | Parâmetro, Valor (linhas `Nome` e `Objetivo` = max/min) |
| `Variaveis` | Nome, Legenda, Tipo, Limite inferior, Limite superior, Coef. objetivo |
| `Restricoes` | Nome, Legenda, *uma coluna por variável*, Sinal, Lado direito |

## 5. Testes

```bash
pip install pytest
pytest -q
```

## 6. Publicar no GitHub

```bash
git init
git add .
git commit -m "Otimizador de Programação Linear"
git branch -M main
git remote add origin https://github.com/<o-teu-utilizador>/otimizador-pl.git
git push -u origin main
```

(Primeiro cria o repositório vazio `otimizador-pl` em github.com → *New repository*.) Em cada `push`, o GitHub Actions corre os testes automaticamente (separador *Actions*).

No VS Code também podes usar o painel **Source Control** (`Ctrl+Shift+G`) → *Publish to GitHub*.

## 7. Pôr a app online (site público gratuito)

1. Entra em [share.streamlit.io](https://share.streamlit.io) com a conta GitHub.
2. **Create app** → *Deploy a public app from GitHub*.
3. Escolhe o repositório `otimizador-pl`, o ramo `main` e o ficheiro `app.py`.
4. Em *Advanced settings*, escolhe Python 3.12.
5. **Deploy**. Em 1 a 2 minutos tens um endereço do tipo `https://otimizador-pl.streamlit.app`.

Cada `git push` para `main` atualiza o site automaticamente.

Alternativas: Hugging Face Spaces (SDK Streamlit) ou Render.

## Conceitos (resumo)

- **Função objetivo**: `Z = c1·x1 + c2·x2 + … + cn·xn`, a maximizar ou minimizar.
- **Restrições**: `ai1·x1 + … + ain·xn (<=, >=, =) bi`.
- **Folga**: quanto sobra de um recurso; restrição **ativa** quando a folga é zero.
- **Preço sombra**: variação de Z por cada unidade adicional no lado direito de uma restrição (válido enquanto a base ótima não muda).
- **Custo reduzido**: variação de Z por cada unidade que se force uma variável a entrar na solução.
- **Método gráfico**: se existe ótimo finito num problema contínuo, ele está num vértice da região admissível.

## Evoluções possíveis

- Intervalos de sensibilidade (limites dentro dos quais os preços sombra se mantêm), usando `highspy` diretamente.
- Análise paramétrica (variar um lado direito e ver Z).
- Explicação dos resultados em linguagem natural com a API do Claude.
