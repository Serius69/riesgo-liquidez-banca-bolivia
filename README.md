# 📰 NLP Aplicado a Noticias Financieras LATAM

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Español-blue?style=flat)
![Status](https://img.shields.io/badge/Estado-Activo-brightgreen)

Pipeline completo de procesamiento de lenguaje natural aplicado a noticias financieras en español del mercado latinoamericano. Incluye análisis de sentimiento, clasificación de categorías con TF-IDF, topic modeling con LDA y correlación con precios de activos.

---

## 🎯 Por Qué NLP Financiero en Español

La mayoría de los modelos de sentimiento financiero están entrenados en inglés (Bloomberg, Reuters). El mercado boliviano y LATAM genera noticias en español con particularidades propias:

- Términos económicos regionales ("encaje legal", "tipo de cambio paralelo", "aguinaldo")
- Entidades locales (BCB, ASFI, YPFB, Tigo Money)
- Contexto político-económico específico

Este proyecto construye ese pipeline desde cero con lexicón financiero personalizado para español.

---

## 🧠 Pipeline Técnico

```
Titulares en español
       ↓
1. Limpieza         → minúsculas, regex, stopwords ES + financieras
       ↓
2. Tokenización     → NLTK con modelo español
       ↓
3. Sentimiento      → VADER + lexicón financiero personalizado
       ↓
4. Clasificación    → TF-IDF (1-2 gramas) + Logistic Regression
       ↓
5. Topic Modeling   → LDA (descubrimiento no supervisado de temas)
       ↓
6. Correlación      → Sentimiento vs retornos de activos (BTC, ILF)
```

---

## 📐 Decisiones Técnicas Clave

**¿Por qué VADER con lexicón personalizado y no un modelo preentrenado?**

VADER es ligero, rápido e interpretable. Para producción en fintech con recursos limitados, es más práctico que cargar un modelo BERT de 500MB. El lexicón financiero personalizado compensa la falta de vocabulario de dominio.

```python
# Lexicón financiero para contexto boliviano/LATAM
lexicon_financiero = {
    'superávit':   +3.0,   'crisis':      -3.0,
    'crecimiento': +2.5,   'devaluación': -3.0,
    'expansión':   +2.0,   'escasez':     -2.5,
    'récord':      +2.0,   'fraude':      -3.5,
    # ... más términos
}
analyzer.lexicon.update(lexicon_financiero)
```

**¿Por qué TF-IDF y no embeddings?**

Con datasets pequeños (< 1000 noticias), TF-IDF supera a modelos de embeddings. Los embeddings necesitan miles de ejemplos para generalizar. TF-IDF con bigramas captura bien los patrones de noticias financieras.

**¿Por qué LDA para topic modeling?**

LDA es probabilístico e interpretable. Cada documento es una mezcla de temas, y cada tema tiene palabras con probabilidades asignadas. Ideal para exploración cuando no tienes categorías predefinidas.

---

## 📊 Fuentes de Datos Reales

Para escalar este análisis con noticias reales, estas fuentes tienen RSS en español:

| Fuente | URL RSS | Cobertura |
|---|---|---|
| El Deber (Bolivia) | `eldeber.com.bo/rss` | Bolivia |
| La Razón (Bolivia) | `la-razon.com/rss` | Bolivia |
| Bloomberg Línea | `bloomberglinea.com/rss` | LATAM |
| El Financiero MX | `elfinanciero.com.mx/rss` | México |
| Portafolio CO | `portafolio.co/rss` | Colombia |
| Gestión PE | `gestion.pe/rss` | Perú |

```python
import feedparser

def leer_rss(url: str) -> list:
    feed = feedparser.parse(url)
    return [{'titulo': e.title, 'fecha': e.published, 'link': e.link}
            for e in feed.entries]
```

---

## 📁 Estructura

```
nlp-noticias-financieras-latam/
├── nlp_noticias_financieras.ipynb    # Análisis completo
├── src/
│   └── monitor_sentimiento.py        # Script de producción con CLI
├── datos/                            # Tus CSVs de noticias
├── modelos/                          # Modelos entrenados (.pkl)
├── img/                              # Gráficas exportadas
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación

```bash
git clone https://github.com/Serius69/nlp-noticias-financieras-latam
cd nlp-noticias-financieras-latam

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p datos modelos img

# Notebook completo
jupyter lab nlp_noticias_financieras.ipynb
```

**Usar el monitor desde CLI:**
```bash
# Analizar un CSV de noticias
python src/monitor_sentimiento.py --csv datos/noticias.csv

# Especificar columna de texto
python src/monitor_sentimiento.py --csv datos/noticias.csv --col-texto headline

# Guardar resultados
python src/monitor_sentimiento.py --csv datos/noticias.csv --output datos/con_sentimiento.csv
```

**Formato CSV de entrada:**
```
fecha,titulo,fuente,categoria,pais
2024-01-05,Bolivia registra superávit comercial,El Deber,macroeconomia,BO
2024-01-08,Crisis de reservas preocupa al BCB,La Razón,macroeconomia,BO
```

---

## 📈 Output del Monitor

```
=================================================================
REPORTE DE SENTIMIENTO — NOTICIAS FINANCIERAS
Generado: 2024-04-15 10:32
=================================================================

📊 Resumen General
  Total noticias:    89
  Período:           2024-01-01 → 2024-04-15
  Score promedio:    -0.0821

🎭 Distribución de Sentimientos
  🟢 POSITIVO   ███████ 34 (38.2%)
  ⚪ NEUTRO      ████ 18 (20.2%)
  🔴 NEGATIVO   ██████████ 37 (41.6%)
```

---

## 🔮 Próximos Pasos

- [ ] Integrar RSS feeds en tiempo real con `feedparser`
- [ ] Modelo preentrenado en español: `PlanTL-GOB-ES/roberta-base-bne`
- [ ] Índice de sentimiento financiero boliviano en tiempo real
- [ ] Alertas automáticas cuando el sentimiento cae por debajo de umbral

---

## 🔗 Proyectos Relacionados

- [💱 Dólar Paralelo Bolivia — ARIMA](https://github.com/Serius69/dolar-paralelo-bolivia-arima)
- [₿ Bitcoin LATAM](https://github.com/Serius69/bitcoin-latam-analisis)
- [🔍 Detección de Fraude](https://github.com/Serius69/deteccion-fraude-transacciones)

---

## 👤 Autor

**Sergio** — Data Scientist en Finanzas | [github.com/Serius69](https://github.com/Serius69)
