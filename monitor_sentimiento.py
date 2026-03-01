"""
src/monitor_sentimiento.py
─────────────────────────────────────────────────────────────────────────────
Monitor de sentimiento financiero en tiempo real.
Lee un CSV de noticias, calcula sentimiento y genera reporte.

Uso:
    python src/monitor_sentimiento.py --csv datos/noticias.csv
    python src/monitor_sentimiento.py --csv datos/noticias.csv --output reporte.csv
"""

import argparse
import re
import numpy as np
import pandas as pd
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Descargar recursos silenciosamente
for r in ['punkt', 'stopwords', 'punkt_tab']:
    nltk.download(r, quiet=True)


# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

STOPWORDS_ES = set(stopwords.words('spanish')) | {
    'millones', 'miles', 'según', 'señaló', 'informó', 'indicó',
    'dijo', 'aseguró', 'afirmó', 'nuevo', 'nueva', 'año', 'mes',
}

LEXICON_FINANCIERO = {
    'superávit': 3.0, 'crecimiento': 2.5, 'récord': 2.0, 'ganancias': 2.5,
    'inversión': 1.5, 'expansión': 2.0, 'adopción': 1.5, 'supera': 2.0,
    'exitoso': 2.5, 'recuperación': 2.0, 'crecen': 2.0, 'aumenta': 1.5,
    'caen': -2.5, 'caída': -2.5, 'crisis': -3.0, 'escasez': -2.5,
    'devaluación': -3.0, 'inflación': -2.0, 'fraude': -3.5, 'suspende': -2.0,
    'mínimo': -2.0, 'contracción': -2.5, 'deuda': -1.5, 'déficit': -2.5,
    'congela': -2.5, 'cierre': -2.0, 'enfrenta': -1.5, 'preocupando': -2.0,
}

# Categorías de entidades financieras bolivianas para detección
ENTIDADES_BO = {
    'instituciones': ['BCB', 'ASFI', 'BDP', 'YPFB', 'ENTEL', 'LAB'],
    'fintechs': ['Tigo Money', 'Simple', 'Pagos Net', 'Fondo Unión'],
    'bancos': ['BNB', 'Banco Bisa', 'Banco Mercantil', 'Banco Sol'],
}


# ─── FUNCIONES CORE ───────────────────────────────────────────────────────────

def limpiar_texto(texto: str) -> str:
    """Limpia y normaliza texto en español."""
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)
    tokens = word_tokenize(texto, language='spanish')
    tokens = [t for t in tokens if t not in STOPWORDS_ES and len(t) > 2]
    return ' '.join(tokens)


def calcular_sentimiento(texto: str, analyzer: SentimentIntensityAnalyzer) -> dict:
    """Calcula sentimiento con VADER + lexicón financiero."""
    scores = analyzer.polarity_scores(texto)
    compound = scores['compound']

    if compound >= 0.05:
        etiqueta = 'POSITIVO'
    elif compound <= -0.05:
        etiqueta = 'NEGATIVO'
    else:
        etiqueta = 'NEUTRO'

    return {
        'score':      round(compound, 4),
        'positivo':   round(scores['pos'], 4),
        'negativo':   round(scores['neg'], 4),
        'neutro':     round(scores['neu'], 4),
        'sentimiento': etiqueta,
    }


def detectar_entidades(texto: str) -> list:
    """Detecta entidades financieras bolivianas en el texto."""
    texto_upper = texto.upper()
    encontradas = []
    for categoria, entidades in ENTIDADES_BO.items():
        for ent in entidades:
            if ent.upper() in texto_upper:
                encontradas.append(f'{ent} ({categoria})')
    return encontradas


def calcular_indice_sentimiento(df: pd.DataFrame,
                                  ventana_dias: int = 7) -> pd.Series:
    """
    Calcula índice de sentimiento rolling.
    Agrega scores diarios y aplica media móvil.
    """
    diario = df.groupby('fecha')['score'].mean()
    return diario.rolling(f'{ventana_dias}D').mean()


def generar_reporte(df: pd.DataFrame) -> None:
    """Imprime reporte ejecutivo de sentimiento."""
    print('\n' + '='*65)
    print('REPORTE DE SENTIMIENTO — NOTICIAS FINANCIERAS')
    print(f'Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('='*65)

    print(f'\n📊 Resumen General')
    print(f'  Total noticias:    {len(df):,}')
    print(f'  Período:           {df["fecha"].min().date()} → {df["fecha"].max().date()}')
    print(f'  Score promedio:    {df["score"].mean():+.4f}')

    print(f'\n🎭 Distribución de Sentimientos')
    for sent in ['POSITIVO', 'NEUTRO', 'NEGATIVO']:
        count = (df['sentimiento'] == sent).sum()
        pct   = count / len(df) * 100
        barra = '█' * int(pct / 3)
        emoji = '🟢' if sent == 'POSITIVO' else ('⚪' if sent == 'NEUTRO' else '🔴')
        print(f'  {emoji} {sent:<10} {barra} {count} ({pct:.1f}%)')

    print(f'\n📰 5 Noticias Más Positivas')
    for _, row in df.nlargest(5, 'score').iterrows():
        print(f'  [{row["score"]:+.3f}] {str(row.get("titulo", row.get("texto", "")))[:70]}')

    print(f'\n📰 5 Noticias Más Negativas')
    for _, row in df.nsmallest(5, 'score').iterrows():
        print(f'  [{row["score"]:+.3f}] {str(row.get("titulo", row.get("texto", "")))[:70]}')

    if 'categoria' in df.columns:
        print(f'\n📂 Sentimiento por Categoría')
        por_cat = df.groupby('categoria')['score'].agg(['mean', 'count']).round(3)
        por_cat = por_cat.sort_values('mean', ascending=False)
        for cat, row in por_cat.iterrows():
            emoji = '🟢' if row['mean'] > 0.05 else ('🔴' if row['mean'] < -0.05 else '⚪')
            print(f'  {emoji} {cat:<20} score: {row["mean"]:+.3f}  ({int(row["count"])} noticias)')

    print('\n' + '='*65)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def procesar_csv(ruta_csv: str, col_texto: str = 'titulo',
                  col_fecha: str = 'fecha') -> pd.DataFrame:
    """Procesa CSV de noticias y agrega columnas de sentimiento."""
    df = pd.read_csv(ruta_csv)

    if col_fecha in df.columns:
        df[col_fecha] = pd.to_datetime(df[col_fecha])
        df = df.rename(columns={col_fecha: 'fecha'})

    if col_texto not in df.columns:
        raise ValueError(f"Columna '{col_texto}' no encontrada. Columnas: {df.columns.tolist()}")

    print(f'📂 Cargadas {len(df)} noticias desde {ruta_csv}')

    # Configurar analizador
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(LEXICON_FINANCIERO)

    print('⚙️  Calculando sentimiento...')
    df['texto_limpio'] = df[col_texto].apply(limpiar_texto)

    resultados = df[col_texto].apply(lambda t: calcular_sentimiento(t, analyzer))
    df = pd.concat([df, pd.DataFrame(resultados.tolist())], axis=1)

    df['entidades'] = df[col_texto].apply(
        lambda t: ', '.join(detectar_entidades(t)) if detectar_entidades(t) else ''
    )

    print(f'✅ Sentimiento calculado para {len(df)} noticias')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor de sentimiento financiero')
    parser.add_argument('--csv',       required=True,     help='Ruta al CSV de noticias')
    parser.add_argument('--col-texto', default='titulo',  help='Columna con el texto (default: titulo)')
    parser.add_argument('--col-fecha', default='fecha',   help='Columna con la fecha (default: fecha)')
    parser.add_argument('--output',    default=None,      help='Guardar resultados en CSV')
    args = parser.parse_args()

    df = procesar_csv(args.csv, args.col_texto, args.col_fecha)
    generar_reporte(df)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f'\n💾 Resultados guardados en {args.output}')
