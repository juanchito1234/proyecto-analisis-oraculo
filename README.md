# Proyecto Oráculo · Grupo 1 (ADAG1)

Análisis y Diseño de Algoritmos · 2026-B

Búsqueda de una **configuración de prompt** (un índice de 0 a 7 para cada una de las ranuras `rol`, `estrategia`, `formato`, `estilo` y `cierre`, más la temperatura) que maximice la fracción de instancias que cumplen **todas** sus restricciones verificables, consultando un oráculo que ejecuta un modelo de lenguaje en GPU. El espacio tiene 8⁵ = 32 768 configuraciones con temperatura 0.0.

## Identificación

| Campo | Dato |
|---|---|
| Grupo / subproyecto | ADAG1 · Grupo 1 · Proyecto Oráculo |
| Integrantes | Felipe Gaviria Arango, Jesús David Milián Saza, Juan José Serna Pérez |
| Semana de entrega | 6 |
| Repositorio | https://github.com/juanchito1234/proyecto-analisis-oraculo.git |
| Rama o versión entregada | `main` |
| Informe técnico | `Informe_Proyecto_Oraculo_ADAG1.docx` |

## Configuración entregada

Está en `entrega.json` y proviene de **backtracking con poda**:

```json
{
 "grupo": "Grupo 1",
 "config": {"rol": 6, "estrategia": 2, "formato": 0, "estilo": 6, "cierre": 3, "temperatura": 0.0},
 "semana": 6
}
```

Es la **mejor configuración encontrada**, no un óptimo global probado: la búsqueda visitó 36 de 32 768 configuraciones (0,11 %).

## Algoritmos implementados

| Algoritmo | Notebook de búsqueda | Configuración (rol · estrategia · formato · estilo · cierre) | Búsqueda | Validación (n = 87) |
|---|---|---|---|---|
| Backtracking con poda por cota superior | `notebooks/OraculoBacktracking.ipynb` | 6 · 2 · 0 · 6 · 3 | 20,0 % (n = 100) | 23,0 % |
| Descenso de coordenadas | `notebooks/descenso_coordenadas_proyecto_oraculo(3).ipynb` | 5 · 1 · 4 · 3 · 7 | 12,0 % (n = 100) | 10,3 % |
| Algoritmo genético | `notebooks/OraculoGenetic.ipynb` | 4 · 3 · 7 · 5 · 0 | 16,0 % (n = 25) | 9,2 % |

Las precisiones de búsqueda no son comparables entre sí (cada algoritmo usó un lote y un punto de partida distintos); la validación sí. Con 87 instancias, diferencias de pocos puntos no son estadísticamente concluyentes. El análisis completo está en el informe.

## Estructura del repositorio

```
.
├── README.md
├── entrega.json                     # configuración oficial (backtracking)
├── entregas/
│   ├── entrega_backtracking_poda.json      # mejor configuración para backtracking con poda
│   ├── entrega_descenso_coordenadas.json   # mejor configuración para descenso de coordenadas
│   └── entrega_genetico.json               # mejor configuración para genético
├── oraculo.py                       # caja negra del curso (se consulta, no se abre)
├── ayudas.py                        # cargar modelo y datos, dividir, ver_prompt, entrega
├── datos_visibles.json              # 450 instancias: 100 búsqueda, 87 validación, 263 descartadas
└── notebooks/
    ├── evaluacion.ipynb             # evaluación unificada sobre los datos de prueba
    ├── backtracking_poda.ipynb      # búsqueda que justifica entrega.json
    ├── descenso_coordenadas.ipynb
    └── genetico.ipynb
```

## Entorno

- **Lenguaje:** Python 3.10 o superior (probado en Google Colab).
- **Hardware:** GPU NVIDIA T4 (Colab: *Entorno de ejecución ▸ Cambiar tipo de entorno de ejecución ▸ T4 GPU*). Sin GPU no se puede cargar el modelo.
- **Modelo:** `Qwen/Qwen3-1.7B` en fp16 (alias `qwen17b`), modo de pensamiento apagado, `max_new_tokens = 384`.
- **Dependencias:** `torch`, `transformers`, `accelerate`, `nltk`, `spacy` (con `en_core_web_sm`), `emoji`, `langdetect`, `immutabledict`, `pandas`, `scipy`, `matplotlib`, `tqdm`.
- **Verificadores:** repositorio `allenai/open-instruct` clonado en `/content/open-instruct` (lo exige `oraculo.py`). Para familias de restricciones nuevas, el notebook de evaluación intenta cargar también `allenai/IFBench`.

Los notebooks instalan lo que falta en su primera celda.

## Cómo ejecutar

**Evaluar la configuración entregada (no requiere editar nada):**

1. Clonar el repositorio en Colab con GPU T4 y abrir `notebooks/evaluacion.ipynb`.
2. Colocar el archivo de prueba en el repositorio (mismo formato que `datos_visibles.json`); el notebook lo detecta solo.
3. *Entorno de ejecución ▸ Ejecutar todo*.

El notebook valida `entrega.json`, evalúa esa configuración con `semilla = 1` y temperatura 0.0, compara las tres propuestas y guarda `resultados_evaluacion.json`. Si no encuentra datos de prueba, corre en **modo ensayo** con las instancias de validación visibles y lo declara; ese resultado no es la evaluación final.

**Reproducir una búsqueda:** abrir el notebook del algoritmo correspondiente en Colab con GPU y ejecutar las celdas en orden. Cada notebook guarda sus evaluaciones en la caché del oráculo, de modo que repetir una corrida no vuelve a gastar GPU.

**Inspeccionar un prompt sin generar texto:**

```python
from ayudas import cargar_datos, dividir, ver_prompt
busqueda, validacion = dividir(cargar_datos())
config = {"rol": 6, "estrategia": 2, "formato": 0, "estilo": 6, "cierre": 3, "temperatura": 0.0}
print(ver_prompt(config, busqueda[0]))
```

## Reproducibilidad

- Temperatura 0.0 (decodificación *greedy*) y `semilla = 1` en todas las evaluaciones: con la misma configuración e instancias el resultado es determinista.
- Semillas de los algoritmos: `random.seed(0)` en backtracking, descenso y genético.
- Datos: `datos_visibles.json` ya trae la partición (`busqueda`, `validacion`, `descartada`); `dividir` la lee sin sorteos.
- Verificación básica de corrección: `entrega.json` es validado por `evaluacion.ipynb` (un entero de 0 a 7 por ranura y temperatura en {0.0, 0.3, 0.7}), y `ver_prompt` permite revisar el texto exacto que recibe el modelo.

## Límites conocidos

- La búsqueda de backtracking quedó truncada por presupuesto (36 configuraciones); no hay garantía de optimalidad global.
- El genético buscó sobre 25 instancias, por lo que su precisión de búsqueda tiene mucha incertidumbre.
- La validación comparte familias de restricciones con la búsqueda; el test privado puede incluir familias nuevas.
- Solo se exploró la temperatura 0.0.
- `evaluacion.ipynb` se probó con un oráculo simulado; falta una corrida real en GPU.
- La validación experimental del crecimiento de T(n) está pendiente (sección 7 del informe).
- El manual técnico se entrega como documento aparte.

## Referencias

- Documento del proyecto semestral ADAG1 2026-2 y guía de primera entrega.
- README, `oraculo.py` y `ayudas.py` del repositorio del curso (DanielMelo404/Proyecto-Oraculo-AyD).
- Chang y Chen, *Naive Prompt Optimization: Rethinking the Need for Complex Prompt Search*, 2026 (citado en el README del curso).