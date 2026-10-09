# Matriz de cumplimiento de la consigna · Proyecto 1

Esta revisión contrasta el proyecto con **todas las obligaciones y preguntas aplicables** de `enunciado_TFM_MDATA2-.pdf`: condiciones comunes de las páginas 1–4 y Proyecto 1 de las páginas 4–6. El PDF `Fuentes-de-datos-TFM.pdf` ofrece fuentes posibles, incluida Kaggle; no exige utilizar todas ellas.

La explicación principal conserva un relato continuo en [README.md](../README.md). Esta matriz permite localizar cada respuesta y su evidencia; no sustituye ese relato ni atribuye una calificación académica.

**Estado:** requisitos técnicos y documentales comprobados el 9 de octubre de 2026. Código, datos, resultados y notebook publicados y verificados en el repositorio público [diego1615/tfm-bank-marketing](https://github.com/diego1615/tfm-bank-marketing); los archivos remotos coinciden con las versiones revisadas. Las comprobaciones automáticas de GitHub también superaron las pruebas. Compartir el enlace en el campus, dentro del plazo institucional, es un paso posterior del alumno.

## Obligaciones comunes y entregables

| ID | Requisito de la consigna | Cumplimiento y evidencia publicada |
|---|---|---|
| C01 | Un repositorio propio para este proyecto en el GitHub personal | Repositorio público independiente indicado arriba, con código, resultados, datos, documentación y notebook publicados |
| C02 | Código fuente completo | [Módulos](../src/bank_marketing/), [scripts](../scripts/), [cuaderno](../notebooks/01_bank_marketing_tfm.ipynb), [Makefile](../Makefile) y [dependencias](../requirements.txt) |
| C03 | Resultados obtenidos, outputs, métricas y ejemplos | [results.json](../reports/results.json), [tablas calculadas](../reports/tables/) y cuaderno con outputs reales |
| C04 | Visualizaciones realizadas | Siete PNG en [reports/figures](../reports/figures/); los gráficos principales también aparecen en el README |
| C05 | Markdown estructurado y coherente que narre problema, datos, modelos, resultados y conclusiones | [README](../README.md): «Problema y criterios de éxito», «Datos y preparación», «Diseño de validación», «Resultados reales» y «Dificultades resueltas y conclusiones» |
| C06 | Describir el proceso y justificar decisiones técnicas | README: «Problema y criterios de éxito», «Datos y preparación» y «Diseño de validación»; ampliación en [INFORME](../reports/INFORME.md), secciones 1–5 |
| C07 | Integrar las respuestas naturalmente, sin un cuestionario aislado | Todas las preguntas se desarrollan en los apartados narrativos señalados en la matriz siguiente |
| C08 | Ejemplos, métricas y visualizaciones relevantes | Cuaderno ejecutado; tablas de selección, prueba y escenarios; figuras de datos, discriminación, calibración, negocio y explicación |
| C09 | Dificultades encontradas y cómo se resolvieron | README: «Dificultades resueltas y conclusiones»; INFORME, sección 9 |
| C10 | Esquema visual o textual del sistema completo | README: «Esquema de la solución»: cinco módulos, sus entradas, salidas y código responsable |
| E01 | Datos utilizados o enlace público | CSV real comprimido [bank-additional-full.csv.gz](../data/raw/bank-additional-full.csv.gz), [procedencia y licencia](../data/DATA_SOURCES.md), enlaces Kaggle/UCI y verificación SHA-256 |
| E02 | Reproducir el flujo completo | `make reproduce`, `make test`, `make notebook`; pipeline y cuaderno ejecutado disponibles |
| E03 | Métricas técnicas y de negocio documentadas | README: «Resultados reales» y «Decisiones de negocio y sensibilidad»; CSV/JSON calculados |
| E04 | Explicar conceptualmente la producción, sin obligación de desplegar | README: «Explicabilidad, riesgos y producción conceptual»; INFORME, sección 10. Se identifica expresamente como propuesta |

## Las seis fases obligatorias del proyecto

| Fase | Cumplimiento y evidencia |
|---|---|
| 1. Objetivo empresarial y preguntas del modelo | Priorización de contactos comerciales bajo presupuesto; suscripción de depósitos observada y capacidad del 10%, en «Problema y criterios de éxito» |
| 2. Exploración, limpieza y preparación | Distribución objetivo, edad, cambio temporal, ausencias codificadas y duplicados; sentinel `pdays=999`; imputación/codificación/escalado dentro del pipeline. [data_quality.json](../reports/data_quality.json), [core.py](../src/bank_marketing/core.py) y figuras 01–02 |
| 3. Entrenar, validar y comparar modelos | Dummy y dos configuraciones por cada una de tres familias; partición cronológica 60/10/10/20, selección, reajuste, calibración y diagnóstico de origen móvil. [model_selection.csv](../reports/tables/model_selection.csv), [forward_validation.csv](../reports/tables/forward_validation.csv), [test_metrics.csv](../reports/tables/test_metrics.csv) |
| 4. Explicabilidad | Importancia por permutación y dependencia parcial; interpretación y limitaciones en README. [permutation_importance.csv](../reports/tables/permutation_importance.csv), figuras 06–07 |
| 5. Traducción a valor de negocio real o potencial | Precision/lift, captación observada bajo capacidad y escenarios contables con supuestos explícitos; comparación al azar bajo igual presupuesto y sensibilidad. [business_scenarios.csv](../reports/tables/business_scenarios.csv), [cost_benefit_sensitivity.csv](../reports/tables/cost_benefit_sensitivity.csv) |
| 6. Producción conceptual | Batch diario, alternativas API/dashboard, tecnologías, seguimiento, reentrenamiento, cambios de features, registro y reversión; README e INFORME, sección 10 |

El problema pertenece a la categoría admitida de **clasificación**. No se exige realizar adicionalmente una regresión o un modelo de series temporales: son alternativas del enunciado.

## Todas las preguntas del Proyecto 1

Las preguntas se transcriben para indexar su respuesta dentro del documento principal; el README desarrolla las respuestas como una narrativa.

| ID | Pregunta de la página 6 | Ubicación de la respuesta en el README y evidencia |
|---|---|---|
| Q01 | ¿Cuál es el objetivo de negocio que intentas resolver? ¿Qué impacto tendría si el modelo funciona correctamente? | «Problema y criterios de éxito» y «Decisiones de negocio y sensibilidad»: ordenar contactos bajo capacidad; impacto potencial de priorización, sin afirmar ROI causal |
| Q02 | ¿Qué tipo de variable predices (numérica, categórica, temporal) y por qué elegiste ese enfoque? | «Problema y criterios de éxito»: `y=yes/no`, clasificación binaria supervisada y justificación por etiqueta observada; no existe monto continuo |
| Q03 | ¿Qué fuentes de datos utilizaste? ¿Cómo limpiaste y transformaste las variables? | «Datos y preparación»: Kaggle/UCI, 41.188 filas, licencia/hash, `unknown`, duplicados, sentinel, whitelist, imputación, escalado y codificación; `core.py` |
| Q04 | ¿Qué modelos probaste y cómo justificas su elección? | «Diseño de validación»: logística regularizada y explicable, random forest para no linealidades/interacciones, boosting secuencial regularizado y dummy; comparación bajo los mismos datos |
| Q05 | ¿Qué métricas técnicas usaste para evaluar el rendimiento y por qué son adecuadas? | «Problema y criterios de éxito»: AP y tasa base, ROC-AUC, Brier, precision, recall y F1 con su razón; «Resultados reales» da los valores |
| Q06 | ¿Qué técnicas de explicabilidad aplicaste? ¿Qué información te aportaron sobre el modelo? | «Explicabilidad, riesgos y producción conceptual»: permutación y PDP, asociaciones de historial, límites por correlación y combinaciones sintéticas; figuras 06–07 |
| Q07 | ¿Qué variables influyen más en las predicciones y cómo interpretas su efecto? | «Explicabilidad, riesgos y producción conceptual»: `pdays`, `poutcome`, `job`, educación y mora con caídas de AP; importancias negativas, dirección limitada de PDP y ausencia de interpretación causal |
| Q08 | ¿Qué métricas de negocio definiste? | «Decisiones de negocio y sensibilidad»: presupuesto, contactos, aceptaciones observadas/esperadas, precision/lift y proxy `beneficio × aceptaciones − coste × contactos`; beneficios/costes hipotéticos |
| Q09 | ¿Qué resultados obtuviste al traducir el modelo a indicadores empresariales? | «Decisiones de negocio y sensibilidad»: 824 contactos, 473 aceptaciones observadas, proxy 43.180 frente a 21.286 esperada al azar; «todos» se distingue por diferente capacidad |
| Q10 | ¿Cómo llevarías el modelo a producción, de forma conceptual? | «Explicabilidad, riesgos y producción conceptual»: arquitectura, requisitos de datos recientes y cinco dimensiones detalladas abajo |
| Q11 | ¿Qué conclusiones extraes y cómo podría mejorarse el sistema en una versión futura? | «Dificultades resueltas y conclusiones»: lift favorable e inestabilidad temporal; IDs, disponibilidad, márgenes, consentimiento y evaluación prospectiva antes de ampliar algoritmos |

### Todas las subpreguntas sobre producción

| ID | Subpregunta | Respuesta explícita en «Explicabilidad, riesgos y producción conceptual» |
|---|---|---|
| Q10.1 | ¿Sería un modelo en tiempo real o por lotes? ¿Qué parte del flujo se ejecutaría? | Batch diario: snapshot CRM, validación, puntuación y cola según capacidad; API autenticada y dashboard como alternativas |
| Q10.2 | ¿Cada cuánto lo reentrenarías y con qué criterio? | Trimestral y revisión anticipada ante cambios de campaña, producto, esquema o política; criterios iniciales propuestos de PSI, lift y Brier |
| Q10.3 | ¿Cómo monitorizarías su rendimiento y detectarías data drift? | Datos/distribuciones/categorías/scores diariamente; prevalencia, AP con tasa base, lift, ROC-AUC, Brier/calibración y grupos mensualmente con etiquetas maduras |
| Q10.4 | Si descubres una nueva feature o mejoras el modelo, ¿cómo lo actualizarías o versionarías? | Fuentes y timestamps versionados, nueva versión del pipeline, pruebas, validación por tiempo/cliente, comparación con vigente, shadow mode y reversión |
| Q10.5 | ¿Qué herramientas o estructura utilizarías para desplegarlo en un entorno real? | Docker para entorno, Airflow para ingesta/calidad/inferencia, MLflow para parámetros/métricas/hashes/versiones, FastAPI y dashboard opcionales |

## Correspondencia con las cinco categorías de evaluación

| Categoría | Peso indicado por la consigna | Evidencia para su evaluación |
|---|---:|---|
| Investigación y justificación técnica | 3 puntos | Definición precontacto, disponibilidad de variables, comparación, protocolo temporal, razones de métricas y limitaciones |
| Resultados y visualizaciones | 2,5 puntos | Resultados calculados, siete figuras, tablas técnicas/empresariales y ejemplos ejecutados |
| Calidad del código y estructura | 2 puntos | Módulos separados, dependencias, comandos reproducibles y siete pruebas relevantes |
| Innovación, creatividad y originalidad | 1,5 puntos | Prevención de fuga mediante whitelist, control temporal, calibración, escenarios con igual capacidad y separación de valor contable/causal |
| Documentación y storytelling | 1 punto | README narrativo completo, informe ampliado, cuaderno con explicación y matriz de localización |

El Proyecto 1 representa el **60%** de la nota final y el Proyecto 2 el **40%**. Las evidencias permiten evaluar la rúbrica; no garantizan una nota ni reemplazan el juicio del evaluador.

## Verificaciones realizadas

- **Siete pruebas superadas** con Python 3.13.9: exclusión de futuro/objetivo, sentinel, particiones completas sin solapamiento, presupuesto/empates, costes, preprocesamiento ajustado con entrenamiento e integridad del origen.
- Cuaderno con **7/7 celdas de código ejecutadas y cero outputs de error**. El método de ejecución queda documentado en sus metadatos.
- Modelo local recargado: sus scores reproducen `test_predictions.csv` con diferencia máxima de aproximadamente `1,11 × 10⁻¹⁶`; las métricas de `results.json` coinciden al recalcularlas.
- Filas de prueba verificadas como el último 20% del orden original; las particiones son exhaustivas y disjuntas. Los duplicados exactos de todas las columnas no cruzan particiones.
- Gzip y CSV comparados byte a byte, con SHA-256 igual al documentado. No se necesita red para restaurar el dato incluido.
- Inferencia CLI probada con 17 registros: 17 scores dentro de `[0,1]` y dos selecciones bajo presupuesto del 10%, conforme a su redondeo.
- Figuras de evaluación y escenarios inspeccionadas; el eje monetario se expresa en miles y es legible.

No se afirma despliegue real, retorno causal, independencia por cliente o estabilidad que los datos no permiten demostrar. Los límites están incluidos en la documentación principal.
