# Bank Marketing: procedencia verificada

Dataset seleccionado en Kaggle: [bank-additional-full.csv](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv), publicado por **Sahista_Patel** (`sahistapatel96`).

La tarjeta de Kaggle remite a la fuente primaria UCI y describe 41.188 ejemplos, 20 predictores y un objetivo binario `y`. El archivo conserva orden cronológico de mayo de 2008 a noviembre de 2010. La página principal de UCI también describe la versión antigua de 45.211 filas; esa cifra no corresponde al archivo adicional seleccionado.

Fuente primaria: [UCI Bank Marketing, dataset 222](https://archive.ics.uci.edu/dataset/222/bank+marketing). DOI: https://doi.org/10.24432/C5K306.

Autores originales: S. Moro, P. Rita y P. Cortez. La fuente primaria establece licencia **Creative Commons Attribution 4.0 International (CC BY 4.0)**. La tarjeta Kaggle declara «Other (specified in description)»; para redistribución se utiliza la autorización CC BY 4.0 de UCI con atribución completa.

Referencia del dataset: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.

Referencia científica: Moro, S., Cortez, P. y Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decision Support Systems*, 62, 22–31.

Fecha de consulta: 9 de octubre de 2026.

Descarga Kaggle: https://www.kaggle.com/api/v1/datasets/download/sahistapatel96/bankadditionalfullcsv?datasetVersionNumber=1

Descarga primaria alternativa: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip (contiene el ZIP interno bank-additional.zip).

## Protocolo recomendado

- Mantener el orden original, guardar índice de fila y utilizar división temporal por orden de observación. No reconstruir fechas exactas: no se publican.
- Excluir `duration`: solo se conoce después de llamar y UCI advierte que produce fuga para predicción operativa.
- Excluir `campaign` de un modelo estrictamente previo a la campaña: incluye el último contacto y puede representar el total final.
- `contact`, `month` y `day_of_week` son plausibles solo si se conoce el calendario/canal de la llamada programada. Documentar esa hipótesis o excluirlos en el escenario conservador.
- Los indicadores macroeconómicos se registraron retrospectivamente; para uso operativo se necesita comprobar calendario de publicación y rezagos. Incluir escenario sin estos indicadores.
- `pdays=999` significa sin contacto previo, no 999 días transcurridos. Derivar indicador `previously_contacted` y tratar los días faltantes dentro del pipeline.
- `unknown` es ausencia informativa, no una categoría social necesariamente definida.
- Separar entrenamiento, validación y test. Decidir modelos/umbrales en validación; usar test una vez. Evaluar PR-AUC, ROC-AUC, Brier/calibración, precision/recall, lift y precision@k.
- `y` es aceptación observada bajo campañas históricas. No identifica efecto causal de contactar, uplift ni rentabilidad incremental. Costes/beneficios hipotéticos requieren análisis de sensibilidad y no son retornos observados.
- No hay identificador de cliente: no se puede garantizar que contactos del mismo cliente no aparezcan en particiones distintas. Explicitar este límite.
