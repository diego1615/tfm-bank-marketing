"""Create and execute a compact Spanish notebook from the reproducible module."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/tfm-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/tfm-jupyter-runtime')
os.environ.setdefault('IPYTHONDIR','/tmp/tfm-ipython')
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
def md(s): return nbformat.v4.new_markdown_cell(s)
def code(s): return nbformat.v4.new_code_cell(s)

nb=nbformat.v4.new_notebook(cells=[
md('''# Priorización de campañas bancarias antes del contacto\n**Proyecto 1 — TFM MDATA2** · Diego Fernández · Octubre de 2026\n\nPregunta: ¿puede priorizarse la suscripción observada de depósitos a plazo con información disponible antes de llamar? Se utiliza Bank Marketing de UCI, descargado de Kaggle (41.188 registros reales). El objetivo es asociación predictiva, no uplift causal.\n\nEste cuaderno ejecuta el pipeline completo y presenta sus resultados; el detalle de implementación está en `src/bank_marketing/`. No requiere conexión a internet al incluir el dataset comprimido con atribución CC BY 4.0.'''),
code('''from pathlib import Path\nimport sys, os, json, subprocess\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nROOT = Path.cwd()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\nassert (ROOT / 'src/bank_marketing').exists(), 'Abrir desde la raíz del repositorio o notebooks/'\nsys.path.insert(0, str(ROOT / 'src'))\nos.environ['MPLCONFIGDIR'] = '/tmp/tfm-matplotlib'\nos.environ['OMP_NUM_THREADS'] = '2'\nsubprocess.run([sys.executable, str(ROOT/'scripts/download_data.py')], check=True, capture_output=True, text=True)\ndata = pd.read_csv(ROOT/'data/raw/bank-additional-full.csv', sep=';')\ndisplay(data.head(5))\nprint(f'{data.shape[0]:,} registros, {data.shape[1]} columnas; suscripción {(data.y=="yes").mean():.2%}')'''),
md('''## Disponibilidad de variables y prevención de fuga\n`duration` se conoce después de llamar. `campaign` incluye el último contacto. Canal, fecha de contacto e indicadores macroeconómicos se excluyen del modelo conservador por falta de información verificable sobre su disponibilidad prospectiva. Se conservan edad, características del cliente e historia de campañas previas. `pdays=999` significa ausencia de contacto anterior. Los valores `unknown` se mantienen explícitos.\n\nCada imputación, escalado y codificación se ajusta dentro del pipeline únicamente con entrenamiento. Las 12 filas idénticas se mantienen: sin identificador de cliente no se puede inferir que sean errores.'''),
code('''from bank_marketing.core import BankFeatures, RAW_FEATURES, FORBIDDEN, chronological_splits\nfeatures = BankFeatures().fit_transform(data.drop(columns='y'))\ndisplay(features.head())\nassert not set(FORBIDDEN) & set(features.columns)\nsplit = chronological_splits(len(data))\npartition_table = pd.DataFrame([{'bloque':k, 'n':len(ix), 'primer_registro':ix[0], 'ultimo_registro':ix[-1], 'suscripcion':(data.iloc[ix].y=='yes').mean()} for k,ix in split.items()])\ndisplay(partition_table)'''),
md('''## Entrenamiento, comparación y evaluación\nSe comparan regresión logística, random forest y hist gradient boosting (dos configuraciones por familia), además de un dummy prior. La selección usa average precision en el siguiente 10% cronológico. La regresión ganadora se reajusta con el primer 70%; una calibración sigmoide usa el siguiente 10%. Ese bloque también decide un umbral contable hipotético. El último 20% se reserva para evaluación.\n\nSe ejecuta además validación de origen móvil en tres cortes del primer 70%, sin usar la prueba. El orden es cronológico según UCI; no se dispone de fechas exactas ni ID de cliente. Los cortes por número de filas no representan intervalos temporales de igual duración.'''),
code('''env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='2')\nrun = subprocess.run([sys.executable, '-m', 'bank_marketing.run', '--root', str(ROOT)], cwd=ROOT, env=env, capture_output=True, text=True, check=True)\nprint(run.stdout)\nresults = json.loads((ROOT/'reports/results.json').read_text())\ndisplay(pd.read_csv(ROOT/'reports/tables/model_selection.csv')[['model','average_precision','roc_auc','brier','lift_at_10pct']])\ndisplay(pd.read_csv(ROOT/'reports/tables/forward_validation.csv')[['fold','prevalence','average_precision','roc_auc','lift_at_10pct']])'''),
code('''display(Image(filename=str(ROOT/'reports/figures/01_data_overview.png')))\ndisplay(pd.read_csv(ROOT/'reports/tables/test_metrics.csv')[['model','training_scope','average_precision','roc_auc','brier','precision_at_10pct','lift_at_10pct']])\ndisplay(Image(filename=str(ROOT/'reports/figures/03_model_evaluation.png')))'''),
md('''## Interpretación crítica\nLa prevalencia pasa de 4,81% en entrenamiento a 30,83% en prueba. El ranking del modelo final concentra un 57,40% de aceptaciones observadas en el 10% priorizado, frente a 30,83% esperado al azar (lift 1,86). Sin embargo, la validación móvil no demuestra estabilidad: dos de tres ventanas tienen lift inferior a uno.\n\nLa calibración reduce Brier de 0,2553 a 0,2305, pero el cambio temporal persiste. Una constante ajustada con la prevalencia del propio test alcanzaría Brier 0,2133; es un diagnóstico retrospectivo que no puede conocerse al desplegar. No se afirma que el modelo esté listo para producción. El dummy, al empatar todos sus scores, muestra expectativas de selección aleatoria en métricas top-k.'''),
code('''display(pd.read_csv(ROOT/'reports/tables/bootstrap_intervals.csv'))\ndisplay(pd.read_csv(ROOT/'reports/tables/business_scenarios.csv'))\ndisplay(Image(filename=str(ROOT/'reports/figures/04_policy_scenarios.png')))'''),
md('''## Decisión de negocio y sensibilidad\nSe usa un beneficio supuesto de 100 unidades por aceptación observada y coste de 5 por contacto. Con presupuesto de 824 contactos, el top 10% obtiene una proxy contable de 43.180, frente a 21.286 al azar. Son escenarios, no ganancias incrementales identificadas. Sin límite de capacidad, contactar a todos obtiene una proxy mayor (212.810), porque los supuestos hacen rentables muchas aceptaciones. La política depende de capacidad y costes; ranking y umbral responden a preguntas distintas.\n\nLos intervalos bootstrap son condicionales y descriptivos: remuestrear filas no resuelve la dependencia temporal ni contactos repetidos de clientes.'''),
code('''display(pd.read_csv(ROOT/'reports/tables/cost_benefit_sensitivity.csv').head(8))\ndisplay(pd.read_csv(ROOT/'reports/tables/permutation_importance.csv'))\ndisplay(pd.read_csv(ROOT/'reports/tables/age_group_audit.csv'))'''),
md('''## Explicabilidad y uso responsable\nLa importancia por permutación mide contribución predictiva y no causalidad. Las curvas de dependencia parcial pueden crear combinaciones que no representan clientes reales. El uso de edad, ocupación y educación puede producir diferencias de selección; la auditoría por grupos de edad es exploratoria y no certifica equidad. El dataset no tiene sexo, etnia, consentimiento, montos ni margen de los depósitos.\n\nAntes de cualquier implementación se necesita una base reciente con IDs, disponibilidad temporal auditada y consentimiento, una política de contactos, evaluación temporal y por clientes, monitorización y prueba prospectiva de decisiones. El despliegue conceptual y los criterios se desarrollan en `reports/INFORME.md`.'''),
code('''# Ejemplo reproducible de inferencia local; el modelo se genera al entrenar.\nimport joblib\nmodel = joblib.load(ROOT/'models/bank_marketing.joblib')\nexample = data.iloc[-5:][RAW_FEATURES].copy()\nexample['score_suscripcion'] = model.predict_proba(example)[:,1]\ndisplay(example)\ntests = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, env=env, capture_output=True, text=True, check=True)\nprint(tests.stdout)'''),
md('''## Fuentes y autoría\n- Kaggle: https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv (Sahista_Patel).\n- UCI, fuente primaria y licencia CC BY 4.0: https://doi.org/10.24432/C5K306.\n- Moro, Cortez y Rita (2014), *A Data-Driven Approach to Predict the Success of Bank Telemarketing*, Decision Support Systems 62, 22–31.\n- Documentación oficial scikit-learn: pipelines, calibración, average precision y TimeSeriesSplit.\n\nAutor: Diego Fernández. Elaboración y programación asistidas por inteligencia artificial; resultados calculados con datos reales, código reproducible y pruebas automáticas. La interpretación y revisión académica final corresponden al autor.''')
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13'}})
path=ROOT/'notebooks/01_bank_marketing_tfm.ipynb'
path.parent.mkdir(parents=True,exist_ok=True)
try:
    NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nb.metadata['execution_strategy']='Jupyter kernel'
except PermissionError:
    # A restricted local sandbox may prohibit the TCP ports required by Jupyter.
    # All cells here are plain Python. Execute their exact source sequentially,
    # capturing real stdout/stderr and rich displays; never fabricate outputs.
    import contextlib, io, traceback, IPython.display as ipdisplay
    namespace={'__name__':'__main__'}
    original_display=ipdisplay.display
    old_cwd=Path.cwd(); os.chdir(ROOT)
    try:
        for count,cell in enumerate([c for c in nb.cells if c.cell_type=='code'],start=1):
            outputs=[]; stdout=io.StringIO(); stderr=io.StringIO()
            def capture_display(*objects,**kwargs):
                for obj in objects:
                    data={'text/plain':repr(obj)}; metadata={}
                    if hasattr(obj,'_repr_mimebundle_'):
                        bundle=obj._repr_mimebundle_()
                        if isinstance(bundle,tuple): data.update(bundle[0]); metadata=bundle[1]
                        elif isinstance(bundle,dict): data.update(bundle)
                    if hasattr(obj,'_repr_html_'):
                        html=obj._repr_html_()
                        if html: data['text/html']=html
                    outputs.append(nbformat.v4.new_output('display_data',data=data,metadata=metadata))
            ipdisplay.display=capture_display
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                exec(compile(cell.source,f'{path.name}:cell-{count}','exec'),namespace)
            if stdout.getvalue(): outputs.append(nbformat.v4.new_output('stream',name='stdout',text=stdout.getvalue()))
            if stderr.getvalue(): outputs.append(nbformat.v4.new_output('stream',name='stderr',text=stderr.getvalue()))
            cell.outputs=outputs; cell.execution_count=count
        nb.metadata['execution_strategy']='Sequential execution of exact pure-Python cells in process; sandbox disallows Jupyter TCP sockets'
    finally:
        ipdisplay.display=original_display; os.chdir(old_cwd)

nbformat.write(nb,path)
print(f'Executed notebook: {path}')
