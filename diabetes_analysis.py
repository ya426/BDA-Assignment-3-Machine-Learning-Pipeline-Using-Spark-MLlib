import argparse, csv, json, shutil, urllib.request
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.functions import vector_to_array

ROOT = Path(__file__).resolve().parent
URL = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
ZERO_MISSING = {'Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age'}

def clean(df):
    for c in FEATURES:
        v = F.col(c).cast('double')
        invalid = v.isNull() | F.isnan(v) | (v < 0) | (F.abs(v) == float('inf'))
        if c in ZERO_MISSING: invalid = invalid | (v == 0)
        df = df.withColumn(c, F.when(invalid, F.lit(None).cast('double')).otherwise(v))
    return df

def save_rows(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--ui', action='store_true'); p.add_argument('--data', type=Path, default=ROOT/'data'/'diabetes.csv'); args=p.parse_args()
    if not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, args.data)
    out=ROOT/'output'; out.mkdir(exist_ok=True)
    spark=SparkSession.builder.appName('BDA Assignment 3 Diabetes MLlib').master('local[2]').config('spark.sql.shuffle.partitions','4').getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    try:
        raw=spark.read.option('header',True).csv(str(args.data))
        if not set(FEATURES+['Outcome']).issubset(raw.columns): raise ValueError('Unexpected dataset columns')
        raw.printSchema(); raw.show(5)
        df=clean(raw).withColumn('label',F.col('Outcome').cast('double'))
        missing=[{'feature':c,'missing_or_invalid':df.filter(F.col(c).isNull()).count()} for c in FEATURES]
        save_rows(out/'missing_values.csv',missing,['feature','missing_or_invalid'])
        valid=df.filter(F.col('label').isin(0.0,1.0)).select(*FEATURES,'label').cache()
        total=raw.count(); valid_count=valid.count()
        train,test=valid.randomSplit([0.8,0.2],seed=42); train=train.cache(); test=test.cache()
        ntrain,ntest=train.count(),test.count(); print('Rows:',total,'Train:',ntrain,'Test:',ntest)
        imputed=[c+'_imputed' for c in FEATURES]
        def stages():
            return [Imputer(inputCols=FEATURES,outputCols=imputed,strategy='median'),VectorAssembler(inputCols=imputed,outputCol='unscaled'),StandardScaler(inputCol='unscaled',outputCol='features',withMean=True,withStd=True)]
        classifiers={'Logistic Regression':LogisticRegression(maxIter=100,regParam=0.01),'Decision Tree':DecisionTreeClassifier(maxDepth=5,seed=42),'Random Forest':RandomForestClassifier(numTrees=50,maxDepth=5,seed=42)}
        metrics=[]; models={}; conf=[]
        for name,clf in classifiers.items():
            spark.sparkContext.setJobGroup(name,name)
            model=Pipeline(stages=stages()+[clf]).fit(train); models[name]=model
            pred=model.transform(test).select('label','prediction')
            counts={(int(r.label),int(r.prediction)):r['count'] for r in pred.groupBy('label','prediction').count().collect()}
            tn,fp,fn,tp=[counts.get(k,0) for k in [(0,0),(0,1),(1,0),(1,1)]]
            div=lambda a,b: a/b if b else 0.0
            precision=div(tp,tp+fp); recall=div(tp,tp+fn)
            row={'model':name,'accuracy':div(tp+tn,ntest),'precision':precision,'recall':recall,'f1':div(2*precision*recall,precision+recall)}
            metrics.append(row); conf.append({'model':name,'tn':tn,'fp':fp,'fn':fn,'tp':tp}); print(row)
        best=max(metrics,key=lambda r:(r['f1'],r['accuracy']))['model']
        save_rows(out/'performance_comparison.csv',metrics,['model','accuracy','precision','recall','f1'])
        save_rows(out/'confusion_matrices.csv',conf,['model','tn','fp','fn','tp'])
        patients=spark.createDataFrame([( 'P001',2.,105.,72.,25.,90.,26.5,.25,28.),('P002',6.,165.,85.,35.,180.,36.2,.75,48.),('P003',1.,125.,70.,0.,0.,30.,.4,32.)],['patient_id']+FEATURES)
        predictions=models[best].transform(clean(patients)).select('patient_id','prediction',vector_to_array('probability')[1].alias('model_probability_positive')).orderBy('patient_id')
        predictions.show(truncate=False); save_rows(out/'new_patient_predictions.csv',[r.asDict() for r in predictions.collect()],['patient_id','prediction','model_probability_positive'])
        models[best].write().overwrite().save(str(out/'best_model'))
        plan=models[best].transform(test)._jdf.queryExecution().toString(); (out/'execution_plan.txt').write_text(plan,encoding='utf-8')
        summary={'dataset_rows':total,'valid_label_rows':valid_count,'training_rows':ntrain,'testing_rows':ntest,'best_model_by_test_f1':best,'metrics':metrics,'interpretation':'Positive class metrics use label 1. Best model is selected on the test set for coursework comparison; independent validation is required for unbiased performance estimation. Not for clinical use.'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2)); print('Spark UI:',spark.sparkContext.uiWebUrl)
        if args.ui: input('Capture Spark UI screenshots, then press Enter to finish: ')
    finally:
        spark.stop()
    shutil.make_archive(str(ROOT/'BDA-Assignment-3-Actual-Output'),'zip',out)

if __name__=='__main__': main()
