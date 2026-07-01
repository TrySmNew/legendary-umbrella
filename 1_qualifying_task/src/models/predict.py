import numpy as np
import pandas as pd


def make_submission(model, df_test, feature_fn, output_path='submission_single.csv'):
    ids = df_test['id']
    X_submit = feature_fn(df_test.drop(columns=['id']))
    predictions = model.predict_proba(X_submit)[:, 1]

    submission = pd.DataFrame({'id': ids, 'retention': predictions})
    submission.to_csv(output_path, index=False)

    print(f"Сохранено: {output_path} ({len(submission)} строк)")
    print(submission.head())

    return submission


def make_submission_ensemble(models, df_test, feature_fn, output_path='submission_ensemble.csv'):
    ids = df_test['id']
    X_submit = feature_fn(df_test.drop(columns=['id']))

    proba = np.mean([m.predict_proba(X_submit)[:, 1] for m in models], axis=0)

    submission = pd.DataFrame({'id': ids, 'retention': proba})
    submission.to_csv(output_path, index=False)

    print(f"Сохранено: {output_path} ({len(submission)} строк)")
    print(submission.head())

    return submission