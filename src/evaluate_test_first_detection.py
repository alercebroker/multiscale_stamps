import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import pickle

from tqdm import tqdm
from typing import List

from data_loader import ra_dec_to_cartesian
from data_loader import normalize_stamps
from data_loader import build_multiscale_ndarray, crop_stamps_ndarray, low_res_stamps_ndarray


def classify_dataset(
        scenario: str,
        dataset: pd.DataFrame,
        stamp_classifier: tf.keras.Model,
        label_list: List[str]) -> pd.DataFrame:

    batch_size = 32
    n_batches = int(np.ceil(len(dataset) / batch_size))

    prediction_dfs = []
    for batch_idx in tqdm(range(n_batches)):
        start = batch_idx*batch_size
        end = (batch_idx + 1) * batch_size
        batch = dataset.iloc[start:end]
        ra = batch['ra'].values
        dec = batch['dec'].values
        science = np.stack(batch['science'], axis=0)
        reference = np.stack(batch['reference'], axis=0)
        diff = np.stack(batch['diff'], axis=0)

        position = ra_dec_to_cartesian(ra, dec)
        stamps = np.stack([science, reference, diff], axis=-1)
        norm_stamps = normalize_stamps(stamps)
        if scenario == 'cropped':
            stamps = crop_stamps_ndarray(norm_stamps, 16)
        elif scenario == 'low_res':
            stamps = low_res_stamps_ndarray(norm_stamps, 4)
        elif scenario == 'multiscale':
            stamps = build_multiscale_ndarray(norm_stamps)
        elif scenario == 'full':
            stamps = norm_stamps
        else:
            raise NotImplementedError()

        predicted_probs = tf.nn.softmax(
            stamp_classifier((stamps, position)))

        predicted_df = pd.DataFrame(
            data=predicted_probs,
            columns=label_list,
            index=batch.index
        )
        prediction_dfs.append(predicted_df)
    predicted_df = pd.concat(prediction_dfs, axis=0)
    return predicted_df


if __name__ == '__main__':
    scenario = 'multiscale'

    LABELS = [
        'agn',
        'asteroid',
        'bogus',
        'satellite',
        'sn',
        'vs'
    ]

    dataset = pd.read_pickle('test_first_stamps_dataset.pkl')

    scenario_results = {}
    for scenario in ['cropped', 'low_res', 'multiscale', 'full']:
        results = []
        for model_idx in range(5):
            print(f'{scenario} {model_idx}')
            stamp_classifier = tf.keras.models.load_model(f'saved_models/{scenario}_run_{model_idx}/')
            predicted_probs = classify_dataset(scenario, dataset, stamp_classifier, LABELS)
            predicted_labels = predicted_probs.idxmax(axis=1)

            run_results = {}
            precision, recall, f1, _ = precision_recall_fscore_support(
                dataset['class'].values, predicted_labels.values, average='macro')

            run_results['precision'] = precision
            run_results['recall'] = recall
            run_results['f1'] = f1

            val_cm = confusion_matrix(
                dataset['class'].values,
                predicted_labels.values,
                labels=LABELS)

            run_results['cm'] = val_cm

            results.append(run_results)
        scenario_results[scenario] = results

    print(scenario_results)
    with open('test_first_detection_results.pkl', 'wb') as f:
        pickle.dump(scenario_results, f)

    metrics = [
        'f1', 'precision', 'recall'
    ]

    for scenario, results in scenario_results.items():
        print(scenario)
        for metric in metrics:
            l = [res[metric] for res in results]
            print(f'{metric.ljust(15)} {np.mean(l) * 100:.2f} +/- {np.std(l) * 100:.2f}')
        print('\n')
