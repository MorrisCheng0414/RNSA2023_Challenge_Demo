import pandas as pd
import sklearn
import numpy as np
import os
import altair as alt
import streamlit as st
from glob import glob

class ParticipantVisibleError(Exception):
    pass

def normalize_probabilities_to_one(df: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    # Normalize the sum of each row's probabilities to 100%.
    # 0.75, 0.75 => 0.5, 0.5
    # 0.1, 0.1 => 0.5, 0.5
    row_totals = df[group_columns].sum(axis=1)
    if row_totals.min() == 0:
        raise ParticipantVisibleError('All rows must contain at least one non-zero prediction')
    for col in group_columns:
        df[col] /= row_totals
    return df

def score(solution_:pd.DataFrame, submission_:pd.DataFrame, row_id_column_name:str, reduction:str ='mean'):
    '''
    Pseudocode:
    1. For every label group (liver, bowel, etc):
        - Normalize the sum of each row's probabilities to 100%.
        - Calculate the sample weighted log loss.
    2. Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    3. Calculate the sample weighted log loss for the new label group
    4. Return the average of all of the label group log losses as the final score.
    '''
    solution = solution_.copy()
    submission = submission_.copy()
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # Run basic QC checks on the inputs
    if not pd.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission.values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if solution.min().min() < 0:
        raise ParticipantVisibleError('All labels must be at least zero')
    if submission.min().min() < 0:
        raise ParticipantVisibleError('All predictions must be at least zero')

    # Calculate the label group log losses
    binary_targets = ['bowel', 'extravasation']
    triple_level_targets = ['kidney', 'liver', 'spleen']
    all_target_categories = binary_targets + triple_level_targets

    label_group_losses = []
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        solution = normalize_probabilities_to_one(solution, col_group)

        for col in col_group:
            if col not in submission.columns:
                raise ParticipantVisibleError(f'Missing submission column {col}')
        submission = normalize_probabilities_to_one(submission, col_group)
        label_group_losses.append(
            sklearn.metrics.log_loss(
                y_true=solution[col_group].values,
                y_pred=submission[col_group].values,
                sample_weight=solution[f'{category}_weight'].values
            )
        )

    # # Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    # healthy_cols = [x + '_healthy' for x in all_target_categories]
    # any_injury_labels = (1 - solution[healthy_cols]).max(axis=1)
    # any_injury_predictions = (1 - submission[healthy_cols]).max(axis=1)
    # any_injury_loss = sklearn.metrics.log_loss(
    #     y_true=any_injury_labels.values,
    #     y_pred=any_injury_predictions.values,
    #     sample_weight=solution['any_injury_weight'].values
    # )

    # label_group_losses.append(any_injury_loss)
    if reduction == 'mean':
        return np.mean(label_group_losses)
    else:
        return label_group_losses

# Assign the appropriate weights to each category
def create_training_solution(y_train):
    sol_train = y_train.copy()

    # bowel healthy|injury sample weight = 1|2
    sol_train['bowel_weight'] = np.where(sol_train['bowel_injury'] == 1, 2, 1)

    # extravasation healthy/injury sample weight = 1|6
    sol_train['extravasation_weight'] = np.where(sol_train['extravasation_injury'] == 1, 6, 1)

    # kidney healthy|low|high sample weight = 1|2|4
    sol_train['kidney_weight'] = np.where(sol_train['kidney_low'] == 1, 2, np.where(sol_train['kidney_high'] == 1, 4, 1))

    # liver healthy|low|high sample weight = 1|2|4
    sol_train['liver_weight'] = np.where(sol_train['liver_low'] == 1, 2, np.where(sol_train['liver_high'] == 1, 4, 1))

    # spleen healthy|low|high sample weight = 1|2|4
    sol_train['spleen_weight'] = np.where(sol_train['spleen_low'] == 1, 2, np.where(sol_train['spleen_high'] == 1, 4, 1))

    # any healthy|injury sample weight = 1|6
    sol_train['any_injury_weight'] = np.where(sol_train['any_injury'] == 1, 6, 1)
    return sol_train

def plot_result(pred_df, true_df):
    pred_long_df = pred_df.melt(var_name = "Class", value_name = "Probability")
    pred_long_df["Metric"] = "Prediction"

    true_long_df = true_df.melt(var_name = "Class", value_name = "Probability")
    true_long_df["Metric"] = "Ground Truth"

    concat_df = pd.concat([pred_long_df, true_long_df])

    chart = alt.Chart(concat_df).mark_bar().encode(
        y = alt.Y("Class", axis = None),
        x = alt.X("Probability", axis = None),
        color = alt.Color("Metric", 
                          title = "Metric", 
                          scale = alt.Scale(domain = ["Prediction", "Ground Truth"]),
                          range = ['#54A24B', '#29B5E8']),
        row = alt.Row("Metric", title = '', header = alt.Header(titleOrient = "bottom", labelOrient = "bottom")),
        tooltip = ["Class", "Metric", "Probability"],
        ).properties(title = "Model Prediction vs. Ground Truth").interactive()
    
    st.altair_chart(chart, use_container_width = True)
    

