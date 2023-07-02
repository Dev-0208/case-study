"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""
import logging
import os.path as op

from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH
)

from ta_lib.data_processing.api import Outlier

logger = logging.getLogger(__name__)


@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """Transform dataset to create training datasets."""

    input_features_ds = "train/sales/features"
    input_target_ds = "train/sales/target"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    cat_columns = train_X.select_dtypes("object").columns
    num_columns = train_X.select_dtypes("number").columns

    # Treating Outliers
    outlier_transformer = Outlier(method=params["outliers"]["method"])
    train_X = outlier_transformer.fit_transform(
        train_X, drop=params["outliers"]["drop"]
    )

    

   


    # Train the feature engg. pipeline prepared earlier. Note that the pipeline is
    # fitted on only the **training data** and not the full dataset.
    # This avoids leaking information about the test dataset when training the model.
    # In the below code train_X, train_y in the fit_transform can be replaced with
    # sample_X and sample_y if required. 
    #train_X = get_dataframe(
        #features_transformer.fit_transform(train_X, train_y),
        #get_feature_names_from_column_transformer(features_transformer),
    #)

    # Note: we can create a transformer/feature selector that simply drops
    # a specified set of columns. But, we don't do that here to illustrate
    # what to do when transformations don't cleanly fall into the sklearn
    # pattern.
    curated_columns = list(
        set(train_X.columns.to_list())
        - set(
            [
              
            ]
        )
    )

    # saving the list of relevant columns and the pipeline.
    save_pipeline(
        curated_columns, op.abspath(op.join(artifacts_folder, "curated_columns.joblib"))
    )
    #save_pipeline(
     #   features_transformer, op.abspath(op.join(artifacts_folder, "features.joblib"))
    #)