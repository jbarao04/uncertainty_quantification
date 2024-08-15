# Uncertainty Quantification
Simple and effective implementation of different UQ methods

This repository is an easy way to start using uncertainty quantification on your ML model. Each of the notebooks applies a different method to the 'customer purchase' dataset with a state-of-the-art machine learning model.

## Example Notebooks
- [Delete-d Jackknife](./notebooks/Delete-d_Jackknife.ipynb): The Delete-d Jackknife is designed to provide prediction intervals using sampling techniques. This approach follows the methodology described in Giordano (2019), Wager & Hastie & Efron (2014) and Shao & Wun (1989)
- [Ensemble Methods](./notebooks/EnsembleMethods.ipynb): Ensemble Methods are important techniques can be adapted and used to provide Uncertainty Quantification through Prediction intervals.
- [Black-Box MetaModel](./notebooks/BB_MetaModel.ipynb): The Metamodel Classification is designed to provide confidence scores for predictions made by black-box classification models using a meta-model approach. This approach follows the methodology described in Chen et al. (2019)

## Dataset

The dataset used was [customer_purchase](./data/customer_purchase_data.csv)
